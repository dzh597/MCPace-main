import torch
import torch.nn.functional as F


class GradientCoordinator(object):
    """Base interface for general-purpose gradient coordination baselines.

    All coordinators consume a tensor ``G`` with shape [M, D], where M is the
    number of coordinated modalities/tasks and D is the flattened head+tail
    representation-gradient dimension. They return per-modality gradients with
    the same shape, so the sum of returned rows corresponds to the final update
    direction used by the method.
    """

    def coordinate(self, grads, losses=None, **kwargs):
        raise NotImplementedError


def project_to_simplex(v):
    """Project ``v`` onto the probability simplex {x | x >= 0, sum(x)=1}."""
    if v.numel() == 1:
        return torch.ones_like(v)

    sorted_v, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(sorted_v, dim=0) - 1.0
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = sorted_v - cssv / ind > 0

    if not torch.any(cond):
        return torch.ones_like(v) / v.numel()

    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return torch.clamp(v - theta, min=0.0)


def _avg_pairwise_cosine(G, eps=1e-12):
    vals = []
    M = G.shape[0]
    for i in range(M):
        for j in range(i + 1, M):
            cos = torch.dot(G[i], G[j]) / (
                torch.norm(G[i]) * torch.norm(G[j]) + eps
            )
            vals.append(cos)
    if not vals:
        return torch.zeros((), device=G.device, dtype=G.dtype)
    return torch.stack(vals).mean()


def _conflict_ratio(G, eps=1e-12):
    vals = []
    M = G.shape[0]
    for i in range(M):
        for j in range(i + 1, M):
            cos = torch.dot(G[i], G[j]) / (
                torch.norm(G[i]) * torch.norm(G[j]) + eps
            )
            vals.append((cos < 0).to(dtype=G.dtype))
    if not vals:
        return torch.zeros((), device=G.device, dtype=G.dtype)
    return torch.stack(vals).mean()


def _tensor_to_float_list(x):
    return [float(v) for v in x.detach().cpu().view(-1)]


class PCGradCoordinator(GradientCoordinator):
    """PCGrad: project away pairwise conflicting gradient components."""

    def __init__(self, eps=1e-12):
        self.eps = eps

    def coordinate(self, grads, losses=None, **kwargs):
        G = grads.detach()
        M = G.shape[0]
        G_pc = G.clone()

        for i in range(M):
            if M <= 1:
                continue
            others = [j for j in range(M) if j != i]
            perm = torch.randperm(len(others), device=G.device).tolist()
            for idx in perm:
                j = others[idx]
                dot = torch.dot(G_pc[i], G[j])
                if dot < 0:
                    G_pc[i] = G_pc[i] - dot / (torch.dot(G[j], G[j]) + self.eps) * G[j]

        return G_pc, {}


class CAGradCoordinator(GradientCoordinator):
    """CAGrad conflict-averse gradient coordination."""

    def __init__(self, c=0.4, num_iters=50, lr=0.1, eps=1e-12):
        self.c = c
        self.num_iters = num_iters
        self.lr = lr
        self.eps = eps

    def coordinate(self, grads, losses=None, **kwargs):
        G = grads.detach()
        M, _ = G.shape
        g0 = G.mean(dim=0)
        g0_norm = torch.norm(g0) + self.eps

        w = torch.ones(M, device=G.device, dtype=G.dtype) / M
        w.requires_grad_(True)

        for _ in range(self.num_iters):
            gw = torch.sum(w[:, None] * G, dim=0)
            obj = torch.dot(gw, g0) + self.c * g0_norm * torch.norm(gw)
            grad_w = torch.autograd.grad(obj, w, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                w -= self.lr * grad_w
                w.copy_(project_to_simplex(w))
            w.requires_grad_(True)

        with torch.no_grad():
            gw = torch.sum(w[:, None] * G, dim=0)
            scale = self.c * g0_norm / (torch.norm(gw) + self.eps)
            beta = torch.ones_like(w) / M + scale * w
            modified_grads = beta[:, None] * G

        return modified_grads, {"weights": beta.detach(), "simplex_weights": w.detach()}


class NashMTLCoordinator(GradientCoordinator):
    """Nash-MTL bargaining-based gradient coordination."""

    def __init__(self, num_iters=100, lr=0.05, eps=1e-8, normalize=True):
        self.num_iters = num_iters
        self.lr = lr
        self.eps = eps
        self.normalize = normalize

    def coordinate(self, grads, losses=None, **kwargs):
        G = grads.detach()
        M, _ = G.shape
        A = G @ G.T
        A = A + self.eps * torch.eye(M, device=G.device, dtype=G.dtype)

        u = torch.zeros(M, device=G.device, dtype=G.dtype, requires_grad=True)
        opt = torch.optim.Adam([u], lr=self.lr)

        for _ in range(self.num_iters):
            alpha = F.softplus(u) + self.eps
            residual = A @ alpha - 1.0 / alpha
            loss = (residual ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            alpha = F.softplus(u) + self.eps
            if self.normalize:
                d = torch.sum(alpha[:, None] * G, dim=0)
                target_norm = torch.norm(G.mean(dim=0)) + self.eps
                current_norm = torch.norm(d) + self.eps
                alpha = alpha * target_norm / current_norm
            modified_grads = alpha[:, None] * G

        return modified_grads, {"weights": alpha.detach()}


class AlignedMTLCoordinator(GradientCoordinator):
    """Aligned-MTL conditioning-based gradient alignment."""

    def __init__(self, eps=1e-8, normalize=True, clamp=True):
        self.eps = eps
        self.normalize = normalize
        self.clamp = clamp

    def coordinate(self, grads, losses=None, **kwargs):
        G = grads.detach()
        M, _ = G.shape
        gram = G @ G.T
        gram = gram + self.eps * torch.eye(M, device=G.device, dtype=G.dtype)

        eigvals, V = torch.linalg.eigh(gram)
        eigvals_clamped = torch.clamp(eigvals, min=self.eps)
        positive = eigvals[eigvals > self.eps]
        if positive.numel() > 0:
            lambda_min = positive.min()
        else:
            lambda_min = eigvals_clamped.min()

        scale_diag = torch.sqrt(lambda_min / eigvals_clamped)
        B = V @ torch.diag(scale_diag) @ V.T
        w = torch.ones(M, device=G.device, dtype=G.dtype) / M
        alpha = B @ w

        if self.clamp:
            alpha = torch.clamp(alpha, min=0.0)
            if alpha.sum() <= self.eps:
                alpha = torch.ones_like(alpha) / M

        if self.normalize:
            d = torch.sum(alpha[:, None] * G, dim=0)
            target_norm = torch.norm(G.mean(dim=0)) + self.eps
            current_norm = torch.norm(d) + self.eps
            alpha = alpha * target_norm / current_norm

        modified_grads = alpha[:, None] * G
        return modified_grads, {"weights": alpha.detach()}


class FairGradCoordinator(GradientCoordinator):
    """FairGrad alpha-fair gradient coordination."""

    def __init__(self, fair_alpha=2.0, num_iters=100, lr=0.05,
                 eps=1e-8, normalize=True):
        self.fair_alpha = fair_alpha
        self.num_iters = num_iters
        self.lr = lr
        self.eps = eps
        self.normalize = normalize

    def coordinate(self, grads, losses=None, **kwargs):
        G = grads.detach()
        M, _ = G.shape
        A = G @ G.T
        A = A + self.eps * torch.eye(M, device=G.device, dtype=G.dtype)

        if abs(self.fair_alpha) < 1e-6:
            w = torch.ones(M, device=G.device, dtype=G.dtype) / M
            modified_grads = w[:, None] * G
            return modified_grads, {"weights": w.detach()}

        u = torch.zeros(M, device=G.device, dtype=G.dtype, requires_grad=True)
        opt = torch.optim.Adam([u], lr=self.lr)
        power = -1.0 / self.fair_alpha

        for _ in range(self.num_iters):
            w = F.softplus(u) + self.eps
            residual = A @ w - torch.pow(w, power)
            loss = (residual ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            w = F.softplus(u) + self.eps
            if self.normalize:
                d = torch.sum(w[:, None] * G, dim=0)
                target_norm = torch.norm(G.mean(dim=0)) + self.eps
                current_norm = torch.norm(d) + self.eps
                w = w * target_norm / current_norm
            modified_grads = w[:, None] * G

        return modified_grads, {"weights": w.detach()}


class GradientBaselineBackwardCoordinator(object):
    """Adapter that exposes the same ``backward`` API as MCPaceCoordinator.

    It uses the current fused KGC/GAN loss as the common gradient source, just as
    the existing MCPace implementation does. Only the gradient-space coordinator
    is changed, which makes these baselines suitable as controlled comparisons.
    """

    def __init__(self, coordinator, modalities=None, name=None, eps=1e-12,
                 log_interval=0):
        self.coordinator = coordinator
        self.modalities = modalities or ["structure", "visual", "textual"]
        self.name = name or coordinator.__class__.__name__
        self.eps = eps
        self.log_interval = log_interval
        self.step = 0
        self.last_stats = {}

    def _safe_grad(self, grad, tensor):
        if grad is None:
            return torch.zeros_like(tensor).detach()
        return grad.detach()

    def _flatten_pair(self, h_grad, t_grad):
        return torch.cat([h_grad.reshape(-1), t_grad.reshape(-1)], dim=0)

    def _split_pair(self, flat_grad, h_shape, t_shape, h_numel):
        h = flat_grad[:h_numel].view(h_shape)
        t = flat_grad[h_numel:].view(t_shape)
        return h, t

    def _record_stats(self, G, modified_grads, info):
        weights = info.get("weights") if isinstance(info, dict) else None
        self.last_stats = {
            "method": self.name,
            "avg_cosine": float(_avg_pairwise_cosine(G, self.eps).detach().cpu()),
            "conflict_ratio": float(_conflict_ratio(G, self.eps).detach().cpu()),
            "grad_norms": _tensor_to_float_list(torch.norm(G, dim=1)),
            "modified_grad_norms": _tensor_to_float_list(torch.norm(modified_grads, dim=1)),
        }
        if weights is not None:
            self.last_stats["weights"] = _tensor_to_float_list(weights)

    def backward(self, loss, modal_context, lr=None):
        modal_tensors = []
        tensor_meta = []
        for name in self.modalities:
            h_tensor, t_tensor = modal_context["modalities"][name]
            modal_tensors.extend([h_tensor, t_tensor])
            tensor_meta.append((name, h_tensor, t_tensor))

        raw_grads = torch.autograd.grad(
            loss,
            modal_tensors,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        modal_grads = {}
        flat_grads = []
        grad_idx = 0
        for name, h_tensor, t_tensor in tensor_meta:
            h_grad = self._safe_grad(raw_grads[grad_idx], h_tensor)
            t_grad = self._safe_grad(raw_grads[grad_idx + 1], t_tensor)
            grad_idx += 2
            modal_grads[name] = (h_grad, t_grad)
            flat_grads.append(self._flatten_pair(h_grad, t_grad))

        G = torch.stack(flat_grads, dim=0).detach()
        modified_grads, info = self.coordinator.coordinate(G)
        modified_grads = modified_grads.detach()
        self._record_stats(G, modified_grads, info or {})

        # First apply the original loss gradient to all parameters, then add a
        # delta-backward through the modal tensors so their upstream branches see
        # the coordinated gradients instead of the raw gradients.
        loss.backward(retain_graph=True)

        delta_tensors = []
        for row_idx, (name, h_tensor, t_tensor) in enumerate(tensor_meta):
            h_grad, t_grad = modal_grads[name]
            h_numel = h_grad.numel()
            h_mod, t_mod = self._split_pair(
                modified_grads[row_idx],
                h_grad.shape,
                t_grad.shape,
                h_numel,
            )
            delta_tensors.extend([
                (h_mod - h_grad).detach(),
                (t_mod - t_grad).detach(),
            ])

        torch.autograd.backward(
            modal_tensors,
            grad_tensors=delta_tensors,
            retain_graph=False,
        )

        self.step += 1
        if self.log_interval and self.step % self.log_interval == 0:
            print("[{}] step {} stats: {}".format(self.name, self.step, self.last_stats))


def build_gradient_baseline(method, modalities=None, eps=1e-8, log_interval=0,
                            cagrad_c=0.4, cagrad_iters=50, cagrad_lr=0.1,
                            nash_iters=100, nash_lr=0.05, nash_normalize=True,
                            aligned_normalize=True, aligned_clamp=True,
                            fair_alpha=2.0, fair_iters=100, fair_lr=0.05,
                            fair_normalize=True):
    method = (method or "none").lower()
    if method in ("none", "", "null"):
        return None
    if method == "pcgrad":
        core = PCGradCoordinator(eps=eps)
    elif method == "cagrad":
        core = CAGradCoordinator(c=cagrad_c, num_iters=cagrad_iters,
                                 lr=cagrad_lr, eps=eps)
    elif method in ("nash", "nashmtl", "nash-mtl"):
        core = NashMTLCoordinator(num_iters=nash_iters, lr=nash_lr,
                                  eps=eps, normalize=bool(nash_normalize))
    elif method in ("aligned", "alignedmtl", "aligned-mtl"):
        core = AlignedMTLCoordinator(eps=eps, normalize=bool(aligned_normalize),
                                     clamp=bool(aligned_clamp))
    elif method == "fairgrad":
        core = FairGradCoordinator(fair_alpha=fair_alpha, num_iters=fair_iters,
                                   lr=fair_lr, eps=eps,
                                   normalize=bool(fair_normalize))
    else:
        raise ValueError("Unknown gradient baseline method: {}".format(method))

    return GradientBaselineBackwardCoordinator(
        coordinator=core,
        modalities=modalities,
        name=method,
        eps=eps,
        log_interval=log_interval,
    )


def build_gradient_baseline_from_args(args, modalities=None):
    return build_gradient_baseline(
        method=getattr(args, "grad_method", "none"),
        modalities=modalities,
        eps=getattr(args, "grad_eps", 1e-8),
        log_interval=getattr(args, "grad_log_interval", 0),
        cagrad_c=getattr(args, "cagrad_c", 0.4),
        cagrad_iters=getattr(args, "cagrad_iters", 50),
        cagrad_lr=getattr(args, "cagrad_lr", 0.1),
        nash_iters=getattr(args, "nash_iters", 100),
        nash_lr=getattr(args, "nash_lr", 0.05),
        nash_normalize=getattr(args, "nash_normalize", 1),
        aligned_normalize=getattr(args, "aligned_normalize", 1),
        aligned_clamp=getattr(args, "aligned_clamp", 1),
        fair_alpha=getattr(args, "fair_alpha", 2.0),
        fair_iters=getattr(args, "fair_iters", 100),
        fair_lr=getattr(args, "fair_lr", 0.05),
        fair_normalize=getattr(args, "fair_normalize", 1),
    )
