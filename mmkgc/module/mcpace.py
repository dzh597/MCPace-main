import torch


class MCPaceCoordinator(object):
    """
    Gradient-space modality coordinator for MCPace.

    This first implementation is intentionally scoped to the three modalities
    requested for DB15K: structure, visual, textual. It reads the modal tensors
    saved by the KGE model, computes their gradients w.r.t. the final KGC/D loss,
    updates RACE energy, and replaces the upstream modal gradients through a
    delta-backward correction.
    """

    def __init__(
        self,
        rel_tot,
        modalities=None,
        mu=0.1,
        num_blocks=4,
        lambda_alpha=1.0,
        eps=1e-8,
        min_rebalance=0.2,
        max_rebalance=5.0,
        log_interval=0,
    ):
        self.modalities = modalities or ["structure", "visual", "textual"]
        self.rel_tot = rel_tot
        self.mu = mu
        self.num_blocks = num_blocks
        self.lambda_alpha = lambda_alpha
        self.eps = eps
        self.min_rebalance = min_rebalance
        self.max_rebalance = max_rebalance
        self.log_interval = log_interval

        self.global_energy = None
        self.relation_energy = None
        self.step = 0
        self.last_stats = {}

    def _ensure_energy(self, device, dtype):
        if self.global_energy is None or self.global_energy.device != device:
            self.global_energy = torch.zeros(
                len(self.modalities), device=device, dtype=dtype
            )
            self.relation_energy = torch.zeros(
                self.rel_tot, len(self.modalities), device=device, dtype=dtype
            )

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

    def _cosine(self, x, y):
        return torch.dot(x, y) / (torch.norm(x) * torch.norm(y) + self.eps)

    def _update_energy(self, modal_grads, batch_r, lr):
        first_mod = self.modalities[0]
        device = modal_grads[first_mod][0].device
        dtype = modal_grads[first_mod][0].dtype
        self._ensure_energy(device, dtype)

        rel_ids = batch_r.detach().long().view(-1).to(device)
        if rel_ids.numel() == 0:
            return

        lr2 = float(lr) * float(lr)
        with torch.no_grad():
            for idx, name in enumerate(self.modalities):
                h_grad, t_grad = modal_grads[name]
                sample_energy = (
                    h_grad.pow(2).sum(dim=-1) + t_grad.pow(2).sum(dim=-1)
                ) * lr2
                self.global_energy[idx] += sample_energy.sum()
                self.relation_energy[:, idx].index_add_(0, rel_ids, sample_energy)

    def _compute_pair_stats(self, flat_grads):
        pair_cos = {}
        balance = {}
        energy = self.global_energy + self.eps

        for i, mi in enumerate(self.modalities):
            for j, mj in enumerate(self.modalities):
                if i == j:
                    continue
                if (mj, mi) in pair_cos:
                    pair_cos[(mi, mj)] = pair_cos[(mj, mi)]
                    balance[(mi, mj)] = balance[(mj, mi)]
                    continue
                c = self._cosine(flat_grads[mi], flat_grads[mj])
                wi = energy[i] / (energy[i] + energy[j] + self.eps)
                wj = 1.0 - wi
                w_bar = 8.0 * wi * wj - 1.0
                pair_cos[(mi, mj)] = c
                balance[(mi, mj)] = w_bar
        return pair_cos, balance

    def _compute_rho(self, pair_cos):
        rho = {}
        for mi in self.modalities:
            denom = None
            vals = {}
            for mj in self.modalities:
                if mi == mj:
                    continue
                val = torch.abs(pair_cos[(mi, mj)]) + self.eps
                vals[mj] = val
                denom = val if denom is None else denom + val
            for mj, val in vals.items():
                rho[(mi, mj)] = val / (denom + self.eps)
        return rho

    def _compute_rebalance(self, pair_cos, rho):
        energy = self.global_energy + self.eps
        total = torch.sum(energy)
        num_modalities = float(len(self.modalities))
        rebalance = {}

        for i, mi in enumerate(self.modalities):
            base = total / (num_modalities * energy[i] + self.eps)
            base = torch.clamp(base, self.min_rebalance, self.max_rebalance)
            c_bar = 0.0
            for mj in self.modalities:
                if mi == mj:
                    continue
                c_bar = c_bar + rho[(mi, mj)] * pair_cos[(mi, mj)]
            exponent = (1.0 + c_bar) / 2.0
            r_tilde = torch.pow(base, exponent)
            rebalance[mi] = torch.clamp(
                r_tilde, self.min_rebalance, self.max_rebalance
            )
        return rebalance

    def _cooperative_candidates(self, gi, gj, ri, rj, w_bar):
        chunks_i = torch.chunk(gi, self.num_blocks)
        chunks_j = torch.chunk(gj, self.num_blocks)
        zi_chunks = []
        zj_chunks = []
        q = 4.0 * w_bar

        for bi, bj in zip(chunks_i, chunks_j):
            ci = self._cosine(bi, bj)
            a = torch.clamp(ci, min=0.0)
            xi = torch.norm(bi)
            xj = torch.norm(bj)

            a11 = a * xi * xi + self.lambda_alpha
            a12 = -a * xi * xj
            a21 = a12
            a22 = a * xj * xj + self.lambda_alpha
            det = a11 * a22 - a12 * a21 + self.eps

            alpha_i = self.lambda_alpha * (a22 - a12) / det
            alpha_j = self.lambda_alpha * (a11 - a21) / det
            alpha_i = torch.clamp(alpha_i, 0.0, self.max_rebalance)
            alpha_j = torch.clamp(alpha_j, 0.0, self.max_rebalance)

            proj_i = alpha_i * bi
            proj_j = alpha_j * bj
            gamma = torch.sigmoid(q * ci)

            zi_chunks.append(ri * (gamma * proj_i + (1.0 - gamma) * bi))
            zj_chunks.append(rj * (gamma * proj_j + (1.0 - gamma) * bj))

        return torch.cat(zi_chunks, dim=0), torch.cat(zj_chunks, dim=0)

    def _conflict_candidates(self, gi, gj, ri, rj, c, w_bar, ei, ej):
        s = torch.dot(gi, gj)
        wi = ei / (ei + ej + self.eps)
        wj = ej / (ei + ej + self.eps)
        denom = (
            torch.norm(gj).pow(2) / (wi + self.eps)
            + torch.norm(gi).pow(2) / (wj + self.eps)
            + self.eps
        )
        oi = -s / denom * gj / (wi + self.eps)
        oj = -s / denom * gi / (wj + self.eps)
        gamma_g = torch.sigmoid(4.0 * w_bar * c)
        zi = ri * gi + (1.0 - gamma_g) * oi
        zj = rj * gj + (1.0 - gamma_g) * oj
        return zi, zj

    def _coordinate(self, modal_grads):
        flat_grads = {
            name: self._flatten_pair(modal_grads[name][0], modal_grads[name][1])
            for name in self.modalities
        }
        pair_cos, balance = self._compute_pair_stats(flat_grads)
        rho = self._compute_rho(pair_cos)
        rebalance = self._compute_rebalance(pair_cos, rho)
        energy = self.global_energy + self.eps

        candidates = {name: [] for name in self.modalities}
        for i, mi in enumerate(self.modalities):
            for j in range(i + 1, len(self.modalities)):
                mj = self.modalities[j]
                c = pair_cos[(mi, mj)]
                w_bar = balance[(mi, mj)]
                if c >= 0:
                    zi, zj = self._cooperative_candidates(
                        flat_grads[mi], flat_grads[mj],
                        rebalance[mi], rebalance[mj], w_bar
                    )
                else:
                    zi, zj = self._conflict_candidates(
                        flat_grads[mi], flat_grads[mj],
                        rebalance[mi], rebalance[mj], c, w_bar,
                        energy[i], energy[j]
                    )
                candidates[mi].append((mj, zi))
                candidates[mj].append((mi, zj))

        modified = {}
        for mi in self.modalities:
            base = rebalance[mi] * flat_grads[mi]
            pair_sum = torch.zeros_like(base)
            for mj, z in candidates[mi]:
                pair_sum = pair_sum + rho[(mi, mj)] * z
            modified[mi] = (base + self.mu * pair_sum) / (1.0 + self.mu)

        self.last_stats = {
            "energy": {
                name: float(self.global_energy[i].detach().cpu())
                for i, name in enumerate(self.modalities)
            },
            "rebalance": {
                name: float(rebalance[name].detach().cpu())
                for name in self.modalities
            },
            "pair_cos": {
                "{}-{}".format(a, b): float(v.detach().cpu())
                for (a, b), v in pair_cos.items()
                if self.modalities.index(a) < self.modalities.index(b)
            },
        }
        return modified

    def backward(self, loss, modal_context, lr):
        """
        Apply MCPace to `loss`.

        The function first obtains gradients w.r.t. saved modal tensors, then
        performs normal loss.backward() for all parameters, and finally applies
        delta-backward on modal tensors so that upstream modal branches receive
        MCPace-modified gradients.
        """
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
        grad_idx = 0
        for name, h_tensor, t_tensor in tensor_meta:
            h_grad = self._safe_grad(raw_grads[grad_idx], h_tensor)
            t_grad = self._safe_grad(raw_grads[grad_idx + 1], t_tensor)
            grad_idx += 2
            modal_grads[name] = (h_grad, t_grad)

        self._update_energy(modal_grads, modal_context["batch_r"], lr)
        modified_flat = self._coordinate(modal_grads)

        # Original gradients for all model parameters.
        loss.backward(retain_graph=True)

        # Delta-backward replaces modal gradients by adding (g_mod - g_orig).
        delta_tensors = []
        for name, h_tensor, t_tensor in tensor_meta:
            h_grad, t_grad = modal_grads[name]
            h_numel = h_grad.numel()
            h_mod, t_mod = self._split_pair(
                modified_flat[name],
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
            print("[MCPace] step {} stats: {}".format(self.step, self.last_stats))
