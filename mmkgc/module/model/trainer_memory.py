import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model

import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import random
from .memory_bank import RelationAwareMultiModalMemoryBank


class RAMMMTrainerKuai16K(object):
    """
    RAMMM = Relation-Aware Multi-Modal Memory
    用 Relation-aware Multi-modal Memory Bank 替代原 WCGTrainerKuai16KGP 中的 GAN generator。
    """

    def __init__(self,
                 model=None,
                 data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="adam",
                 save_steps=100,
                 checkpoint_dir=None,
                 tester=None,
                 memory_bank=None,
                 memory_size_per_relation=512,
                 memory_neg_num=2,
                 warmup_epochs=1,
                 mu=0.5):
        """
        Args:
            model:
                一般是 NegativeSamplingGP(...)
            data_loader:
                TrainDataLoader
            train_times:
                epoch数
            alpha:
                学习率
            use_gpu:
                是否用GPU
            opt_method:
                目前默认Adam
            save_steps:
                验证步长
            checkpoint_dir:
                checkpoint路径
            tester:
                Tester(...)
            memory_bank:
                外部传入MemoryBank；若为None则内部自动创建
            memory_size_per_relation:
                每个relation保留多少条memory
            memory_neg_num:
                每个样本从memory中取多少个hard negatives
            warmup_epochs:
                前几个epoch只用随机负采样，先把memory攒起来
            mu:
                memory loss权重
        """
        self.work_threads = 8
        self.train_times = train_times
        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir
        self.tester = tester

        self.batch_size = self.model.batch_size
        self.memory_neg_num = memory_neg_num
        self.warmup_epochs = warmup_epochs
        self.mu = mu

        rel_tot = self.model.model.rel_tot
        device = "cuda" if use_gpu else "cpu"

        if memory_bank is None:
            self.memory_bank = RelationAwareMultiModalMemoryBank(
                rel_tot=rel_tot,
                capacity_per_relation=memory_size_per_relation,
                device=device
            )
        else:
            self.memory_bank = memory_bank

    def to_var(self, x, use_gpu):
        if torch.is_tensor(x):
            if use_gpu:
                return x.cuda()
            return x
        else:
            if use_gpu:
                return Variable(torch.from_numpy(x).cuda())
            else:
                return Variable(torch.from_numpy(x))

    def _get_positive_batch(self, data):
        """
        从原 dataloader 输出中取正样本部分。
        原工程中通常前 batch_size 个是 positive，后面是 negative。
        """
        batch_h = self.to_var(data['batch_h'], self.use_gpu)
        batch_t = self.to_var(data['batch_t'], self.use_gpu)
        batch_r = self.to_var(data['batch_r'], self.use_gpu)

        pos_h = batch_h[:self.batch_size]
        pos_t = batch_t[:self.batch_size]
        pos_r = batch_r[:self.batch_size]
        return pos_h, pos_t, pos_r

    def _extract_multimodal_embeddings(self, ent_ids):
        """
        依赖原模型:
            self.model.model.get_batch_ent_multimodal_embs(ent_ids)

        假设返回:
            struct, img, text, audio, video
        """
        batch_s, batch_i, batch_t, batch_a, batch_v = \
            self.model.model.get_batch_ent_multimodal_embs(ent_ids)
        return batch_s, batch_i, batch_t, batch_a, batch_v

    def _build_memory_samples_from_positive_batch(self, pos_h, pos_t, pos_r, pos_score):
        """
        从当前正样本batch构造memory item。
        存真实实体的多模态表征，供后续relation-aware检索。
        """
        h_s, h_i, h_tx, h_a, h_v = self._extract_multimodal_embeddings(pos_h)
        t_s, t_i, t_tx, t_a, t_v = self._extract_multimodal_embeddings(pos_t)

        samples = []
        batch_size = pos_h.shape[0]
        for i in range(batch_size):
            item = {
                'h': pos_h[i].detach().cpu(),
                't': pos_t[i].detach().cpu(),
                'r': pos_r[i].detach().cpu(),

                'h_struct': h_s[i].detach().cpu(),
                't_struct': t_s[i].detach().cpu(),

                'h_img': h_i[i].detach().cpu(),
                't_img': t_i[i].detach().cpu(),

                'h_text': h_tx[i].detach().cpu(),
                't_text': t_tx[i].detach().cpu(),

                'h_audio': h_a[i].detach().cpu(),
                't_audio': t_a[i].detach().cpu(),

                'h_video': h_v[i].detach().cpu(),
                't_video': t_v[i].detach().cpu(),

                'score': float(pos_score[i].detach().cpu().item())
            }
            samples.append(item)
        return samples

    def _update_memory_bank(self, pos_h, pos_t, pos_r, pos_score):
        samples = self._build_memory_samples_from_positive_batch(
            pos_h, pos_t, pos_r, pos_score
        )
        self.memory_bank.push(pos_r, samples)

    def _sample_memory_negatives(self, pos_h, pos_t, pos_r):
        """
        对每个正样本，按 relation 从 memory 中采样 hard negatives。
        简化策略：
            从相同 relation 的 memory 中采样若干条样本，
            用其中的 head 或 tail 替换当前三元组，构造 hard negatives。

        返回:
            mem_h, mem_t, mem_r
        shape:
            [B * memory_neg_num]
        """
        sampled = self.memory_bank.sample(pos_r, num_samples=self.memory_neg_num)

        mem_h_list = []
        mem_t_list = []
        mem_r_list = []

        for i, candidates in enumerate(sampled):
            if len(candidates) == 0:
                continue

            cur_h = pos_h[i].detach().cpu()
            cur_t = pos_t[i].detach().cpu()
            cur_r = pos_r[i].detach().cpu()

            for item in candidates:
                # 交替替换 head / tail，构造 relation-aware negatives
                if random.random() < 0.5:
                    # replace tail
                    mem_h_list.append(cur_h.clone())
                    mem_t_list.append(item['t'].clone())
                    mem_r_list.append(cur_r.clone())
                else:
                    # replace head
                    mem_h_list.append(item['h'].clone())
                    mem_t_list.append(cur_t.clone())
                    mem_r_list.append(cur_r.clone())

        if len(mem_h_list) == 0:
            return None, None, None

        mem_h = torch.stack(mem_h_list, dim=0).long()
        mem_t = torch.stack(mem_t_list, dim=0).long()
        mem_r = torch.stack(mem_r_list, dim=0).long()

        if self.use_gpu:
            mem_h = mem_h.cuda()
            mem_t = mem_t.cuda()
            mem_r = mem_r.cuda()

        return mem_h, mem_t, mem_r

    def _calc_memory_loss(self, pos_h, pos_t, pos_r, mem_h, mem_t, mem_r, mode='normal'):
        """
        Memory loss:
            希望 positive score > memory negative score

        这里不再走 GAN fake_score，
        而是直接调用 模型的普通打分接口。

        你需要确保底层模型有以下函数之一：
            1) self.model.model.predict_score(batch_h, batch_r, batch_t, mode)
            或
            2) self.model.model.forward({...}) 返回 score
            或
            3) self.model.model.cal_triplet_score(batch_h, batch_r, batch_t, mode)

        我这里优先按 predict_score 写。
        """
        # 正样本评分
        pos_data = {
            'batch_h': pos_h,  # numpy.ndarray
            'batch_t': pos_t,  # numpy.ndarray
            'batch_r': pos_r,  # numpy.ndarray
            'mode': mode       # str
        }
        pos_score = self.model.model.forward(pos_data)

        # 记忆样本评分
        mem_data = {
            'batch_h': mem_h,  # numpy.ndarray
            'batch_t': mem_t,  # numpy.ndarray
            'batch_r': mem_r,  # numpy.ndarray
            'mode': mode       # str
        }
        mem_score = self.model.model.forward(mem_data)
        # pos_score = self.model.model.forward(
        #     batch_h=pos_h,
        #     batch_r=pos_r,
        #     batch_t=pos_t,
        #     mode=mode
        # )

        # mem_score = self.model.model.forward(
        #     batch_h=mem_h,
        #     batch_r=mem_r,
        #     batch_t=mem_t,
        #     mode=mode
        # )

        # margin ranking style
        # 希望 pos_score 更大
        # 如果你原模型是 “越小越好”，这里把符号改掉即可
        margin = self.model.model.margin
        loss_mem = torch.relu(margin - pos_score.mean() + mem_score.mean())
        return loss_mem, pos_score, mem_score

    def train_one_step(self, data, epoch):
        self.optimizer.zero_grad()

        # -----------------------------
        # 1) 原始随机负采样loss
        # -----------------------------
        loss, p_score, real_embs = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })

        # -----------------------------
        # 2) 取正样本，更新 memory
        # -----------------------------
        pos_h, pos_t, pos_r = self._get_positive_batch(data)

        # p_score 默认应对应正样本分数；如果 p_score 包含更多内容，可截前 batch_size
        if p_score.shape[0] > self.batch_size:
            pos_score_for_memory = p_score[:self.batch_size]
        else:
            pos_score_for_memory = p_score

        self._update_memory_bank(pos_h, pos_t, pos_r, pos_score_for_memory)

        # -----------------------------
        # 3) warmup后再引入 memory negatives
        # -----------------------------
        loss_mem = torch.tensor(0.0).cuda() if self.use_gpu else torch.tensor(0.0)

        if epoch >= self.warmup_epochs:
            mem_h, mem_t, mem_r = self._sample_memory_negatives(pos_h, pos_t, pos_r)

            if mem_h is not None:
                loss_mem, _, _ = self._calc_memory_loss(
                    pos_h=pos_h,
                    pos_t=pos_t,
                    pos_r=pos_r,
                    mem_h=mem_h,
                    mem_t=mem_t,
                    mem_r=mem_r,
                    mode=data['mode']
                )
                loss = loss + self.mu * loss_mem

        loss.backward()
        self.optimizer.step()

        return loss.item(), float(loss_mem.detach().cpu().item())

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
            print("Learning Rate: {}".format(self.alpha))
        else:
            raise NotImplementedError

        print("Finish initializing RAMMM trainer...")

        for epoch in range(self.train_times):
            res = 0.0
            res_mem = 0.0

            for data in self.data_loader:
                loss, loss_mem = self.train_one_step(data, epoch)
                res += loss
                res_mem += loss_mem

            print("Epoch {} | total loss: {:.6f}, memory loss: {:.6f}, memory size: {}".format(
                epoch, res, res_mem, len(self.memory_bank))
            )

            if self.save_steps and (epoch + 1) % self.save_steps == 0:
                print("Epoch {} has finished, validate...".format(epoch))
                if self.tester is not None:
                    self.tester.run_link_prediction(type_constrain=False)

    def set_model(self, model):
        self.model = model

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if checkpoint_dir is not None:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
