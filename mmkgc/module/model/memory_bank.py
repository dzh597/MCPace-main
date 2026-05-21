import torch
import torch.autograd as autograd
import torch.nn as nn
from .Model import Model


import random
from collections import defaultdict, deque
import torch


class RelationAwareMultiModalMemoryBank(object):
    """
    Relation-aware Multi-modal Memory Bank

    每个 relation 对应一个 memory queue:
        memory[r] = deque([sample1, sample2, ...], maxlen=capacity_per_relation)

    每个 sample 里存:
        {
            'h': LongTensor scalar,
            't': LongTensor scalar,
            'r': LongTensor scalar,
            'h_struct': Tensor [dim]
            't_struct': Tensor [dim]
            'h_img': Tensor [dim]
            't_img': Tensor [dim]
            'h_text': Tensor [dim]
            't_text': Tensor [dim]
            'h_audio': Tensor [dim]
            't_audio': Tensor [dim]
            'h_video': Tensor [dim]
            't_video': Tensor [dim]
            'score': float
        }
    """

    def __init__(self, rel_tot, capacity_per_relation=512, device="cuda"):
        self.rel_tot = rel_tot
        self.capacity_per_relation = capacity_per_relation
        self.device = device
        self.memory = {
            r: deque(maxlen=capacity_per_relation) for r in range(rel_tot)
        }

    def __len__(self):
        total = 0
        for r in self.memory:
            total += len(self.memory[r])
        return total

    def clear(self):
        for r in self.memory:
            self.memory[r].clear()

    def push(self, relation_ids, samples):
        """
        relation_ids: list[int] or Tensor [B]
        samples: list[dict], len == B
        """
        if torch.is_tensor(relation_ids):
            relation_ids = relation_ids.detach().cpu().tolist()

        for rid, sample in zip(relation_ids, samples):
            self.memory[int(rid)].append(sample)

    def sample(self, relation_ids, num_samples=1):
        """
        对每个 relation，采样 num_samples 个 memory negatives
        返回:
            sampled_list: list[list[dict]]
                外层长度 = batch_size
                每个元素是该 relation 采样到的样本列表
        """
        if torch.is_tensor(relation_ids):
            relation_ids = relation_ids.detach().cpu().tolist()

        sampled_list = []
        for rid in relation_ids:
            bank = self.memory[int(rid)]
            if len(bank) == 0:
                sampled_list.append([])
                continue

            if len(bank) >= num_samples:
                sampled = random.sample(list(bank), num_samples)
            else:
                sampled = random.choices(list(bank), k=num_samples)
            sampled_list.append(sampled)

        return sampled_list

    def has_relation_memory(self, relation_id):
        return len(self.memory[int(relation_id)]) > 0
