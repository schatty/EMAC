import time
import numpy as np
import torch as t

MEM_DTYPE = t.float16


class MemBuffer:

    def __init__(self, state_dim, action_dim, capacity, k, mem_dim,
                 cosine=False,
                 device="cuda:1"):
        self.max_size = capacity
        self.ptr = 0
        self.size = 0
        self.k = k

        self.sa_cuda = t.zeros(capacity, mem_dim, dtype=MEM_DTYPE).to(device)
        self.q = np.zeros((capacity, 1))
        self.cosine = cosine
        self.device = device

        self.mapping_cpu = np.random.randn(state_dim + action_dim, mem_dim)
        self.mapping = t.from_numpy(self.mapping_cpu).to(self.device, dtype=MEM_DTYPE)

        self.cos = t.nn.CosineSimilarity(dim=2)

    def store(self, state, action, q):
        sa = np.concatenate([state, action], axis=0).reshape(1, -1)
        sa = np.dot(sa, self.mapping_cpu)

        self.sa_cuda[self.ptr] = t.from_numpy(sa)
        self.q[self.ptr] = q

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def _calc_l2_dist(self, v1, v2):
        v2.unsqueeze_(1)
        l2 = t.pow(t.sum(t.pow(t.abs(v1 - v2), 2), dim=-1), 0.5)
        return l2

    def _calc_cosine_dist(self, v1, v2):
        v1_ = v1.unsqueeze(0)
        v2_ = v2.unsqueeze(1)

        d = self.cos(v1_, v2_)

        return d

    def retrieve_cuda(self, states, actions, k=None):
        if k is None:
            k = self.k

        sa = t.cat([states, actions], dim=1).to(self.device, dtype=MEM_DTYPE)
        sa = t.mm(sa, self.mapping)

        # TODO: Bug here, I take only first self.size elements
        if self.cosine:
            dists_all = self._calc_cosine_dist(self.sa_cuda[:self.size], sa)
            soft = t.nn.Softmax(dim=1)
        else:
            dists_all = self._calc_l2_dist(self.sa_cuda[:self.size], sa)
            soft = t.nn.Softmin(dim=1)
        dists, inds = t.topk(dists_all, k, dim=1, largest=False)

        #softmin = t.nn.Softmin(dim=1)
        weights = soft(dists)

        inds = inds.cpu().numpy()
        weights = weights.cpu().numpy()

        qs = self.q[inds]
        weights = np.expand_dims(weights, 2)
        qs = np.multiply(qs, weights) 
        qs = np.sum(qs, axis=1)

        return qs

    def save(self, file_name):
        np.save(f"{file_name}.npy", {
            "sa": self.sa,
            "q": self.q,
        })
        print("Memory module saved.")
