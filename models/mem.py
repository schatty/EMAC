import numpy as np

import torch as t
import torch.nn as nn


class MemBuffer:

    def __init__(self, state_dim, action_dim, capacity, mem_dim,
                 device="cuda:1"):
        self.max_size = capacity
        self.ptr = 0
        self.size = 0

        self.sa = np.zeros((capacity, mem_dim))
        self.sa_cuda = t.from_numpy(self.sa).float().to(device)
        self.q = np.zeros((capacity, 1))
        self._cuda_memory_update = 2000
        self._prev_cuda_ptr = 0
        self.device = device

        self._mem_mapper = np.random.randn(state_dim + action_dim, mem_dim)

    def store(self, state, action, q):
        sa = np.concatenate([state, action], axis=0).reshape(1, -1)
        sa = np.dot(sa, self._mem_mapper)

        self.sa[self.ptr] = sa
        self.q[self.ptr] = q

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def retrieve(self, state, action):
        sa = np.concatenate([state, action], axis=0)
        l2 = np.linalg.norm(self.sa[:self.size] - sa, axis=1)
        min_i = np.argmin(l2)
        return self.q[min_i]

    def retrieve_vec(self, states, actions):
        sa = np.concatenate([states, actions], axis=1)
        sa = np.dot(sa, self._mem_mapper)
        sa = np.expand_dims(sa, axis=1)

        l2 = np.linalg.norm(self.sa[:self.size] - sa, axis=2)
        min_inds = np.argmin(l2, axis=1)

        qs = self.q[min_inds]
        return qs

    def retrieve_cuda(self, states, actions, step):
        if step % self._cuda_memory_update == 0:
            print("Reallocating memory to CUDA...", self._prev_cuda_ptr, self.size)
            self.sa_cuda[self._prev_cuda_ptr:self.size] = \
                    t.from_numpy(self.sa[self._prev_cuda_ptr:self.size]).float().to(self.device)
            self._prev_cuda_ptr = self.size

        sa = t.cat([states, actions], dim=1).to(self.device)
        mapping = t.from_numpy(self._mem_mapper).float().to(self.device)
        sa = t.mm(sa, mapping)
        sa.unsqueeze_(1)

        l2 = t.pow(t.sum(t.pow(t.abs(self.sa_cuda[:self.size] - sa), 2), dim=-1), 0.5)
        min_inds = t.argmin(l2, dim=1).cpu().numpy()

        qs = self.q[min_inds]
        return qs

    def save(self, file_name):
        np.save(f"{file_name}.npy", {
            "sa": self.sa,
            "q": self.q,
        })
        print("Memory module saved.")
