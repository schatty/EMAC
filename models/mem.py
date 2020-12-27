import numpy as np
import torch as t


class MemBuffer:

    def __init__(self, state_dim, action_dim, capacity, k, mem_dim,
                 device="cuda:1"):
        self.max_size = capacity
        self.ptr = 0
        self.size = 0
        self.k = k

        self.sa_cuda = t.zeros(capacity, mem_dim).float().to(device)
        self.q = np.zeros((capacity, 1))
        self._prev_cuda_ptr = 0
        self.device = device

        self._mem_mapper = np.random.randn(state_dim + action_dim, mem_dim)

    def store(self, state, action, q):
        sa = np.concatenate([state, action], axis=0).reshape(1, -1)
        sa = np.dot(sa, self._mem_mapper)

        self.sa_cuda[self.ptr] = t.from_numpy(sa).float()
        self.q[self.ptr] = q

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def retrieve_cuda(self, states, actions, step):
        sa = t.cat([states, actions], dim=1).to(self.device)
        mapping = t.from_numpy(self._mem_mapper).float().to(self.device)
        sa = t.mm(sa, mapping)
        sa.unsqueeze_(1)

        # TODO: Bug here, I take only first self.size elements
        l2 = t.pow(t.sum(t.pow(t.abs(self.sa_cuda[:self.size] - sa), 2), dim=-1), 0.5)
        dists, inds = t.topk(l2, self.k, dim=1, largest=False)

        softmin = t.nn.Softmin(dim=1)
        weights = softmin(dists)

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
