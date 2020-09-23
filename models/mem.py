import numpy as np


class MemBuffer:

    def __init__(self, state_dim, action_dim, capacity):
        self.max_size = capacity
        self.ptr = 0
        self.size = 0

        self.sa = np.zeros((capacity, state_dim + action_dim))
        self.q = np.zeros((capacity, 1))

    def store(self, state, action, q):
        sa = np.concatenate([state, action], axis=0)
        self.sa[self.ptr] = sa
        self.q[self.ptr] = q

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def retrieve(self, state, action):
        sa = np.concatenate([state, action], axis=0)
        l2 = np.linalg.norm(self.sa[:self.size] - sa, axis=1)
        min_i = np.argmin(l2)
        return self.q[min_i]

    def save(self, file_name):
        np.save(f"{file_name}.npy", {
            "sa": self.sa,
            "q": self.q,
        })
        print("Memory module saved.")
