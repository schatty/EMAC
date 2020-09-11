import time
import gym

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cuda"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, file_name):
        np.save(f"{file_name}.npy", {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "not_done": self.not_done
        })


class EpisodicReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cuda"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.ep_state = []
        self.ep_action = []
        self.ep_next_state = []
        self.ep_reward = []

        self.device = device

    def _add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add(self, state, action, next_state, reward, done):
        self.ep_state.append(state)
        self.ep_action.append(action)
        self.ep_next_state.append(next_state)
        self.ep_reward.append(reward)

        if done == True:
            dones = [0] * (len(self.ep_state) - 1) + [1]
            for s, a, ns, r, d in zip(self.ep_state, self.ep_action,
                                      self.ep_next_state, self.ep_reward, dones):
                self._add(s, a, ns, r, d)

            self.ep_state.clear()
            self.ep_action.clear()
            self.ep_next_state.clear()
            self.ep_reward.clear()

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, file_name):
        np.save(f"{file_name}.npy", {
            "state": self.state,
            "action": self.action,
            "next_state": self.next_state,
            "reward": self.reward,
            "not_done": self.not_done
        })


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def estimate_true_q(policy, env_name, discount, buffer, eval_episodes=1000):
    t1 = time.time()
    eval_env = gym.make(env_name)

    qs = []
    for _ in range(eval_episodes):
        eval_env.reset()

        ind = np.random.choice(buffer.size)
        state = buffer.next_state[ind]
        reward = buffer.reward[ind]

        qpos = state[:eval_env.model.nq-1]
        qvel = state[eval_env.model.nq-1:]
        qpos = np.concatenate([[0], qpos])

        eval_env.set_state(qpos, qvel)

        q = reward
        s_i = 1
        while True:
            action = policy.select_action(np.array(state))
            state, r, d, _ = eval_env.step(action)
            q += r * discount ** s_i

            s_i += 1

            if d:
                break
        qs.append(q)

    print("Estimation took: ", time.time() - t1)

    return np.mean(qs)
