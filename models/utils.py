import os
import json
import time
import gym

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(200_000),
            device="cuda", **kwargs):
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
    def __init__(self, state_dim, action_dim, mem,
                 max_size=int(1e6), device="cuda", prioritized=False, pr_alpha=0.0, 
                 start_timesteps=0, **kwargs):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.mem = mem
        self.expl_noise = kwargs["expl_noise"]

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.q = np.zeros((max_size, 1))
        self.p = np.ones(max_size)
        self.pr_alpha = pr_alpha

        self.ep_state = []
        self.ep_action = []
        self.ep_next_state = []
        self.ep_reward = []

        self.ep_length = 1000
        self.prioritized = prioritized
        self.start_timesteps = start_timesteps

        self.device = device

    def _add_replay_buffer(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add(self, state, action, next_state, reward, done_env, done_limit, env, policy, step):
        self.ep_state.append(state)
        self.ep_action.append(action)
        self.ep_next_state.append(next_state)
        self.ep_reward.append(reward)

        if done_limit:
            dones = [0] * (len(self.ep_state) - 1) + [1]

            # Calculate Q-values
            if not done_env:
                for i_add_step in range(1000):
                    # TODO: Range of (-1, 1) is for HalfCheetah, Walker, Hopper only
                    action_dim = env.action_space.shape[0]
                    action = (
                            policy.select_action(np.array(state))
                            + np.random.normal(0, self.expl_noise, size=action_dim)
                    ).clip(-1, 1)
                    _, r, d, _ = env.step(action)
                    self.ep_reward.append(r)

                    if d:
                        print("Extended only for ", i_add_step)
                        break

            qs = []
            reward_np = np.asarray(self.ep_reward)

            n = len(self.ep_reward)
            for i in range(min(1000, len(self.ep_reward))):
                slide = min(n-i, 1000)
                gamma = np.power(np.ones(slide) * 0.99, np.arange(slide))

                q = np.sum(reward_np[i:i+slide] * gamma)
                qs.append(q)

            # Add to memory
            for s, a, q in zip(self.ep_state, self.ep_action, qs):
                self.mem.store(s, a, q)

            for s, a, ns, r, d in zip(self.ep_state, self.ep_action,
                                      self.ep_next_state, self.ep_reward,
                                      dones):
                self._add_replay_buffer(s, a, ns, r, d)

            self.ep_state.clear()
            self.ep_action.clear()
            self.ep_next_state.clear()
            self.ep_reward.clear()

            if self.prioritized and step >= self.start_timesteps:
                self._recalc_priorities()

    def _recalc_priorities(self):
        self.p[:self.size] = self.mem.q[:self.size].flatten()
        min_mc = np.min(self.p[:self.size])
        if min_mc < 0:
            self.p[:self.size] += np.abs(min_mc)
        
        self.p[:self.size] **= self.pr_alpha
        d_s = np.sum(self.p[:self.size]) 
        self.p[:self.size] /= d_s

    def sample(self, batch_size, step=None):
        if step is None or (step < self.start_timesteps or (not self.prioritized)):
            p = None
        else:
            p = self.p[:self.size]

        ind = np.random.choice(self.size, batch_size, p=p)
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
        i_step = 0
        while not done and i_step < 1000:
            i_step += 1
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


class RewardLogger:
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.data = []

        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

    def log(self, v, step):
        self.data.append([0, step, v])

    def dump(self, fn):
        with open(os.path.join(self.exp_dir, fn), "w") as f:
            json.dump(self.data, f)
        print("Rewards dumped to ", self.exp_dir)
