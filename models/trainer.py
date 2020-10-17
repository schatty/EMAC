import gym
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from models.utils import EpisodicReplayBuffer
from models.TD3 import TD3
from models.DDPG import DDPG
from models.CCMEMv0 import CCMEMv00
from models.CCMEMv01 import CCMEMv01
from models.CCMEMv02 import CCMEMv02

from .utils import eval_policy
from .mem import MemBuffer


class Trainer:

    def __init__(self, config):
        self.c = config

    def train(self, exp_dir):
        expl_noise = self.c["expl_noise"]
        max_timesteps = self.c["max_timesteps"]
        start_timesteps = self.c["start_timesteps"]
        batch_size = self.c["batch_size"]
        eval_freq = self.c["eval_freq"]
        save_model = self.c["save_model"]
        save_buffer = self.c["save_buffer"]
        save_memory = self.c["save_memory"]
        save_model_every = self.c["save_model_every"]
        device = self.c["device"]
        env_name = self.c["env"]
        env = gym.make(self.c["env"])

        # Logger
        tb_logger = SummaryWriter(f"{exp_dir}/tb")

        # Set seeds
        seed = self.c["seed"]
        env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": self.c["discount"],
            "tau": self.c["tau"],
            "device": self.c["device"],
            "log_dir": f"{exp_dir}/tb"
        }

        # Initialize policy
        policy = self.c["policy"]
        if policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = self.c["policy_noise"] * max_action
            kwargs["noise_clip"] = self.c["noise_clip"] * max_action
            kwargs["policy_freq"] = self.c["policy_freq"]
            policy = TD3(**kwargs)
        elif policy == "DDPG":
            policy = DDPG(**kwargs)
        elif policy == "CCMEMv00":
            policy = CCMEMv00(**kwargs)
        elif policy == "CCMEMv01":
            kwargs["alpha"] = self.c["alpha"]
            policy = CCMEMv01(**kwargs)
        elif policy == "CCMEMv02":
            kwargs["alpha"] = self.c["alpha"]
            policy = CCMEMv02(**kwargs)

        load_model = self.c["load_model"]
        if load_model != "":
            policy.load(f"{exp_dir}/models/{load_model}")

        mem = MemBuffer(state_dim, action_dim, 100000,
                        mem_dim=4, device=kwargs["device"])
        replay_buffer = EpisodicReplayBuffer(state_dim, action_dim, mem,
                                             device=device, expl_noise=self.c["expl_noise"])

        # Evaluate untrained policy
        ep_reward = eval_policy(policy, env_name, seed)
        tb_logger.add_scalar("agent/eval_reward", ep_reward, 0)

        state = env.reset()
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0

        for t in range(1, int(max_timesteps)+1):
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < start_timesteps:
                action = env.action_space.sample()
            else:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, done_env, _ = env.step(action)
            done_limit = done_env if episode_timesteps < self.c["ep_len"] else True

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_env, done_limit, env, policy)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= start_timesteps:
                policy.train(replay_buffer, batch_size)

            if done_limit:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                # Reset environment
                state = env.reset()
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Logging buffer size
            if t % 250 == 0:
                tb_logger.add_scalar("trainer/buffer_size", replay_buffer.size, t)
                tb_logger.add_scalar("memory/size", replay_buffer.mem.size, t)

            # Evaluate episode
            if t % eval_freq == 0:
                ep_reward = eval_policy(policy, env_name, seed)
                tb_logger.add_scalar("agent/eval_reward", ep_reward, t)

            # Save model
            if save_model and t % save_model_every == 0:
                print("Saving model...")
                policy.save(f"{exp_dir}/models/model_step_{t}")

            if t % 250000 == 0 and save_buffer:
                print(f"Saving buffer at {t} timestep...")
                replay_buffer.save(f"{exp_dir}/buffers/replay_buffer")

            if t % 250000 == 0 and save_memory:
                print(f"Saving memory at {t} timesteps...")
                replay_buffer.mem.save(f"{exp_dir}/buffers/memory")
