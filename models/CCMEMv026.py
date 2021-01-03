import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from itertools import chain


class Encoder(nn.Module):
    def __init__(self, state_dim, out_dim):
        super(Encoder, self).__init__()

        self.l1 = nn.Linear(state_dim, out_dim)

    def forward(self, state):
        e = F.relu(self.l1(state))
        return e


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, s_emb):
        a = F.relu(self.l2(s_emb))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(action_dim, 64)
        self.l2 = nn.Linear(256 + 64, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, s_emb, action):
        q_a = self.l1(action)
        q = torch.cat([s_emb, q_a], 1)
        q = F.relu(self.l2(q))
        return self.l3(q)


class CCMEMv026(object):
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, alpha=0.0,
            tau=0.005, device="cuda", log_dir="tb"):

        self.encoder = Encoder(state_dim, 256).to(device)

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(chain(*[self.actor.parameters(), self.encoder.parameters()]))
        print("Actor optimizer created.")

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        print("Critic optimizer created.")

        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.device = device
        self.q = 0

        self.step = 0
        self.tb_logger = SummaryWriter(log_dir)

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_emb = self.encoder(state)
        return self.actor(state_emb).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        state_emb = self.encoder(state)
        next_state_emb = self.encoder(next_state).detach()

        # Compute the target Q value
        target_Q = self.critic_target(next_state_emb, self.actor_target(next_state_emb))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        time1 = time.time()
        mem_q = replay_buffer.mem.retrieve_cuda(state, action, self.step)
        mem_q = torch.from_numpy(mem_q).float().to(self.device)

        #mem_contrib = torch.sum(min_inds).item() / batch_size
        mem_time = time.time() - time1

        # Get current Q estimate
        current_Q = self.critic(state_emb, action)

        # Compute critic loss
        q_loss = F.mse_loss(current_Q, target_Q)
        q_loss_mem = F.mse_loss(current_Q, mem_q)
        critic_loss = (1 - self.alpha) * q_loss + self.alpha * q_loss_mem

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state_emb, self.actor(state_emb.detach())).mean()

        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Logging
        if self.step % 250 == 0:
            q = np.mean(current_Q.detach().cpu().numpy())
            self.tb_logger.add_scalar("algo/q", q, self.step)
            q_mem = np.mean(mem_q.cpu().numpy())
            self.tb_logger.add_scalar("algo/q_mem", q_mem, self.step)
            q_loss = q_loss.detach().cpu().item()
            self.tb_logger.add_scalar("algo/q_cur_loss", q_loss, self.step)
            q_mem_loss = q_loss_mem.detach().cpu().item()
            self.tb_logger.add_scalar("algo/q_mem_loss", q_mem_loss, self.step)
            q_total_loss = q_loss + q_mem_loss
            self.tb_logger.add_scalar("algo/critic_loss", q_total_loss, self.step)
            pi_loss = actor_loss.detach().cpu().item()
            self.tb_logger.add_scalar("algo/pi_loss", pi_loss, self.step)
            self.tb_logger.add_scalar("algo/mem_retrieve_time", mem_time, self.step)
            # self.tb_logger.add_scalar("algo/mem_contribution", mem_contrib, self.step)
        self.step += 1

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
            
