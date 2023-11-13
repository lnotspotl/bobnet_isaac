#!/usr/bin/env python3

from anymal_rl.legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from anymal_rl.legged_gym.envs import *
from anymal_rl.legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import torch.nn as nn

import tqdm

import copy

n_envs = 4096
proprioceptive_size = 3 + 3 + 3 + 3 + 12 + 12 + 3*12 + 2 * 12 + 2 * 16 + 8
exteroceptive_latent_size = 4 * 24
priviliged_latent_size = 24
hidden_size = 1024
exteroceptive_size=4*52
priviliged_size=41

class BeliefEncoder(nn.Module):
    def __init__(self, n_envs, hidden_size, device):
        super().__init__()
        self.n_envs = n_envs
        self.hidden_size = hidden_size
        self.device = device

        self.gru_input_size = proprioceptive_size + exteroceptive_latent_size

        # RNN
        self.gru = nn.GRUCell(input_size=self.gru_input_size, hidden_size=self.hidden_size)
        self.hidden = self.init_hidden()

        # encoders
        self.ga = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=1),
        )

        self.gb_out_size = exteroceptive_latent_size + priviliged_latent_size
        self.exteroceptive_latent_size = exteroceptive_latent_size
        self.gb = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=512),
            nn.ELU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ELU(),
            nn.Linear(in_features=256, out_features=self.gb_out_size)
        )

        self.reset_every = 50
        self.i = 0

    def forward(self, proprioceptive, exteroceptive, hidden=None):

        # Forward pass through the RNN
        if hidden is None:
            hidden = self.hidden
        rnn_input = self.concat(proprioceptive, exteroceptive)
        hidden_new = self.gru(rnn_input, hidden)

        # Gate
        alpha = torch.sigmoid(self.ga(hidden_new))
        exteroceptive_attenuated = alpha * exteroceptive

        belief_state = self.gb(hidden_new)

        belief_state[:, :self.exteroceptive_latent_size] += exteroceptive_attenuated

        self.hidden = hidden_new
        return belief_state, self.hidden
    
    def reset_graph(self):
        self.i += 1

        if self.i == self.reset_every:
            self.hidden = self.hidden.detach()
            self.i = 0

    def concat(self, proprio, extero_latent):
        return torch.cat((proprio, extero_latent), dim=-1)

    def init_hidden(self):
        return torch.zeros(self.n_envs, self.hidden_size, device=self.device)

class BeliefDecoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # Gate
        self.ga = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=128),
            nn.ELU(),
            nn.Linear(in_features=128, out_features=1),
        )

        # Exteroceptive decoder
        self.exteroceptive_decoder = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=512),
            nn.ELU(),
            nn.Linear(in_features=512, out_features=exteroceptive_size),
        )

        # Priviliged decoder
        self.priviliged_decoder = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=512),
            nn.ELU(),
            nn.Linear(in_features=512, out_features=priviliged_size),
        )


    def forward(self, hidden, exteroceptive):
        alpha = torch.sigmoid(self.ga(hidden)) # (n_envs, 1)
        exteroceptive_attenuated = alpha * exteroceptive

        exteroceptive_decoded = self.exteroceptive_decoder(hidden) + exteroceptive_attenuated
        priviliged_decoded = self.priviliged_decoder(hidden)

        return exteroceptive_decoded, priviliged_decoded


class StudentPolicy(nn.Module):
    def __init__(self, n_envs, teacher_policy, device="cuda"):
        super().__init__()
        self.n_envs = n_envs
        self.device = device
        self.mlp = copy.deepcopy(teacher_policy.actor).to(self.device)


        self.belief_encoder = BeliefEncoder(n_envs, hidden_size, self.device).to(self.device)
        self.belief_decoder = BeliefDecoder(hidden_size).to(self.device)

        self.ge = copy.deepcopy(teacher_policy.heights_encoder).to(self.device)
        for param in self.ge.parameters():
            param.requires_grad = False

    def reset(self, dones):
        if not dones.any():
            return
        self.belief_encoder.hidden[dones] = self.belief_encoder.init_hidden()[dones]

    def forward(self, proprioceptive, exteroceptive, hidden=None):
        n_envs = proprioceptive.shape[0]
        exteroceptive_encoded = self.ge(exteroceptive).view(n_envs, -1)

        belief_state, hidden = self.belief_encoder(proprioceptive, exteroceptive_encoded, hidden=hidden)

        mlp_in = torch.cat((proprioceptive, belief_state), dim=-1)
        action = self.mlp(mlp_in)

        reconstructed = self.belief_decoder(hidden, exteroceptive)

        return action, torch.cat(reconstructed, dim=-1)
    
    def reset_graph(self):
        self.belief_encoder.reset_graph()
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    actor_critic = ppo_runner.alg.actor_critic

    n_envs = env_cfg.env.num_envs
    student_policy = StudentPolicy(n_envs, actor_critic, device=env.device)

    optimizer = torch.optim.Adam(student_policy.parameters(), lr=2e-4)
    
    j = 0
    total_loss = 0

    lr_updated = False

    for i in tqdm.tqdm(range(30*int(env.max_episode_length))):
        j += 1
        with torch.no_grad():
            actions = policy(obs.detach())
        obs, priviliged, rews, dones, infos = env.step(actions.detach())

        student_policy.reset(dones)
        
        proprioceptive = obs[:, :proprioceptive_size]
        exteroceptive = obs[:, proprioceptive_size: proprioceptive_size + exteroceptive_size]
        priviliged = obs[:, proprioceptive_size + exteroceptive_size:]

        reconstructed_target = torch.cat((exteroceptive, priviliged), dim=-1)

        action_student, reconstructed_student = student_policy(proprioceptive, exteroceptive)

        assert not actions.requires_grad
        assert action_student.requires_grad
        assert not reconstructed_target.requires_grad
        assert reconstructed_student.requires_grad

        action_loss = nn.functional.mse_loss(action_student, actions)
        reconstruction_loss = nn.functional.mse_loss(reconstructed_student, reconstructed_target)

        loss = action_loss + 0.5 * reconstruction_loss
        total_loss += loss.item()
        loss.backward(retain_graph=True)
        student_policy.reset_graph()

        if j == 32:
            for param in student_policy.parameters():
                param.grad = param.grad / 32 if param.grad is not None else None
            total_loss /= 32
            j = 0
            print(f"step: {i} | loss {total_loss} | lr {optimizer.param_groups[0]['lr']}")
            optimizer.step()
            optimizer.zero_grad()

        if not lr_updated and i > 17*int(env.max_episode_length):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 3e-5
            lr_updated = True

    student_policy.save_weights("student2.pt")

if __name__ == '__main__': 
    args = get_args()
    play(args)
