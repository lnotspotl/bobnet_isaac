# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn

class HeightsEncoder(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.n_heights = 52
        self.n_legs = 4
        self.net = nn.Sequential(
            nn.Linear(self.n_heights, 80),
            activation,
            nn.Linear(80,60),
            activation,
            nn.Linear(60, 24),
            nn.Tanh()
        )

    def forward(self, heights):
        n_envs = heights.shape[0]
        return self.net(heights.view(n_envs, self.n_legs, self.n_heights))
    
class PriviligedEncoder(nn.Module):
    def __init__(self, activation):
        super().__init__()
        foot_contacts = 4
        thigh_contacts = 4
        shank_contacts = 4
        airtime = 4
        friction = 1
        contact_forces = 4 * 3
        size_in = foot_contacts + thigh_contacts + shank_contacts + airtime + friction + contact_forces
        self.net = nn.Sequential(
            nn.Linear(size_in, 64),
            activation,
            nn.Linear(64, 32),
            activation,
            nn.Linear(32, 24),
            nn.Tanh()
        )

    def forward(self, priviliged_info):
        return self.net(priviliged_info)

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        self.heights_encoder = HeightsEncoder(activation)
        self.priviliged_encoder = PriviligedEncoder(activation)

        # mlp_input_dim_a = 6 + 3 + 3 + 12 + 12 + 12 + 24 + 4*24
        mlp_input_dim_a = 3 + 3 + 3 + 3 + 12 + 12 + 8 + 3*12 + 2 * 12 + 2 * 16 + 24 + 4*24
        mlp_input_dim_c = mlp_input_dim_a
        num_actions=16

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(0 * init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def preprocess_input(self, observation):
        # measured velocity, orientation, command, joint_pos_res, joint_velocity, last_action
        # PROPRIOCEPTIVE_SIZE = 6 + 3 + 3 + 12 + 12 + 12
        PROPRIOCEPTIVE_SIZE = 3 + 3 + 3 + 3 + 12 + 12 + 3*12 + 2 * 12 + 2 * 16 + 8
        PROPRIOCEPTIVE_START = 0

        # print("[Preprocess input] | observation shape:", observation.shape)

        # 4 x 52 height samples
        HEIGHTS_SCAN_SIZE = 4 * 52
        HEIGHTS_START = PROPRIOCEPTIVE_SIZE

        # foot contacts, thigh contacs, shank contacts, airtime, friction coefficients
        PRIVILIGED_SIZE = 4 + 4 + 4 + 4 + 1 + 4*3
        PRIVILIGED_START = PROPRIOCEPTIVE_SIZE + HEIGHTS_SCAN_SIZE

        n_envs = observation.shape[0]
        heights_encoded = self.heights_encoder(observation[:, HEIGHTS_START:PRIVILIGED_START]).reshape(n_envs, -1)

        priviliged_encoded = self.priviliged_encoder(observation[:, PRIVILIGED_START:])

        obs = torch.cat([
            observation[:, PROPRIOCEPTIVE_START:HEIGHTS_START],
            heights_encoded,
            priviliged_encoded
        ], dim = 1)
        return obs

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_mean(self, observation):
        return self.actor(self.preprocess_input(observation))

    def update_distribution(self, observations):
        mean = self.get_mean(observation=observations)
        self.distribution = Normal(mean, mean*0. + self.std.exp())

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.get_mean(observation=observations)
        return actions_mean

    def evaluate(self, observation, **kwargs):
        return self.critic(self.preprocess_input(observation))

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
