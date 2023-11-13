#!/usr/bin/env python3

import torch
import torch.nn as nn

import copy

# hardcoded constants - TODO: remove these
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

    @torch.no_grad()    
    def inference(self, proprioceptive, exteroceptive):
        return self(proprioceptive, exteroceptive)

## helper functions
def proprioceptive_from_observation(obs):
    return obs[:, :proprioceptive_size]

def exteroceptive_from_observation(obs):
    return obs[:, proprioceptive_size: proprioceptive_size + exteroceptive_size]

def priviliged_from_observation(obs):
    return obs[:, proprioceptive_size + exteroceptive_size:]

def priviliged_from_decoded(decoded):
    return decoded[:, -priviliged_size:]

def exteroceptive_from_decoded(decoded):
    return decoded[:, :exteroceptive_size]