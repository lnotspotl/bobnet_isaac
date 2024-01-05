#!/usr/bin/env python3

import torch

N_ENVS = 4096

class CentralPatternGenerator:
    def __init__(self, period, initial_offsets,  n_envs = N_ENVS, device="cpu"):
        self.period = period
        self.initial_offsets = initial_offsets.to(device)
        self.device = device
        self.n_envs = n_envs
        self.time = torch.zeros((self.n_envs,)).to(device)

    def step(self, dt):
        self.time += dt

    def get_observation(self):
        phases = self.compute_phases()
        return torch.cat([phases.cos(), phases.sin()], dim=-1)

    def compute_phases(self, phase_offsets = None):
        leg_times = self.time.unsqueeze(1) + self.initial_offsets.view(-1, 4)
        # leg_phases = 2 * torch.pi * torch.remainder(leg_times,self.period) / self.period    # <-- alternative
        self.phases = 2 * torch.pi * torch.frac(leg_times / self.period) 

        if phase_offsets is not None:
            self.phases = (self.phases + phase_offsets) % (2 * torch.pi)

        return self.phases

    def reset(self, env_idxs = None):
        if env_idxs is None:
            env_idxs = slice(0, self.n_envs)
        self.time[env_idxs] = 0.0

    def leg_heights(self, phase_offsets = None):
        phases = self.compute_phases(phase_offsets)
        heights = torch.zeros_like(phases, device=self.device)

        # swing - going up
        swing_up_indeces = (phases <= torch.pi / 2)
        time_up = phases[swing_up_indeces] * (2 / torch.pi)
        heights[swing_up_indeces] = 0.2 * (-2 * torch.pow(time_up, 3) + 3 * torch.pow(time_up, 2))

        # swing - going down
        swing_down_indeces = torch.logical_and((phases <= torch.pi), torch.logical_not(swing_up_indeces))
        time_down = phases[swing_down_indeces] * (2 / torch.pi) - 1.0
        heights[swing_down_indeces] = 0.2 * (2 * torch.pow(time_down, 3) - 3 * torch.pow(time_down, 2) + 1.0)

        return heights
    
class AnymalInverseKinematics:
    def __init__(self, device = "cpu"):
        height = 0.573
        self.device = device

        self.d2 = 0.19716
        self.a3 = 0.285
        self.a4 = 0.34923

        self.default_positions = torch.Tensor([
             0.00790786, 0.05720384, -height,  # LF
            -0.00790786, 0.05720384, -height,  # LH
             0.00790786, -0.05720384, -height, # RF
            -0.00790786, -0.05720384, -height  #  RH
        ]).to(self.device)

    def compute_ik(self, heights):
        n_envs = heights.shape[0]
        positions = self.default_positions.repeat(n_envs, 1)
        positions[:, [2,5,8,11]] +=  heights
        return self._ik_vectorized(positions)

    def _ik_vectorized(self, positions):
        n_envs = positions.shape[0]

        d2_t = torch.tensor([self.d2], device=self.device)
        d2_ts = torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device) * d2_t
        a3_t = torch.tensor([self.a3], device=self.device)
        a4_t = torch.tensor([self.a4], device=self.device)

        theta4_multipliets = torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device)

        x_indices = [0, 3, 6, 9]
        y_indices = [1,4,7,10]
        z_indices = [2,5,8,11]
        yz_indeces = [1,2,4,5,7,8,10,11]

        Es = torch.pow(positions[:, yz_indeces].view(n_envs, 4, -1), 2).sum(dim=2) - d2_ts.pow(2).unsqueeze(0)
        Es_sqrt = Es.sqrt()

        theta1s = torch.atan2(Es_sqrt, d2_ts) + torch.atan2(positions[:, z_indices], positions[:, y_indices])

        Ds = (Es + torch.pow(positions[:, x_indices], 2) - a3_t.pow(2) - a4_t.pow(2)) / (2 * a3_t * a4_t)
        Ds[Ds > 1.0] = 1.0
        Ds[Ds < -1.0] = -1.0
        theta4_offset = torch.tensor([0.254601], device=self.device)
        theta4s = -torch.atan2(torch.sqrt(1 - Ds.pow(2)), Ds)
        theta4s_final = theta4s + theta4_offset

        theta4s *= theta4_multipliets
        theta4s_final *= theta4_multipliets

        theta3s = torch.atan2(-positions[:, x_indices], Es_sqrt) - torch.atan2(a4_t * torch.sin(theta4s), a3_t + a4_t * torch.cos(theta4s))

        joint_angles = torch.cat([theta1s, theta3s, theta4s_final], dim=1)[:, [4 * i + j for j in range(4) for i in range(3)]]
        return joint_angles


class SpotInverseKinematics:
    def __init__(self, device = "cpu"):
        height = 0.46
        self.device = device

        self.d2 = 0.1108
        self.a3 = 0.32097507691408067
        self.a4 = 0.335

        self.default_positions = torch.Tensor([
             0.00490786, 0.1108, -height,  # LF
            -0.06, 0.1108, -height,  # LH
             0.00490786, -0.1108, -height, # RF
            -0.06, -0.1108, -height  #  RH
        ]).to(self.device)

    def compute_ik(self, heights):
        n_envs = heights.shape[0]
        positions = self.default_positions.repeat(n_envs, 1)
        positions[:, [2,5,8,11]] +=  heights
        return self._ik_vectorized(positions)

    def _ik_vectorized(self, positions):
        n_envs = positions.shape[0]

        d2_t = torch.tensor([self.d2], device=self.device)
        d2_ts = torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device) * d2_t
        a3_t = torch.tensor([self.a3], device=self.device)
        a4_t = torch.tensor([self.a4], device=self.device)

        theta4_multipliets = torch.tensor([1.0, 1.0, 1.0, 1.0], device=self.device)

        x_indices = [0, 3, 6, 9]
        y_indices = [1,4,7,10]
        z_indices = [2,5,8,11]
        yz_indeces = [1,2,4,5,7,8,10,11]

        Es = torch.pow(positions[:, yz_indeces].view(n_envs, 4, -1), 2).sum(dim=2) - d2_ts.pow(2).unsqueeze(0)
        Es_sqrt = Es.sqrt()

        theta1s = torch.atan2(Es_sqrt, d2_ts) + torch.atan2(positions[:, z_indices], positions[:, y_indices])

        Ds = (Es + torch.pow(positions[:, x_indices], 2) - a3_t.pow(2) - a4_t.pow(2)) / (2 * a3_t * a4_t)
        Ds[Ds > 1.0] = 1.0
        Ds[Ds < -1.0] = -1.0
        theta4_offset = torch.tensor([-0.0779666], device=self.device)
        theta4s = -torch.atan2(torch.sqrt(1 - Ds.pow(2)), Ds)
        theta4s_final = theta4s + theta4_offset

        theta4s *= theta4_multipliets
        theta4s_final *= theta4_multipliets

        theta3_offset = torch.tensor([0.07796663], device=self.device)
        theta3s = torch.atan2(-positions[:, x_indices], Es_sqrt) - torch.atan2(a4_t * torch.sin(theta4s), a3_t + a4_t * torch.cos(theta4s)) + theta3_offset

        joint_angles = torch.cat([theta1s, theta3s, theta4s_final], dim=1)[:, [4 * i + j for j in range(4) for i in range(3)]]

        return joint_angles