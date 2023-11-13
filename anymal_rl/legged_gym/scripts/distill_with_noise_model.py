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

from student_policy import StudentPolicy, proprioceptive_from_observation, exteroceptive_from_observation, priviliged_from_observation, priviliged_from_decoded, priviliged_size


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
    env_cfg.domain_rand.randomize_friction = True

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

    for i in tqdm.tqdm(range(10*int(env.max_episode_length))):
        j += 1
        with torch.no_grad():
            actions = policy(obs.detach())

        
        proprioceptive = obs[:, :proprioceptive_size]
        exteroceptive = obs[:, proprioceptive_size: proprioceptive_size + exteroceptive_size] + torch.randn_like(obs[:, proprioceptive_size: proprioceptive_size + exteroceptive_size]) * 0.2

        priviliged = obs[:, proprioceptive_size + exteroceptive_size:]

        reconstructed_target = torch.cat((exteroceptive, priviliged), dim=-1)

        action_student, reconstructed_student = student_policy(proprioceptive, exteroceptive)

        obs, priviliged, rews, dones, infos = env.step(action_student.detach())
        # obs, priviliged, rews, dones, infos = env.step(actions.detach())

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

        if not lr_updated and i > 7*int(env.max_episode_length):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 3e-5
            lr_updated = True
        student_policy.reset(dones)

    student_policy.save_weights("student_terrain_noise.pt")

if __name__ == '__main__': 
    args = get_args()
    play(args)
