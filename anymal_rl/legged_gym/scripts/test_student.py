#!/usr/bin/env python3

from anymal_rl.legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from anymal_rl.legged_gym.envs import *
from anymal_rl.legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import torch.nn as nn

from noise_model import ExteroceptiveNoiseGenerator


import tqdm

import copy

from anymal_rl.legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from anymal_rl.legged_gym.envs import *
from anymal_rl.legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import matplotlib.pyplot as plt

from student_policy import StudentPolicy, proprioceptive_from_observation, exteroceptive_from_observation, priviliged_from_observation, priviliged_from_decoded, exteroceptive_from_decoded

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
    env_cfg.domain_rand.randomize_friction = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    actor_critic = ppo_runner.alg.actor_critic

    n_envs = env_cfg.env.num_envs
    student_policy = StudentPolicy(1024, actor_critic, device=env.device)

    student_policy_path = "anymal_c_teacher_06.pt"
    student_policy.load_weights(student_policy_path)
    print(f"loaded weights from {student_policy_path}")
    student_policy.belief_encoder.hidden = torch.zeros(2, n_envs, 50).to(env.device)
    student_policy.belief_encoder.n_envs = n_envs
    
    noise_generator = ExteroceptiveNoiseGenerator(52, env_cfg.env.num_envs, env.max_episode_length, n_legs=4)
    noise_generator.ck = 1.0
    noise_generator.reset(torch.arange(env_cfg.env.num_envs).to(env.device))

    for i in tqdm.tqdm(range(10*int(env.max_episode_length))):
        proprioceptive = proprioceptive_from_observation(obs)
        exteroceptive = exteroceptive_from_observation(obs)

        exteroceptive_with_noise, points = env.get_heights_observation_with_noise(noise_generator)

        x_points = points[0,:,0].view(-1)
        y_points = points[0,:,1].view(-1)
        z_points = exteroceptive_with_noise[0, :].view(-1)

        print(env.last_actions[0])

        # priviliged = priviliged_from_observation(obs)

        action_student, reconstructed_student = student_policy.inference(proprioceptive, exteroceptive_with_noise)

        extero = exteroceptive_from_decoded(reconstructed_student)

        extero_diff = (extero - exteroceptive) / 5.0

        # print(torch.abs(extero_diff).mean())

        obs, _, rews, dones, infos = env.step(action_student)
        env.draw_spheres(x_points, y_points, z_points, reset=True)
        env.draw_spheres(x_points, y_points, exteroceptive_from_decoded(reconstructed_student.detach())[0,:].view(-1), reset=False, color=(0,0,1))
        student_policy.reset(dones)

if __name__ == '__main__':
    args = get_args()
    play(args)
