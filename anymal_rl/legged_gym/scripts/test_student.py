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

from anymal_rl.legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from anymal_rl.legged_gym.envs import *
from anymal_rl.legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import matplotlib.pyplot as plt

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
    student_policy = StudentPolicy(n_envs, actor_critic, device=env.device)

    student_policy_path = "./student3.pt"
    student_policy.load_weights(student_policy_path)
    print(f"loaded weights from {student_policy_path}")

    for i in tqdm.tqdm(range(10*int(env.max_episode_length))):
        proprioceptive = proprioceptive_from_observation(obs)
        exteroceptive = exteroceptive_from_observation(obs)

        # priviliged = priviliged_from_observation(obs)

        action_student, reconstructed_student = student_policy.inference(proprioceptive, exteroceptive)

        obs, _, rews, dones, infos = env.step(action_student)
        student_policy.reset(dones)

if __name__ == '__main__':
    args = get_args()
    play(args)
