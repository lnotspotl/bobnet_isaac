#!/usr/bin/env python3

from anymal_rl.legged_gym.envs import *
from anymal_rl.legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import argparse
import torch
import torch.nn as nn
import tqdm

from student_policy import StudentPolicy, proprioceptive_from_observation, exteroceptive_from_observation, priviliged_from_observation, priviliged_from_decoded, priviliged_size, exteroceptive_from_decoded, StudentPolicyJitted
from noise_model import ExteroceptiveNoiseGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='Distill policy with noise model')
    parser.add_argument('--policy_path', type=str, help='Path where to store the distilled policy')
    args, unknown = parser.parse_known_args()
    policy_path = args.policy_path
    return policy_path

def play(args, policy_path):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    total_envs = 1
    env_cfg.env.num_envs = total_envs
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_friction = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # policy = ppo_runner.get_inference_policy(device=env.device)

    actor_critic = ppo_runner.alg.actor_critic

    import torch
    import torch.nn as nn

    student_jitted = StudentPolicyJitted(actor_critic)
    student_jitted.load_from_file(policy_path)
    student_jitted.student_policy.belief_encoder.hidden = torch.zeros(1, 2048)
    student_jitted.to("cpu")

    jitted = torch.jit.script(student_jitted)
    print(jitted)

    jitted.save(policy_path + "_jitted.pt")

if __name__ == '__main__': 
    policy_path = parse_args()
    args = get_args()
    play(args, policy_path)
