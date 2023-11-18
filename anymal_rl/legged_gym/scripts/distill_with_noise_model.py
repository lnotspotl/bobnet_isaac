#!/usr/bin/env python3

from anymal_rl.legged_gym.envs import *
from anymal_rl.legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import argparse
import torch
import torch.nn as nn
import tqdm

from student_policy import StudentPolicy, proprioceptive_from_observation, exteroceptive_from_observation, priviliged_from_observation, priviliged_from_decoded, priviliged_size, exteroceptive_from_decoded
from noise_model import ExteroceptiveNoiseGenerator

def parse_args():
    parser = argparse.ArgumentParser(description='Distill policy with noise model')
    parser.add_argument('--policy_path', type=str, help='Path where to store the distilled policy')
    args, unknown = parser.parse_known_args()
    policy_path = args.policy_path
    return policy_path

def play(args, policy_path):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    total_envs = 200
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
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    actor_critic = ppo_runner.alg.actor_critic

    n_envs = env_cfg.env.num_envs
    student_policy = StudentPolicy(n_envs, actor_critic, device=env.device)

    optimizer = torch.optim.Adam(student_policy.parameters(), lr=1e-3)
    
    j = 0
    total_loss = 0

    lr_updated = False

    noise_generator = ExteroceptiveNoiseGenerator(52, total_envs, env.max_episode_length, n_legs=4)

    for i in tqdm.tqdm(range(10*int(env.max_episode_length))):
        j += 1

        # Unpack observation
        proprioceptive = proprioceptive_from_observation(obs)
        exteroceptive = exteroceptive_from_observation(obs)
        priviliged = priviliged_from_observation(obs)

        exteroceptive_with_noise, points = env.get_heights_observation_with_noise(noise_generator)
        noise_generator.step()

        x_points = points[0,:,0].view(-1)
        y_points = points[0,:,1].view(-1)
        z_points = exteroceptive_with_noise[0, :].view(-1)


        # exteroceptive_diff = exteroceptive_with_noise[0] - exteroceptive[0]

        # Generate input to student policy
        reconstructed_target = torch.cat((exteroceptive, priviliged), dim=-1)

        # Generate student policy output
        action_student, reconstructed_student = student_policy(proprioceptive, exteroceptive_with_noise)
        # env.draw_spheres(x_points, y_points, z_points, reset=True)
        # env.draw_spheres(x_points, y_points, exteroceptive_from_decoded(reconstructed_student.detach())[0,:].view(-1), reset=False, color=(0,0,1))

        # What would the teacher do?
        with torch.no_grad():
            actions = policy(obs.detach())

        # Step environment
        obs, priviliged, rews, dones, infos = env.step(action_student.detach())

        # Reset envs
        noise_generator.reset(dones.nonzero().squeeze())

        assert not actions.requires_grad
        assert action_student.requires_grad
        assert not reconstructed_target.requires_grad
        assert reconstructed_student.requires_grad

        # Calculate action reconstruction loss
        action_loss = nn.functional.mse_loss(action_student, actions)

        # Calculate exteroceptive and priviliged information loss
        reconstruction_loss = nn.functional.mse_loss(reconstructed_student, reconstructed_target)

        # Calculate total loss
        loss = action_loss + 0.6 * reconstruction_loss

        # Book keeping
        loss.backward(retain_graph=True)
        student_policy.reset_graph()

        total_loss += loss.item()
        if j == 32:
            for param in student_policy.parameters():
                param.grad = param.grad / 32 if param.grad is not None else None
            total_loss /= 32
            j = 0
            print(f"step: {i} | loss {total_loss} | lr {optimizer.param_groups[0]['lr']}")
            optimizer.step()
            optimizer.zero_grad()

        # Learning rate schedule
        if not lr_updated and i > 7*int(env.max_episode_length):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 3e-5
            lr_updated = True

        # Reset student policy RNN computational graph
        student_policy.reset(dones)

    print(f"Storing distilled policy at {policy_path}")
    student_policy.save_weights(policy_path)
    print("Done")

if __name__ == '__main__': 
    policy_path = parse_args()
    args = get_args()
    play(args, policy_path)
