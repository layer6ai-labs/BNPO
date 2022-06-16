import os
import sys
import argparse
import time
import datetime
import json
import random
import numpy as np
import torch
import gym

from bnp_options import *
from utils import *
from train import split_train_test
from env.atari_env import AtariEnv
from env.augmented_atari_env import AugmentedAtariEnv

sys.path.append('../stable-baselines3')
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3 import PPO


def get_args():
    parser = argparse.ArgumentParser(
        description='Trains BNP learning of options')

    # Environment settings
    parser.add_argument('--pretrained-path', type=str, default='',
                        help='model run directory (with config + checkpoint')
    parser.add_argument('--env-name', type=str, default='AlienNoFrameskip-v4',
                        help='type of environment')
    parser.add_argument('--random-seed', type=int, default=0,
                        help='Used to seed random number generators')
    parser.add_argument('--n-envs', type=int, default=10,
                        help='Numbers of environments created in ppo')
    parser.add_argument('--max-aug-steps', type=int, default=15,
                        help='maximum number of steps when taking a high-level action')

    # Training settings
    parser.add_argument('--training-steps', type=int, default=int(1e6),
                        help='Number of ppo training steps')  

    # Baseline settings
    parser.add_argument('--baseline-ddo', action="store_true", default=False,
                        help='use ddo baseline instead of bnp options')
    parser.add_argument('--baseline-compile', action="store_true", default=False,
                        help='use compile baseline instead of bnp options')
    parser.add_argument('--baseline-compile-np', action="store_true", default=False,
                        help='use nonparametric compile baseline instead of bnp options')

    # Misc settings
    parser.add_argument('--save-dir', type=str, default='',
                        help='directory where model and config are saved')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    params = vars(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import baselines if needed
    if args.baseline_ddo:
        sys.path.append('../ddo_baseline_pytorch')
        from models.AtariRamModel import AtariRamModel
    if args.baseline_compile:
        sys.path.append('../compile')
        from modules import CompILE
    if args.baseline_compile_np:
        sys.path.append('../compile_np')
        from modules import CompILE

    # Folder for this run
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    start_time_s = time.time()
    run_ID = f"atari_exp_{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}"
    if args.save_dir == '':
        run_dir = f"runs/{run_ID}"
    else:
        run_dir = args.save_dir
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        f.write(json.dumps(params, indent=4))

    if args.pretrained_path == '':
        ### If no pretrained model is given, will trained on the basic gym environment
        
        base_env = make_vec_env(args.env_name, n_envs=args.n_envs, seed=args.random_seed)

        ppo = PPO('CnnPolicy', base_env, n_steps=512, verbose=1, tensorboard_log=run_dir)
        ppo.learn(total_timesteps=args.training_steps, tb_log_name='tensorboard_logs_base')

        ppo.save(os.path.join(run_dir, "ppo_base"))
    
    else:
        ### Training on an augmented environment using a pretrained model
        
        ## Setup augmented environment

        # This will be used to generate the seeds for other RNGs.
        random_seed = args.random_seed
        rng_master = np.random.RandomState(random_seed)
        np.random.seed(random_seed) # there were some issue with reproducibility
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Create gym environment
        env = gym.make(args.env_name)
        state_dim = 1024
        action_dim = env.action_space.n

        # Load pretrained model
        with open(os.path.join(args.pretrained_path, 'config.json'), 'r') as config_file:
            config = json.load(config_file)
        if args.baseline_ddo:
            model = AtariRamModel(config['K'], statedim=(state_dim,), actiondim=(action_dim,))
        elif args.baseline_compile or args.baseline_compile_np:
            model = CompILE(state_dim, action_dim, config['hidden_dim'], config['latent_dim'],
                            config['num_segments'], latent_dist=config['latent_dist'], device=device)
            model.to(device)
        else:
            model = BNPOptions(None, state_dim, action_dim, device, rng=rng_master, **config)
        
        model.load(os.path.join(args.pretrained_path, 'checkpoint.pth'))

        # Create augmented environment
        def augmented_atari_wrapper(env, model):
            env = AtariWrapper(env)
            env = AugmentedAtariEnv(env, model, max_steps=args.max_aug_steps)
            return env

        augmented_env = make_vec_env(args.env_name, n_envs=args.n_envs, seed=args.random_seed, 
            wrapper_class=lambda env: augmented_atari_wrapper(env, model)
        )

        ## Training

        ppo = PPO('CnnPolicy', augmented_env, n_steps=512, verbose=1, tensorboard_log=run_dir, custom_buffer=True)
        ppo = ppo.learn(total_timesteps=args.training_steps, tb_log_name='tensorboard_logs_aug')

        ppo.save(os.path.join(run_dir, "ppo_augmented"))



