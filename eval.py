import os
import sys
import random
import numpy as np
import torch

from bnp_options import *
from utils import *
from env.room_env import RoomEnv
from env.atari_env import AtariEnv


def get_options_policies(model, env, test_data, device):
    states = []
    if env.env_type == 'room':
        states.append(np.array([[4,0,-1]]))
    elif env.env_type == 'atari':
        states = test_data[0][:, :-1, :]

    states = torch.from_numpy(np.array(states)).float().to(device)
    policies, terminations = model.rollout(states)

    return policies, terminations

def compute_score(model, env, test_data, device):
    policies, terminations = get_options_policies(model, env, test_data, device)
    if env.env_type == 'room':
        policies = torch.squeeze(torch.cat(policies), dim=1)
        room_scores = [0]*env.n_rooms
        for room in range(1, env.n_rooms+1):
            room_scores[room-1] = float(torch.max(policies[:, room]))
        score = np.mean(room_scores)
        return score

    elif env.env_type == 'atari':
        probs = []
        for policy in policies:
            log_probs = torch.distributions.OneHotCategorical(policy).log_prob(torch.from_numpy(test_data[1])).unsqueeze(-1)
            probs.append(torch.exp(log_probs))
        max_probs = probs[0]
        for prob in probs[1:]:
            max_probs = torch.max(max_probs, prob)
        # print(test_data[1][0])
        # print(max_probs[0])
        traj_probs = torch.mean(max_probs, dim=1)
        score = torch.mean(traj_probs)
        return score

    return 0.

def evaluate_options_usage(model, test_data, device):
    states = torch.tensor(test_data[0])[:, :-1].to(device)
    actions = torch.tensor(test_data[1]).to(device)
    options_probas = model.get_options_probas(states, actions)
    return options_probas
