import os
import itertools
from functools import reduce

import torch.nn as nn
import torch.nn.functional as F

from utils import *
from networks import DiscretePolicy, ContinuousPolicy, Termination, Encoder, StickBreakingKumaraswamy

torch.set_printoptions(sci_mode=False)

class BNPOptions:

    def __init__(self, data, state_dim, action_dim, device, rng, **kwargs):
        self.K = kwargs['K']
        self.random_seed = kwargs['random_seed']
        self.tol = kwargs['tolerance']
        if data is not None:
            self.data_states, self.data_actions, _ = data
        self.n_traj = np.shape(self.data_states)[0] if data is not None else 0
        self.device = device
        self.action_space = kwargs['action_space']
        # Will initialize several NNs below.
        # For reproducibility, need to seed pytorch first.
        torch_seed = rng.randint(100000)
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)    # Seed for all GPUs
        if self.action_space == 'discrete':
            self.policy = DiscretePolicy(d_states=state_dim, K=self.K, d_actions=action_dim,
                                         hidden_layer_sizes=kwargs['hidden_layer_sizes_policy'], device=device)
        else:
            self.policy = ContinuousPolicy(d_states=state_dim, K=self.K, d_actions=action_dim,
                                         hidden_layer_sizes=kwargs['hidden_layer_sizes_policy'], device=device)
        self.termination = Termination(d_states=state_dim, K=self.K,
                                       hidden_layer_sizes=kwargs['hidden_layer_sizes_termination'], device=device)
        self.relaxation_type = kwargs['relaxation_type']
        self.encoder = Encoder(d_states=state_dim, K=self.K, d_actions=action_dim,
                               hidden_size_LSTM=kwargs['LSTM_hidden_layer_size'],
                               hidden_layer_sizes=kwargs['LSTM_MLP_hidden_layer_sizes'], 
                               device=device, relaxation_type=self.relaxation_type)
        self.high_level_posterior = StickBreakingKumaraswamy(self.K, device)

        self.policy.to(device)
        self.termination.to(device)
        self.encoder.to(device)
        self.high_level_posterior.to(device)

        self.model_list = [self.policy, self.termination, self.encoder, self.high_level_posterior]
        self.parameter_list = sum([list(model.parameters()) for model in self.model_list], [])
        self.lr = kwargs['learning_rate']
        self.optimizer = torch.optim.Adam(self.parameter_list, lr=kwargs['learning_rate'], 
                                          weight_decay=0.1*kwargs['learning_rate'])

        perm_seed = rng.randint(100000)
        rng_perm = np.random.RandomState(perm_seed)
        self.perm = PermManager(self.n_traj, kwargs['batch_size'], rng=rng_perm)
        self.relaxation_type = kwargs['relaxation_type']
        self.temp = kwargs['temperature']
        self.temp_ratio = kwargs['temperature_ratio']
        self.max_epochs = kwargs['max_epochs']
        self.clip = kwargs['clip']
        self.entropy_factor = kwargs['entropy_factor']
        self.entropy_ratio = kwargs['entropy_ratio']
        self.eps = 10e-5
        self.fixed_options = kwargs['fixed_options']
        self.counter = 0
        self.check_options_interval = kwargs.get('check_options_usage', 10)
        self.new_option_hist = []

    def compute_negative_elbo(self, batch_states, batch_actions):
        # computes a stochastic approximation to the ELBO
        new_option = False
        eta, pre_sb_eta = self.high_level_posterior.sample_mean(return_pre_sb=True, nb_samples=30)  # [K] and [K]
        self.encoder.encode_trajectories(torch.cat([batch_states, batch_actions], axis=2))
        options, relaxed_opts = [], []
        terminations, relaxed_termins = [], []
        previous_option = torch.cat([
            torch.ones((batch_states.shape[0], 1, 1), device=self.device),
            torch.reshape(eta, (1, 1, self.K)).repeat((batch_states.shape[0], 1, 1))
        ], axis=2)
        for timestep in range(batch_states.shape[1]):
            enc = self.encoder.forward(timestep, previous_option) # [batch, 1, K+1]
            if self.relaxation_type == 'GS':
                options.append(enc[:, :, 1:])
                terminations.append(enc[:, :, 0:1])
                relaxed_opt_distr = torch.distributions.RelaxedOneHotCategorical(self.temp, logits=options[-1])
                relaxed_termin_distr = torch.distributions.RelaxedBernoulli(self.temp, logits=terminations[-1])
            else:
                raise NotImplementedError
            relaxed_opts.append(relaxed_opt_distr.rsample()) # [batch, 1, K]
            relaxed_termins.append(relaxed_termin_distr.rsample())  # [batch, 1, 1]
            previous_option = torch.cat([relaxed_termins[-1], relaxed_opts[-1]], axis=2)
        options = torch.cat(options, axis=1)
        relaxed_opts = torch.cat(relaxed_opts, axis=1)
        terminations = torch.cat(terminations, axis=1)
        relaxed_termins = torch.cat(relaxed_termins, axis=1)
        policies = self.policy(batch_states, relaxed_opts)
        # policies has shape [batch, max_length-1, action_dim] for discrete action spaces and 
        # is a tuple (mean: [batch, max_length-1, action_dim], log_std: [batch, max_length-1, action_dim]) 
        # for continuous action spaces
        termin_funcs = self.termination(batch_states[:, 1:], relaxed_opts[:, :-1])
        # termin funcs has shape [batch, max_length-2, 1]
        log_b0_term = stable_log(relaxed_delta_binary(relaxed_termins[:, :1]), self.eps) # [batch, 1, 1]
        eta_ht_terms = relaxed_policy_eval(torch.reshape(eta, (1, 1, -1)), relaxed_opts) # [batch, max_length-1, 1]

        bt_eq_1_terms = relaxed_termins[:, 1:] * termin_funcs * eta_ht_terms[:, 1:]  # [batch, max_length-2, 1]
        relaxed_delta_terms = relaxed_delta_one_hot(relaxed_opts[:, 1:], relaxed_opts[:, :-1])
        # relaxed_delta terms has shape [batch, max_length-2, 1]

        bt_eq_0_terms = (1. - relaxed_termins[:, 1:]) * (1. - termin_funcs) * relaxed_delta_terms
        # bt_eq_0_terms has shape [batch, max_length-2, 1]
        log_opts_terminations_terms = stable_log(bt_eq_1_terms + bt_eq_0_terms, self.eps)  # [batch, max_length-2, 1]
        if self.action_space == 'discrete':
            log_policies_terms = stable_log(torch.sum(batch_actions * policies, axis=2, keepdim=True), self.eps)
            # log_policies_terms has shape [batch, max_length-1, 1]
        else:
            log_policies_terms = torch.distributions.Independent(torch.distributions.Normal(
                policies[0], torch.exp(policies[1])), 1).log_prob(batch_actions).unsqueeze(-1)
            # log_policies_terms has shape [batch, max_length-1, 1]
        log_p_xi_zeta_given_eta = log_b0_term + stable_log(eta_ht_terms[:, 0].reshape((-1, 1, 1)), self.eps) +\
                                  torch.sum(log_opts_terminations_terms, axis=1, keepdim=True) +\
                                  torch.sum(log_policies_terms, axis=1, keepdim=True)
        # log_p_xi_zeta_given_eta has shape [batch, 1, 1]

        # q(zeta|eta,xi) is taken as its non-relaxed version (i.e. Gumbel-Softmax density is not evaluated)
        log_q_termins_given_eta_xi = torch.sum(-F.binary_cross_entropy_with_logits(input=terminations,
                                                                                   target=relaxed_termins,
                                                                                   reduction='none'), axis=1,
                                               keepdim=True)
        # [batch, 1, 1]

        # the cross entropy function from pytorch does not admit soft labels
        log_q_opts_given_eta_xi = torch.sum(relaxed_opts * F.log_softmax(options, dim=2), axis=[1, 2], keepdim=True)
        # log_q_opts_given_eta_xi has shape [batch, 1, 1]

        log_q_zeta_given_eta_xi = log_q_termins_given_eta_xi + log_q_opts_given_eta_xi

        kl_eta = self.high_level_posterior.compute_kl(self.K, pre_sb_eta, self.eps)

        # Entropy term
        entropy_term = torch.sum(relaxed_opts.mean(axis=[0,1]) * stable_log(relaxed_opts.mean(axis=[0,1])))

        if self.counter <= self.perm.epoch:
            with torch.no_grad():
                if self.action_space == 'discrete':
                    opt_policies = []
                    for option in range(self.K):
                        opt_vector = torch.zeros_like(relaxed_opts)
                        opt_vector[:, :, option] = 1.
                        opt_policy = self.policy(batch_states, opt_vector)
                        opt_policy = torch.sum(batch_actions * opt_policy, axis=2, keepdim=True)
                        opt_policies.append(opt_policy)
                    opt_policies_cat = torch.cat(opt_policies, dim=-1)
                    opt_policies_cat_amax = torch.argmax(opt_policies_cat, axis=-1)
                    option_usage = torch.bincount(opt_policies_cat_amax.view(-1), minlength=self.K)/(opt_policies_cat_amax.shape[0]*opt_policies_cat_amax.shape[1])
                    print("Option usage:", option_usage)
                    new_option = check_new_option(option_usage, tol=self.tol)

                    rec_acc = torch.mean(torch.sum(batch_actions * policies, axis=2, keepdim=True))
                    print('Reconstruction accuracy (with trained encoder):', rec_acc)
                    rec_acc2 = torch.mean(torch.max(opt_policies_cat, axis=-1).values)
                    print('Reconstruction accuracy (with perfect encoder):', rec_acc2)
                else:
                    opt_policies = []
                    for option in range(self.K):
                        opt_vector = torch.zeros_like(relaxed_opts)
                        opt_vector[:, :, option] = 1.
                        opt_policy = self.policy(batch_states, opt_vector)[0]
                        opt_policies.append(opt_policy)
                    opt_policies_cat = torch.stack(opt_policies, dim=-1)
                    opt_policies_dist = torch.sum((opt_policies_cat-batch_actions.unsqueeze(-1).repeat((1, 1, 1, self.K)))**2, dim=-2)
                    opt_policies_cat_amax = torch.argmin(opt_policies_dist, axis=-1)
                    option_usage = torch.bincount(opt_policies_cat_amax.view(-1), minlength=self.K)/(opt_policies_cat_amax.shape[0]*opt_policies_cat_amax.shape[1])
                    print("Option usage:", option_usage)
                    new_option = check_new_option(option_usage, tol=self.tol)

            self.counter += self.check_options_interval

        loss = torch.mean(log_q_zeta_given_eta_xi - log_p_xi_zeta_given_eta) + self.entropy_factor*entropy_term + kl_eta/self.n_traj

        return loss, self.K, new_option

    def _gradient_step(self, is_last):
        batch = self.perm.get_indices()
        # Select the states and actions, (s_t, a_t) for the episodes that make up this batch.
        # Note that the final state is dropped since it does not have an associated action
        # (we never transition 'out' of that state).
        batch_states = torch.tensor(self.data_states[batch])[:, :-1].to(self.device)
        batch_actions = torch.tensor(self.data_actions[batch]).to(self.device)
        # First index is element of the batch, second is time in the episode.
        # Third is component of the state space / component of the action space.
        assert batch_states.shape[0:2] == batch_actions.shape[0:2]

        self.optimizer.zero_grad()
        negative_elbo, k, new_option = self.compute_negative_elbo(batch_states, batch_actions)
        negative_elbo.backward()
        nn.utils.clip_grad_norm_(self.parameter_list, self.clip)
        self.optimizer.step()
        return negative_elbo.detach().cpu().numpy(), k, new_option

    def train(self, verbose=True):
        is_last = False
        new_option_counter = 0
        while True:
            start_epoch = self.perm.epoch
            if start_epoch >= self.max_epochs:
                is_last = True
            negative_elbo_np, k_np, new_option = self._gradient_step(is_last)
            if new_option and not(self.fixed_options):
                new_option_counter += 1
            # Test if we have completed an epoch.
            if start_epoch < self.perm.epoch:
                self.temp *= self.temp_ratio
                self.entropy_factor *= self.entropy_ratio
                if new_option_counter >= 1:
                    new_option_counter = 0
                    print("Adding a new option")
                    self.K += 1
                    for model in self.model_list:
                        model.add_option(self.optimizer)
                    self.new_option_hist.append(self.perm.epoch)
                    new_option_counter = 0
                if verbose:
                    print(f'Finished epoch {start_epoch}\twith loss: {negative_elbo_np:f}\tand k: {self.K:d}')
            if is_last:
                break


    def play_from_observation(self, option, obs):
        with torch.no_grad():
            state = torch.tensor(obs).unsqueeze(0).unsqueeze(0).to(self.device).float()
            o_vector = torch.zeros((1, 1, self.K)).to(self.device).float()
            o_vector[0,0,option] = 1
            if self.action_space == 'discrete':
                policy = self.policy.forward(state, o_vector).cpu()
            else:
                policy = self.policy.forward(state, o_vector)[0].cpu()
            termination = self.termination.forward(state, o_vector).cpu()
        return np.argmax(policy), termination


    def rollout(self, states):
        with torch.no_grad():
            policies = []
            terminations = []
            for option in range(self.K):
                o_vector = torch.zeros(self.K).to(self.device).float()
                o_vector[option] = 1
                o_vector = o_vector.repeat((states.shape[0], states.shape[1], 1))
                if self.action_space == 'discrete':
                    policy = self.policy.forward(states, o_vector).cpu()
                else:
                    mean, std = self.policy.forward(states, o_vector)
                    policy = mean.cpu(), std.cpu()
                policies.append(policy)
                termination = self.termination.forward(states, o_vector).cpu()
                terminations.append(termination)
        return policies, terminations


    def get_options_probas(self, states, actions):
        with torch.no_grad():
            eta, pre_sb_eta = self.high_level_posterior.sample_mean(return_pre_sb=True, nb_samples=30)  # [K] and [K]
            self.encoder.encode_trajectories(torch.cat([states, actions], axis=2))
            options, relaxed_opts = [], []
            terminations, relaxed_termins = [], []
            previous_option = torch.cat([
                torch.ones((states.shape[0], 1, 1), device=self.device),
                torch.reshape(eta, (1, 1, self.K)).repeat((states.shape[0], 1, 1))
            ], axis=2)
            for timestep in range(states.shape[1]):
                enc = self.encoder.forward(timestep, previous_option) # [batch, 1, K+1]
                options.append(enc[:, :, 1:])
                terminations.append(enc[:, :, 0:1])
                if self.relaxation_type == 'GS':
                    relaxed_opt_distr = torch.distributions.RelaxedOneHotCategorical(self.temp, logits=options[-1])
                    relaxed_termin_distr = torch.distributions.RelaxedBernoulli(self.temp, logits=terminations[-1])
                else:
                    raise NotImplementedError
                relaxed_opts.append(relaxed_opt_distr.rsample()) # [batch, 1, K]
                relaxed_termins.append(relaxed_termin_distr.rsample())  # [batch, 1, 1]
                previous_option = torch.cat([relaxed_termins[-1], relaxed_opts[-1]], axis=2)
            options = torch.cat(options, axis=1)
            relaxed_opts = torch.cat(relaxed_opts, axis=1)
            terminations = torch.cat(terminations, axis=1)
            relaxed_termins = torch.cat(relaxed_termins, axis=1)
            policies = self.policy.forward(states, relaxed_opts)
            termin_funcs = self.termination.forward(states[:, 1:], relaxed_opts[:, :-1])
        return {
            'options': options,
            'relaxed_opts': relaxed_opts,
            'terminations': terminations,
            'relaxed_termins': relaxed_termins,
            'policies': policies,
            'termin_funcs': termin_funcs,
        }

    def save(self, path):
        checkpoint = {
            'K': self.K,
            'encoder': self.encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'termination': self.termination.state_dict(),
            'high_level_posterior': self.high_level_posterior.state_dict(),
            'GS_temp': self.temp
        }
        torch.save(checkpoint, path)

    def load(self, path):
        checkpoint = torch.load(path)
        for _ in range(checkpoint['K']-self.K):
            for model in self.model_list:
                model.add_option(self.optimizer)
        self.K = checkpoint['K']
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.termination.load_state_dict(checkpoint['termination'])
        self.high_level_posterior.load_state_dict(checkpoint['high_level_posterior'])
        self.temp = checkpoint['GS_temp']
