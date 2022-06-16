import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import stable_log, sb
from functools import reduce


class DiscretePolicy(nn.Module):
    def __init__(self, d_states, K, d_actions, hidden_layer_sizes, device):
        super(DiscretePolicy, self).__init__()
        self.K = K
        self.d_actions = d_actions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.device = device
        layers = []
        self.prev_layer_size = d_states
        for h_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_features=self.prev_layer_size, out_features=h_size))
            self.prev_layer_size = h_size
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.output_layers = nn.ModuleList(
            [nn.Linear(self.prev_layer_size, d_actions) for _ in range(K)]
        )
        self.activation_layer = nn.Softmax(dim=2)

    def forward(self, x, option):
        x = self.net(x)
        outputs = [self.activation_layer(layer(x)).unsqueeze(2) for layer in self.output_layers]
        x = option.unsqueeze(-1) * torch.cat(outputs, dim=2)
        x = x.sum(dim=2)
        return x

    def add_option(self, optimizer):
        self.output_layers.append(nn.Linear(self.prev_layer_size, self.d_actions).to(self.device))
        self.K += 1
        optimizer.add_param_group({"params" : self.output_layers[-1].parameters()})


class ContinuousPolicy(nn.Module):
    def __init__(self, d_states, K, d_actions, hidden_layer_sizes, device, log_std_init=1.):
        super(ContinuousPolicy, self).__init__()
        self.K = K
        self.d_actions = d_actions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.device = device
        layers = []
        self.prev_layer_size = d_states
        for h_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_features=self.prev_layer_size, out_features=h_size))
            self.prev_layer_size = h_size
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.output_layers = nn.ModuleList(
            [nn.Linear(self.prev_layer_size, d_actions) for _ in range(K)]
        )
        self.activation_layer = nn.Tanh()
        self.log_std = nn.Parameter(torch.ones(d_actions) * log_std_init, requires_grad=True)

    def forward(self, x, option):
        x = self.net(x)
        outputs = [self.activation_layer(layer(x)).unsqueeze(2) for layer in self.output_layers]
        x = option.unsqueeze(-1) * torch.cat(outputs, dim=2)
        x = x.sum(dim=2)
        return x, self.log_std.repeat(x.shape[0], x.shape[1], 1)

    def add_option(self, optimizer):
        self.output_layers.append(nn.Linear(self.prev_layer_size, self.d_actions).to(self.device))
        self.K += 1
        optimizer.add_param_group({"params" : self.output_layers[-1].parameters()})


class Termination(nn.Module):
    def __init__(self, d_states, K, hidden_layer_sizes, device):
        super(Termination, self).__init__()
        self.K = K
        self.hidden_layer_sizes = hidden_layer_sizes
        self.device = device
        # variable size input layer
        layers = []
        self.prev_layer_size = d_states
        for h_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_features=self.prev_layer_size, out_features=h_size))
            self.prev_layer_size = h_size
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        layers.append(nn.Sigmoid())
        self.output_layers = nn.ModuleList([nn.Linear(self.prev_layer_size, 1) for _ in range(K)])
        self.activation_layer = nn.Sigmoid()

    def forward(self, x, option):
        x = self.net(x)
        outputs = [self.activation_layer(layer(x)).unsqueeze(2) for layer in self.output_layers]
        x = option.unsqueeze(-1) * torch.cat(outputs, dim=2)
        x = x.sum(dim=2)
        return x

    def add_option(self, optimizer):
        self.output_layers.append(nn.Linear(self.prev_layer_size, 1).to(self.device))
        self.K += 1
        optimizer.add_param_group({"params" : self.output_layers[-1].parameters()})


class Encoder(nn.Module):
    def __init__(self, d_states, d_actions, K, hidden_size_LSTM, hidden_layer_sizes, device, relaxation_type='GS'):
        super(Encoder, self).__init__()
        self.K = K
        self.device = device
        self.hidden_size_LSTM = hidden_size_LSTM
        self.relaxation_type = relaxation_type
        self.attention_net = nn.Sequential(
            nn.Linear(d_states + d_actions, hidden_size_LSTM),
            nn.ReLU(),
            nn.Linear(hidden_size_LSTM, hidden_size_LSTM),
            nn.LSTM(input_size=hidden_size_LSTM, hidden_size=hidden_size_LSTM, batch_first=True, num_layers=1),
        )
        # self.attention_net = nn.LSTM(input_size=d_states + d_actions, hidden_size=hidden_size_LSTM, batch_first=True)
        self.encoded_trajectories = None

        self.hidden_layer_sizes = hidden_layer_sizes
        # variable size input layer
        self.input_layer = nn.ParameterList(
            [nn.Parameter(torch.empty(hidden_layer_sizes[0], hidden_size_LSTM + 1))]
        )
        nn.init.xavier_normal_(self.input_layer[-1])
        for _ in range(self.K):
            self.input_layer.append(nn.Parameter(torch.empty(self.hidden_layer_sizes[0], 1, device=self.device)))
            nn.init.xavier_normal_(self.input_layer[-1])
        self.input_layer_bias = nn.Parameter(torch.zeros(hidden_layer_sizes[0]))
        layers = []
        self.prev_layer_size = hidden_layer_sizes[0]
        for h_size in hidden_layer_sizes[1:]:
            layers.append(nn.Linear(in_features=self.prev_layer_size, out_features=h_size))
            self.prev_layer_size = h_size
            layers.append(nn.ReLU())
        if layers == []:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(*layers)
        # variable size output layer
        if self.relaxation_type == 'GS':
            self.output_layer = nn.ParameterList([nn.Parameter(torch.empty(1, hidden_layer_sizes[-1]))])
            self.output_layer_bias = nn.ParameterList([nn.Parameter(torch.zeros(1))])
            nn.init.xavier_normal_(self.output_layer[-1])
            for _ in range(self.K):
                self.output_layer.append(nn.Parameter(torch.empty(1, self.hidden_layer_sizes[-1], device=self.device)))
                self.output_layer_bias.append(nn.Parameter(torch.empty(1, device=self.device)))
                nn.init.xavier_normal_(self.output_layer[-1])
            # output here has shape [batch, seq_length, K+1]
        else:
            raise NotImplementedError

    def forward(self, timestep, previous_option):
        # returns logits
        # concatenate eta to the current timestep lstm output
        x = torch.cat([self.encoded_trajectories[:, timestep:timestep+1, :], previous_option], axis=2)
        # x here has shape [batch, 1, hidden_size_LSTM + K + 1]
        x = F.relu(F.linear(x, reduce(lambda x,y: torch.cat((x,y), 1), self.input_layer)) + self.input_layer_bias)
        x = self.net(x)
        x = F.linear(x, reduce(lambda x,y: torch.cat((x,y), 0), self.output_layer)) + \
            reduce(lambda x,y: torch.cat((x,y), 0), self.output_layer_bias)
        # output here has shape [batch, 1, K+1]
        # x[:, :, 1:] = F.softmax(x[:, :, :-1], dim=-1)  # last K coordinates correspond to options
        # x[:, :, :1] = torch.sigmoid(x[:, :, -1:], dim=-1)  # first  coordinate correspond to termination
        return x

    def encode_trajectories(self, states_actions):
        x = self.attention_net(states_actions.flip(1))
        if type(x) == tuple:
            x, _ = x
        self.encoded_trajectories = x.flip(1)
        # encoded_trajectories has shape [batch, seq_length, hidden_size_LSTM]

    def add_option(self, optimizer):
        self.input_layer.append(nn.Parameter(torch.empty(self.hidden_layer_sizes[0], 1, device=self.device)))
        if self.relaxation_type == 'GS':
            self.output_layer.append(nn.Parameter(torch.empty(1, self.prev_layer_size, device=self.device)))
            self.output_layer_bias.append(nn.Parameter(torch.empty(1, device=self.device)))
        nn.init.xavier_normal_(self.input_layer[-1])
        nn.init.xavier_normal_(self.output_layer[-1])
        self.K += 1
        optimizer.add_param_group({"params" : [self.input_layer[-1], self.output_layer[-1], self.output_layer_bias[-1]]})


class StickBreakingKumaraswamy(nn.Module):
    """
    Contains parameters for K independent Kumaraswamy distributions and allows to sample each of them through the
    reparameterization trick. The stick breaking procedure is then applied.
    """
    def __init__(self, K, device):
        super(StickBreakingKumaraswamy, self).__init__()
        self.K = K
        self.device = device
        self.log_kuma_params = nn.ParameterList([nn.Parameter(torch.randn([1, 2]))])
        for _ in range(self.K-1):
            self.log_kuma_params.append(nn.Parameter(torch.randn(1, 2, device=self.device)))
        log_alpha_fixed = False
        if log_alpha_fixed:
            self.log_alpha = torch.tensor(2).float().to(device)
        else:
            self.log_alpha = nn.Parameter(torch.randn([1]))
        self.soft_plus = nn.Softplus()

    def add_option(self, optimizer):
        self.log_kuma_params.append(nn.Parameter(torch.randn(1, 2, device=self.device)))
        self.K += 1
        optimizer.add_param_group({"params" : self.log_kuma_params[-1]})

    def compute_kl(self, k=None, pre_sb=None, eps=10e-6):
        # returns an approximation of the KL between the product of Kumaraswamys and a product of Betas(1, alpha)
        # uses only the first k distributions and ignores the rest

        calculate_with_taylor_expansion = False
        log_kuma_params = reduce(lambda x,y: torch.cat((x,y), 0), self.log_kuma_params)
        kuma_params = torch.exp(log_kuma_params)
        alpha_param = torch.exp(self.log_alpha)
        if pre_sb is None:
            _, pre_sb = self.sample(return_pre_sb=True)
        if k is None:
            k = self.K

        if not calculate_with_taylor_expansion:
            # calc by samplings
            clamped_pre_sb = (pre_sb[:k]-0.5)*(1-2*eps) + 0.5
            beta_log_pdf = torch.distributions.Beta(1., alpha_param).log_prob(clamped_pre_sb)

            kuma_log_pdf = torch.sum(log_kuma_params[:k], axis=1) +\
                           (kuma_params[:k, 0] - 1.) * stable_log(clamped_pre_sb, eps) +\
                           (kuma_params[:k, 1] - 1.) * stable_log(1. - torch.pow(clamped_pre_sb, kuma_params[:k, 0]), eps)
            return torch.sum(kuma_log_pdf - beta_log_pdf)

        else:
            # calc taylor expansion
            beta =  alpha_param
            alpha = torch.tensor(1).float()

            kl = 1. /(1. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(1./kuma_params[:k, 0], kuma_params[:k, 1])
            kl += 1. /(2. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(2./kuma_params[:k, 0], kuma_params[:k, 1])
            kl += 1. / (3. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(3. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (4. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(4. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (5. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(5. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (6. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(6. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (7. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(7. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (8. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(8. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (9. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(9. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl += 1. / (10. + kuma_params[:k, 0] * kuma_params[:k, 1]) * self.beta_fn(10. / kuma_params[:k, 0],
                                                                                     kuma_params[:k, 1])
            kl *= (kuma_params[:k,1]) * (beta - 1)

            psi_b_taylor_approx = stable_log(kuma_params[:k, 1], eps) - 1. / (2 * kuma_params[:k, 1]) - 1. / (12 * torch.pow(kuma_params[:k, 1],2))
            kl += (kuma_params[:k, 0] - alpha) / kuma_params[:k, 0] * (-0.57721 - psi_b_taylor_approx - 1/kuma_params[:k, 1])
            kl += log_kuma_params[:k, 0] * log_kuma_params[:k, 1] + stable_log(self.beta_fn(alpha, beta), eps)


            kl += -(kuma_params[:k, 1] - 1) / kuma_params[:k, 1]

            return torch.sum(kl)

    def sample(self, return_pre_sb=False):
        log_kuma_params = reduce(lambda x,y: torch.cat((x,y), 0), self.log_kuma_params)
        kuma_params = torch.exp(log_kuma_params)
        u = torch.rand(self.K).to(self.device)
        pre_sb = torch.pow(1. - torch.pow(1. - u, 1. / kuma_params[:, 1]), 1. / kuma_params[:, 0])
        if return_pre_sb:
            return sb(pre_sb, self.device), pre_sb
        else:
            return sb(pre_sb, self.device)

    def sample_mean(self, return_pre_sb=False, nb_samples=20):
        log_kuma_params = reduce(lambda x,y: torch.cat((x,y), 0), self.log_kuma_params)
        kuma_params = torch.exp(log_kuma_params)
        u = torch.rand((nb_samples, self.K)).to(self.device)
        pre_sb = torch.pow(1. - torch.pow(1. - u, 1. / kuma_params[:, 1]), 1. / kuma_params[:, 0])
        post_sb = sb(pre_sb, self.device)
        pre_sb = torch.mean(pre_sb, dim=0)
        post_sb = torch.mean(post_sb, dim=0)
        if return_pre_sb:
            return post_sb, pre_sb
        else:
            return post_sb

    def beta_fn(self, a, b):
        return torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a+b))
