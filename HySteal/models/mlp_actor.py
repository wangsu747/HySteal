
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal, MultivariateNormal

from custom.MultiOneHotCategorical import MultiOneHotCategorical
from custom.MultiSoftMax import MultiSoftMax
from utils.torch_util import resolve_activate_function


def check_data(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
class Actor(nn.Module):
    def __init__(self, num_states, num_actions, num_discrete_actions=0, discrete_actions_sections: Tuple = (0,),
                 action_log_std=0, use_multivariate_distribution=False,
                 num_hiddens: Tuple = (64, 64), activation: str = "relu",
                 drop_rate=None, num_states_2 =64):















        super(Actor, self).__init__()
        self.num_states = num_states
        self.num_states_2 = num_states_2
        self.num_actions = num_actions
        self.drop_rate = drop_rate
        self.use_multivariate_distribution = use_multivariate_distribution
        self.num_discrete_actions = num_discrete_actions
        assert sum(discrete_actions_sections) == num_discrete_actions, f"Expected sum of discrete actions's " \
                                                                       f"dimension =  {num_discrete_actions}"
        self.discrete_action_sections = discrete_actions_sections

        self.action_log_std = nn.Parameter(torch.ones(1, self.num_actions - self.num_discrete_actions) * action_log_std,
                                           requires_grad=True)


        _module_units1 = [num_states]
        _module_units1.extend(num_hiddens)
        _module_units1 += (64,)

        self._layers_units1 = [(_module_units1[i], _module_units1[i + 1]) for i in range(len(_module_units1) - 1)]
        activation1 = resolve_activate_function(activation)

        self._module_list1 = nn.ModuleList()
        for idx, module_unit in enumerate(self._layers_units1):
            n_units_in, n_units_out = module_unit
            self._module_list1.add_module(f"Branch1_Layer_{idx + 1}_Linear", nn.Linear(n_units_in, n_units_out))
            if idx != len(self._layers_units1) - 1:
                self._module_list1.add_module(f"Branch1_Layer_{idx + 1}_Activation", activation1())
                self._module_list1.add_module(f"Branch1_Layer_{idx + 1}_LayerNorm", nn.LayerNorm(n_units_out))
            if self.drop_rate and idx != len(self._layers_units1) - 1:
                self._module_list1.add_module(f"Branch1_Layer_{idx + 1}_Dropout", nn.Dropout(self.drop_rate))

        self._module_list1.add_module(f"Branch1_Layer_{idx + 1}_Activation", nn.Tanh())


        _module_units2 = [num_states_2]
        _module_units2.extend(num_hiddens)
        _module_units2 += (64,)

        self._layers_units2 = [(_module_units2[i], _module_units2[i + 1]) for i in range(len(_module_units2) - 1)]
        activation2 = resolve_activate_function(activation)

        self._module_list2 = nn.ModuleList()
        for idx, module_unit in enumerate(self._layers_units2):
            n_units_in, n_units_out = module_unit
            self._module_list2.add_module(f"Branch2_Layer_{idx + 1}_Linear", nn.Linear(n_units_in, n_units_out))
            if idx != len(self._layers_units2) - 1:
                self._module_list2.add_module(f"Branch2_Layer_{idx + 1}_Activation", activation2())
                self._module_list2.add_module(f"Branch2_Layer_{idx + 1}_LayerNorm", nn.LayerNorm(n_units_out))
            if self.drop_rate and idx != len(self._layers_units2) - 1:
                self._module_list2.add_module(f"Branch2_Layer_{idx + 1}_Dropout", nn.Dropout(self.drop_rate))

        self._module_list2.add_module(f"Branch2_Layer_{idx + 1}_Activation", nn.Tanh())


        _module_units_combined = [128]
        _module_units_combined.extend(num_hiddens)
        _module_units_combined += (self.num_actions,)

        self._layers_units_combined = [(_module_units_combined[i], _module_units_combined[i + 1]) for i in
                                       range(len(_module_units_combined) - 1)]
        activation_combined = resolve_activate_function(activation)

        self._module_list_combined = nn.ModuleList()
        for idx, module_unit in enumerate(self._layers_units_combined):
            n_units_in, n_units_out = module_unit
            self._module_list_combined.add_module(f"Combined_Layer_{idx + 1}_Linear",
                                                  nn.Linear(n_units_in, n_units_out))
            if idx != len(self._layers_units_combined) - 1:
                self._module_list_combined.add_module(f"Combined_Layer_{idx + 1}_Activation", activation_combined())
                self._module_list_combined.add_module(f"Combined_Layer_{idx + 1}_LayerNorm", nn.LayerNorm(n_units_out))
            if self.drop_rate and idx != len(self._layers_units_combined) - 1:
                self._module_list_combined.add_module(f"Combined_Layer_{idx + 1}_Dropout", nn.Dropout(self.drop_rate))

        self._module_list_combined.add_module(f"Combined_Layer_{idx + 1}_Activation", nn.Tanh())

        if self.num_discrete_actions:
            self._module_list_combined.add_module(f"Layer_{idx + 1}_Custom_Softmax",
                                                  MultiSoftMax(0, self.num_discrete_actions,
                                                               self.discrete_action_sections))



    def forward(self, x, obs):





        for module1 in self._module_list1:
            x = module1(x)
            check_data(x, "x after module1")

        for module2 in self._module_list2:
            obs = module2(obs)
            check_data(obs, "obs after module2")

        combined_input = torch.cat((x, obs), dim=1)
        for module3 in self._module_list_combined:
            combined_input = module3(combined_input)
            check_data(combined_input, "combined_input after module3")
        dist_discrete = None
        if self.num_discrete_actions:
            dist_discrete = MultiOneHotCategorical(combined_input[..., :self.num_discrete_actions],
                                                   sections=self.discrete_action_sections)

        continuous_action_mean = combined_input[..., self.num_discrete_actions:]
        continuous_action_log_std = self.action_log_std.expand_as(continuous_action_mean)
        continuous_action_std = torch.exp(continuous_action_log_std)
        if self.use_multivariate_distribution:
            dist_continuous = MultivariateNormal(continuous_action_mean, torch.diag_embed(continuous_action_std))
        else:
            dist_continuous = Normal(continuous_action_mean, continuous_action_std)

        return dist_discrete, dist_continuous

    def get_action_log_prob(self, states,obs):








        dist_discrete, dist_continuous = self.forward(states,obs)
        eps = 1e-6
        action = dist_continuous.sample()

        if self.use_multivariate_distribution:
            log_prob = dist_continuous.log_prob(action)
        else:
            log_prob = dist_continuous.log_prob(action).sum(dim=-1)
        if dist_discrete:
            discrete_action = dist_discrete.sample()
            discrete_log_prob = dist_discrete.log_prob(discrete_action)
            action = torch.cat([discrete_action, action], dim=-1)

            """
            How to deal with log prob?
            
            1. Add discrete log_prob and continuous log_prob, consider their dependency;
            2. Concat them together
            """
            log_prob = (log_prob + discrete_log_prob)

        log_prob.unsqueeze_(-1)
        return action, log_prob


    def agent_get_action_log_prob(self, states,obs):








        dist_discrete, dist_continuous = self.forward(states,obs)
        action = dist_continuous.sample()
        eps = 1e-6


        if self.use_multivariate_distribution:
            log_prob = dist_continuous.log_prob(action)
        else:
            log_prob = dist_continuous.log_prob(action).sum(dim=-1)
        action = (torch.tanh(action) + 1) / 2

        action = action * 4 + 1

        action = torch.round(action)
        if dist_discrete:
            discrete_action = dist_discrete.sample()
            discrete_log_prob = dist_discrete.log_prob(discrete_action)
            action = torch.cat([discrete_action, action], dim=-1)

            """
            How to deal with log prob?

            1. Add discrete log_prob and continuous log_prob, consider their dependency;
            2. Concat them together
            """
            log_prob = (log_prob + discrete_log_prob)

        log_prob.unsqueeze_(-1)
        return action, log_prob

    def get_log_prob(self, states, actions,obs):







        dist_discrete, dist_continuous = self.forward(states,obs)
        if self.use_multivariate_distribution:
            log_prob = dist_continuous.log_prob(actions[..., self.num_discrete_actions:])
        else:
            log_prob = dist_continuous.log_prob(actions[..., self.num_discrete_actions:]).sum(dim=-1)
        if dist_discrete:
            discrete_log_prob = dist_discrete.log_prob(actions[..., :self.num_discrete_actions])
            log_prob = log_prob + discrete_log_prob
        return log_prob.unsqueeze(-1)

    def get_entropy(self, states,obs):





        dist_discrete, dist_continuous = self.forward(states,obs)
        ent_discrete = dist_discrete.entropy()
        ent_continuous = dist_continuous.entropy()

        ent = torch.cat([ent_discrete, ent_continuous], dim=-1).unsqueeze_(-1)
        return ent

    def get_kl(self, states):





        pass
