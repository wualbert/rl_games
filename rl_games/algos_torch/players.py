from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch
from torch import nn
import numpy as np

import copy


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class PpoPlayerContinuous(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        # The network is here
        self.network = self.config["network"]
        self.actions_num = self.action_space.shape[0]
        self.actions_low = (
            torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        )
        self.actions_high = (
            torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        )
        self.mask = [False]

        self.normalize_input = self.config["normalize_input"]
        self.normalize_value = self.config.get("normalize_value", False)

        obs_shape = self.obs_shape
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "num_seqs": self.num_agents,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }
        if not hasattr(self.network, "__iter__"):
            # Single model is not iterable
            self.model = [self.network.build(config).to(self.device)]
            self.model[0].eval()
            self.is_rnn = self.model[0].is_rnn()
        else:
            # multiple models
            self.model = []
            for network in self.network:
                model = network.build(config).to(self.device)
                model.eval()
                self.model.append(model)
            self.is_rnn = self.model[0].is_rnn()

    def get_action(self, obs, is_deterministic=False):
        # get actions here
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)  # batch
        if self.policy_selection_idx is not None:
            assert self.policy_selection_idx == -1
            obs_trimmed = obs[:, : self.policy_selection_idx]
            policy_idxs = (
                obs[:, self.policy_selection_idx].to(torch.long).to(self.device)
            )
        else:
            obs_trimmed = obs
            policy_idxs = torch.zeros_like(
                obs[:, 0], dtype=torch.long, device=self.device
            )
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs_trimmed,
            "rnn_states": self.states,
        }
        res_mus = torch.zeros(self.num_policies, self.batch_size, self.actions_num).to(
            self.device
        )
        res_actions = torch.zeros(
            self.num_policies, self.batch_size, self.actions_num
        ).to(self.device)
        for policy_idx, model in enumerate(self.model):
            with torch.no_grad():
                # FIXME: why does the forward call modify the input dictionary?
                res_dict = model(copy.deepcopy(input_dict))
                res_mus[policy_idx] = res_dict["mus"]
                res_actions[policy_idx] = res_dict["actions"]

        # TODO: vectorize
        # Initialize with the first one, replace values later if needed
        if self.is_rnn:
            raise NotImplementedError
            self.states = res_dicts[0]["rnn_states"]
        mu = torch.take_along_dim(res_mus, policy_idxs.view(1, -1, 1), 0).squeeze(0)
        action = torch.take_along_dim(
            res_actions, policy_idxs.view(1, -1, 1), 0
        ).squeeze(0)
        # change the ones where policy_idx is not 0
        # for env_idx, policy_idx in enumerate(policy_idxs):
        #     mu[env_idx] = res_dicts[policy_idx]["mus"][env_idx]
        #     action[env_idx] = res_dicts[policy_idx]["actions"][env_idx]
        #     if self.is_rnn:
        #         self.states[env_idx] = res_dicts[policy_idx]["rnn_states"][env_idx]

        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())
        if self.clip_actions:
            return rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        else:
            return current_action

    def restore(self, fn):
        if isinstance(fn, str):
            checkpoint = torch_ext.load_checkpoint(fn)
            self.model[0].load_state_dict(checkpoint["model"])
            if self.normalize_input and "running_mean_std" in checkpoint:
                self.model[0].running_mean_std.load_state_dict(
                    checkpoint["running_mean_std"]
                )
            env_state = checkpoint.get("env_state", None)
            if self.env is not None and env_state is not None:
                self.env.set_env_state(env_state)
        else:
            for idx, fn_i in enumerate(fn):
                checkpoint = torch_ext.load_checkpoint(fn_i)
                self.model[idx].load_state_dict(checkpoint["model"])
                if self.normalize_input and "running_mean_std" in checkpoint:
                    self.model[idx].running_mean_std.load_state_dict(
                        checkpoint["running_mean_std"]
                    )
            # FIXME: the env_state should be the same across models?
            env_state = checkpoint.get("env_state", None)
            if self.env is not None and env_state is not None:
                raise NotImplementedError
                self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()


class PpoPlayerDiscrete(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)

        self.network = self.config["network"]
        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_num = self.action_space.n
            self.is_multi_discrete = False
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        self.mask = [False]
        self.normalize_input = self.config["normalize_input"]
        self.normalize_value = self.config.get("normalize_value", False)
        obs_shape = self.obs_shape
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "num_seqs": self.num_agents,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.is_rnn = self.model.is_rnn()

    def get_masked_action(self, obs, action_masks, is_deterministic=True):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        action_masks = torch.Tensor(action_masks).to(self.device).bool()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "action_masks": action_masks,
            "rnn_states": self.states,
        }
        self.model.eval()

        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict["logits"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if self.is_multi_discrete:
            if is_deterministic:
                action = [
                    torch.argmax(logit.detach(), axis=-1).squeeze() for logit in logits
                ]
                return torch.stack(action, dim=-1)
            else:
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:
                return action.squeeze().detach()

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)

        self.model.eval()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict["logits"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if self.is_multi_discrete:
            if is_deterministic:
                action = [
                    torch.argmax(logit.detach(), axis=1).squeeze() for logit in logits
                ]
                return torch.stack(action, dim=-1)
            else:
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:
                return action.squeeze().detach()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint["model"])
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

        env_state = checkpoint.get("env_state", None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()


class SACPlayer(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config["network"]
        self.actions_num = self.action_space.shape[0]
        self.action_range = [
            float(self.env_info["action_space"].low.min()),
            float(self.env_info["action_space"].high.max()),
        ]

        obs_shape = self.obs_shape
        self.normalize_input = False
        config = {
            "obs_dim": self.env_info["observation_space"].shape[0],
            "action_dim": self.env_info["action_space"].shape[0],
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": False,
            "normalize_input": self.normalize_input,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.sac_network.actor.load_state_dict(checkpoint["actor"])
        self.model.sac_network.critic.load_state_dict(checkpoint["critic"])
        self.model.sac_network.critic_target.load_state_dict(
            checkpoint["critic_target"]
        )
        if self.normalize_input and "running_mean_std" in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

        env_state = checkpoint.get("env_state", None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        dist = self.model.actor(obs)
        actions = dist.sample() if is_deterministic else dist.mean
        actions = actions.clamp(*self.action_range).to(self.device)
        if self.has_batch_dimension == False:
            actions = torch.squeeze(actions.detach())
        return actions

    def reset(self):
        pass
