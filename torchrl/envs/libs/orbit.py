# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Union
import numpy as np
from torchrl.envs.libs.gym import GymWrapper, GymEnv
import torch
from torchrl.envs import make_composite_from_td
from torch import Tensor
from numpy import ndarray
from tensordict import TensorDict
import itertools
import gym

class OrbitEnv(GymEnv):
        def __init__(self, env_name, **kwargs):
                # kwargs["batch_size"] = torch.Size([kwargs["cfg"]["env"]["num_envs"]])
                num_envs = kwargs["cfg"]["env"]["num_envs"]
                super().__init__(env_name, batch_size=torch.Size([num_envs]), **kwargs)
                
                # self._env = gym.wrappers.RecordVideo(
                #         self._env, "videos", step_trigger=lambda step: step % 1000 == 0, video_length=250
                #     )

        def _make_specs(self, env: "gym.Env") -> None:
            super()._make_specs(env, batch_size=self.batch_size)
            # remove batch from 'pixels' observation (one viewport instead of batch_size)
            if "pixels" in self.observation_spec:
                self.observation_spec["pixels"].shape = self.observation_spec["pixels"].shape[1:]
    
        def read_obs(
            self, observations: Union[Dict[str, Any], torch.Tensor, np.ndarray]
        ) -> Dict[str, Any]:
            """Reads an observation from the environment and returns an observation compatible with the output TensorDict.

            Args:
                observations (observation under a format dictated by the inner env): observation to be read.

            """

            if isinstance(observations, dict):
                if "policy" in observations and "observation" not in observations:
                    observations["observation"] = observations.pop("policy")
                if "state" in observations and "observation" not in observations:
                    observations["observation"] = observations.pop("state")

            if not isinstance(observations, (TensorDict, dict)):
                (key,) = itertools.islice(self.observation_spec.keys(True, True), 1)
                observations = {key: observations}
                      
            return observations
        
        def read_done(self, done):
            return done.bool(), done.any()
        
        def read_action(self, action): 
            return action
