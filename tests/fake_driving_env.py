"""
Fake driving environment for integration testing.

Matches CARLA's interface (continuous actions, image observations)
without requiring CARLA server or GPU. Uses old gym API (4-tuple step)
per dreamerv3-torch's convention.

This is the HIGHEST VALUE piece of test infrastructure in the project.
It enables testing the full Dreamer pipeline on CPU without any
external dependencies.
"""

import gym
import gym.spaces
import numpy as np


class FakeDrivingEnv(gym.Env):
    """
    Minimal driving env matching CARLA's observation/action contract.

    Observations:
        image: (H, W, 3) uint8 — random noise "camera" image

    Actions:
        Box(-1, 1, shape=(2,)) — [steer, throttle]

    Reward:
        Small random value + bonus for "going straight" (action[0] near 0).
        Terminates randomly with low probability to simulate collisions.

    Episode:
        Max 100 steps (short for fast testing).
    """

    metadata = {"render.modes": []}

    def __init__(self, image_size=(64, 64), max_steps=100, seed=None):
        super().__init__()
        self.image_size = image_size
        self.max_steps = max_steps
        self._step_count = 0
        self._rng = np.random.RandomState(seed)

        # Observation space: dict with 'image' + dreamerv3-torch required keys
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0, high=255,
                shape=(image_size[0], image_size[1], 3),
                dtype=np.uint8,
            ),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
        })

        # Action space: continuous [steer, throttle]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _make_obs(self, is_first=False, is_terminal=False):
        """Generate an observation dict with required dreamerv3-torch keys."""
        image = self._rng.randint(
            0, 256,
            size=(self.image_size[0], self.image_size[1], 3),
            dtype=np.uint8,
        )
        return {
            "image": image,
            "is_first": is_first,
            "is_terminal": is_terminal,
        }

    def reset(self):
        """Reset and return initial observation."""
        self._step_count = 0
        return self._make_obs(is_first=True)

    def step(self, action):
        """
        Take one step.

        Returns: (obs, reward, done, info) — old gym 4-tuple API.
        """
        self._step_count += 1

        # Simple reward: small base + bonus for straight driving
        reward = 0.1 + 0.5 * (1.0 - abs(float(action[0])))

        # Random "collision" termination (5% chance)
        collision = self._rng.random() < 0.05

        # Episode done if collision or max steps
        done = collision or (self._step_count >= self.max_steps)

        obs = self._make_obs(is_first=False, is_terminal=collision)

        info = {
            "collision": collision,
            "step": self._step_count,
        }

        if done and not collision:
            info["discount"] = np.array(1.0, dtype=np.float32)
        elif collision:
            info["discount"] = np.array(0.0, dtype=np.float32)

        return obs, float(reward), done, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
