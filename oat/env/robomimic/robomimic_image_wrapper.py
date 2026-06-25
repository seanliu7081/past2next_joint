"""
Gymnasium-compatible wrapper for robomimic / robosuite environments.

Bridges two API conventions:
    - robomimic (old gym): reset() -> obs,  step() -> (obs, rew, done, info)
    - OAT infra (gymnasium): reset() -> (obs, info),  step() -> (obs, rew, term, trunc, info)

Image observations are converted from robomimic's CHW float [0,1]
to HWC uint8 [0,255] for consistency with OAT's ZarrDataset and
RobomimicRgbEncoder (which expects [B, To, H, W, C] input).
"""
from typing import Optional, Dict
import numpy as np
import gymnasium
from gymnasium import spaces
from robomimic.envs.env_robosuite import EnvRobosuite


class RobomimicImageWrapperGymnasium(gymnasium.Env):
    """Wraps a robomimic EnvRobosuite into a gymnasium-compatible env.

    Observations:
        rgb ports  -> (H, W, C) uint8, values in [0, 255]
        state ports -> (D,) float32
    """

    def __init__(
        self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key: str = 'agentview_image',
    ):
        super().__init__()
        self.env = env
        self.shape_meta = shape_meta
        self.init_state = init_state
        self.render_obs_key = render_obs_key
        self.render_cache = None
        self.has_reset_before = False
        self._seed = None
        self.seed_state_map = dict()

        # --- parse shape_meta into rgb / state keys ---
        self.rgb_keys = []
        self.state_keys = []
        for key, attr in shape_meta['obs'].items():
            port_type = attr.get('type', 'state')
            if port_type == 'rgb':
                self.rgb_keys.append(key)
            elif port_type == 'state':
                self.state_keys.append(key)

        # --- build observation & action spaces ---
        action_shape = shape_meta['action']['shape']
        self.action_space = spaces.Box(
            low=-1, high=1, shape=action_shape, dtype=np.float32)

        obs_spaces = {}
        for key, attr in shape_meta['obs'].items():
            shape = tuple(attr['shape'])
            port_type = attr.get('type', 'state')
            if port_type == 'rgb':
                obs_spaces[key] = spaces.Box(
                    low=0, high=255, shape=shape, dtype=np.uint8)
            else:
                obs_spaces[key] = spaces.Box(
                    low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

    # ---------- internal helpers ----------

    def _raw_to_obs(self, raw_obs: dict) -> dict:
        """Convert robomimic raw observation to OAT-compatible format."""
        obs = {}
        for key in self.rgb_keys:
            # robomimic returns images as CHW float [0, 1]
            img = raw_obs[key]                          # (C, H, W) float
            img = np.moveaxis(img, 0, -1)               # (H, W, C)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            obs[key] = img
        for key in self.state_keys:
            obs[key] = raw_obs[key].astype(np.float32)

        # cache for render()
        self.render_cache = obs.get(self.render_obs_key, None)
        return obs

    # ---------- gymnasium API ----------

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = seed

        if self.init_state is not None:
            if not self.has_reset_before:
                self.env.reset()
                self.has_reset_before = True
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            s = self._seed
            if s in self.seed_state_map:
                raw_obs = self.env.reset_to({'states': self.seed_state_map[s]})
            else:
                np.random.seed(seed=s)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[s] = state
            self._seed = None
        else:
            raw_obs = self.env.reset()

        obs = self._raw_to_obs(raw_obs)
        return obs, {}

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self._raw_to_obs(raw_obs)
        # gymnasium convention: terminated, truncated
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self, mode='rgb_array', **kwargs):
        if self.render_cache is None:
            raise RuntimeError('Must call reset() or step() before render().')
        return self.render_cache  # already HWC uint8

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

    @property
    def task_name(self):
        """Convenience for runner init_fn pattern."""
        return getattr(self.env, 'name', 'robomimic_env')
