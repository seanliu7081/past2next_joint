"""
Environment runner for robomimic / MimicGen single-task evaluation.

Follows OAT's BaseRunner interface and uses OAT's gymnasium infrastructure
(MultiStepWrapper, AsyncVectorEnv, VideoRecordingWrapper).

Modelled after:
    - oat.env_runner.libero_runner.LiberoRunner  (OAT's runner pattern)
    - equi_diffpo.env_runner.robomimic_image_runner  (robomimic env setup)
"""
import os
import collections
import math
import pathlib

import wandb
import wandb.sdk.data_types.video as wandb_video
import numpy as np
import torch
import tqdm
import dill
import h5py

from oat.gymnasium_util.multistep_wrapper import MultiStepWrapper
from oat.gymnasium_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from oat.gymnasium_util.async_vector_env import AsyncVectorEnv
from oat.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapperGymnasium
from oat.env_runner.base_runner import BaseRunner
from oat.policy.base_policy import BasePolicy
from oat.common.pytorch_util import dict_apply

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

# register mimicgen task environments with robosuite
try:
    import mimicgen_envs
except ImportError:
    print("Warning: mimicgen_envs not installed. "
          "MimicGen-specific tasks will not be available.")


def _create_robomimic_env(env_meta, shape_meta, enable_render=True):
    """Create a robomimic env with OAT-compatible modality mapping."""
    # map OAT type names to robomimic modality names
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        oat_type = attr.get('type', 'state')
        if oat_type == 'rgb':
            modality_mapping['rgb'].append(key)
        elif oat_type == 'state':
            modality_mapping['low_dim'].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


def _maybe_to_torch(x, device, dtype):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    return x


class RobomimicRunner(BaseRunner):
    """
    Evaluation runner for robomimic / MimicGen tasks.

    Creates test environments with random seeds, runs policy rollouts,
    and logs success rates and videos.

    Returns log_data with key 'mean_success_rate' compatible with
    OAT's checkpoint monitor.
    """

    def __init__(
        self,
        output_dir: str,
        dataset_path: str,
        shape_meta: dict,
        n_train: int = 6,
        n_train_vis: int = 2,
        train_start_idx: int = 0,
        n_test: int = 50,
        n_test_vis: int = 4,
        test_start_seed: int = 100000,
        n_obs_steps: int = 2,
        n_action_steps: int = 16,
        max_episode_steps: int = 400,
        fps: int = 10,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        n_parallel_envs: int = None,
        render_obs_key: str = 'agentview_image',
    ):
        super().__init__(output_dir)

        n_total = n_train + n_test
        if n_parallel_envs is None:
            n_parallel_envs = n_total
        n_parallel_envs = min(n_parallel_envs, n_total)

        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # --- read env metadata from dataset ---
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta['env_kwargs']['use_object_obs'] = False

        # --- build env factory functions ---
        def make_env_fn():
            robomimic_env = _create_robomimic_env(
                env_meta=env_meta,
                shape_meta=shape_meta,
                enable_render=True,
            )
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapperGymnasium(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps, codec='h264',
                        input_pix_fmt='rgb24', crf=crf,
                        thread_type='FRAME', thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
                reward_agg_method='max',
            )

        def make_dummy_env_fn():
            robomimic_env = _create_robomimic_env(
                env_meta=env_meta,
                shape_meta=shape_meta,
                enable_render=False,
            )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapperGymnasium(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps, codec='h264',
                        input_pix_fmt='rgb24', crf=crf,
                        thread_type='FRAME', thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
                reward_agg_method='max',
            )

        # --- set up per-env init functions ---
        env_fns = [make_env_fn] * n_parallel_envs
        env_seeds = []
        env_prefixes = []
        env_init_fn_dills = []

        # train rollouts: replay on known demo initial states
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state,
                            enable_render=enable_render):
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wandb_video.util.generate_id() + '.mp4')
                        filename.parent.mkdir(parents=True, exist_ok=True)
                        env.env.file_path = str(filename)

                    assert isinstance(env.env.env, RobomimicImageWrapperGymnasium)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixes.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        # test rollouts: random seeds
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wandb_video.util.generate_id() + '.mp4')
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    env.env.file_path = str(filename)

                assert isinstance(env.env.env, RobomimicImageWrapperGymnasium)
                env.env.env.init_state = None
                env.env.env.seed(seed)

            env_seeds.append(seed)
            env_prefixes.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # --- create vectorized env ---
        # Use 'forkserver' to avoid inheriting EGL context from main process.
        # fork() causes EGL_BAD_ALLOC when workers try to create new GL contexts.
        env = AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            dummy_env_fn=make_dummy_env_fn,
            context='forkserver',
        )

        # --- store attributes ---
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixes = env_prefixes
        self.env_init_fn_dills = env_init_fn_dills
        self.env_meta = env_meta
        self.n_train = n_train
        self.n_test = n_test
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.shape_meta = shape_meta

    @torch.inference_mode()
    def run(self, policy: BasePolicy, **kwargs) -> dict:
        device = policy.device
        dtype = policy.dtype
        policy_name = policy.get_policy_name()
        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_success = [False] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each(
                'run_dill_function',
                args_list=[(x,) for x in this_init_fns],
            )

            # start rollout
            obs, _ = env.reset()
            policy.reset()

            env_name = self.env_meta.get('env_name', 'robomimic')
            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Eval {policy_name} {env_name} {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            while not done and pbar.n < pbar.total:
                # convert obs to torch
                obs_dict = dict_apply(
                    obs,
                    lambda x: _maybe_to_torch(x, device=device, dtype=dtype),
                )

                # run policy — select only the ports the policy uses
                with torch.inference_mode():
                    action = policy.predict_action({
                        port: obs_dict[port]
                        for port in policy.get_observation_ports()
                    }, **kwargs)['action'].detach().cpu().numpy()

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("NaN or Inf action")

                # step env
                obs, reward, terminated, _, _ = env.step(action)

                # track success (reward >= 1 means task success)
                for j in range(this_n_active_envs):
                    if reward[j] >= 1.0:
                        all_success[start + j] = True

                # check if all active envs are done
                done_flags = np.array([
                    terminated[j] or all_success[start + j]
                    for j in range(this_n_active_envs)
                ])
                done = np.all(done_flags)

                pbar.update(action.shape[1])
            pbar.close()

            # collect videos
            all_video_paths[this_global_slice] = env.render()[this_local_slice]

        # clear video buffer
        _ = env.reset()

        # --- build log ---
        log_data = dict()

        # per-episode videos
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixes[i]
            video_path = all_video_paths[i]
            if video_path is not None:
                log_data[f'{prefix}video_{seed}'] = wandb.Video(
                    video_path, format='mp4')

        # split train / test success
        train_success = [all_success[i] for i in range(n_inits)
                         if self.env_prefixes[i] == 'train/']
        test_success = [all_success[i] for i in range(n_inits)
                        if self.env_prefixes[i] == 'test/']

        if train_success:
            log_data['train/mean_success_rate'] = np.mean(train_success).item()
        if test_success:
            log_data['test/mean_success_rate'] = np.mean(test_success).item()

        # aggregate: mean_success_rate uses TEST only (OAT checkpoint monitor key)
        log_data['mean_success_rate'] = np.mean(test_success).item() if test_success else 0.0
        log_data['test_mean_score'] = log_data['mean_success_rate']

        return log_data

    def close(self):
        self.env.close()
