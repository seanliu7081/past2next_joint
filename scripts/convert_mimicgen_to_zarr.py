"""
Convert MimicGen HDF5 dataset to OAT's zarr format.

Usage:
    python scripts/convert_mimicgen_to_zarr.py \
        --hdf5_path data/robomimic/datasets/stack_three_d1/stack_three_d1.hdf5 \
        --zarr_path data/mimicgen/stack_three_d1_N100.zarr \
        -n 100

Zarr structure produced:
    data/
        action:                    (total_steps, 7) float32
        agentview_image:           (total_steps, 84, 84, 3) uint8
        robot0_eye_in_hand_image:  (total_steps, 84, 84, 3) uint8
        robot0_eef_pos:            (total_steps, 3) float32
        robot0_eef_quat:           (total_steps, 4) float32
        robot0_gripper_qpos:       (total_steps, 2) float32
    meta/
        episode_ends:              (n_episodes,) int64
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib
import click
import h5py
import numpy as np
import zarr
import tqdm

from oat.common.replay_buffer import ReplayBuffer


OBS_KEYS = [
    'agentview_image',
    'robot0_eye_in_hand_image',
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
]


def convert_mimicgen_hdf5_to_zarr(
    hdf5_path: str,
    obs_keys: list = OBS_KEYS,
    num_demos: int = None,
) -> ReplayBuffer:
    replay_buffer = ReplayBuffer.create_empty_zarr()

    with h5py.File(hdf5_path, 'r') as f:
        data_grp = f['data']
        # discover demo keys sorted numerically
        demo_keys = sorted(
            [k for k in data_grp.keys() if k.startswith('demo_')],
            key=lambda k: int(k.split('_')[1]),
        )
        total_demos = len(demo_keys)
        if num_demos is None:
            num_demos = total_demos
        num_demos = min(num_demos, total_demos)

        selected = np.random.choice(total_demos, num_demos, replace=False)
        selected_keys = [demo_keys[i] for i in sorted(selected)]

        for demo_key in tqdm.tqdm(selected_keys, desc="Converting MimicGen"):
            demo = data_grp[demo_key]
            episode_data = {
                'action': demo['actions'][:].astype(np.float32),
            }
            for obs_key in obs_keys:
                arr = demo['obs'][obs_key][:]
                # keep images as uint8, cast proprioception to float32
                if arr.dtype == np.uint8:
                    episode_data[obs_key] = arr
                else:
                    episode_data[obs_key] = arr.astype(np.float32)
            replay_buffer.add_episode(episode_data)

    print(f"Converted {replay_buffer.n_episodes} episodes, "
          f"{replay_buffer.n_steps} total steps")
    return replay_buffer


@click.command()
@click.option('--hdf5_path', type=str, required=True,
              help='Path to MimicGen HDF5 dataset')
@click.option('--zarr_path', type=str, default=None,
              help='Output zarr path (auto-generated if omitted)')
@click.option('-n', '--num_demos', type=int, default=None,
              help='Number of demos to convert (default: all)')
@click.option('--seed', type=int, default=42,
              help='Random seed for demo sampling')
def main(hdf5_path, zarr_path, num_demos, seed):
    np.random.seed(seed)

    hdf5_path = os.path.expanduser(hdf5_path)
    if not os.path.isfile(hdf5_path):
        raise FileNotFoundError(f"HDF5 not found: {hdf5_path}")

    replay_buffer = convert_mimicgen_hdf5_to_zarr(
        hdf5_path=hdf5_path,
        num_demos=num_demos,
    )

    # auto-generate zarr_path if not given
    if zarr_path is None:
        task_name = pathlib.Path(hdf5_path).stem
        zarr_path = f"data/mimicgen/{task_name}_N{replay_buffer.n_episodes}.zarr"

    zarr_path = os.path.expanduser(zarr_path)
    if os.path.exists(zarr_path):
        resp = input(f"{zarr_path} already exists. Overwrite? [y/N]: ").strip().lower()
        if resp != 'y':
            print("Abort")
            return
        import shutil
        shutil.rmtree(zarr_path)

    pathlib.Path(zarr_path).mkdir(parents=True, exist_ok=True)
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
    replay_buffer.save_to_path(zarr_path, compressor=compressor)
    print(f"Saved to {zarr_path}")


if __name__ == "__main__":
    main()
