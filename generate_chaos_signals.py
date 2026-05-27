# %%
"""Generate Lorenz and Rossler signal datasets.

The training and analysis scripts expect paired files named
``Chaos_Signals/data1_<id>.npy`` and ``Chaos_Signals/data2_<id>.npy``.
Here ``data1`` is the normalized Lorenz trajectory and ``data2`` is the
normalized Rossler trajectory.  Each output has shape
``(data_size, wave_length - burn_in, 3)``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from sklearn import preprocessing


# ================================
# Settings from the original notebooks
# ================================
DATA_SIZE = 5000
RAW_WAVE_LENGTH = 14000
BURN_IN = 4000
LORENZ_TIME_LEN = 100.0
ROSSLER_TIME_LEN = 700.0
LORENZ_ARGS = (10.0, 28.0, 8.0 / 3.0)
ROSSLER_ARGS = (0.2, 0.2, 5.7)
OUTPUT_DIR = Path("Chaos_Signals")
OUTPUT_DTYPE = "float32"


def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    return [-sigma * x + sigma * y, -x * z + rho * x - y, x * y - beta * z]


def rossler(t, state, a, b, c):
    x, y, z = state
    return [-y - z, x + a * y, b + x * z - c * z]


def solve_normalized_trajectory(task):
    system, init, end_time, wave_length, system_args, method = task
    t_eval = np.linspace(0.0, end_time, wave_length)
    solution = solve_ivp(
        system,
        [0.0, end_time],
        init,
        method=method,
        t_eval=t_eval,
        args=system_args,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    normalized = preprocessing.minmax_scale(solution.y, axis=1)
    return normalized.T


def generate_initial_conditions(rng, data_size):
    return rng.uniform(0.0, 1.0, size=(data_size, 3))


def generate_system(system, initial_conditions, end_time, wave_length, system_args, method, workers):
    tasks = [
        (system, init, end_time, wave_length, system_args, method)
        for init in initial_conditions
    ]
    if workers == 1:
        trajectories = [solve_normalized_trajectory(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            trajectories = list(executor.map(solve_normalized_trajectory, tasks, chunksize=100))
    return np.asarray(trajectories)


def generate_pair(data_id, args):
    rng = np.random.default_rng(None if args.seed is None else args.seed + data_id)
    lorenz_init = generate_initial_conditions(rng, args.data_size)
    rossler_init = generate_initial_conditions(rng, args.data_size)

    print(f"[{data_id}] generating Lorenz trajectories")
    lorenz_data = generate_system(
        lorenz,
        lorenz_init,
        args.lorenz_time_len,
        args.raw_wave_length,
        LORENZ_ARGS,
        args.method,
        args.workers,
    )

    print(f"[{data_id}] generating Rossler trajectories")
    rossler_data = generate_system(
        rossler,
        rossler_init,
        args.rossler_time_len,
        args.raw_wave_length,
        ROSSLER_ARGS,
        args.method,
        args.workers,
    )

    lorenz_data = lorenz_data[:, args.burn_in :, :].astype(args.dtype, copy=False)
    rossler_data = rossler_data[:, args.burn_in :, :].astype(args.dtype, copy=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / f"data1_{data_id}.npy", lorenz_data)
    np.save(args.output_dir / f"data2_{data_id}.npy", rossler_data)
    print(
        f"[{data_id}] saved data1/data2 with shapes "
        f"{lorenz_data.shape} and {rossler_data.shape}"
    )


def generate_pair_from_task(task):
    data_id, args = task
    generate_pair(data_id, args)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=103)
    parser.add_argument("--data-size", type=int, default=DATA_SIZE)
    parser.add_argument("--raw-wave-length", type=int, default=RAW_WAVE_LENGTH)
    parser.add_argument("--burn-in", type=int, default=BURN_IN)
    parser.add_argument("--lorenz-time-len", type=float, default=LORENZ_TIME_LEN)
    parser.add_argument("--rossler-time-len", type=float, default=ROSSLER_TIME_LEN)
    parser.add_argument("--method", default="RK45")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of worker processes used for trajectories within each system.",
    )
    parser.add_argument(
        "--pair-workers",
        type=int,
        default=1,
        help="Number of dataset IDs generated in parallel.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=OUTPUT_DTYPE,
        help="Dtype used for saved .npy files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_ids = range(args.start_id, args.end_id + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pair_workers == 1:
        for data_id in data_ids:
            generate_pair(data_id, args)
    else:
        with ProcessPoolExecutor(max_workers=args.pair_workers) as executor:
            list(executor.map(generate_pair_from_task, zip(data_ids, repeat(args))))


if __name__ == "__main__":
    main()
