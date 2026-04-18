"""Run train_for_search.py locally without cluster_utils or slurm.

Usage:
    conda activate depRLCaT
    cd /home/franzleo/Projects/depRLCaT
    python cluster_utils/hyfydy_search/debug_local.py [env_name]

Stubs initialize_job/finalize_job with a fabricated params dict and shortens
training (steps=5e4, parallel=2) so errors surface in under a minute.
Any unhandled exception prints a full Python traceback.
"""

import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import train_for_search  # noqa: E402


FAKE_PARAMS = {
    "id": 0,
    "working_dir": tempfile.mkdtemp(prefix="hyfydy_debug_"),
    # Optimized params -- any valid option from the search space.
    "max_velocity_cost_coeff": 1.0,
    "nmuscle_cost_coeff": 1.0,
    "smooth_cost_coeff": 0.0,
    "target_grf": 1.4,
    "target_delta": 0.1,
    "target_muscle_usage_percentage": 0.33,
    "target_hip_torque": 80,
    "target_knee_torque": 80,
    "target_ankle_torque": 80,
    "target_lumbar_torque_trunk": 80,
    "target_lumbar_torque_offset": 0,
    # Fixed params from search.toml.
    "min_velocity_cost_coeff": 1.0,
    "grf_cost_coeff": 1.0,
    "hip_limit_cost_coeff": 1.0,
    "knee_limit_cost_coeff": 1.0,
    "ankle_limit_cost_coeff": 1.0,
    "lumbar_limit_cost_coeff": 1.0,
}


def stub_initialize_job(*_args, **_kwargs):
    return FAKE_PARAMS


def stub_finalize_job(metrics, _params):
    print("\n=== finalize_job ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def short_build_config(template, params, env_name):
    cfg = train_for_search._orig_build_config(template, params, env_name)
    cfg["tonic"]["trainer"] = (
        "deprl.custom_trainer.Trainer("
        "steps=int(5e4), "
        "epoch_steps=int(1e4), "
        "save_steps=int(5e4))"
    )
    cfg["tonic"]["parallel"] = 2
    cfg["tonic"]["sequential"] = 2
    return cfg


def main():
    override_env = sys.argv[1] if len(sys.argv) > 1 else None
    if override_env is not None:
        assert override_env in train_for_search.ENV_NAMES, (
            f"env must be one of {train_for_search.ENV_NAMES}"
        )
        import numpy as np
        np.random.default_rng = lambda *a, **kw: type(
            "R", (), {"choice": lambda self, xs: override_env}
        )()

    train_for_search.initialize_job = stub_initialize_job
    train_for_search.finalize_job = stub_finalize_job
    train_for_search._orig_build_config = train_for_search.build_config
    train_for_search.build_config = short_build_config
    train_for_search.EARLY_KILL_STEP = 10 ** 12

    print(f"Working dir: {FAKE_PARAMS['working_dir']}")
    try:
        train_for_search.main()
    finally:
        print(f"\nWorking dir (inspect if needed): {FAKE_PARAMS['working_dir']}")


if __name__ == "__main__":
    main()
