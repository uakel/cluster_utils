"""Entry point launched by cluster_utils for the hyfydy cost-limit search.

Each invocation samples an environment uniformly from {h0918, h1622, h2190},
patches `template.yaml` with the params cluster_utils provides, runs training
via `deprl.main.train`, and reports the mean of `test/gait_match/overall`
across all logged test epochs as the objective `score`.

Early kill: once the agent has been trained for >= 15M steps, the first test
epoch whose mean `test/episode_length` is below 100 aborts the run and
reports score 0.
"""

import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from cluster_utils import finalize_job, initialize_job

from deprl.main import train
from deprl.vendor.tonic.utils import logger as tonic_logger
from deprl.vendor.tonic.utils.logger import Logger


TEMPLATE_PATH = Path(__file__).parent / "template.yaml"
ENV_NAMES = ("h0918", "h1622", "h2190")
EARLY_KILL_STEP = 15_000_000
EARLY_KILL_MIN_EPISODE_LEN = 800

EARLY_KILL_STATE = {"triggered": False, "reason": ""}

COST_COEFF_KEYS = (
    "min_velocity_cost_coeff",
    "max_velocity_cost_coeff",
    "grf_cost_coeff",
    "nmuscle_cost_coeff",
    "hip_limit_cost_coeff",
    "knee_limit_cost_coeff",
    "ankle_limit_cost_coeff",
    "lumbar_limit_cost_coeff",
    "smooth_cost_coeff",
)

LIMIT_KEYS = (
    "target_grf",
    "target_delta",
    "target_muscle_usage_percentage",
    "target_hip_torque",
    "target_knee_torque",
    "target_ankle_torque",
)


def build_config(template: Dict[str, Any], params: Any, env_name: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(template)

    cfg["tonic"]["environment"] = (
        f"deprl.environments.Gym('sconerun_{env_name}_cat-v0', scaled_actions=False)"
    )
    cfg["tonic"]["name"] = f"hyfydy_search_{env_name}_{params['id']}"
    cfg["tonic"]["seed"] = 0

    for k in COST_COEFF_KEYS + LIMIT_KEYS:
        if k in params:
            cfg["env_args"][k] = float(params[k])

    trunk = float(params["target_lumbar_torque_trunk"])
    offset = float(params["target_lumbar_torque_offset"])
    cfg["env_args"]["target_lumbar_torque"] = (
        trunk if env_name == "h2190" else trunk + offset
    )

    return cfg


def install_early_kill() -> None:
    """Wrap Logger.dump so training aborts when the agent is clearly falling."""
    original_dump = Logger.dump

    def patched(self):
        train_steps_vals = list(self.epoch_dict.get("train/steps", []))
        ep_len_vals = list(self.epoch_dict.get("test/episode_length", []))
        original_dump(self)
        if EARLY_KILL_STATE["triggered"]:
            return
        if not train_steps_vals or not ep_len_vals:
            return
        latest_step = float(np.mean(train_steps_vals))
        latest_len = float(np.mean(ep_len_vals))
        if latest_step >= EARLY_KILL_STEP and latest_len < EARLY_KILL_MIN_EPISODE_LEN:
            EARLY_KILL_STATE["triggered"] = True
            EARLY_KILL_STATE["reason"] = (
                f"test/episode_length={latest_len:.1f} < "
                f"{EARLY_KILL_MIN_EPISODE_LEN} at step {latest_step:.0f}"
            )
            raise RuntimeError("EARLY_KILL: " + EARLY_KILL_STATE["reason"])

    Logger.dump = patched


def compute_score() -> float:
    log_path = Path(tonic_logger.get_path()) / "log.csv"
    if not log_path.exists():
        return 0.0
    df = pd.read_csv(log_path)
    col = "test/gait_match/overall/mean"
    if col not in df.columns:
        col = "test/gait_match/overall"
    if col not in df.columns:
        return 0.0
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(values) == 0:
        return 0.0
    return float(values.mean())


def main() -> None:
    params = initialize_job()

    rng = np.random.default_rng()
    env_name = str(rng.choice(ENV_NAMES))

    with open(TEMPLATE_PATH, "r") as f:
        template = yaml.safe_load(f)
    config = build_config(template, params, env_name)

    install_early_kill()

    train(config)

    if EARLY_KILL_STATE["triggered"]:
        tonic_logger.log(f"Early kill: {EARLY_KILL_STATE['reason']}")
        finalize_job(
            {"score": 0.0, "env_name": env_name, "early_killed": 1},
            params,
        )
        return

    score = compute_score()
    finalize_job(
        {"score": score, "env_name": env_name, "early_killed": 0},
        params,
    )


if __name__ == "__main__":
    main()
