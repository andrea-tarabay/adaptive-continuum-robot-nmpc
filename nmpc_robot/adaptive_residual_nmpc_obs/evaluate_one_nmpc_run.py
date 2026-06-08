"""
Evaluate ONE NMPC hardware run.

Input can be either:
1) a CSV log file, or
2) a run folder that contains a CSV log.

It prints:
- tracking error metrics
- staircase / stick-slip metrics
- tension-rate command metrics
-  dT saturation metrics using that run's max_du_step
- friction/current diagnostics
- observer diagnostics
- solve time diagnostics

Examples:
    python evaluate_one_nmpc_run.py "hardware_validation/run_folder/log.csv" --out one_run_eval.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def safe_col(df: pd.DataFrame, name: str, default: float = np.nan) -> np.ndarray:
    if name and name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(df), default, dtype=float)


def first_existing_col(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def finite_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "rms": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "min": np.nan,
        }

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
        "min": float(np.min(x)),
    }


def median_dt(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return np.nan
    return float(np.median(dt))


def longest_true_streak(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    best = 0
    cur = 0
    for val in mask:
        if val:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


# ---------------------------------------------------------------------
# Log/config discovery
# ---------------------------------------------------------------------

def choose_log_file(path: Path) -> Path:
   
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV log file, got: {path}")
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    csvs = []
    for p in path.rglob("*.csv"):
        name = p.name.lower()
        if any(bad in name for bad in [
            "history_angles",
            "history_u_rate",
            "history_u_tendon",
            "bend_reference",
            "reference",
            "summary",
            "evaluation",
        ]):
            continue
        csvs.append(p)

    if not csvs:
        raise RuntimeError(f"No CSV logs found inside folder: {path}")

    preferred = [p for p in csvs if "log" in p.name.lower()]
    candidates = preferred if preferred else csvs
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def parse_max_du_from_text(text: str) -> Optional[float]:
    patterns = [
        r'["\']max_du_step["\']\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
        r'\bmax_du_step\b\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
        r'["\']max_dT_step["\']\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
        r'\bmax_dT_step\b\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
    ]

    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                val = float(m.group(1))
                if np.isfinite(val) and val > 0:
                    return val
            except Exception:
                pass
    return None


def parse_max_du_from_json(path: Path) -> Optional[float]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k) in {"max_du_step", "max_du_step_N", "max_dT_step", "dT_limit_N"}:
                    try:
                        val = float(v)
                        if np.isfinite(val) and val > 0:
                            return val
                    except Exception:
                        pass
                found = walk(v)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for item in obj:
                found = walk(item)
                if found is not None:
                    return found
        return None

    return walk(data)


def detect_du_limit(df: pd.DataFrame, log_path: Path, cli_du_limit: Optional[float]) -> tuple[float, str]:
    
    if cli_du_limit is not None:
        if cli_du_limit <= 0:
            raise ValueError("--du-limit must be positive")
        return float(cli_du_limit), "command_line"

    column_candidates = [
        "max_du_step",
        "max_du_step_N",
        "mpc_max_du_step",
        "du_max_N",
        "dT_limit_N",
        "dT_max_N",
        "max_dT_N",
        "max_dT_step_N",
    ]

    for col in column_candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals):
                return float(np.nanmedian(vals)), f"csv_column:{col}"

    config_hints = ["experiment_config", "params", "config", "settings"]
    exts = {".txt", ".py", ".json", ".yaml", ".yml"}

    folders = [log_path.parent, log_path.parent.parent, log_path.parent.parent.parent]
    seen = set()

    for folder in folders:
        if not folder.exists() or folder in seen:
            continue
        seen.add(folder)

        candidates = []
        for f in folder.iterdir():
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            if any(h in f.name.lower() for h in config_hints):
                candidates.append(f)

        candidates = sorted(candidates, key=lambda f: f.stat().st_mtime, reverse=True)

        for cfg in candidates:
            if cfg.suffix.lower() == ".json":
                val = parse_max_du_from_json(cfg)
            else:
                try:
                    val = parse_max_du_from_text(cfg.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    val = None

            if val is not None:
                return float(val), f"config:{cfg.name}"

    return np.nan, "not_found"


# ---------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------

def get_dT_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        ["dT1_N", "dT2_N", "dT3_N"],
        ["dT1", "dT2", "dT3"],
        ["du1_N", "du2_N", "du3_N"],
        ["du1", "du2", "du3"],
    ]
    for cols in candidates:
        existing = [c for c in cols if c in df.columns]
        if len(existing) >= 2:
            return existing
    return []


def get_T_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        ["T1_N", "T2_N", "T3_N"],
        ["T1", "T2", "T3"],
        ["u1_N", "u2_N", "u3_N"],
        ["u1", "u2", "u3"],
    ]
    for cols in candidates:
        existing = [c for c in cols if c in df.columns]
        if len(existing) >= 2:
            return existing
    return []


def get_friction_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        ["I_fric1", "I_fric2", "I_fric3"],
        ["I_fric1_mA", "I_fric2_mA", "I_fric3_mA"],
    ]
    for cols in candidates:
        existing = [c for c in cols if c in df.columns]
        if len(existing) >= 2:
            return existing
    return []


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_one_run(log_path: Path, ignore_first_s: float, du_limit_cli: Optional[float]) -> dict:
    df = pd.read_csv(log_path)

    time_col = first_existing_col(df, ["time_s", "t", "time", "timestamp_s"])
    if time_col is None:
        raise ValueError("Missing time column. Expected time_s, t, time, or timestamp_s.")

    t_all = safe_col(df, time_col)
    valid_time = np.isfinite(t_all)
    if not np.any(valid_time):
        raise ValueError("Invalid time values.")

    t0 = np.nanmin(t_all)
    mask = valid_time & (t_all >= t0 + ignore_first_s)
    df2 = df.loc[mask].copy()

    if len(df2) < 5:
        raise ValueError("Not enough samples after ignoring initial seconds.")

    t = safe_col(df2, time_col)
    duration_s = float(np.nanmax(t) - np.nanmin(t))
    dt_med = median_dt(t)

    du_limit, du_limit_source = detect_du_limit(df, log_path, du_limit_cli)

    # ----------------------------
    # Tracking error
    # ----------------------------
    if "bend_error_deg" in df2.columns:
        err = safe_col(df2, "bend_error_deg")
    else:
        bx = safe_col(df2, "bx_deg")
        by = safe_col(df2, "by_deg")
        bx_ref = safe_col(df2, "target_bx_deg")
        by_ref = safe_col(df2, "target_by_deg")
        err = np.hypot(bx - bx_ref, by - by_ref)

    err_stats = finite_stats(err)

    # ----------------------------
    # Bend movement / staircase
    # ----------------------------
    bx = safe_col(df2, "bx_deg")
    by = safe_col(df2, "by_deg")
    bx_ref = safe_col(df2, "target_bx_deg")
    by_ref = safe_col(df2, "target_by_deg")

    db = np.hypot(np.diff(bx), np.diff(by))
    db_ref = np.hypot(np.diff(bx_ref), np.diff(by_ref))

    valid_db = np.isfinite(db) & np.isfinite(db_ref)
    db = db[valid_db]
    db_ref = db_ref[valid_db]

    bend_step_stats = finite_stats(db)

    if len(db) > 5:
        target_moving = db_ref > 0.15
        measured_stuck = db < 0.08
        measured_jump = db > 2.0

        plateau_fraction = float(np.mean(measured_stuck[target_moving])) if np.any(target_moving) else np.nan
        jump_fraction = float(np.mean(measured_jump))

        stuck_then_jump = 0
        for k in range(2, len(db)):
            if db[k - 2] < 0.08 and db[k - 1] < 0.08 and db[k] > 2.0:
                stuck_then_jump += 1

        stuck_then_jump_per_s = stuck_then_jump / max(duration_s, 1e-9)

        stair_score = (
            1.0 * np.nan_to_num(plateau_fraction, nan=0.0)
            + 2.0 * jump_fraction
            + 0.5 * stuck_then_jump_per_s
        )
    else:
        plateau_fraction = np.nan
        jump_fraction = np.nan
        stuck_then_jump_per_s = np.nan
        stair_score = np.nan

    # ----------------------------
    # dT / tension-rate commands + saturation
    # ----------------------------
    dT_cols = get_dT_columns(df2)

    if dT_cols:
        dT = df2[dT_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        abs_dT_mat = np.abs(dT)
        abs_dT = abs_dT_mat[np.isfinite(abs_dT_mat)]

        dT_stats = finite_stats(abs_dT)
        total_variation_dT_per_s = float(np.nansum(abs_dT) / max(duration_s, 1e-9))

        if np.isfinite(du_limit) and du_limit > 0:
            sat_threshold = 0.95 * du_limit
            finite_mat = np.isfinite(abs_dT_mat)
            sat_mat = (abs_dT_mat >= sat_threshold) & finite_mat

            sat_fraction_all = float(np.sum(sat_mat) / max(np.sum(finite_mat), 1))
            any_sat_per_step = np.any(sat_mat, axis=1)
            any_sat_fraction = float(np.mean(any_sat_per_step))

            sat_fraction_by_tendon = []
            for j in range(abs_dT_mat.shape[1]):
                finite_j = np.isfinite(abs_dT_mat[:, j])
                sat_fraction_by_tendon.append(
                    float(np.mean(sat_mat[finite_j, j])) if np.any(finite_j) else np.nan
                )

            max_sat_streak_samples = longest_true_streak(any_sat_per_step)
            max_sat_streak_s = float(max_sat_streak_samples * dt_med) if np.isfinite(dt_med) else np.nan

            dT_ratio = abs_dT / du_limit
            dT_ratio_stats = finite_stats(dT_ratio)

            valid_err = err[np.isfinite(err)]
            if len(valid_err) and len(any_sat_per_step) == len(df2):
                high_err_threshold = float(np.percentile(valid_err, 75))
                high_error_mask = err > high_err_threshold
                sat_while_high_error_fraction = (
                    float(np.mean(any_sat_per_step[high_error_mask]))
                    if np.any(high_error_mask)
                    else np.nan
                )
            else:
                sat_while_high_error_fraction = np.nan
        else:
            sat_fraction_all = np.nan
            any_sat_fraction = np.nan
            sat_fraction_by_tendon = [np.nan, np.nan, np.nan]
            max_sat_streak_samples = np.nan
            max_sat_streak_s = np.nan
            dT_ratio_stats = finite_stats(np.array([]))
            sat_while_high_error_fraction = np.nan
    else:
        dT_stats = finite_stats(np.array([]))
        total_variation_dT_per_s = np.nan
        sat_fraction_all = np.nan
        any_sat_fraction = np.nan
        sat_fraction_by_tendon = [np.nan, np.nan, np.nan]
        max_sat_streak_samples = np.nan
        max_sat_streak_s = np.nan
        dT_ratio_stats = finite_stats(np.array([]))
        sat_while_high_error_fraction = np.nan

    # ----------------------------
    # Tensions
    # ----------------------------
    T_cols = get_T_columns(df2)
    if T_cols:
        T = df2[T_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        T_mean = float(np.nanmean(T))
        T_min = float(np.nanmin(T))
        T_max = float(np.nanmax(T))
        T_range_mean = float(np.nanmean(np.nanmax(T, axis=1) - np.nanmin(T, axis=1)))
    else:
        T_mean = T_min = T_max = T_range_mean = np.nan

    # ----------------------------
    # Friction currents
    # ----------------------------
    Ifric_cols = get_friction_columns(df2)
    if Ifric_cols:
        Ifric = df2[Ifric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        abs_fric = np.abs(Ifric[np.isfinite(Ifric)])
        fric_stats = finite_stats(abs_fric)
    else:
        fric_stats = finite_stats(np.array([]))

    # ----------------------------
    # Observer diagnostics
    # ----------------------------
    innovation_col = first_existing_col(df2, ["observer_innovation_deg", "innovation_deg"])
    innovation = safe_col(df2, innovation_col) if innovation_col else np.full(len(df2), np.nan)
    innovation_stats = finite_stats(innovation)

    dist_norm_col = first_existing_col(df2, ["dist_norm_rad_s2", "dist_norm"])
    dist_norm = safe_col(df2, dist_norm_col) if dist_norm_col else np.full(len(df2), np.nan)
    dist_stats = finite_stats(dist_norm)

    # ----------------------------
    # Solve time
    # ----------------------------
    solve_col = first_existing_col(df2, ["solve_time_s", "mpc_solve_time_s", "solver_time_s"])
    solve_time = safe_col(df2, solve_col) if solve_col else np.full(len(df2), np.nan)
    solve_stats = finite_stats(solve_time)

    opt_col = first_existing_col(df2, ["opt_ok", "solver_ok", "success"])
    opt_ok = safe_col(df2, opt_col) if opt_col else np.full(len(df2), np.nan)
    opt_success_fraction = np.nan if np.all(np.isnan(opt_ok)) else float(np.nanmean(opt_ok.astype(float)))

    result = {
        "file": str(log_path),
        "run_folder": str(log_path.parent),
        "samples_used": int(len(df2)),
        "duration_s": duration_s,
        "median_dt_s": dt_med,

        "du_limit_N": du_limit,
        "du_limit_source": du_limit_source,

        "mean_error_deg": err_stats["mean"],
        "median_error_deg": err_stats["median"],
        "rms_error_deg": err_stats["rms"],
        "p95_error_deg": err_stats["p95"],
        "max_error_deg": err_stats["max"],

        "bend_step_mean_deg": bend_step_stats["mean"],
        "bend_step_p95_deg": bend_step_stats["p95"],
        "bend_step_max_deg": bend_step_stats["max"],

        "plateau_fraction": plateau_fraction,
        "jump_fraction": jump_fraction,
        "stuck_then_jump_per_s": stuck_then_jump_per_s,
        "stair_score": stair_score,

        "mean_abs_dT_N": dT_stats["mean"],
        "p95_abs_dT_N": dT_stats["p95"],
        "max_abs_dT_N": dT_stats["max"],
        "total_variation_dT_per_s": total_variation_dT_per_s,

        "sat_fraction_all": sat_fraction_all,
        "any_sat_fraction": any_sat_fraction,
        "sat_fraction_dT1": sat_fraction_by_tendon[0] if len(sat_fraction_by_tendon) > 0 else np.nan,
        "sat_fraction_dT2": sat_fraction_by_tendon[1] if len(sat_fraction_by_tendon) > 1 else np.nan,
        "sat_fraction_dT3": sat_fraction_by_tendon[2] if len(sat_fraction_by_tendon) > 2 else np.nan,
        "max_sat_streak_samples": max_sat_streak_samples,
        "max_sat_streak_s": max_sat_streak_s,
        "sat_while_high_error_fraction": sat_while_high_error_fraction,

        "mean_dT_ratio_to_limit": dT_ratio_stats["mean"],
        "p95_dT_ratio_to_limit": dT_ratio_stats["p95"],
        "max_dT_ratio_to_limit": dT_ratio_stats["max"],

        "T_mean_N": T_mean,
        "T_min_N": T_min,
        "T_max_N": T_max,
        "T_range_mean_N": T_range_mean,

        "mean_abs_friction_mA": fric_stats["mean"],
        "p95_abs_friction_mA": fric_stats["p95"],
        "max_abs_friction_mA": fric_stats["max"],

        "mean_innovation_deg": innovation_stats["mean"],
        "p95_innovation_deg": innovation_stats["p95"],
        "max_innovation_deg": innovation_stats["max"],

        "mean_dist_norm": dist_stats["mean"],
        "p95_dist_norm": dist_stats["p95"],
        "max_dist_norm": dist_stats["max"],

        "mean_solve_time_s": solve_stats["mean"],
        "p95_solve_time_s": solve_stats["p95"],
        "max_solve_time_s": solve_stats["max"],

        "opt_success_fraction": opt_success_fraction,
    }

    return result


def print_report(result: dict) -> None:
    def f(key, digits=3):
        val = result.get(key, np.nan)
        if isinstance(val, str):
            return val
        if val is None or not np.isfinite(val):
            return "nan"
        return f"{val:.{digits}f}"

    print("\n================ ONE RUN EVALUATION ================\n")
    print(f"File: {result['file']}")
    print(f"Samples used: {result['samples_used']}")
    print(f"Duration: {f('duration_s', 2)} s")
    print(f"Median dt: {f('median_dt_s', 4)} s")
    print(f"du limit: {f('du_limit_N', 4)} N/step  ({result['du_limit_source']})")

    print("\n--- Tracking error ---")
    print(f"Mean error:   {f('mean_error_deg')} deg")
    print(f"Median error: {f('median_error_deg')} deg")
    print(f"RMS error:    {f('rms_error_deg')} deg")
    print(f"95% error:    {f('p95_error_deg')} deg")
    print(f"Max error:    {f('max_error_deg')} deg")

    print("\n--- Staircase / stick-slip ---")
    print(f"Stair score:             {f('stair_score', 4)}  lower is better")
    print(f"Plateau fraction:        {f('plateau_fraction', 4)}")
    print(f"Jump fraction:           {f('jump_fraction', 4)}")
    print(f"Stuck-then-jump per sec: {f('stuck_then_jump_per_s', 4)}")
    print(f"Bend step p95:           {f('bend_step_p95_deg')} deg/sample")
    print(f"Bend step max:           {f('bend_step_max_deg')} deg/sample")

    print("\n--- Tension-rate commands dT ---")
    print(f"Mean |dT|:        {f('mean_abs_dT_N', 4)} N")
    print(f"95% |dT|:         {f('p95_abs_dT_N', 4)} N")
    print(f"Max |dT|:         {f('max_abs_dT_N', 4)} N")
    print(f"TV |dT| per sec:  {f('total_variation_dT_per_s', 4)} N/s")

    print("\n--- TRUE dT saturation against du limit ---")
    if not np.isfinite(result.get("du_limit_N", np.nan)):
        print("Could not determine du limit. Use --du-limit VALUE.")
    else:
        print(f"Saturation fraction all entries: {f('sat_fraction_all', 4)}")
        print(f"Any tendon saturated per step:   {f('any_sat_fraction', 4)}")
        print(f"dT1 sat fraction:                {f('sat_fraction_dT1', 4)}")
        print(f"dT2 sat fraction:                {f('sat_fraction_dT2', 4)}")
        print(f"dT3 sat fraction:                {f('sat_fraction_dT3', 4)}")
        print(f"Max saturation streak:           {f('max_sat_streak_s', 3)} s")
        print(f"Sat while high error:            {f('sat_while_high_error_fraction', 4)}")
        print(f"95% dT / limit:                  {f('p95_dT_ratio_to_limit', 4)}")
        print(f"Max dT / limit:                  {f('max_dT_ratio_to_limit', 4)}")

    print("\n--- Tension levels ---")
    print(f"T mean:       {f('T_mean_N', 3)} N")
    print(f"T min/max:    {f('T_min_N', 3)} / {f('T_max_N', 3)} N")
    print(f"T range mean: {f('T_range_mean_N', 3)} N")

    print("\n--- Friction current ---")
    print(f"Mean |I_fric|: {f('mean_abs_friction_mA', 3)} mA")
    print(f"95% |I_fric|:  {f('p95_abs_friction_mA', 3)} mA")
    print(f"Max |I_fric|:  {f('max_abs_friction_mA', 3)} mA")

    print("\n--- Observer / solver ---")
    print(f"Mean innovation: {f('mean_innovation_deg', 3)} deg")
    print(f"95% innovation:  {f('p95_innovation_deg', 3)} deg")
    print(f"Max innovation:  {f('max_innovation_deg', 3)} deg")
    print(f"Mean solve time: {f('mean_solve_time_s', 4)} s")
    print(f"95% solve time:  {f('p95_solve_time_s', 4)} s")
    print(f"Max solve time:  {f('max_solve_time_s', 4)} s")
    print(f"Opt success:     {f('opt_success_fraction', 4)}")

    print("\nQuick interpretation:")
    mean_err = result.get("mean_error_deg", np.nan)
    sat = result.get("any_sat_fraction", np.nan)
    stair = result.get("stair_score", np.nan)

    if np.isfinite(mean_err):
        if mean_err < 5:
            print("- Tracking is good.")
        elif mean_err < 8:
            print("- Tracking is acceptable but can still improve.")
        else:
            print("- Tracking error is high.")

    if np.isfinite(sat):
        if sat > 0.4:
            print("- dT saturation is high: controller is often banging against rate limits.")
        elif sat > 0.15:
            print("- dT saturation is moderate: watch smoothness and robustness.")
        else:
            print("- dT saturation is low/moderate.")

    if np.isfinite(stair):
        if stair > 0.15:
            print("- Staircase/stick-slip score is noticeable.")
        else:
            print("- Staircase/stick-slip score is relatively low.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="CSV log file OR run folder containing the log CSV",
    )
    parser.add_argument(
        "--ignore-first",
        type=float,
        default=2.0,
        help="Ignore first N seconds of the run",
    )
    parser.add_argument(
        "--du-limit",
        type=float,
        default=None,
        help="max_du_step for this run [N/step]. Overrides auto-detection.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output CSV. Default: one_run_evaluation.csv in the log folder.",
    )

    args = parser.parse_args()

    input_path = Path(args.path).expanduser().resolve()
    log_path = choose_log_file(input_path)

    result = evaluate_one_run(
        log_path=log_path,
        ignore_first_s=args.ignore_first,
        du_limit_cli=args.du_limit,
    )

    print_report(result)

    out_path = Path(args.out).expanduser().resolve() if args.out else log_path.parent / "one_run_evaluation.csv"
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f"\nSaved CSV summary to:\n{out_path}")


if __name__ == "__main__":
    main()
