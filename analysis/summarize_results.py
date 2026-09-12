from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd


def parse_gamma(token: str) -> float:
    return float(token.replace("p", "."))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="rerun_results_v3")
    a = p.parse_args()
    root = Path(a.root)
    rows = []
    base_pat = re.compile(r"(iid|noniid)_(2|5)c_baseline_seed(\d+)\.csv$")
    exp_pat = re.compile(r"(iid|noniid)_(2|5)c_(noise|crowd)_g([0-9p]+)_seed(\d+)\.csv$")

    for path in root.glob("*.csv"):
        mb = base_pat.match(path.name)
        me = exp_pat.match(path.name)
        if not mb and not me:
            continue
        if mb:
            partition, clients, seed = mb.group(1), int(mb.group(2)), int(mb.group(3))
            condition, gamma = "baseline", 0.0
        else:
            partition, clients = me.group(1), int(me.group(2))
            condition, gamma, seed = me.group(3), parse_gamma(me.group(4)), int(me.group(5))

        df = pd.read_csv(path)
        tail = df.tail(5)
        last = df.iloc[-1]
        rows.append({
            "partition": partition,
            "clients": clients,
            "condition": condition,
            "gamma": gamma,
            "seed": seed,
            "final5_mse": tail["test_mse"].mean(),
            "final5_r2": tail["test_r2"].mean(),
            "total_upload_bytes": df["bytes_up_round"].sum(),
            "total_crowd_reviewed": last["crowd_reviewed_cumulative"],
            "total_crowd_fixed": last["crowd_fixed_cumulative"],
            "initial_corrupted": last["initial_corrupted"],
            "final_remaining_corrupted": last["remaining_corrupted"],
            "selection_precision": last["selection_precision_cumulative"],
            "correction_recall": last["correction_recall_cumulative"],
            "noise_sigma_actual": last["noise_sigma_actual"],
            "mean_round_wall_s": df["round_wall_time_s"].mean(),
        })

    raw = pd.DataFrame(rows)
    if raw.empty:
        print("No V3 experiment CSV files found.")
        return
    raw = raw.sort_values(["partition", "clients", "condition", "gamma", "seed"])
    raw.to_csv(root / "per_seed_summary.csv", index=False)

    summary = raw.groupby(["partition", "clients", "condition", "gamma"], dropna=False).agg(
        mse_mean=("final5_mse", "mean"), mse_sd=("final5_mse", "std"),
        r2_mean=("final5_r2", "mean"), r2_sd=("final5_r2", "std"),
        upload_mean=("total_upload_bytes", "mean"),
        reviewed_mean=("total_crowd_reviewed", "mean"),
        fixed_mean=("total_crowd_fixed", "mean"),
        initial_corrupted_mean=("initial_corrupted", "mean"),
        remaining_mean=("final_remaining_corrupted", "mean"),
        precision_mean=("selection_precision", "mean"),
        recall_mean=("correction_recall", "mean"),
        sigma_actual_mean=("noise_sigma_actual", "mean"),
        round_wall_mean_s=("mean_round_wall_s", "mean"),
    ).reset_index()
    summary.to_csv(root / "summary_mean_sd.csv", index=False)

    # Paired effects for each nonzero gamma.  Baseline/noise/crowd share split,
    # partition, seed, and initialization policy; Noise and Crowd also share the
    # exact corruption indices and perturbations.
    paired_rows = []
    baselines = raw[raw.condition == "baseline"]
    for _, b in baselines.iterrows():
        candidates = raw[(raw.partition == b.partition) &
                         (raw.clients == b.clients) &
                         (raw.seed == b.seed) &
                         (raw.condition != "baseline")]
        for gamma in sorted(candidates.gamma.unique()):
            n = candidates[(candidates.condition == "noise") & (candidates.gamma == gamma)]
            c = candidates[(candidates.condition == "crowd") & (candidates.gamma == gamma)]
            if len(n) != 1 or len(c) != 1:
                continue
            n, c = n.iloc[0], c.iloc[0]
            denom_r2 = b.final5_r2 - n.final5_r2
            denom_mse = n.final5_mse - b.final5_mse
            paired_rows.append({
                "partition": b.partition,
                "clients": int(b.clients),
                "seed": int(b.seed),
                "gamma": gamma,
                "baseline_r2": b.final5_r2,
                "noise_r2": n.final5_r2,
                "crowd_r2": c.final5_r2,
                "noise_delta_r2": n.final5_r2 - b.final5_r2,
                "crowd_gain_r2": c.final5_r2 - n.final5_r2,
                "r2_recovery_fraction": ((c.final5_r2 - n.final5_r2) / denom_r2
                                         if denom_r2 > 0 else np.nan),
                "baseline_mse": b.final5_mse,
                "noise_mse": n.final5_mse,
                "crowd_mse": c.final5_mse,
                "noise_delta_mse": n.final5_mse - b.final5_mse,
                "crowd_gain_mse": n.final5_mse - c.final5_mse,
                "mse_recovery_fraction": ((n.final5_mse - c.final5_mse) / denom_mse
                                          if denom_mse > 0 else np.nan),
                "reviewed": c.total_crowd_reviewed,
                "fixed": c.total_crowd_fixed,
                "selection_precision": c.selection_precision,
                "correction_recall": c.correction_recall,
                "noise_sigma_actual": c.noise_sigma_actual,
            })
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired.to_csv(root / "paired_effects.csv", index=False)

    display_cols = [
        "partition", "clients", "condition", "gamma", "mse_mean", "mse_sd",
        "r2_mean", "r2_sd", "reviewed_mean", "fixed_mean", "precision_mean",
        "recall_mean", "sigma_actual_mean"
    ]
    print(summary[display_cols].to_string(index=False))
    if not paired.empty:
        print("\nPAIRED EFFECTS")
        print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
