import argparse
import csv
import os
import time
from typing import Any, Dict, List, Tuple

import flwr as fl


def aggregate_eval(metrics: List[Tuple[int, Dict[str, Any]]]):
    tot = sum(n for n, _ in metrics) or 1
    return {
        "mse": sum(n * float(m.get("mse", 0.0)) for n, m in metrics) / tot,
        "r2": sum(n * float(m.get("r2", 0.0)) for n, m in metrics) / tot,
    }


def aggregate_fit(metrics: List[Tuple[int, Dict[str, Any]]]):
    tot = sum(n for n, _ in metrics) or 1
    initial = sum(int(m.get("initial_corrupted", 0)) for _, m in metrics)
    reviewed_cum = sum(int(m.get("crowd_reviewed_cumulative", 0)) for _, m in metrics)
    fixed_cum = sum(int(m.get("crowd_fixed_cumulative", 0)) for _, m in metrics)
    remaining = sum(int(m.get("remaining_corrupted", 0)) for _, m in metrics)
    sigma_vals = [float(m.get("noise_sigma_actual", 0.0)) for _, m in metrics]
    sigma_actual = max(sigma_vals) if sigma_vals else 0.0
    precision = fixed_cum / reviewed_cum if reviewed_cum else 0.0
    recall = fixed_cum / initial if initial else 0.0
    return {
        "bytes_up_round": float(sum(int(m.get("bytes_up", 0)) for _, m in metrics)),
        "avg_client_fit_time_s": sum(n * float(m.get("fit_time_s", 0.0)) for n, m in metrics) / tot,
        "max_client_fit_time_s": max((float(m.get("fit_time_s", 0.0)) for _, m in metrics), default=0.0),
        "crowd_reviewed_round": float(sum(int(m.get("crowd_reviewed_round", 0)) for _, m in metrics)),
        "crowd_fixed_round": float(sum(int(m.get("crowd_fixed_round", 0)) for _, m in metrics)),
        "crowd_reviewed_cumulative": float(reviewed_cum),
        "crowd_fixed_cumulative": float(fixed_cum),
        "initial_corrupted": float(initial),
        "remaining_corrupted": float(remaining),
        "selection_precision_cumulative": float(precision),
        "correction_recall_cumulative": float(recall),
        "noise_sigma_actual": float(sigma_actual),
    }


def fit_cfg(local_epochs, crowd_warmup, total_rounds):
    def f(server_round):
        return {
            "local_epochs": int(local_epochs),
            "round": int(server_round),
            "crowd_warmup": int(crowd_warmup),
            "total_rounds": int(total_rounds),
        }
    return f


class LoggingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, log_csv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_csv = log_csv
        self.buf = {}
        self.round_start = {}
        os.makedirs(os.path.dirname(log_csv) or ".", exist_ok=True)
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "round", "bytes_up_round", "test_mse", "test_r2",
                "avg_client_fit_time_s", "max_client_fit_time_s",
                "round_wall_time_s", "crowd_reviewed_round", "crowd_fixed_round",
                "crowd_reviewed_cumulative", "crowd_fixed_cumulative",
                "initial_corrupted", "remaining_corrupted",
                "selection_precision_cumulative", "correction_recall_cumulative",
                "noise_sigma_actual"
            ])

    def configure_fit(self, server_round, parameters, client_manager):
        self.round_start[server_round] = time.perf_counter()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        out = super().aggregate_fit(server_round, results, failures)
        vals = self.fit_metrics_aggregation_fn([(r.num_examples, r.metrics) for _, r in results])
        self.buf.setdefault(server_round, {}).update(vals)
        return out

    def aggregate_evaluate(self, server_round, results, failures):
        out = super().aggregate_evaluate(server_round, results, failures)
        if out is not None:
            _, metrics = out
            self.buf.setdefault(server_round, {}).update({
                "test_mse": float(metrics.get("mse", 0.0)),
                "test_r2": float(metrics.get("r2", 0.0)),
            })
        start = self.round_start.pop(server_round, None)
        if start is not None:
            self.buf.setdefault(server_round, {})["round_wall_time_s"] = time.perf_counter() - start
        self._write(server_round)
        return out

    def _write(self, rnd):
        r = self.buf.get(rnd, {})
        keys = [
            "bytes_up_round", "test_mse", "test_r2", "avg_client_fit_time_s",
            "max_client_fit_time_s", "round_wall_time_s", "crowd_reviewed_round",
            "crowd_fixed_round", "crowd_reviewed_cumulative", "crowd_fixed_cumulative",
            "initial_corrupted", "remaining_corrupted", "selection_precision_cumulative",
            "correction_recall_cumulative", "noise_sigma_actual"
        ]
        if all(k in r for k in keys):
            with open(self.log_csv, "a", newline="") as f:
                csv.writer(f).writerow([rnd] + [r[k] for k in keys])
            self.buf.pop(rnd, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--min-fit-clients", type=int, required=True)
    p.add_argument("--crowd-warmup", type=int, default=3)
    p.add_argument("--server", default="0.0.0.0:8090")
    p.add_argument("--log", required=True)
    a = p.parse_args()
    strategy = LoggingFedAvg(
        log_csv=a.log,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=a.min_fit_clients,
        min_evaluate_clients=a.min_fit_clients,
        min_available_clients=a.min_fit_clients,
        on_fit_config_fn=fit_cfg(a.local_epochs, a.crowd_warmup, a.rounds),
        evaluate_metrics_aggregation_fn=aggregate_eval,
        fit_metrics_aggregation_fn=aggregate_fit,
    )
    fl.server.start_server(
        server_address=a.server,
        config=fl.server.ServerConfig(num_rounds=a.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
