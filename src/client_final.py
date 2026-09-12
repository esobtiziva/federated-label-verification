import argparse
import json
import math
import time
from pathlib import Path
from typing import List

import flwr as fl
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data_utils import prepare_global_data, make_partitions


def load_optimizer_config(path="selected_config.json"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            "selected_config.json not found. Run: python calibrate_baseline.py"
        )
    with p.open() as f:
        return json.load(f)


class FLClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_clients: int, iid: bool, seed: int,
                 noise_frac: float, noise_gamma: float,
                 crowd_budget_lifetime: int, opt_cfg: dict):
        self.cid = int(cid)
        self.num_clients = int(num_clients)
        self.iid = bool(iid)
        self.seed = int(seed)
        self.noise_frac = float(noise_frac)
        self.noise_gamma = float(noise_gamma)
        self.crowd_budget_lifetime = int(crowd_budget_lifetime)
        self.crowd_used = 0

        X_train_all, X_test, y_train_all, y_test = prepare_global_data(seed=seed)
        parts = make_partitions(X_train_all, y_train_all, num_clients, iid, seed)
        idx = parts[self.cid]

        self.X_train = X_train_all[idx]
        self.y_true = y_train_all[idx].copy()
        self.y_work = self.y_true.copy()
        self.X_test = X_test
        self.y_test = y_test

        # Scale-relative noise severity computed from TRAINING TARGETS ONLY.
        # The same value is used by every client in a given experiment.
        self.train_target_sd = float(np.std(y_train_all, ddof=1))
        self.noise_sigma_actual = self.noise_gamma * self.train_target_sd

        # Hidden experimental bookkeeping.  `corrupted` is NEVER consulted when
        # deciding which samples to review.  It is consulted only after selection
        # to simulate whether perfect annotation reveals a wrong stored target.
        self.corrupted = np.zeros(len(self.y_true), dtype=bool)
        self.reviewed = np.zeros(len(self.y_true), dtype=bool)
        self.noise_applied = False
        self.initial_corrupted = 0
        self.reviewed_cumulative = 0
        self.fixed_cumulative = 0

        self.model = SGDRegressor(
            loss="squared_error",
            learning_rate=opt_cfg.get("learning_rate", "constant"),
            eta0=float(opt_cfg["eta0"]),
            alpha=float(opt_cfg["alpha"]),
            max_iter=1,
            tol=None,
            random_state=self.seed + self.cid,
            fit_intercept=True,
            warm_start=True,
            average=bool(opt_cfg.get("average", False)),
            shuffle=True,
        )
        self._prime()

    def _prime(self):
        n_features = self.X_train.shape[1]
        if not hasattr(self.model, "coef_"):
            self.model.partial_fit(np.zeros((1, n_features)), np.zeros(1))

    def get_parameters(self, config):
        self._prime()
        return [self.model.coef_.astype(np.float64).copy(),
                np.asarray(self.model.intercept_, dtype=np.float64).reshape(1).copy()]

    def set_parameters(self, parameters: List[np.ndarray]):
        self._prime()
        self.model.coef_ = np.asarray(parameters[0], dtype=np.float64).copy()
        self.model.intercept_ = np.asarray(parameters[1], dtype=np.float64).reshape(1).copy()

    def _inject_noise_once(self):
        """Inject a deterministic corruption pattern once per client.

        Noise and Crowd runs with the same seed/cid use the same RNG seed, hence
        the same corrupted indices AND the same Gaussian perturbations.  Baseline
        uses noise_frac=0.  No test-set information is used here.
        """
        if self.noise_applied or self.noise_frac <= 0:
            self.noise_applied = True
            return
        rng = np.random.default_rng(self.seed * 100_000 + 1000 + self.cid)
        n_noisy = int(round(self.noise_frac * len(self.y_work)))
        if n_noisy > 0:
            idx = rng.choice(len(self.y_work), size=n_noisy, replace=False)
            eps = rng.normal(0.0, self.noise_sigma_actual, size=n_noisy)
            self.corrupted[idx] = True
            self.y_work[idx] = self.y_true[idx] + eps
            self.initial_corrupted = int(n_noisy)
        self.noise_applied = True

    def _round_budget(self, round_id: int, warmup: int, total_rounds: int) -> int:
        """Spread each client's finite lifetime budget across remaining FL rounds."""
        remaining_budget = self.crowd_budget_lifetime - self.crowd_used
        if remaining_budget <= 0 or round_id <= warmup:
            return 0
        remaining_rounds = total_rounds - round_id + 1
        if remaining_rounds <= 0:
            return 0
        # Evenly spread a finite lifetime budget instead of granting B every round.
        return min(remaining_budget, int(math.ceil(remaining_budget / remaining_rounds)))

    def _selective_crowd_review(self, round_id: int, warmup: int, total_rounds: int):
        """Select samples by observable residual only, then simulate verification.

        Selection has no access to the hidden corruption mask.  A clean reviewed
        sample still consumes budget.  A reviewed corrupted sample is restored to
        its clean target, simulating perfect crowd verification after selection.
        """
        k_budget = self._round_budget(round_id, warmup, total_rounds)
        if k_budget <= 0:
            return 0, 0

        candidates = np.flatnonzero(~self.reviewed)
        if candidates.size == 0:
            return 0, 0

        pred = self.model.predict(self.X_train[candidates])
        residual = np.abs(self.y_work[candidates] - pred)
        k = min(k_budget, candidates.size)
        if k < candidates.size:
            # Top-k residuals; no corruption labels enter this ranking.
            pos = np.argpartition(-residual, k - 1)[:k]
            chosen = candidates[pos]
        else:
            chosen = candidates

        self.reviewed[chosen] = True
        self.crowd_used += int(k)
        self.reviewed_cumulative += int(k)

        # Hidden oracle is consulted only AFTER selection to simulate annotation.
        truly_bad = chosen[self.corrupted[chosen]]
        if truly_bad.size:
            self.y_work[truly_bad] = self.y_true[truly_bad]
            self.corrupted[truly_bad] = False
        fixed = int(truly_bad.size)
        self.fixed_cumulative += fixed
        return int(k), fixed

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        round_id = int(config.get("round", 1))
        local_epochs = int(config.get("local_epochs", 3))
        crowd_warmup = int(config.get("crowd_warmup", 3))
        total_rounds = int(config.get("total_rounds", 30))

        self._inject_noise_once()
        reviewed, fixed = self._selective_crowd_review(
            round_id, crowd_warmup, total_rounds
        )

        t0 = time.perf_counter()
        for _ in range(local_epochs):
            self.model.partial_fit(self.X_train, self.y_work)
        fit_time = time.perf_counter() - t0

        params = self.get_parameters({})
        bytes_up = int(sum(np.asarray(p, dtype=np.float64).nbytes for p in params))
        precision = (self.fixed_cumulative / self.reviewed_cumulative
                     if self.reviewed_cumulative else 0.0)
        recall = (self.fixed_cumulative / self.initial_corrupted
                  if self.initial_corrupted else 0.0)
        return params, len(self.y_work), {
            "fit_time_s": float(fit_time),
            "bytes_up": bytes_up,
            "crowd_reviewed_round": int(reviewed),
            "crowd_fixed_round": int(fixed),
            "crowd_reviewed_cumulative": int(self.reviewed_cumulative),
            "crowd_fixed_cumulative": int(self.fixed_cumulative),
            "initial_corrupted": int(self.initial_corrupted),
            "remaining_corrupted": int(self.corrupted.sum()),
            "selection_precision_cumulative": float(precision),
            "correction_recall_cumulative": float(recall),
            "noise_sigma_actual": float(self.noise_sigma_actual if self.noise_frac > 0 else 0.0),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        pred = self.model.predict(self.X_test)
        mse = float(mean_squared_error(self.y_test, pred))
        r2 = float(r2_score(self.y_test, pred))
        return mse, len(self.y_test), {"mse": mse, "r2": r2}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--client-id", type=int, required=True)
    p.add_argument("--num-clients", type=int, required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--iid", action="store_true")
    mode.add_argument("--non-iid", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--noise-frac", type=float, default=0.0)
    p.add_argument("--noise-gamma", type=float, default=0.5,
                   help="Gaussian SD as gamma * SD(y_train)")
    p.add_argument("--crowd-budget-lifetime", type=int, default=0,
                   help="Per-client TOTAL review budget for the whole FL run")
    p.add_argument("--server", default="127.0.0.1:8090")
    p.add_argument("--config", default="selected_config.json")
    a = p.parse_args()
    cfg = load_optimizer_config(a.config)
    client = FLClient(a.client_id, a.num_clients, a.iid, a.seed,
                      a.noise_frac, a.noise_gamma,
                      a.crowd_budget_lifetime, cfg)
    fl.client.start_numpy_client(server_address=a.server, client=client)


if __name__ == "__main__":
    main()
