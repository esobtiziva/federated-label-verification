# Budget-Constrained Simulated Verification in Federated Regression

Reproducibility package for the experiments reported in the manuscript on label-induced data-quality distortion and budget-constrained simulated verification in federated learning.

## Experimental scope

The package implements federated regression on the California Housing dataset under IID and controlled non-IID client partitions. It compares three paired conditions:

- **Baseline:** clean training targets.
- **Noise:** 30% of training targets are corrupted once with Gaussian perturbations.
- **Crowd:** starts from the exact same corruption realization as the paired Noise condition and applies budget-constrained simulated verification.

The term **Crowd** denotes an idealized simulated external verifier. No human participants or crowd workers were recruited. Candidate ranking has no access to the hidden corruption mask or original clean target. The clean target is revealed only after a sample has been selected for review.

## Final experimental design

- Dataset: California Housing, 20,640 observations, 8 numerical features.
- Fixed train/test split: 80/20, `random_state=42`.
- Scaling: `StandardScaler` fitted only on the training split.
- Client counts: 2 and 5.
- Partitions: IID random balanced allocation; non-IID allocation by sorting training observations by median income and assigning contiguous shards.
- Communication rounds: 30.
- Local epochs: 3 per round.
- Model: `SGDRegressor` with constant learning rate, `eta0=0.0001`, `alpha=1e-5`, `average=False`.
- Label corruption fraction: 30% of training targets.
- Noise: `sigma = gamma * SD(y_train)`, with `gamma` = 0.5 or 1.0.
- Paired design: Noise and Crowd use the same corrupted indices and the same Gaussian perturbations for a given seed/configuration.
- Verification score: absolute residual `|observed target - prediction|` among unreviewed training samples.
- Global lifetime verification budget: `B=100` for the complete FL run.
- Budget allocation: 50 reviews/client for 2 clients; 20 reviews/client for 5 clients.
- Warm-up: 3 rounds for 2 clients; 5 rounds for 5 clients.
- Seeds: 1, 2, 3, 4, 5.
- Aggregation: FedAvg.

A clean reviewed sample consumes budget. A sample cannot be reviewed twice. If a selected sample is corrupted, its clean training target is restored before ordinary local training in that same verification round and remains corrected in subsequent rounds. Test labels are never used for corruption correction or candidate selection.

## Tested environment

The final experiments were run with:

- Python 3.9.20
- Flower 1.5.0
- NumPy 1.26.4
- pandas 2.2.3
- scikit-learn 1.3.2

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── selected_config.json
├── run_suite.py
├── src/
│   ├── client_final.py
│   ├── server_final.py
│   └── data_utils.py
├── analysis/
│   ├── summarize_results.py
│   ├── aggregate_v3_five_seeds.py
│   └── plot_appendix_A1.py
└── results/
    ├── final_v3_five_seed_combined/
    └── seed_runs/
        ├── seed1/
        ├── seed2/
        ├── seed3/
        ├── seed4/
        └── seed5/
```

## Running the experiments

`run_suite.py` launches the Flower server and clients for all four federated configurations and the Baseline, Noise, and Crowd conditions for the requested seeds.

The original scripts expect `client_final.py`, `server_final.py`, and `data_utils.py` in the same working directory as `run_suite.py`. To preserve the exact scientific source while keeping the repository readable, copy the three files from `src/` to the repository root before executing, or run from a working directory containing these four scripts.

For the five final seeds at `gamma=0.5`:

```bash
python run_suite.py --seeds 1,2,3,4,5 --noise-gamma 0.5 --outdir final_v3_gamma0p5
```

For the five final seeds at `gamma=1.0`:

```bash
python run_suite.py --seeds 1,2,3,4,5 --noise-gamma 1.0 --outdir final_v3_gamma1
```

The stored `results/` directory contains the final experimental outputs used for manuscript analysis. These files are supplied so that numerical results can be inspected without rerunning the full federated experiment.

## Main reported quantities

The analysis records MSE, R-squared, model-update upload payload, wall-clock time, reviewed/fixed counts, remaining corruptions, selection precision, correction recall, and the actual injected noise scale. Paired effects are defined as:

- `Delta R2_noise = R2_noise - R2_baseline`
- `Delta R2_crowd = R2_crowd - R2_noise`

A positive `Delta R2_crowd` indicates higher R-squared after simulated verification relative to the paired noisy run. The experiments do not assume that corruption must always reduce performance or that verification must always improve it.

## Data

The California Housing dataset is obtained through scikit-learn. The repository does not redistribute a separate copy of the source dataset. The held-out test partition is used only for evaluation.

## Reproducibility note

The stored result files are the authoritative outputs corresponding to the five-seed final experiment. Calibration, exploratory sensitivity runs, and earlier three-seed validation artifacts are intentionally excluded from this release to avoid ambiguity about which results support the manuscript. 'selected_config.json' records the configuration-selection stage and is retained for provenance; the files under 'results/' contain the authoritative outputs corresponding to the final experiments reported in the manuscript.

## License and citation
The source code in this repository is released under the MIT License. Citation metadata are provided in 'CITATION.cff'.

The version 1.0.0 reproducibility package is permanently archived in Zenodo: https://doi.org/10.5281/zenodo.23109763
