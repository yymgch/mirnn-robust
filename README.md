# Mutual-information minimization in recurrent neural networks

This directory contains the public analysis and reproduction scripts for:

**Tomoda, Yamaguti, "Mutual-information minimization enhances noise robustness and induces fault containment in functionally differentiated recurrent neural networks"**

The scripts train GRU-based source-separation models, evaluate noise and dropout robustness, and generate the figures used in the manuscript.

## Environment

Two environments have been tested and both reproduce the results. The code uses
the Keras 2 API and is compatible with NumPy 1.x and 2.x.

- **Original**: TensorFlow 2.10 with NumPy 1.x (the pinned set in
  `requirements.txt`).
- **Recent GPUs (Ampere / Ada / Blackwell, e.g. RTX 40/50 series)**: a recent
  TensorFlow with the legacy Keras 2 API, e.g.

  ```bash
  pip install "tensorflow[and-cuda]" tf-keras
  pip install numpy scipy scikit-learn statsmodels pandas matplotlib networkx
  export TF_USE_LEGACY_KERAS=1   # make tf.keras resolve to Keras 2 via tf-keras
  ```

  The results reported in the paper were produced with TensorFlow 2.21,
  `tf-keras`, and NumPy 2.x on an RTX 5090.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # original TensorFlow 2.10 / NumPy 1.x stack
```

## Expected files

Run the scripts from this directory or from a project root where these relative paths are available.

- `Chaos_Signals/data1_<id>.npy` and `Chaos_Signals/data2_<id>.npy`: paired chaotic signal datasets
- `model_<id>*`, `model_nm<id>*`, `model_nr<id>*`: saved TensorFlow weight files used by the analysis scripts
- `Fig/`: output directory for generated figures

## Scripts

- `train_and_save_models.py`: trains and saves models. Use `--condition` to switch among `mine_l2`, `l2_only`, `mine_no_l2`, and `plain`. Each run also writes an acceptance log (see below).
- `select_accepted_models.py`: aggregates the acceptance logs into the model-selection report (total trained, number passing `R2 > threshold`, R2 distribution before/after selection, final sample size) and writes `accepted_models.json`, which the noise/ablation scripts consume. Use `--min-r2` to re-select at a stricter threshold (e.g. 0.997) than the one used during training.
- `determine_orientations.py`: assigns each accepted model a primary/non-primary subgroup orientation from its per-neuron output correlations, and writes `model_orientations.json`, which the ablation script consumes.
- `generate_chaos_signals.py`: generates normalized Lorenz (`data1`) and Rossler (`data2`) signal datasets.
- `make_fig2.py`: generates the representative model analysis for Fig. 2.
- `generate_fig3_noise_robustness_data.py`: evaluates trained models under input noise and saves `.npy` summary data (reads `accepted_models.json` when present).
- `plot_fig3_noise_robustness.py`: plots Fig. 3 from saved noise-robustness data.
- `generate_fig4_dropout_robustness_data.py`: evaluates structured hidden-unit dropout and saves per-model `.pkl` summary data (reads `model_orientations.json` when present; column index 0 is the unablated baseline, k=0).
- `plot_dropout_robustness_from_saved.py`: plots Fig. 4 from saved dropout-robustness data (across-model mean with 95% confidence interval).
- `analyze_noise_metrics.py`: curve-level noise-robustness metrics (AUC, degradation slope, threshold-crossing noise level) with per-condition 95% CIs and pairwise effect sizes (Cohen's d) and Holm-corrected Welch tests.
- `analyze_ablation_stats.py`: primary-minus-non-primary ablation gap with 95% CIs and a condition x target x k interaction analysis.
- `analyze_mechanism.py`: per-condition mechanism indicators (inter-subgroup correlation, final MINE MI, readout-weight segregation).
- `run_smoke_test.sh`: runs a lightweight end-to-end check in a temporary directory.

Example:

```bash
# 1. Generate the chaotic signal datasets (ids 1..100 train, 101 MINE, 102/103 eval).
python generate_chaos_signals.py --start-id 1 --end-id 103

# 2. Train models for each condition until the target number are accepted
#    (R2 > 0.997). Repeat per condition; runs are independent and can be split
#    across machines. Each run appends to its own acceptance log.
python train_and_save_models.py --condition mine_l2    --num-models 100 --start-index 1
python train_and_save_models.py --condition l2_only    --num-models 100 --start-index 1
python train_and_save_models.py --condition mine_no_l2 --num-models 100 --start-index 1

# 3. Select the accepted models and produce the model-selection report.
#    Writes accepted_models.json + model_selection_summary.csv + model_selection_r2.npz.
python select_accepted_models.py

# 4. Noise experiment on the selected models, then plot.
python generate_fig3_noise_robustness_data.py     # reads accepted_models.json
python plot_fig3_noise_robustness.py

# 5. Assign each accepted model a primary/non-primary orientation, then run the
#    ablation experiment on the oriented models and plot.
python determine_orientations.py                  # writes model_orientations.json
python generate_fig4_dropout_robustness_data.py   # reads model_orientations.json
python plot_dropout_robustness_from_saved.py
```

The intended flow is therefore: **train (logs) -> select_accepted_models (selection report + `accepted_models.json`) -> noise experiment on the accepted models -> determine_orientations (`model_orientations.json`) -> ablation experiment on the oriented models**. If `accepted_models.json` / `model_orientations.json` are absent, the figure scripts fall back to their built-in model-id lists.

Training writes acceptance-rate records to
`training_acceptance_log_<condition>_<start-index>.csv` and
`training_acceptance_log_<condition>_<start-index>_summary.json`. These files
record accepted models, R2-based rejections, and non-finite-loss rejections.

Signal generation can be parallelized. `--workers` controls trajectory-level
parallelism within each Lorenz/Rossler system, and `--pair-workers` controls
dataset-ID-level parallelism.

```bash
python generate_chaos_signals.py --start-id 1 --end-id 103 --pair-workers 4 --workers 2
```

For a small server-side check before full simulations:

```bash
./run_smoke_test.sh
```

Set `SMOKE_RUN_FIG2=0` to skip the heaviest plotting check, or
`SMOKE_WORKDIR=/path/to/workdir` to choose the temporary workspace.
