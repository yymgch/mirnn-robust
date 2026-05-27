# Mutual-information minimization in recurrent neural networks

This directory contains the sorce codes for:

**Tomoda, Yamaguti, "Mutual information minimization enhances noise robustness and induces fault containment in functionally differentiated recurrent neural networks"**

The scripts train GRU-based source-separation models, evaluate noise and dropout robustness, and generate the figures used in the manuscript.

## Environment

The original experiments were run with TensorFlow 2.10. NumPy 2.x has not been tested, so `requirements.txt` pins NumPy to the 1.x series.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected files

Run the scripts from this directory or from a project root where these relative paths are available.

- `Chaos_Signals/data1_<id>.npy` and `Chaos_Signals/data2_<id>.npy`: paired chaotic signal datasets
- `model_<id>*`, `model_nm<id>*`, `model_nr<id>*`: saved TensorFlow weight files used by the analysis scripts
- `Fig/`: output directory for generated figures

## Scripts

- `train_and_save_models.py`: trains and saves models. Use `--condition` to switch among `mine_l2`, `l2_only`, `mine_no_l2`, and `plain`.
- `generate_chaos_signals.py`: generates normalized Lorenz (`data1`) and Rossler (`data2`) signal datasets.
- `make_fig2.py`: generates the representative model analysis for Fig. 2.
- `generate_fig3_noise_robustness_data.py`: evaluates trained models under input noise and saves `.npy` summary data.
- `plot_fig3_noise_robustness.py`: plots Fig. 3 from saved noise-robustness data.
- `generate_fig4_dropout_robustness_data.py`: evaluates structured hidden-unit dropout and saves `.pkl` summary data.
- `plot_dropout_robustness_from_saved.py`: plots Fig. 4 from saved dropout-robustness data.
- `run_smoke_test.sh`: runs a lightweight end-to-end check in a temporary directory.

Example:

```bash
python generate_chaos_signals.py --start-id 1 --end-id 103
python train_and_save_models.py --condition mine_l2 --num-models 1 --start-index 1
python generate_fig3_noise_robustness_data.py
python plot_fig3_noise_robustness.py
```

Signal generation can be parallelized. `--workers` controls trajectory-level
parallelism within each Lorenz/Rossler system, and `--pair-workers` controls
dataset-ID-level parallelism.

```bash
python generate_chaos_signals.py --start-id 1 --end-id 103 --pair-workers 4 --workers 2
```
