#!/usr/bin/env bash
set -euo pipefail

# Lightweight end-to-end check for the public reproduction scripts.
#
# This is not a scientific reproduction run.  It uses tiny datasets, one-epoch
# training, and dummy model weights where appropriate, so that dependency,
# path, TensorFlow weight-loading, data-generation, and plotting failures show
# up before launching the full simulations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${SMOKE_WORKDIR:-/tmp/tomoda_public_smoke_$(date +%Y%m%d_%H%M%S)}"
KEEP_WORKDIR="${SMOKE_KEEP_WORKDIR:-0}"
RUN_FIG2="${SMOKE_RUN_FIG2:-1}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "Smoke-test workspace: ${WORKDIR}"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
mkdir -p Chaos_Signals Fig

echo
echo "== 1. Syntax check =="
python3 -m py_compile \
  "${SCRIPT_DIR}/generate_chaos_signals.py" \
  "${SCRIPT_DIR}/train_and_save_models.py" \
  "${SCRIPT_DIR}/make_fig2.py" \
  "${SCRIPT_DIR}/generate_fig3_noise_robustness_data.py" \
  "${SCRIPT_DIR}/plot_fig3_noise_robustness.py" \
  "${SCRIPT_DIR}/generate_fig4_dropout_robustness_data.py" \
  "${SCRIPT_DIR}/plot_dropout_robustness_from_saved.py"

echo
echo "== 1b. Dependency import check =="
python3 - <<'PY'
import matplotlib
import networkx
import numpy
import pandas
import scipy
import sklearn
import statsmodels
import tensorflow

print("Dependency imports succeeded")
print("TensorFlow:", tensorflow.__version__)
print("NumPy:", numpy.__version__)
PY

echo
echo "== 2. Generate tiny Lorenz/Rossler datasets =="
python3 "${SCRIPT_DIR}/generate_chaos_signals.py" \
  --start-id 1 \
  --end-id 1 \
  --data-size 4 \
  --raw-wave-length 260 \
  --burn-in 0 \
  --workers 1 \
  --output-dir Chaos_Signals
cp Chaos_Signals/data1_1.npy Chaos_Signals/data1_101.npy
cp Chaos_Signals/data2_1.npy Chaos_Signals/data2_101.npy
cp Chaos_Signals/data1_1.npy Chaos_Signals/data1_102.npy
cp Chaos_Signals/data2_1.npy Chaos_Signals/data2_102.npy
cp Chaos_Signals/data1_1.npy Chaos_Signals/data1_103.npy
cp Chaos_Signals/data2_1.npy Chaos_Signals/data2_103.npy

echo
echo "== 3. Mini training run =="
python3 - <<'PY'
from argparse import Namespace
from pathlib import Path

import numpy as np
import train_and_save_models as train

train.DATA_DIR = Path("Chaos_Signals")
train.OUTPUT_DIR = Path(".")
train.BATCH_SIZE = 4
train.DATA_SIZE = 4
train.WAVE_LENGTH = 260
train.TRAIN_DATA_IDS = np.array([1])
train.EVAL_DATA_ID = 101

args = Namespace(
    condition="l2_only",
    num_models=1,
    start_index=9001,
    max_attempts=1,
    epochs=1,
    learning_rate=0.001,
    r2_threshold=-1e12,
    output_dir=Path("."),
    acceptance_log_name=None,
)
train.train_until_accepted(args)
PY

echo
echo "== 4. Prepare lightweight model weights for robustness scripts =="
python3 - <<'PY'
from pathlib import Path

import generate_fig3_noise_robustness_data as fig3

ids = ["1", "nm1", "nr1", "2", "nm2", "nr2"]
for model_id in ids:
    model = fig3.MyModel()
    _ = model(fig3.tf.zeros((1, 260, 3)), training=False)
    model.save_weights("model_" + model_id)
print(f"Saved dummy weights for {len(ids)} models")
PY

echo
echo "== 5. Fig. 3 data generation smoke test =="
python3 - <<'PY'
from pathlib import Path

import numpy as np
import generate_fig3_noise_robustness_data as fig3

fig3.DATA1_PATH = Path("Chaos_Signals/data1_103.npy")
fig3.DATA2_PATH = Path("Chaos_Signals/data2_103.npy")
fig3.OUTPUT_DIR = Path(".")
fig3.WAVE_LENGTH = 260
fig3.EVAL_SAMPLES = 2
fig3.BATCH_SIZE = 2
fig3.NOISE_LEVELS = np.array([0.0, 0.1, 0.2])
fig3.NOISE_TRIALS = 1
fig3.STRICT_WEIGHTS = True
fig3.GROUPS = [
    {"name": "MINE + L2", "model_ids": ["1"], "output_file": "ketteikeisu3_MINEあり.npy"},
    {"name": "MINE / No L2", "model_ids": ["nr1"], "output_file": "ketteikeisu3_正則化なし.npy"},
    {"name": "L2 only", "model_ids": ["nm1"], "output_file": "ketteikeisu3_MINEなし.npy"},
]
fig3.main()
PY

echo
echo "== 6. Fig. 3 plotting smoke test =="
python3 - <<'PY'
from pathlib import Path

import numpy as np
import plot_fig3_noise_robustness as plot3

plot3.NOISE_LEVELS = np.array([0.0, 0.1, 0.2])
plot3.OUTPUT_PATH = Path("Fig/noise_robustness_3groups_smoke.pdf")
plot3.NORMALIZED_OUTPUT_PATH = Path("Fig/noise_robustness_normalized_smoke.pdf")
plot3.main()
PY

echo
echo "== 7. Fig. 4 data generation smoke test =="
python3 - <<'PY'
from pathlib import Path

import generate_fig4_dropout_robustness_data as fig4

fig4.DATA1_PATH = Path("Chaos_Signals/data1_102.npy")
fig4.DATA2_PATH = Path("Chaos_Signals/data2_102.npy")
fig4.OUTPUT_DIR = Path(".")
fig4.WAVE_LENGTH = 260
fig4.EVAL_SAMPLES = 2
fig4.MAX_K = 2
fig4.DROPOUT_TRIALS = 1
fig4.BATCH_SIZE = 2
fig4.STRICT_WEIGHTS = True
fig4.SUFFIXES_TO_EVALUATE = ["_m", "_nm", "_nr"]
fig4.GROUPS = {
    "backslash_m": ["1"],
    "slash_m": ["2"],
    "backslash_nm": ["nm1"],
    "slash_nm": ["nm2"],
    "backslash_nr": ["nr1"],
    "slash_nr": ["nr2"],
}
fig4.main()
PY

echo
echo "== 8. Fig. 4 plotting smoke test =="
python3 "${SCRIPT_DIR}/plot_dropout_robustness_from_saved.py"

if [[ "${RUN_FIG2}" == "1" ]]; then
  echo
  echo "== 9. Fig. 2 plotting smoke test =="
  python3 - <<'PY'
from pathlib import Path

import numpy as np
import generate_fig3_noise_robustness_data as fig3

rng = np.random.default_rng(123)
Path("Chaos_Signals").mkdir(exist_ok=True)
np.save("Chaos_Signals/data1_103.npy", rng.normal(size=(50, 10000, 3)).astype("float32"))
np.save("Chaos_Signals/data2_103.npy", rng.normal(size=(50, 10000, 3)).astype("float32"))

model = fig3.MyModel()
_ = model(fig3.tf.zeros((1, 10000, 3)), training=False)
model.save_weights("model_13")
PY
  python3 "${SCRIPT_DIR}/make_fig2.py"
else
  echo
  echo "== 9. Fig. 2 plotting smoke test skipped =="
fi

echo
echo "Smoke test completed successfully."
echo "Outputs are in: ${WORKDIR}"

if [[ "${KEEP_WORKDIR}" != "1" ]]; then
  echo "Set SMOKE_KEEP_WORKDIR=1 to keep the workspace for inspection."
fi
