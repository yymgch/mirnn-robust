# %%
"""Generate noise-robustness data for Fig. 3.

The trained model weight files are assumed to already exist as model_<id>.
For each model and noise level, Gaussian input noise is sampled several times;
the stored value is the mean R2 over those repeated noise trials.
"""
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Dense, GRU


# ================================
# Settings
# ================================
DATA1_PATH = Path("Chaos_Signals/data1_103.npy")
DATA2_PATH = Path("Chaos_Signals/data2_103.npy")
MODEL_PREFIX = "model_"
OUTPUT_DIR = Path(".")

WAVE_LENGTH = 10000
EVAL_SAMPLES = 1000
BATCH_SIZE = 50
NOISE_LEVELS = np.linspace(0.0, 0.30, 31)
NOISE_TRIALS = 20
NOISE_MODE = "gaussian"
METRIC = "r2"
STRICT_WEIGHTS = False

# Historical saved data were produced without an explicit seed in the notebook.
# Set this to None to match that stochastic style, or to an integer for reruns.
RANDOM_SEED = 0

MINE_L2_MODEL_IDS = [
    "10", "12", "13", "15", "16", "17", "22", "24", "25", "26",
    "28", "29", "30", "31", "32", "33", "34", "44", "45", "46", "47",
    "71", "73", "74", "75", "76", "100", "101", "102", "103", "104",
    "106", "123", "125", "126", "127", "128", "129", "130", "131",
    "150", "151", "153", "154", "155", "156", "157", "158", "159",
    "160", "161", "162", "163", "164", "165", "166", "167", "168",
    "171", "172", "201", "202", "203", "204", "205", "209", "210",
    "211", "212", "213", "214", "231", "232", "234", "235", "236",
    "237", "238", "239", "240", "241", "242", "262", "264", "265",
    "266", "267", "268", "269", "300", "301", "302", "303", "304",
    "305", "306", "308", "309", "310", "311", "312",
]

GROUPS = [
    {
        "name": "MINE + L2",
        "model_ids": MINE_L2_MODEL_IDS,
        "output_file": "ketteikeisu3_MINEあり.npy",
    },
    {
        "name": "MINE / No L2",
        "model_ids": [f"nr{i}" for i in range(1, 101)],
        "output_file": "ketteikeisu3_正則化なし.npy",
    },
    {
        "name": "L2 only",
        "model_ids": [f"nm{i}" for i in range(1, 101)],
        "output_file": "ketteikeisu3_MINEなし.npy",
    },
]


# ================================
# Model and data helpers
# ================================
def configure_gpu_memory_growth():
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.gru = GRU(100, return_sequences=True, input_shape=(None, 3))
        self.dense1 = Dense(6)

    def call(self, inputs, training=False):
        x = self.gru(inputs, training=training)
        return self.dense1(x)


def load_evaluation_data():
    data1 = np.load(DATA1_PATH)
    data2 = np.load(DATA2_PATH)
    x = data1[:, :WAVE_LENGTH, :] + data2[:, :WAVE_LENGTH, :]
    y = np.concatenate([data1[:, :WAVE_LENGTH, :], data2[:, :WAVE_LENGTH, :]], axis=2)
    return x[:EVAL_SAMPLES], y[:EVAL_SAMPLES]


def load_model(model_id, example_batch):
    model = MyModel()
    _ = model(example_batch[:1], training=False)
    model.load_weights(MODEL_PREFIX + model_id)
    return model


def add_noise(data, noise_level, rng):
    if NOISE_MODE == "gaussian":
        noise = rng.normal(0.0, noise_level, data.shape)
    elif NOISE_MODE == "uniform":
        noise = rng.uniform(-noise_level, noise_level, data.shape)
    else:
        raise ValueError("NOISE_MODE must be gaussian or uniform")
    return data + noise


def predict_in_batches(model, x, batch_size):
    preds = []
    for start in range(0, len(x), batch_size):
        y_batch = model(x[start:start + batch_size], training=False)
        preds.append(y_batch.numpy() if isinstance(y_batch, tf.Tensor) else y_batch)
    return np.concatenate(preds, axis=0)


def score_prediction(y_true, y_pred):
    if y_pred.shape[-1] != y_true.shape[-1]:
        min_dim = min(y_pred.shape[-1], y_true.shape[-1])
        y_pred = y_pred[..., :min_dim]
        y_true = y_true[..., :min_dim]

    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    if METRIC == "r2":
        return r2_score(y_true_flat, y_pred_flat, multioutput="uniform_average")
    if METRIC == "mse":
        return mean_squared_error(y_true_flat, y_pred_flat)
    raise ValueError("METRIC must be r2 or mse")


def evaluate_model(model, x_val, y_val, noise_levels, noise_trials, rng):
    model_scores = []
    for noise_level in noise_levels:
        trial_scores = []
        for _ in range(noise_trials):
            x_noisy = add_noise(x_val, noise_level, rng)
            y_pred = predict_in_batches(model, x_noisy, BATCH_SIZE)
            trial_scores.append(score_prediction(y_val, y_pred))
        model_scores.append(np.mean(trial_scores))
    return np.asarray(model_scores)


def evaluate_group(group, x_val, y_val, rng):
    results = []
    for index, model_id in enumerate(group["model_ids"], start=1):
        print(f"[{group['name']}] {index}/{len(group['model_ids'])}: model_{model_id}")
        try:
            model = load_model(model_id, x_val)
        except Exception as exc:
            if STRICT_WEIGHTS:
                raise
            print(f"  skipped: {exc}")
            continue

        scores = evaluate_model(model, x_val, y_val, NOISE_LEVELS, NOISE_TRIALS, rng)
        print(f"  R2 noise=0: {scores[0]:.5f}, noise=max: {scores[-1]:.5f}")
        results.append(scores)

    if not results:
        raise RuntimeError(f"No models were evaluated for group {group['name']}")
    return np.asarray(results)


# ================================
# Main
# ================================
def main():
    configure_gpu_memory_growth()
    rng = np.random.default_rng(RANDOM_SEED)
    x_val, y_val = load_evaluation_data()

    print(f"Evaluation data: x={x_val.shape}, y={y_val.shape}")
    print(f"Noise levels: {len(NOISE_LEVELS)} from {NOISE_LEVELS[0]:.2f} to {NOISE_LEVELS[-1]:.2f}")
    print(f"Noise trials per model per level: {NOISE_TRIALS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for group in GROUPS:
        results = evaluate_group(group, x_val, y_val, rng)
        output_path = OUTPUT_DIR / group["output_file"]
        np.save(output_path, results)
        print(f"Saved {output_path}: shape={results.shape}")


if __name__ == "__main__":
    main()
