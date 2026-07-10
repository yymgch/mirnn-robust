# %%
"""Generate structured-dropout robustness data for Fig. 4.

The trained model weight files are assumed to already exist as model_<id>.
This script reproduces the saved pickle files consumed by
plot_dropout_robustness_from_saved.py.
"""
from pathlib import Path
import json
import pickle
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, GRU


# ================================
# Settings
# ================================
DATA1_PATH = Path("Chaos_Signals/data1_102.npy")
DATA2_PATH = Path("Chaos_Signals/data2_102.npy")
MODEL_PREFIX = "model_"
OUTPUT_DIR = Path(".")

WAVE_LENGTH = 10000
EVAL_SAMPLES = 1000
HIDDEN_DIM = 100
MAX_K = 20
DROPOUT_TRIALS = 20
MASK_SEED = 42
BATCH_SIZE = 50
STRICT_WEIGHTS = True

# Keep _nr for compatibility with the original exploratory notebooks.
# The Fig. 4 plotting script currently uses only _m and _nm.
SUFFIXES_TO_EVALUATE = ["_m", "_nm", "_nr"]

OUTPUT_FILES = {
    "nonprimary_upper": "combined_results3_1.pkl",
    "nonprimary_lower": "combined_results3_2.pkl",
    "primary_upper": "combined_results3_1warui.pkl",
    "primary_lower": "combined_results3_2warui.pkl",
}


# ================================
# Model groups
# ================================
# These lists contain the models retained for the ablation analysis after
# filtering by baseline performance: mean R2 > 0.997.
#
# "backslash" and "slash" denote the two possible assignments of the two
# hidden-unit subgroups to the two output systems. For backslash models, the
# front hidden subgroup is primary for the upper/Lorenz outputs and the back
# hidden subgroup is primary for the lower/Rossler outputs. For slash models,
# this assignment is reversed.
BACKSLASH_M = [
    "10", "17", "22", "25", "26", "45", "46", "74", "101", "102", "103", "104", "106", "123", "125", "128",
    "130", "158", "159", "161", "163", "164", "165", "166", "167", "172", "201", "202", "204", "205", "209",
    "212", "213", "214", "232", "234", "235", "236", "237", "238", "239", "240", "262", "269", "300", "301", "303", "305", "310", "312",
]

SLASH_M = [
    "12", "13", "15", "16", "24", "28", "29", "30", "31", "32", "33", "34", "44", "47", "71", "73", "75", "76",
    "100", "126", "127", "311", "131", "150", "151", "153", "154", "155", "156", "157", "160", "162", "168",
    "171", "203", "210", "211", "231", "241", "242", "264", "265", "266", "267", "268", "302", "304", "306", "308", "309",
]

BACKSLASH_NM = [
    "nm2", "nm3", "nm6", "nm11", "nm14", "nm16", "nm17", "nm18", "nm21", "nm22", "nm23", "nm25", "nm26", "nm32", "nm34",
    "nm35", "nm39", "nm40", "nm41", "nm42", "nm44", "nm47", "nm48", "nm52", "nm53", "nm57", "nm58", "nm59", "nm62", "nm63",
    "nm66", "nm74", "nm80", "nm83", "nm85", "nm86", "nm93", "nm94", "nm95", "nm98",
]

SLASH_NM = [
    "nm1", "nm4", "nm5", "nm7", "nm8", "nm9", "nm10", "nm12", "nm13", "nm15", "nm19", "nm20", "nm24", "nm27", "nm28", "nm29",
    "nm30", "nm31", "nm33", "nm36", "nm37", "nm38", "nm43", "nm45", "nm46", "nm49", "nm50", "nm51", "nm54", "nm55", "nm56", "nm60",
    "nm61", "nm64", "nm65", "nm67", "nm68", "nm69", "nm70", "nm71", "nm72", "nm73", "nm75", "nm76", "nm77", "nm78", "nm79", "nm81",
    "nm82", "nm84", "nm87", "nm88", "nm89", "nm90", "nm91", "nm92", "nm96", "nm97", "nm99", "nm100",
]

BACKSLASH_NR = [
    "nr2", "nr6", "nr8", "nr11", "nr12", "nr14", "nr15", "nr17", "nr18", "nr19", "nr20", "nr21", "nr22", "nr23", "nr24", "nr26", "nr28", "nr31",
    "nr32", "nr34", "nr35", "nr37", "nr39", "nr43", "nr46", "nr47", "nr48", "nr49", "nr50", "nr54", "nr56", "nr59", "nr63", "nr65", "nr66", "nr67",
    "nr68", "nr72", "nr76", "nr77", "nr79", "nr82", "nr84", "nr86", "nr87", "nr88", "nr93", "nr98", "nr100",
]

SLASH_NR = [
    "nr1", "nr3", "nr4", "nr5", "nr7", "nr9", "nr10", "nr13", "nr16", "nr25", "nr27", "nr29", "nr30", "nr33", "nr36", "nr38", "nr40", "nr41", "nr42",
    "nr44", "nr45", "nr51", "nr52", "nr53", "nr55", "nr57", "nr58", "nr60", "nr61", "nr62", "nr64", "nr69", "nr70", "nr71", "nr73", "nr74", "nr75", "nr78",
    "nr80", "nr81", "nr83", "nr85", "nr89", "nr90", "nr91", "nr92", "nr94", "nr95", "nr96", "nr97", "nr99",
]

GROUPS = {
    "backslash_m": BACKSLASH_M,
    "slash_m": SLASH_M,
    "backslash_nm": BACKSLASH_NM,
    "slash_nm": SLASH_NM,
    "backslash_nr": BACKSLASH_NR,
    "slash_nr": SLASH_NR,
}


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


def generate_common_dropout_masks(hidden_dim, max_k, n_trials, region, seed):
    rng = np.random.RandomState(seed)
    if region == "back":
        candidate_indices = np.arange(hidden_dim // 2, hidden_dim)
    elif region == "front":
        candidate_indices = np.arange(0, hidden_dim // 2)
    else:
        raise ValueError("region must be front or back")

    # Index 0 corresponds to k=0 (no ablation, baseline): a single empty mask.
    all_masks = [[np.array([], dtype=int)]]
    for k in range(1, max_k + 1):
        trial_masks = []
        for _ in range(n_trials):
            trial_masks.append(rng.choice(candidate_indices, size=k, replace=False))
        all_masks.append(trial_masks)
    return all_masks


def predict_with_structured_dropout(model, x, drop_indices, batch_size):
    dense_weights, dense_bias = model.dense1.get_weights()
    mask = np.ones_like(dense_weights)
    mask[drop_indices, :] = 0.0
    masked_weights = dense_weights * mask

    preds = []
    for start in range(0, len(x), batch_size):
        batch = x[start:start + batch_size]
        gru_output = model.gru(batch, training=False)
        gru_output = tf.cast(gru_output, tf.float32)
        y_pred = tf.matmul(gru_output, masked_weights) + dense_bias
        preds.append(y_pred.numpy())
    return np.concatenate(preds, axis=0)


def output_slice(output_region):
    if output_region == "front":
        return slice(0, 3)
    if output_region == "back":
        return slice(3, 6)
    raise ValueError("output_region must be front or back")


def evaluate_model_group(model_ids, dropout_masks, x_val, y_val, drop_region, output_region):
    scores_by_model = []
    target_slice = output_slice(output_region)
    total_models = len(model_ids)
    start_time = time.time()

    print(f"\n=== {drop_region.upper()} drop -> {output_region.upper()} output ({total_models} models) ===")
    for index, model_id in enumerate(model_ids, start=1):
        print(f"[{index}/{total_models}] model_{model_id} ... ", end="", flush=True)
        try:
            model = load_model(model_id, x_val)
        except Exception as exc:
            if STRICT_WEIGHTS:
                raise
            print(f"skipped: {exc}")
            continue

        scores_for_k = []
        for trial_masks in dropout_masks:
            trial_scores = []
            for drop_indices in trial_masks:
                y_pred = predict_with_structured_dropout(model, x_val, drop_indices, BATCH_SIZE)
                y_true_flat = y_val.reshape(-1, y_val.shape[-1])
                y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
                trial_scores.append(
                    r2_score(
                        y_true_flat[:, target_slice],
                        y_pred_flat[:, target_slice],
                        multioutput="uniform_average",
                    )
                )
            scores_for_k.append(np.mean(trial_scores))
        scores_by_model.append(scores_for_k)
        print("done")

    print(f"Elapsed: {time.time() - start_time:.2f} sec")
    if not scores_by_model:
        # No models in this orientation group: return a correctly shaped empty
        # array (0, n_k) so it concatenates with the other orientation.
        return np.empty((0, len(dropout_masks)))
    return np.asarray(scores_by_model)


def summarize_by_suffix(results_by_group, suffixes):
    combined = {}
    for suffix in suffixes:
        data = np.concatenate(
            [results_by_group[f"backslash{suffix}"], results_by_group[f"slash{suffix}"]],
            axis=0,
        )
        # Store the full per-model matrix (n_models, n_k) so downstream analyses
        # (mean, 95% CI, primary-vs-non-primary gap, interaction tests) can be
        # computed. Column index 0 is k=0 (no ablation, baseline).
        combined[suffix] = data
    return combined


def evaluate_condition(x_val, y_val, front_masks, back_masks, condition, output_region):
    results = {}
    for suffix in SUFFIXES_TO_EVALUATE:
        for orientation in ["backslash", "slash"]:
            group_name = f"{orientation}{suffix}"
            model_ids = GROUPS[group_name]

            if condition == "nonprimary" and output_region == "front":
                drop_region = "back" if orientation == "backslash" else "front"
            elif condition == "nonprimary" and output_region == "back":
                drop_region = "front" if orientation == "backslash" else "back"
            elif condition == "primary" and output_region == "front":
                drop_region = "front" if orientation == "backslash" else "back"
            elif condition == "primary" and output_region == "back":
                drop_region = "back" if orientation == "backslash" else "front"
            else:
                raise ValueError("Unknown condition/output_region combination")

            masks = front_masks if drop_region == "front" else back_masks
            results[group_name] = evaluate_model_group(
                model_ids,
                masks,
                x_val,
                y_val,
                drop_region=drop_region,
                output_region=output_region,
            )

    return summarize_by_suffix(results, SUFFIXES_TO_EVALUATE)


def save_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file)
    print(f"Saved {path}")


# ================================
# Main
# ================================
# Orientation assignment produced by determine_orientations.py. When present it
# replaces the hardcoded BACKSLASH_*/SLASH_* lists, so the ablation analysis runs
# on the freshly selected and oriented models (Reviewer 1, point 2 / point 4).
ORIENTATION_JSON = Path("model_orientations.json")


def main():
    global GROUPS, SUFFIXES_TO_EVALUATE
    configure_gpu_memory_growth()

    if ORIENTATION_JSON.exists():
        GROUPS = json.loads(ORIENTATION_JSON.read_text())
        suffixes = sorted({key.split("_", 1)[1] for key in GROUPS}, key=lambda s: ("_" + s))
        SUFFIXES_TO_EVALUATE = ["_" + s for s in suffixes]
        print(f"Using orientations from {ORIENTATION_JSON}: "
              f"{sum(len(v) for v in GROUPS.values())} models, suffixes={SUFFIXES_TO_EVALUATE}")
    else:
        print(f"{ORIENTATION_JSON} not found; using built-in BACKSLASH_*/SLASH_* lists")

    x_val, y_val = load_evaluation_data()
    print(f"Evaluation data: x={x_val.shape}, y={y_val.shape}")

    front_masks = generate_common_dropout_masks(HIDDEN_DIM, MAX_K, DROPOUT_TRIALS, "front", MASK_SEED)
    back_masks = generate_common_dropout_masks(HIDDEN_DIM, MAX_K, DROPOUT_TRIALS, "back", MASK_SEED)

    nonprimary_upper = evaluate_condition(x_val, y_val, front_masks, back_masks, "nonprimary", "front")
    nonprimary_lower = evaluate_condition(x_val, y_val, front_masks, back_masks, "nonprimary", "back")
    primary_upper = evaluate_condition(x_val, y_val, front_masks, back_masks, "primary", "front")
    primary_lower = evaluate_condition(x_val, y_val, front_masks, back_masks, "primary", "back")

    save_pickle(OUTPUT_DIR / OUTPUT_FILES["nonprimary_upper"], nonprimary_upper)
    save_pickle(OUTPUT_DIR / OUTPUT_FILES["nonprimary_lower"], nonprimary_lower)
    save_pickle(OUTPUT_DIR / OUTPUT_FILES["primary_upper"], primary_upper)
    save_pickle(OUTPUT_DIR / OUTPUT_FILES["primary_lower"], primary_lower)


if __name__ == "__main__":
    main()
