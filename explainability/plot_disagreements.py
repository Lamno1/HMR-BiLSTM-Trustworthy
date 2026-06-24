import yaml
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from configs.paths import get_run_id, build_paths, RLSTM_CKPT, INTER_TRAIN, INTER_TEST
from report_results import load_hmr_bilstm
from explainability.data_attribution import plot_waveform_pair

def main():
    with open("configs/experiment_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_id = get_run_id(cfg)
    paths = build_paths(run_id)
    out_dir = paths["out_explain"]
    wave_dir = out_dir / "tracin_waveforms"
    wave_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
    model.eval()

    print("Loading data...")
    test = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)

    train = np.load(INTER_TRAIN)
    X_train = train["X"].astype(np.float32)
    y_train = train["y"].astype(np.int64)

    csv_path = out_dir / "confusable_samples_all.csv"
    print(f"Reading confusable list from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Filter for self-disagreements
    disagreements = df[df.train_label != df.model_self_pred]
    print(f"Found {len(disagreements)} self-disagreements. Generating plots...")

    # Class name to integer mapping
    name_to_int = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}

    for idx, row in disagreements.iterrows():
        rank = int(row["rank"])
        t_idx = int(row["train_idx"])
        te_idx = int(row["test_idx"])
        train_label_str = row["train_label"]
        test_true_str = row["test_true"]
        test_pred_str = row["test_pred"]
        self_pred_str = row["model_self_pred"]
        self_conf = float(row["model_self_conf"])
        influence_val = float(row["cosine_influence"])

        train_label_int = name_to_int[train_label_str]
        test_true_int = name_to_int[test_true_str]
        test_pred_int = name_to_int[test_pred_str]
        self_pred_int = name_to_int[self_pred_str]

        self_info = {
            "model_self_pred": self_pred_str,
            "model_self_conf": self_conf,
            "agrees_noise": True, # Make border red so we can see it clearly
            "disagreement_str": f"model predicts {self_pred_str} (conf={self_conf:.2f}) != annotated {train_label_str}"
        }

        fname = f"disagree_rank_{rank}_tr{t_idx}_te{te_idx}.png"
        out_path = wave_dir / fname

        plot_waveform_pair(
            train_wave=X_train[t_idx],
            train_label_int=train_label_int,
            train_idx=t_idx,
            test_wave=X_test[te_idx],
            test_true_int=test_true_int,
            test_pred_int=test_pred_int,
            test_idx=te_idx,
            influence_val=influence_val,
            self_disagree_info=self_info,
            out_path=out_path
        )
        print(f"Saved {out_path}")

    print("Complete.")

if __name__ == "__main__":
    main()
