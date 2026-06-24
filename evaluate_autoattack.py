import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score
import torchattacks
from hmr_bilstm_ablation import RLSTMClassifier

class DenormWrapper(nn.Module):
    """
    Wraps the base model to denormalize inputs from [0, 1] back to the original scale
    before passing them to the model. This allows torchattacks (which assumes [0, 1] inputs)
    to attack the model correctly.
    """
    def __init__(self, base_model, d_min, d_max):
        super().__init__()
        self.base_model = base_model
        self.d_min = d_min
        self.d_max = d_max

    def forward(self, x_norm):
        # torchattacks expects 4D inputs [B, C, H, W]. 
        # We pass [B, 1, 187, 1], so we must squeeze out the dummy channel dim.
        if x_norm.dim() == 4 and x_norm.shape[1] == 1:
            x_norm = x_norm.squeeze(1)
            
        # Denormalize: [0, 1] -> [d_min, d_max]
        x_denorm = x_norm * (self.d_max - self.d_min) + self.d_min
        
        out = self.base_model(x_denorm)
        if isinstance(out, tuple):
            out = out[0]
            
        return out

def load_data(data_dir):
    test_data = np.load(f"{data_dir}/inter_test.npz")
    X, y = test_data["X"], test_data["y"]
    return X, y

def get_stratified_subset(X, y, target_N_size=2000, seed=42):
    np.random.seed(seed)
    # Get indices of each class
    idx_N = np.where(y == 0)[0]
    idx_S = np.where(y == 1)[0]
    idx_V = np.where(y == 2)[0]
    idx_F = np.where(y == 3)[0]
    
    # Subsample N
    if len(idx_N) > target_N_size:
        idx_N_sub = np.random.choice(idx_N, target_N_size, replace=False)
    else:
        idx_N_sub = idx_N
        
    # Combine all indices
    subset_idx = np.concatenate([idx_N_sub, idx_S, idx_V, idx_F])
    np.random.shuffle(subset_idx)
    
    X_sub = X[subset_idx]
    y_sub = y[subset_idx]
    
    print(f"Subsampled test set size: {len(y_sub)}")
    print(f"Class distribution - N: {len(idx_N_sub)}, S: {len(idx_S)}, V: {len(idx_V)}, F: {len(idx_F)}")
    
    return X_sub, y_sub

def load_model(ckpt_path, device, variant_flags):
    model = RLSTMClassifier(
        input_size=1, hidden_size=96, dropout=0.25, num_classes=5,
        cnn_out_channels=64, num_layers=2,
        use_rmc=variant_flags["use_rmc"],
        use_cnn=variant_flags["use_cnn"],
        use_attention=variant_flags["use_attention"]
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def run_autoattack(model, wrapper, X_norm, y, eps_norm, device, batch_size=128, n_queries_square=1000):
    # AutoAttack version 'standard' contains APGD-CE, APGDT, FAB-T, Square.
    attack = torchattacks.AutoAttack(wrapper, norm='Linf', eps=eps_norm, version='standard', n_classes=5)
    
    # Customize Square Attack queries to save execution time
    if hasattr(attack, "_autoattack") and len(attack._autoattack.attacks) >= 4:
        attack._autoattack.attacks[3].n_queries = n_queries_square
        # Enable accumulation of records for step-by-step breakdown
        attack._autoattack._accumulate_multi_atk_records = True
        attack._autoattack._multi_atk_records = [0.0] * (len(attack._autoattack.attacks) + 1)
        print(f"Configured AutoAttack Standard with Square Attack queries={n_queries_square}")
        
    ds = TensorDataset(torch.from_numpy(X_norm).float().unsqueeze(1), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    all_adv_preds = []
    all_clean_preds = []
    
    for i, (bx, by) in enumerate(loader):
        bx, by = bx.to(device), by.to(device)
        
        # Clean predictions
        with torch.no_grad():
            clean_logits = wrapper(bx)
            clean_preds = clean_logits.argmax(dim=1).cpu().numpy()
            all_clean_preds.extend(clean_preds)
            
        # Adversarial predictions
        bx_adv = attack(bx, by)
        with torch.no_grad():
            adv_logits = wrapper(bx_adv)
            adv_preds = adv_logits.argmax(dim=1).cpu().numpy()
            all_adv_preds.extend(adv_preds)
            
        print(f"  [Batch {i+1}/{len(loader)}] attacked and verified.")
        
    step_accs = []
    if hasattr(attack, "_autoattack") and attack._autoattack._accumulate_multi_atk_records:
        records = attack._autoattack._multi_atk_records
        total_eval = records[0]
        if total_eval > 0:
            step_accs = [float(r / total_eval) for r in records]
        else:
            step_accs = [0.0] * (len(attack._autoattack.attacks) + 1)
            
    return np.array(all_clean_preds), np.array(all_adv_preds), step_accs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load data
    print("Loading data...")
    X, y = load_data("data/processed/splits")
    
    # Use stratified subset: keep all minority class samples (S, V, F, Q),
    # subsample N class to target_N_size=2000 for manageable AutoAttack runtime.
    # This is the approach recommended in P1_PROTOCOL_UNIFICATION_PLAN.md §6.
    X_sub, y_sub = get_stratified_subset(X, y, target_N_size=2000)
    
    d_min = float(X_sub.min())
    d_max = float(X_sub.max())
    print(f"Data range: [{d_min:.4f}, {d_max:.4f}]")
    
    # Normalize input to [0, 1] for torchattacks
    X_norm = (X_sub - d_min) / (d_max - d_min)
    
    # Clinical epsilons of interest: 0.02 (training eps), 0.03, 0.05
    epsilons = [0.02, 0.03, 0.05]
    
    variants = [
        {
            "name": "No-Adv",
            "ckpt": "results/ablation/inter/checkpoints/best_rlstm_no_adv.pt",
            "flags": {"use_rmc": True, "use_cnn": True, "use_attention": True}
        },
        {
            "name": "Full HMR",
            "ckpt": "results/ablation/inter/checkpoints/best_rlstm_full.pt",
            "flags": {"use_rmc": True, "use_cnn": True, "use_attention": True}
        }
    ]
    
    os.makedirs("results/robustness", exist_ok=True)
    all_results = []
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Evaluating model variant: {variant['name']}")
        print(f"{'='*60}")
        
        # Load baseline model
        base_model = load_model(variant["ckpt"], device, variant["flags"])
        wrapper = DenormWrapper(base_model, d_min, d_max).to(device)
        wrapper.eval()
        
        # Try compiling the wrapper to speed up sequential loop execution (PyTorch 2.0+)
        try:
            print("Attempting to compile model wrapper using torch.compile...")
            compiled_wrapper = torch.compile(wrapper, dynamic=False)
            # Warm up compiler with a dummy forward pass
            dummy_x = torch.zeros(1, 1, 187, 1, device=device)
            with torch.no_grad():
                _ = compiled_wrapper(dummy_x)
            wrapper = compiled_wrapper
            print("Model compiled successfully! Sequential loops should run faster.")
        except Exception as e:
            print(f"torch.compile failed or is not supported: {e}. Running in standard mode.")
            wrapper = DenormWrapper(base_model, d_min, d_max).to(device)
            wrapper.eval()
        
        for eps in epsilons:
            print(f"\n--- Running evaluation for epsilon = {eps} ---")
            eps_norm = eps / (d_max - d_min)
            
            clean_preds, adv_preds, step_accs = run_autoattack(
                base_model, wrapper, X_norm, y_sub, eps_norm, device, batch_size=128, n_queries_square=1000
            )
            
            # Compute general metrics
            clean_acc = accuracy_score(y_sub, clean_preds)
            robust_acc = accuracy_score(y_sub, adv_preds)
            clean_correct = (clean_preds == y_sub)
            adv_wrong = (adv_preds != y_sub)
            overall_asr = float((clean_correct & adv_wrong).sum()) / float(clean_correct.sum()) if clean_correct.sum() > 0 else 0.0
            
            # Compute class V metrics (label 2) - to prove attack works on well-learned class
            idx_V = (y_sub == 2)
            clean_recall_V = recall_score(y_sub, clean_preds, labels=[2], average=None, zero_division=0)[0]
            robust_recall_V = recall_score(y_sub, adv_preds, labels=[2], average=None, zero_division=0)[0]
            clean_correct_V = (clean_preds[idx_V] == 2)
            adv_wrong_V = (adv_preds[idx_V] != 2)
            asr_V = float((clean_correct_V & adv_wrong_V).sum()) / float(clean_correct_V.sum()) if clean_correct_V.sum() > 0 else 0.0
            
            # Compute class F metrics (label 3) - for clinical completeness
            clean_recall_F = recall_score(y_sub, clean_preds, labels=[3], average=None, zero_division=0)[0]
            robust_recall_F = recall_score(y_sub, adv_preds, labels=[3], average=None, zero_division=0)[0]
            
            # Report progress in stdout
            print(f"  [Summary] Overall Clean Acc: {clean_acc*100:.2f}% | Robust Acc: {robust_acc*100:.2f}% | ASR: {overall_asr*100:.2f}%")
            print(f"  [Class V] Clean Recall-V: {clean_recall_V*100:.2f}% | Robust Recall-V: {robust_recall_V*100:.2f}% | ASR-V: {asr_V*100:.2f}%")
            print(f"  [Class F] Clean Recall-F: {clean_recall_F*100:.2f}% | Robust Recall-F: {robust_recall_F*100:.2f}%")
            
            # Print attack step-by-step accuracy breakdown (useful for detecting gradient masking)
            if len(step_accs) >= 5:
                print(f"  [Degradation] Clean Acc: {step_accs[0]*100:.2f}% -> APGD-CE: {step_accs[1]*100:.2f}% -> APGDT: {step_accs[2]*100:.2f}% -> FAB: {step_accs[3]*100:.2f}% -> Square: {step_accs[4]*100:.2f}%")
                acc_apgd_ce = step_accs[1]
                acc_apgdt = step_accs[2]
                acc_fab = step_accs[3]
                acc_square = step_accs[4]
            else:
                acc_apgd_ce = acc_apgdt = acc_fab = acc_square = robust_acc
                
            all_results.append({
                "Model": variant["name"],
                "Epsilon": eps,
                "Clean_Acc": clean_acc,
                "Robust_Acc": robust_acc,
                "Overall_ASR": overall_asr,
                "Clean_Recall_V": clean_recall_V,
                "Robust_Recall_V": robust_recall_V,
                "ASR_V": asr_V,
                "Clean_Recall_F": clean_recall_F,
                "Robust_Recall_F": robust_recall_F,
                "Acc_after_APGD_CE": acc_apgd_ce,
                "Acc_after_APGDT": acc_apgdt,
                "Acc_after_FAB": acc_fab,
                "Acc_after_Square": acc_square
            })
            
    df = pd.DataFrame(all_results)
    df.to_csv("results/robustness/autoattack_results.csv", index=False)
    
    print("\n" + "="*90)
    print(" AUTOATTACK DETAILED ROBUSTNESS REPORT ")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)

if __name__ == "__main__":
    main()
