import torch
from hmr_bilstm_ablation import RLSTMClassifier, RLSTMLoss

def main():
    device = torch.device("cpu")
    print("=== Verify No-RMC Dummy Weights ===")
    
    # 1. Initialize No-RMC model
    model = RLSTMClassifier(
        input_size=1, hidden_size=96, dropout=0.25, num_classes=5,
        cnn_out_channels=64, num_layers=2,
        use_rmc=False, use_hybrid=True, use_interaction=True
    ).to(device)
    
    # 2. Dummy forward & backward pass
    criterion = RLSTMLoss(lambda_smooth=0.003, class_weights=None, use_focal=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    dummy_x = torch.randn(16, 187, 1).to(device)
    dummy_y = torch.randint(0, 4, (16,)).to(device)
    
    model.train()
    optimizer.zero_grad()
    
    logits, internals = model(dummy_x, return_internals=True)
    loss, _ = criterion(logits, dummy_y, r_fwd=internals["r_fwd"], r_bwd=internals["r_bwd"])
    loss.backward()
    
    # 3. Check gradients of RMC components
    rmc_params = ["W_c.weight", "W_h_rmc.weight", "cell.layer_norm.weight", "cell.layer_norm.bias", "W_alpha.weight", "W_alpha.bias", "W_beta.weight", "W_beta.bias"]
    
    total_grad_abs = 0.0
    for name, param in model.named_parameters():
        if any(rmc_name in name for rmc_name in rmc_params):
            if param.grad is not None:
                grad_sum = param.grad.abs().sum().item()
                if grad_sum > 0:
                    print(f"  {name:40s} grad sum = {grad_sum}")
                total_grad_abs += grad_sum
            else:
                print(f"  {name:40s} grad is None")
    
    print(f"\nTotal absolute gradient of RMC parameters: {total_grad_abs}")
    if total_grad_abs == 0:
        print("[VERIFIED] RMC weights receive exactly 0 gradients. They are dead weights.")
    else:
        print("[WARNING] RMC weights receive gradients! Leakage detected.")

if __name__ == "__main__":
    main()
