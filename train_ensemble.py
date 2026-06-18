import subprocess
import os
from pathlib import Path

def main():
    seeds = [123, 456]
    
    # Ensure directory exists
    ensemble_dir = Path("results/checkpoints/ensemble")
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    for seed in seeds:
        ckpt_path = ensemble_dir / f"model_{seed}.pt"
        if ckpt_path.exists():
            print(f"Ensemble member for seed {seed} already exists. Skipping training.")
            continue
            
        print(f"\n==========================================")
        print(f"Training ensemble member with seed: {seed}")
        print(f"==========================================\n")
        
        env = os.environ.copy()
        env["ENSEMBLE_SEED"] = str(seed)
        env["ENSEMBLE_CKPT_NAME"] = f"ensemble/model_{seed}.pt"
        env["ENSEMBLE_LOG_NAME"] = f"ensemble_log_{seed}.json"
        
        cmd = ["venv/Scripts/python", "train_inter_patient.py"]
        subprocess.run(cmd, env=env, check=True)
        print(f"\nFinished training seed {seed}. Saved to {ckpt_path}\n")

if __name__ == "__main__":
    main()
