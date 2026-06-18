import os
import time
import train_inter_patient

os.environ["ENSEMBLE_SEED"] = "123"
os.environ["ENSEMBLE_CKPT_NAME"] = "ensemble/test_model.pt"
os.environ["ENSEMBLE_LOG_NAME"] = "test_log.json"

train_inter_patient.CONFIG["epochs"] = 1
t0 = time.time()
train_inter_patient.main()
elapsed = time.time() - t0
print(f"\n[SPEED TEST] 1 epoch completed in: {elapsed:.2f} seconds\n")
