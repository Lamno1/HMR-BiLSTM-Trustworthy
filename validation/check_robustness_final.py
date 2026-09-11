import json, sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "results/robustness_final/results.json")
d = json.loads(p.read_text())
assert d["protocol_version"] == "ROBUST_FINAL_V1"
for phase in ("fgsm", "pgd20"):
    assert set(d[phase]) == {"LSTM", "BiLSTM", "ResMemECG"}
    for name, r in d[phase].items():
        assert r["state_unchanged"] and r["clean_probe_max_abs_diff"] == 0
        assert r["linf_max"] <= r["epsilon"] + 1e-5
        assert 0 <= r["conditional_asr"] <= 1
assert set(d["cw"]) == {"ResMemECG"}
r = d["cw"]["ResMemECG"]
assert r["state_unchanged"] and r["clean_probe_max_abs_diff"] == 0
assert 0 <= r["conditional_asr"] <= 1
print("PASS", p)

if len(sys.argv) > 2:
    q = Path(sys.argv[2])
    other = json.loads(q.read_text())
    for obj in (d, other):
        for phase in ("fgsm", "pgd20", "cw"):
            for result in obj[phase].values():
                result.pop("runtime_seconds", None)
    assert d == other, "Reproducibility mismatch outside runtime"
    print("BIT_IDENTICAL_EXCEPT_RUNTIME", q)
