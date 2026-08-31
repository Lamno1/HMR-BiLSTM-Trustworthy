"""Fast preflight for the S-only conformal-calibration record split.

Run before a long LOPO job:
    python validation/check_lopo_conformal_splits.py
"""
import numpy as np

from train_lopo_cv import POOL, assign_folds, carve_conformal_s_cal, carve_inner_val


def main():
    d = np.load(POOL)
    y, record, source = d["y"], d["record"], d["source"]
    rec_cls = {int(r): np.bincount(y[record == r], minlength=5).astype(float)
               for r in np.unique(record)}
    rec_src = {int(r): int(source[record == r][0]) for r in np.unique(record)}
    fold_of = assign_folds(rec_cls, rec_src, 5, 0)
    for fold in range(5):
        test = {r for r, f in fold_of.items() if f == fold}
        trval = [r for r in fold_of if r not in test]
        val = carve_inner_val(trval, rec_cls, rec_src, 0.15, fold)
        train = set(trval) - val
        cal = carve_conformal_s_cal(sorted(train), rec_cls, 0.15, 2000 + fold)
        assert not (cal & test or cal & val)
        assert all(rec_cls[r][1] > 0 and rec_cls[r][3] == 0 for r in cal)
        n_s = int(sum(rec_cls[r][1] for r in cal))
        n_f_val = int(sum(rec_cls[r][3] for r in val))
        n_f_train = int(sum(rec_cls[r][3] for r in train))
        max_f_frac_val = max((rec_cls[r][3] for r in val), default=0) / max(n_f_val, 1)
        print(f"fold={fold}: S-cal records={len(cal)}, S beats={n_s}")
        print(f"  inner-val F beats={n_f_val}, model-train F beats={n_f_train}")
        print(f"  inner-val F: largest-record share={max_f_frac_val:.0%}")


if __name__ == "__main__":
    main()
