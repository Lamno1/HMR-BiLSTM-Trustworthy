"""Recompute BEL Tables 1, 3-8, own rows of Table 2 and MLII/ranking supplement tables
from saved results and compare them with the LaTeX source.

Sources (no retraining):
  Table 3 tab:matched     results/matched_recipe/<arch>_seed<s>.json
  Table 4 tab:cellimpl    the same + results/tables/matched_diagnostics.json (record bootstrap)
  Table 5 tab:bootstrap   results/tables/matched_diagnostics.json
  Table 6 tab:transfer    results/tables/mechanism_transfer.json, mechanism_lbbb_train.json,
                          mechanism_lbbb.json
  Table 7 tab:sgcv-folds  results/controlled_mit_subject_cv/fold<f>_seed<s>_flat.json
  Table 8 tab:sgcv        results/tables/sgcv_arch_runs.csv + results/sgcv_arch/<arch>/*.json
                          (F1-S/F1-V); the fold-bootstrap interval is recomputed here
                          (2,000 resamples of the five folds, seed 0) because no saved script
                          produced the printed one.

Every printed number is compared with the recomputed one; a number printed with d decimals
passes if they differ by at most half a unit in the last place. Structural mismatches
also fail. Table 1 is independently recomputed from the saved five-class confusion matrices.
Exit code 1 if any cell or table shape fails.

    python -m validation.check_bel_tables [path/to/bel-article.tex]
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "results" / "matched_recipe"
TB = ROOT / "results" / "tables"
FLAT = ROOT / "results" / "controlled_mit_subject_cv"
SGCV = ROOT / "results" / "sgcv_arch"
SEEDS = (42, 123, 456)
PUBLIC_SNAPSHOT = False
NUM = re.compile(r"[+-]?\d[\d,]*\.?\d*")


# ---------------------------------------------------------------- LaTeX side
def table_rows(tex: str, label: str) -> list[list[float]]:
    """Numbers of every body row (after the first \\midrule) of the table with this label."""
    i = tex.index(f"\\label{{{label}}}")
    body = tex[tex.index("\\begin{tabular", i):tex.index("\\end{tabular}", i)]
    body = body[body.index("\\midrule"):]
    rows = []
    for line in body.split("\\\\"):
        if "&" not in line:
            continue
        cells = line.split("&")
        # Literature rows contain an input-description column (e.g. channel 0).
        if label == "tab:literature":
            numeric_cells = cells[2:]
        elif label in {"tab:ranking-val", "tab:ranking-test"}:
            numeric_cells = cells  # The first column is a numeric fold identifier.
        else:
            numeric_cells = cells[1:]
        text = "&".join(numeric_cells)
        text = re.sub(r"\\texttt\{[^}]*\}|\\beta_t=1|\$\{?\}?", " ", text).replace("\\,", "")
        text = text.replace("$-$", "-").replace("--", " ")
        rows.append([decimals(t) for t in NUM.findall(text.replace("$", ""))])
    return rows


def decimals(s: str):
    s = s.replace(",", "")
    return float(s), (len(s.split(".")[1]) if "." in s else 0)


# ---------------------------------------------------------------- results side
def mr(arch, seed):
    return json.loads((MR / f"{arch}_seed{seed}.json").read_text(encoding="utf-8"))["test"]


def macro(arch):
    return np.array([mr(a, s)["f1_macro"] for a, s in [(arch, s) for s in SEEDS]])


def cls(arch, c, key="f1"):
    return float(np.mean([mr(arch, s)["per_class"][c][key] for s in SEEDS]))


def boot(pair):
    return json.loads((TB / "matched_diagnostics.json").read_text())["record_bootstrap"]["pairs"][pair]


def expect_matched():
    rows = []
    for a in ("bilstm", "normc", "full", "nohybrid", "reslstm", "bilstm_ce", "full_ce",
              "bilstm_nocw", "full_nocw"):
        m = macro(a)
        rows.append([m.mean(), m.std(ddof=1), cls(a, "S"), cls(a, "V"), cls(a, "F"), cls(a, "F", "auc")])
    return rows


def expect_baselines():
    data = json.loads((TB / "per_class_ds2.json").read_text())
    names = ["Logistic regression", "Decision tree", "LSTM", "BiLSTM", "ResNet1D",
             "InceptionTime", "TCN", "ECG-Transformer", "ResMemECG_s42",
             "ResMemECG_s123", "ResMemECG_s456"]
    rows = []
    for name in names:
        # All three ResMemECG rows now use matched-harness reruns.
        if name.startswith("ResMemECG_s"):
            seed = int(name.split("_s")[1])
            if PUBLIC_SNAPSHOT:
                summary = json.loads((TB / "matched_diagnostics.json").read_text())
                cm = summary["confusion_full"][str(seed)]
            else:
                with np.load(MR / f"full_seed{seed}_test_logits.npz") as d:
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(d["y"], d["logits"].argmax(axis=1), labels=list(range(5)))
        else:
            cm = data[name]["confusion_NSVFQ"]
        cm = np.asarray(cm, dtype=float)
        tp = cm.diagonal()
        p = np.divide(tp, cm.sum(axis=0), out=np.zeros(5), where=cm.sum(axis=0) != 0)
        r = np.divide(tp, cm.sum(axis=1), out=np.zeros(5), where=cm.sum(axis=1) != 0)
        f = np.divide(2 * p * r, p + r, out=np.zeros(5), where=p + r != 0)
        rows.append([tp.sum() / cm.sum(), f[:4].mean(), f[0],
                     *[v for c in (1, 2, 3) for v in (p[c], r[c], f[c])]])
    return rows


def expect_own_literature():
    rows = []
    for seed in SEEDS:
        t = mr("full", seed)["per_class"]
        rows.append([v for c in "SV" for v in
                     (100 * t[c]["recall"], 100 * t[c]["precision"], t[c]["f1"])])
    return rows


def expect_mlii():
    rows = []
    for arch in ("full", "normc", "bilstm"):
        for seed in SEEDS:
            d = json.loads((ROOT / "results" / "matched_recipe_mlii" /
                            f"{arch}_seed{seed}.json").read_text())
            t = d["test"]
            rows.append([seed, d["best_epoch"], t["f1_macro"],
                         *[t["per_class"][c]["f1"] for c in "SVF"]])
    return rows


def expect_cellimpl():
    def row(a, ref):
        m = macro(a)
        r = [m.mean(), m.std(ddof=1)]
        if ref:
            b = boot(f"{a}-{ref}")
            r += list(m - macro(ref)) + b["ci95"]
        return r + [cls(a, "S"), cls(a, "V"), cls(a, "F")]
    d = macro("full_nodrop") - macro("normc_nodrop")
    return [row("normc", None), row("normc_nodrop", "normc"), row("normc_nofb", "normc"),
            row("normc_torchinit", "normc"), row("normc_lstmlike", "normc"), row("full", None),
            row("full_nodrop", "full"), list(d) + boot("full_nodrop-normc_nodrop")["ci95"]]


def expect_bootstrap():
    pairs = ["full-normc", "full-bilstm", "normc-bilstm", "nohybrid-normc", "bilstm_ce-bilstm",
             "full_ce-bilstm_ce", "full_nocw-bilstm_nocw", "full-reslstm", "normc-reslstm",
             "normc_nodrop-normc", "normc_nofb-normc", "normc_torchinit-normc",
             "normc_lstmlike-normc", "full_nodrop-full", "full_nodrop-normc_nodrop"]
    return [[b["point"], *b["ci95"], b["p_gt0"], *b["loo_range"]] for b in map(boot, pairs)]


def expect_transfer():
    mt = json.loads((TB / "mechanism_transfer.json").read_text())
    lt = json.loads((TB / "mechanism_lbbb_train.json").read_text())["train"]
    lb = json.loads((TB / "mechanism_lbbb.json").read_text())
    rows = []
    for a in ("bilstm", "bilstm_ce", "bilstm_nocw", "normc_lstmlike", "normc", "normc_nodrop",
              "full", "full_nodrop"):
        r = [float(np.mean([x[p]["auc_F_vs_N"] for x in mt[a]])) for p in ("train", "val", "test")]
        if a in lt:
            r.append(float(np.mean([x["auc_F_vs_L"] for x in lt[a]])))
        r.append(float(np.mean([x["auc_F_vs_L"] for x in lb[a]])))
        rows.append(r)
    return rows


def fold_metrics(f, s):
    """Use all five classes in predictions, including Q false positives/negatives."""
    t = json.loads((FLAT / f"fold{f}_seed{s}_flat.json").read_text())["test_metrics"]
    if PUBLIC_SNAPSHOT:
        counts = json.loads((FLAT / f"fold{f}_seed{s}_flat.json").read_text())["confusion_NSVFQ"]
        counts = np.asarray(counts, dtype=float)
        support = counts.sum(axis=1)
        predicted = counts.sum(axis=0)
        true_positive = counts.diagonal()
        p = np.divide(true_positive, predicted, out=np.zeros(5), where=predicted != 0)
        r = np.divide(true_positive, support, out=np.zeros(5), where=support != 0)
        score = np.divide(2 * p * r, p + r, out=np.zeros(5), where=p + r != 0)
    else:
        from sklearn.metrics import precision_recall_fscore_support
        with np.load(FLAT / f"fold{f}_seed{s}_flat_preds.npz") as d:
            p, r, score, support = precision_recall_fscore_support(
                d["y"], d["logits"].argmax(axis=1), labels=list(range(5)), zero_division=0)
    out = {}
    for c in "SF":
        k = "NSVFQ".index(c)
        out[c] = (support[k], p[k], r[k], score[k], t["auc_per_class"][c])
    return out


def expect_sgcv_folds():
    rows, per = [], []
    for f in range(5):
        a, b = fold_metrics(f, 42), fold_metrics(f, 123)
        r = [a["S"][0], a["F"][0]] + [np.mean([a[c][k], b[c][k]]) for c in "SF" for k in (1, 2, 3)] \
            + [np.mean([a["S"][4], b["S"][4]]), np.mean([a["F"][4], b["F"][4]])]
        rows.append(r)
        per.append(r)
    per = np.array(per)
    rows.append([per[:, 0].sum(), per[:, 1].sum(), *per[:, 2:].mean(axis=0)])
    return rows


def expect_sgcv():
    runs = list(csv.DictReader(open(TB / "sgcv_arch_runs.csv", encoding="utf-8")))
    val = {(r["core"], int(r["fold"]), int(r["seed"])): float(r["inner_val_macro_f1"]) for r in runs}
    tst = {(r["core"], int(r["fold"]), int(r["seed"])): float(r["outer_test_macro_f1"]) for r in runs}
    cores = ["full", "normc", "normc_nodrop", "bilstm", "bilstm_nocw", "bilstm_ce"]
    cells = sorted({(f, s) for _, f, s in tst})
    pick = {c: 0 for c in cores}
    best = {c: 0 for c in cores}
    for f, s in cells:
        pick[max(cores, key=lambda c: val[(c, f, s)])] += 1
        best[max(cores, key=lambda c: tst[(c, f, s)])] += 1

    def per_class(c, k):
        vals = []
        for f, s in cells:
            p = (FLAT / f"fold{f}_seed{s}_flat.json") if c == "full" else (SGCV / c / f"fold{f}_seed{s}.json")
            vals.append(json.loads(p.read_text())["test_metrics"]["f1_per_class"][k])
        return float(np.mean(vals))

    rng = np.random.default_rng(0)
    rows = []
    for c in cores:
        v = np.array([val[(c, f, s)] for f, s in cells])
        t = np.array([tst[(c, f, s)] for f, s in cells])
        r = [v.mean(), t.mean(), t.std(ddof=1)]
        if c != "normc":
            d = np.array([[tst[(c, f, s)] - tst[("normc", f, s)] for s in (42, 123)] for f in range(5)]).mean(axis=1)
            bs = d[rng.integers(0, 5, size=(2000, 5))].mean(axis=1)
            r += [d.mean(), *np.percentile(bs, [2.5, 97.5])]
        rows.append(r + [per_class(c, "S"), per_class(c, "V"), pick[c], best[c]])
    return rows


CHECKS = [("tab:baselines", expect_baselines, None),
          ("tab:literature", expect_own_literature, None),
          ("tab:matched", expect_matched, None), ("tab:cellimpl", expect_cellimpl, None),
          ("tab:bootstrap", expect_bootstrap, None), ("tab:transfer", expect_transfer, None),
          ("tab:sgcv-folds", expect_sgcv_folds, None), ("tab:sgcv", expect_sgcv, None)]


def expect_ranking_scores(partition):
    """Read the original SG-CV summary, independently of the ranking audit."""
    cores = ["full", "normc", "normc_nodrop", "bilstm", "bilstm_nocw", "bilstm_ce"]
    with (TB / "sgcv_arch_runs.csv").open(newline="", encoding="utf-8") as handle:
        scores = {
            (row["core"], int(row["fold"]), int(row["seed"])): float(row[partition])
            for row in csv.DictReader(handle)
        }
    return [[fold, seed, *[scores[core, fold, seed] for core in cores]]
            for fold in range(5) for seed in (42, 123)]


def check_public_snapshot(snapshot):
    """Verify exact published bytes and recompute numeric tables from aggregates."""
    global ROOT, MR, TB, FLAT, SGCV, PUBLIC_SNAPSHOT
    snapshot = snapshot.resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = (snapshot / entry["path"]).resolve()
        if not path.is_relative_to(snapshot):
            raise ValueError("Snapshot manifest path leaves its directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {entry['path']}")
    ROOT = snapshot
    MR = snapshot / "results/matched_recipe"
    TB = snapshot / "results/tables"
    FLAT = snapshot / "results/controlled_mit_subject_cv"
    SGCV = snapshot / "results/sgcv_arch"
    PUBLIC_SNAPSHOT = True
    expected = json.loads((snapshot / "table_values.json").read_text())["tables"]
    functions = {label: fn for label, fn, _ in CHECKS}
    functions["tab:mlii-sensitivity"] = expect_mlii
    functions["tab:ranking-val"] = lambda: expect_ranking_scores("inner_val_macro_f1")
    functions["tab:ranking-test"] = lambda: expect_ranking_scores("outer_test_macro_f1")
    count = 0
    for label, fn in functions.items():
        actual = fn()
        reference = expected[label]["values"]
        if len(actual) != len(reference):
            raise ValueError(f"Row count mismatch for {label}")
        for index, (row, saved) in enumerate(zip(actual, reference)):
            if len(row) != len(saved) or not np.allclose(row, saved, rtol=0, atol=1e-12):
                raise ValueError(f"Numeric mismatch for {label}, row {index + 1}")
            count += len(row)
        print(f"{label:22s} OK")
    print(f"\n{len(manifest['files'])} file checksums verified; {count} values reproduced")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--public-snapshot":
        if len(sys.argv) != 3:
            raise SystemExit("Usage: python -m validation.check_bel_tables --public-snapshot DIR")
        check_public_snapshot(Path(sys.argv[2]))
        return
    tex_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "latex" / "bel-article.tex"
    tex = tex_path.read_text(encoding="utf-8")
    n_ok = n_bad = 0
    checks = [(tex, label, fn) for label, fn, _ in CHECKS]
    supplement = ROOT / "latex" / "bel-supplement.tex"
    if supplement.exists():
        supplement_tex = supplement.read_text(encoding="utf-8")
        checks.append((supplement_tex, "tab:mlii-sensitivity", expect_mlii))
        for label, partition in [("tab:ranking-val", "inner_val_macro_f1"),
                                 ("tab:ranking-test", "outer_test_macro_f1")]:
            if f"\\label{{{label}}}" in supplement_tex:
                checks.append((supplement_tex, label,
                               lambda partition=partition: expect_ranking_scores(partition)))
    for source, label, fn in checks:
        printed, expected = table_rows(source, label), fn()
        if label == "tab:literature":
            printed = printed[-3:]  # Own results only; external literature requires source review.
        bad = []
        if len(printed) != len(expected):
            n_bad += 1
            bad.append(f"  row count: printed {len(printed)}, recomputed {len(expected)}")
        for i, (p, e) in enumerate(zip(printed, expected)):
            if len(p) != len(e):
                n_bad += 1
                bad.append(f"  row {i + 1}: {len(p)} numbers printed, {len(e)} recomputed")
                continue
            for j, ((val, dec), exp) in enumerate(zip(p, e)):
                tol = 0.5 * 10 ** -dec + 1e-9
                if abs(val - float(exp)) <= tol:
                    n_ok += 1
                else:
                    n_bad += 1
                    bad.append(f"  row {i + 1} number {j + 1}: printed {val}, recomputed {float(exp):.4f}")
        print(f"{label:16s} {'OK' if not bad else 'MISMATCH'}")
        for b in bad:
            print(b)
    print(f"\n{n_ok} numbers match, {n_bad} differ")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
