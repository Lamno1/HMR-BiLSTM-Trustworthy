import json
import csv
from pathlib import Path

# Load clean metrics
with open('results/ablation/ablation_results.json', 'r', encoding='utf-8') as f:
    clean_data = json.load(f)

# Create a mapping from variant to clean metrics
clean_metrics = {}
for item in clean_data:
    variant = item['variant']
    clean_metrics[variant] = {
        'label': item['label'],
        'n_params': item['n_params'],
        'f1_macro': item['test_metrics']['f1_macro'],
        'f1_s': item['test_metrics'].get('f1_S', 0.0),
        'f1_v': item['test_metrics'].get('f1_V', 0.0),
        'f1_f': item['test_metrics'].get('f1_F', 0.0),
        'auc': item['test_metrics']['auc_ovr']
    }

# Load robustness metrics
rob_csv = Path('results/tables/ablation_robustness.csv')
rob_metrics = {}
if rob_csv.exists():
    with open(rob_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rob_metrics[row['variant']] = row

# Combine
rows = []
for variant, d in clean_metrics.items():
    rob = rob_metrics.get(variant, {})
    row = {
        'Variant': d['label'],
        'Params': f"{d['n_params']:,}",
        'F1-Clean': f"{d['f1_macro']:.4f}",
        'F1-Adv (0.02)': rob.get('fgsm_f1_002', '-'),
        'F1-Adv (0.05)': rob.get('fgsm_f1_005', '-'),
        # F1-S/F1-V = per-class F1 on clean test set (Supraventricular / Ventricular)
        # This matches Table 5 in the paper, NOT robustness metrics.
        'F1-S': f"{d['f1_s']:.4f}",
        'F1-V': f"{d['f1_v']:.4f}",
        'F1-F': f"{d['f1_f']:.4f}",
        'AUC': f"{d['auc']:.4f}",
    }
    rows.append(row)

# Save to CSV
cols = list(rows[0].keys())
out_csv = Path('results/tables/ablation_table_final.csv')
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

# Save to TeX
out_tex = Path('results/tables/ablation_table_final.tex')
with open(out_tex, 'w', encoding='utf-8') as f:
    col_fmt = "l" + "r" * (len(cols) - 1)
    f.write("% Requires \\usepackage{booktabs}\n")
    f.write("\\begin{table}[h]\n")
    f.write("\\centering\n")
    f.write("\\caption{Ablation Study --- F1 Clean vs F1 Adv (FGSM)}\n")
    f.write("\\label{tab:ablation_final}\n")
    f.write(f"\\begin{{tabular}}{{{col_fmt}}}\n")
    f.write("\\toprule\n")
    f.write(" & ".join(cols) + " \\\\\n")
    f.write("\\midrule\n")
    for r in rows:
        line = " & ".join(r[c] for c in cols)
        if r['Variant'].startswith("HMR-BiLSTM (full)"):
            # bold the whole line
            line = " & ".join(f"\\textbf{{{r[c]}}}" for c in cols)
        f.write(line + " \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\end{table}\n")

print(f"[OK] Saved {out_csv} and {out_tex}")
