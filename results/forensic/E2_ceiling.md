# E2 — MIT-BIH fusion-label ceiling

Generated: 2026-09-05  
Review type: scoped literature search + transparent rule-vs-label proxy  
Compute: CPU only; no trained model loaded

## Research question

Is there direct evidence for inter-annotator agreement on MIT-BIH ventricular–normal
fusion beats (`F`), and can the available data support a numerical ceiling for
single-lead beat-level F classification?

## Literature result

The MIT-BIH records were independently annotated by at least two cardiologists;
beat-level discrepancies were highlighted and resolved by consensus before the
released reference annotation was produced. The public database therefore contains
the consensus layer, not the two original independent label layers
([PhysioNet database description](https://physionet.org/content/mitdb/1.0.0/mitdbdir/),
[annotation procedure](https://physionet.org/physiobank/database/html/mitdbdir/intro.htm)).
The canonical database paper is Moody and Mark (2001), DOI
[10.1109/51.932724](https://doi.org/10.1109/51.932724).

The search found **no published F-specific disagreement rate, Cohen's kappa, or
pairwise F1 between the original MIT-BIH annotators**. Consensus quality control is
evidence that the released labels were curated; it is not evidence that annotators
agreed easily, and it prevents retrospective computation of raw agreement.

There is direct qualitative evidence that F is a continuum rather than a compact
morphological category. The official note for record 213 says its fusion-PVC
morphology ranges from almost normal to almost identical to its PVCs
([PhysioNet record notes](https://www.physionet.org/physiobank/database/html/mitdbdir/records.htm)).
Record 213 contains 362 F beats and dominates the conventional DS2 F evaluation.

### Search log

| Source | Query/scope | Included evidence |
|---|---|---|
| PubMed | MIT-BIH annotation agreement; fusion-beat interobserver agreement | Moody & Mark database paper; no F-specific statistic |
| PhysioNet | MIT-BIH annotation protocol, symbols, record notes | independent annotation + consensus procedure; F definition; record-213 continuum note |
| Broad scholarly web search | `MIT-BIH fusion beat inter-annotator agreement`, variants | no recoverable independent F label layer or F-specific agreement number |

Excluded results discussed unrelated ST/T annotation, rhythm-level agreement,
algorithm-vs-consensus classification, paced-normal fusion (`f`, a different label),
or non-MIT-BIH datasets.

## Rule-vs-label proxy

Because literature did not yield a direct number, the pre-registered proxy was run
on all 17 F-bearing MIT-BIH records in the LOPO pool. The binary comparison used
41,996 N/V/F beats and 802 consensus-F beats. The transparent rule required:

1. derivative-based QRS width between the record-specific N and V medians;
2. a pre-QRS component correlated with the record's median-N template;
3. an early-upstroke score displaced from N toward the record-specific V median.

Main-rule result:

| Scope | F1 | Precision | Recall | Cohen κ | Rule-F | True F |
|---|---:|---:|---:|---:|---:|---:|
| Pooled 17 records | 0.051 | 0.033 | 0.113 | 0.022 | 2,786 | 802 |
| Record 208 | 0.000 | 0.000 | 0.000 | −0.088 | 175 | 372 |
| Record 213 | 0.233 | 0.340 | 0.177 | 0.169 | 188 | 362 |

Across nine threshold combinations, pooled F1 remained 0.042–0.070 and κ
0.014–0.039. Full definitions, confusion matrices, per-record feature medians and
sensitivity results are in `E2_rule_proxy.json`.

## Interpretation

**This proxy does not estimate a human or Bayes ceiling.** Its low agreement is
compatible with at least three explanations:

1. consensus-F is not identifiable by the three proposed single-beat rules;
2. the operational P-wave/QRS/upstroke measurements are too crude;
3. annotators used two leads, longer rhythm context, patient-specific morphology,
   or expert gestalt unavailable to the rule.

The 208-versus-213 contradiction is particularly informative: the same rule has
zero true positives and negative κ on 208, but detects a subset of 213 F beats.
That supports strong patient-specific variation and refutes a universal fixed-rule
definition. It does not distinguish label ambiguity from missing context.

## Current ceiling verdict

**No defensible numerical inter-annotator ceiling can be derived from available
MIT-BIH data or the located literature.** In particular, the hypothesized ceiling
of 0.4–0.5 is not confirmed. Claiming that number would be an overreach.

What is supported:

- F morphology can span the N↔V continuum within a single major record.
- A fixed transparent single-lead rule is grossly insufficient and non-transferable
  across the two dominant F patients.
- Together with E1's low-rank F errors and LOPO collapse, H3 (task/label
  non-identifiability) becomes more plausible, but remains entangled with H1
  (insufficient patients) and missing-context/representation explanations.

What would resolve the ceiling question:

- access to the original independent MIT-BIH annotations, if preserved; or
- a blinded independent relabeling study of a stratified F/N/V sample by at least
  two electrophysiologists, retaining raw labels before adjudication.

## Classification of E2

Literature step: **inconclusive for a numeric ceiling**.  
Rule implementation: **completed and falsified as a useful universal detector**.  
H3: **supported qualitatively, not quantitatively confirmed**.
