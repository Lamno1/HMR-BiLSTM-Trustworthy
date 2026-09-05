# P2 Final Report

Locked verdict: **NO-GO**

Qualifying ranking-reliable MIT-BIH folds: **0/3**

## Locked GO criterion

| model | fold | alpha | risk | S coverage | S precision | referral | qualifies |
|---|---:|---:|---:|---:|---:|---:|---|
| r2 | 0 | 0.0900 | 0.0143 | 0.0000 | 0.0000 | 0.4289 | False |
| r2 | 1 | 0.1550 | 0.0177 | 0.0000 | NA | 0.8525 | False |
| r2 | 2 | 0.5000 | 0.0072 | 0.0263 | 0.9643 | 0.2876 | False |
| r2 | 3 | 0.4650 | 0.0198 | 0.3788 | 0.2990 | 0.1168 | False |
| r2 | 4 | NA | NA | NA | NA | NA | False |
| flat | 0 | 0.0800 | 0.0194 | 0.0172 | 0.3889 | 0.5313 | False |
| flat | 1 | 0.0150 | 0.0147 | 0.0000 | NA | 0.9968 | False |
| flat | 2 | 0.5000 | 0.0051 | 0.0000 | NA | 0.7727 | False |
| flat | 3 | 0.5000 | 0.0089 | 0.0152 | 0.1013 | 0.5336 | False |
| flat | 4 | NA | NA | NA | NA | NA | False |

## Structural-failure check

S-ECE is binary classwise ECE on the outer fold; it is diagnostic, not fitted there.

- r2 fold 1: any S singleton=True; max S coverage=0.6861; S-ECE 0.0225->0.0142.
- r2 fold 2: any S singleton=True; max S coverage=0.0468; S-ECE 0.0120->0.0130.
- r2 fold 3: any S singleton=True; max S coverage=0.5152; S-ECE 0.0408->0.0382.
- flat fold 1: any S singleton=True; max S coverage=0.2080; S-ECE 0.0705->0.0300.
- flat fold 2: any S singleton=False; max S coverage=0.0000; S-ECE 0.0921->0.0356.
- flat fold 3: any S singleton=True; max S coverage=0.0227; S-ECE 0.0681->0.0373.
