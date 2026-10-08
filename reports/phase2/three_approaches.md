## Three approaches, chronological holdout

Eligible orders: 91,171. Train: 58,349, validation: 14,587, test: 18,235. Test negative-review rate: 8.46%.

| approach | policy | threshold | train_rows | test_rows | test_negative_rate | precision | recall | f1 | pr_auc | roc_auc | flagged | false_alarms | missed_negatives | detected_negatives |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | default_0.50 | 0.50 | 72936 | 18235 | 0.0846 | 0.3853 | 0.0272 | 0.0508 | 0.1812 | 0.6440 | 109 | 67 | 1501 | 42 |
| natural | validation_f1 | 0.13 | 72936 | 18235 | 0.0846 | 0.2317 | 0.2275 | 0.2296 | 0.1812 | 0.6440 | 1515 | 1164 | 1192 | 351 |
| class_weight_balanced | default_0.50 | 0.50 | 72936 | 18235 | 0.0846 | 0.1774 | 0.3487 | 0.2351 | 0.1793 | 0.6460 | 3033 | 2495 | 1005 | 538 |
| class_weight_balanced | validation_f1 | 0.57 | 72936 | 18235 | 0.0846 | 0.2229 | 0.2463 | 0.2340 | 0.1793 | 0.6460 | 1705 | 1325 | 1163 | 380 |
| undersample_50_50 | default_0.50 | 0.50 | 14172 | 18235 | 0.0846 | 0.1728 | 0.3474 | 0.2308 | 0.1778 | 0.6413 | 3101 | 2565 | 1007 | 536 |
| undersample_50_50 | validation_f1 | 0.58 | 14172 | 18235 | 0.0846 | 0.2329 | 0.2249 | 0.2288 | 0.1778 | 0.6413 | 1490 | 1143 | 1196 | 347 |
