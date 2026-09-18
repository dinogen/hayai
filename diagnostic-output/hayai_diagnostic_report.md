# HAYAI v2 Diagnostic Report

## 1. Experiment

- **experiment_date**: 2026-09-09T18:57:19.003250+00:00
- **git_commit**: c6e341589a322e9f76bfeaa117be2c3118c0c4c4
- **model**: stock_model v2
- **artifact**: /home/dinogen/Progetti/hayai/model/stock_model/v2
- **feature_count**: 24
- **rows**: 133310
- **assets**: 113
- **period**: 2021-12-01 -> 2026-08-28
- **target_definition**: clip(log(close(t+5) / close(t)) / vol_20(t), -3, 3)
- **dataset_integrity**: {'rows': 133310, 'assets': 113, 'first_date': '2021-12-01', 'last_date': '2026-08-28', 'duplicate_symbol_dates': 0, 'invalid_prices': 0, 'nan_features_or_target': 0}

## 2. Dataset integrity

- **rows**: 133310
- **assets**: 113
- **first_date**: 2021-12-01
- **last_date**: 2026-08-28
- **duplicate_symbol_dates**: 0
- **invalid_prices**: 0
- **nan_features_or_target**: 0

## 3. Target verification

Target: `clip(log(close(t+5) / close(t)) / vol_20(t), -3, 3)`

## 4. Feature timing and leakage

- Rolling and cross-sectional features are computed at date `t` by the existing builder.
- **Potential contamination**: v2 winsorization/scaler statistics are computed globally before the random split.
- Status: `UNCERTAIN` until a train-only feature pipeline is compared.

## 5. Train/validation/test split

- Random split: **YES** (`test_size=0.2`, `random_state=42`)
- Chronological split: **YES** (diagnostic 70/15/15)
- v2 random test is also used as early-stopping validation: **methodological contamination**.

## 6. Predictive power

### random_test
#### target_metrics
- n: 26662.0000
- pearson: 0.3094
- spearman: 0.2746
- mae: 0.9269
- rmse: 1.2011
- r2: 0.0919
- hit_rate: 59.14%
#### realized_return_metrics
- n: 26662.0000
- pearson: 0.2608
- spearman: 0.2596
- mae: 0.0321
- rmse: 0.0489
- r2: 0.0446
- hit_rate: 59.14%
### chronological_test
#### target_metrics
- n: 20105.0000
- pearson: 0.3556
- spearman: 0.3208
- mae: 0.9160
- rmse: 1.1878
- r2: 0.1240
- hit_rate: 60.86%
#### realized_return_metrics
- n: 20105.0000
- pearson: 0.2963
- spearman: 0.2736
- mae: 0.0343
- rmse: 0.0520
- r2: 0.0757
- hit_rate: 60.86%

## 7. Quintile analysis

- Status: `OK`
- Q1: -0.0114
- Q5: 0.0143
- Q5-Q1: 0.0257

## 8. Signal ranges

| signal_range | n | signal_mean | future_5d_mean | win_rate |
| --- | --- | --- | --- | --- |
| -3/-2 | 26 | -2.347015619277954 | -0.16224334553101344 | 0.038461538461538464 |
| -2/-1 | 316 | -1.2925188541412354 | -0.03114168246449183 | 0.17405063291139242 |
| -1/0 | 8010 | -0.2864843010902405 | -0.007405827431081471 | 0.41972534332084893 |
| 0/1 | 11174 | 0.26617398858070374 | 0.009892391618165465 | 0.6078396277071774 |
| 1/2 | 545 | 1.3434809446334839 | 0.032962180732444515 | 0.8954128440366973 |
| 2/3 | 34 | 2.182575225830078 | 0.05319608394206725 | 1.0 |

## 9. Long vs short and inverted signal

- Long observations: 11753
- Long mean return: 0.0111
- Long median return: 0.0088
- Long win rate: 62.23%
- Short observations: 8352
- Short mean position return: 0.0088
- Short median position return: 0.0065
- Short win rate: 58.93%

- Inverted signal return: -82.83%
- Inverted signal volatility: 27.45%
- Inverted signal max drawdown: -81.58%

## 10. Baselines

| strategy | n | return | volatility | max_drawdown |
| --- | --- | --- | --- | --- |
| Model | 36 | 4.824774940918509 | 0.27452288139332237 | -0.014109124996906086 |
| Momentum 20d | 36 | 1.2021543664960315 | 0.4373115019038709 | -0.1765183134527537 |
| Equal Weight | 36 | 0.0852164134797413 | 0.10058015928445214 | -0.05628382300435175 |
| SPY | 36 | 0.09892091979100792 | 0.11241332161662823 | -0.05529988106895467 |
| Zero model | 36 | 0.0 | 0.0 | 0.0 |
| Random equal-weight | 36 | -0.03814775517776592 | 0.16254658156516544 | -0.10598976681466343 |

## 11. Quant vs Quant + News

- Status: `BLOCKED`
- Observed rows: 0

## 12. Diagnostic section

Dataset/target audit: `OK`.

## 13. Diagnostic section

Feature timing/leakage: `PARTIAL`; train-only preprocessing differences are measured below.

## 14. Diagnostic section

Split audit: `OK`, with random-test reuse for early stopping documented above.

## 15. Diagnostic section

Signal-to-weight and concentration: `PARTIAL`; persisted weights are included where available.

## 16. Diagnostic section

Turnover and transaction costs: `BLOCKED` unless historical trades and cost configuration are available.

## 17. Diagnostic section

Execution timing: `PARTIAL`; daily trade dates and execution prices are available, intraday timestamps are not.

## 20. Diagnostic section

Asset-level analysis: `PARTIAL`; prediction rows are available, executed trade grouping is available.

## 24. Diagnostic section

Final metrics: `PARTIAL`; model, baseline and NAV reconciliation are included, fees/slippage are unavailable.

## 25. Diagnostic section

Automatic conclusion: `OK` for available evidence.

## 26. Diagnostic section

Prediction CSV: `OK`.

## 21. Temporal stability

| period | start | end | n | return | volatility | max_drawdown | win_rate | q5_q1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Period 1 | 2025-12-11 | 2026-02-17 | 9 | 0.44484381566703335 | 0.24134935765594417 | -0.008313411696877382 | 0.7777777777777778 | 0.04088902551990029 |
| Period 2 | 2026-02-18 | 2026-04-22 | 9 | 0.6422845086249147 | 0.27643562035923985 | 0.0 | 1.0 | 0.05512091839836696 |
| Period 3 | 2026-04-23 | 2026-06-26 | 9 | 0.8114969402391421 | 0.3312473069746625 | -0.014109124996906197 | 0.8888888888888888 | 0.06601706025286032 |
| Period 4 | 2026-06-29 | 2026-08-28 | 9 | 0.35510280786019477 | 0.22335276659356196 | 0.0 | 1.0 | 0.03376414715700113 |

## 22. Market regimes

| regime | n | return | volatility | win_rate |
| --- | --- | --- | --- | --- |
| Bear | 1 | 0.016504213365592824 |  | 1.0 |
| Bull | 6 | 0.6523461582639873 | 0.2609272456133659 | 1.0 |
| High volatility | 1 | 0.0015739942799726236 |  | 1.0 |
| Low volatility | 5 | 0.1453949489862858 | 0.25442363742520385 | 0.8 |
| Sideways | 26 | 2.2257559541718748 | 0.48733545328026434 | 0.8461538461538461 |

## 23. Robustness sensitivity

| test | holding_days | positions_per_side | threshold | return | volatility | max_drawdown | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| holding_positions | 1 | 5 | 0.0 | 0.4301746728729605 | 0.1860035343097905 | -0.10446061150534569 | 0.6388888888888888 |
| holding_positions | 1 | 10 | 0.0 | 0.18063159900247938 | 0.12994091492408477 | -0.0830142266978332 | 0.5833333333333334 |
| holding_positions | 1 | 20 | 0.0 | 0.028816060961502687 | 0.11433419828860461 | -0.05956432468504336 | 0.5833333333333334 |
| holding_positions | 1 | 50 | 0.0 | -0.01994516839478855 | 0.06704526197052203 | -0.054718979827818814 | 0.5277777777777778 |
| holding_positions | 5 | 5 | 0.0 | 3.0714996324839996 | 0.630142207769658 | -0.34081777483357 | 0.8857142857142857 |
| holding_positions | 5 | 10 | 0.0 | 2.060814256511014 | 0.3409480765772211 | -0.18073572070476807 | 0.9142857142857143 |
| holding_positions | 5 | 20 | 0.0 | 0.9903938966511816 | 0.2085621559787614 | -0.10762771177912722 | 0.8285714285714286 |
| holding_positions | 5 | 50 | 0.0 | 0.5285727278352774 | 0.13511421764396256 | -0.035547259505843476 | 0.7142857142857143 |
| holding_positions | 10 | 5 | 0.0 | 1.9598929107923393 | 1.6075529949209215 | -0.7628280972255412 | 0.8235294117647058 |
| holding_positions | 10 | 10 | 0.0 | 1.6711053667267657 | 0.8418024299348865 | -0.5059377435485491 | 0.8235294117647058 |
| holding_positions | 10 | 20 | 0.0 | 1.2695840942585481 | 0.4294650036395217 | -0.2674301657756025 | 0.7647058823529411 |
| holding_positions | 10 | 50 | 0.0 | 0.6252672914991124 | 0.21964033082443346 | -0.13012945693711941 | 0.7647058823529411 |
| holding_positions | 20 | 5 | 0.0 | 1.8001205686845183 | 1.3368855068560888 | -0.7634536876295678 | 0.78125 |
| holding_positions | 20 | 10 | 0.0 | 1.6274250778691397 | 0.7447866567077047 | -0.5153909078468069 | 0.71875 |
| holding_positions | 20 | 20 | 0.0 | 1.8814963357163803 | 0.38050753640665536 | -0.16552753188799207 | 0.8125 |
| holding_positions | 20 | 50 | 0.0 | 0.778401066125392 | 0.24890133818417054 | -0.0859759563387622 | 0.625 |
| signal_threshold | 5 | 5 | 0.0 | 4.824774940918509 | 0.27452288139332237 | -0.014109124996906086 | 0.9166666666666666 |
| signal_threshold | 5 | 5 | 0.25 | 4.723016363160667 | 0.2835335386806572 | -0.014273488688929636 | 0.8888888888888888 |
| signal_threshold | 5 | 5 | 0.5 | 1.2530146533540591 | 0.41087181317045157 | -0.0826519956491012 | 0.6538461538461539 |
| signal_threshold | 5 | 5 | 1.0 | 0.34212413126907615 | 0.4673046657770262 | -0.016551140120444452 | 0.6666666666666666 |

## 13. Preprocessing audit

- Status: `OK`
- Features affected: 20
- Maximum global/train difference: 0.4692
- Mean global/train difference: 0.0327

## Model comparison: v2 vs chronological artifact

### v2
- Split: random
- Train period: 2021-12-01 -> 2026-08-28
- Test period: 2021-12-01 -> 2026-08-28
- Test rows: 26662
- Spearman target: 0.2746
- R2 target: 0.0919
- Hit rate: 59.14%
- Strategy return: 1018.13%

### v4
- Split: time
- Train period: 2021-12-01 -> 2025-02-28
- Test period: 2025-11-13 -> 2026-08-28
- Test rows: 22233
- Spearman target: -0.0108
- R2 target: -0.0072
- Hit rate: 51.98%
- Strategy return: 50.40%

## 11. News retention coverage

- Status: `BLOCKED`
- News rows in model period: 0
- Date coverage: 0.00%
- First news: n/a
- Last news: n/a

## 18. Paper portfolio reconstruction

- Status: `OK`
- Trade rows: 7
- Reconstructed final NAV: 5000.01
- Saved final NAV: 5000.00
- Final difference: 0.01

## 19. Trade-by-trade analysis

- Status: `OK`
- Trades: 7
- Winning cash-flow rows: 2
- Losing cash-flow rows: 5
- Win rate: 28.57%
- Average winner: 615.68
- Average loser: -620.98
- Largest winner: 935.35
- Largest loser: -847.52
- Profit factor: 0.397

## Realized P&L by asset

- Status: `OK`
- Realized P&L: 0.00
## Diagnostic conclusion

### Model predictive power
WEAK / NONE on the chronological holdout

### Data leakage
UNCERTAIN / METHODOLOGICAL CONTAMINATION

### Portfolio construction
PARTIAL

### Execution
OBSERVED

### News correction
PARTIAL

### Main suspected problem
The apparent v2 edge is not reproduced by the chronological v4 holdout. The random split, early-stopping reuse and global preprocessing make the v2 result unreliable as an out-of-sample estimate.

### Evidence
v2 random test: Spearman 0.2746, R2 0.0919, hit rate 59.14%.
v4 time test: Spearman -0.0108, R2 -0.0072, hit rate 51.98%.
Global/train preprocessing maximum difference: 0.4692.
Historical news sentiment coverage: 0.00%.

### Recommended next investigation
Do not tune v2 from the random metrics. Rebuild the chronological training pipeline with train-only winsorization/scaling, then validate on a newly frozen future holdout before using news or portfolio optimization to explain performance.
