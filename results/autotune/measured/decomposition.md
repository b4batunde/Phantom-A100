# Coverage and conditional accuracy (Measured; decomposition defined in ADDENDUM-2)

Source: `scores.json` from NVIDIA A100-SXM4-80GB, BH=324, N=128; Nsight Compute: no_permission.

Coverage = share of claims where the predictor committed to a direction instead of a tie. Conditional accuracy = share of committed predictions in the hardware direction. Confidently wrong = committed predictions opposite to hardware. Tie resolution = predicted ties that hardware resolved into a real difference. Overall accuracy = the pre-registered metric. Unobservable hardware values are excluded. Counts in parentheses.

## Simulator (observed simulated direction)

#### By transformation family

| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |
|---|---|---|---|---|---|---|
| structural | 38 | 95% (36) | 89% (32/36) | 6% | 0% (0/2) | 89% |
| tuning | 64 | 31% (20) | 90% (18/20) | 10% | 50% (22/44) | 62% |

#### By counter

| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |
|---|---|---|---|---|---|---|
| Registers per thread | 51 | 57% (29) | 90% (26/29) | 3% | 0% (0/22) | 94% |
| Performance direction | 51 | 53% (27) | 89% (24/27) | 11% | 92% (22/24) | 51% |

#### By mapping equivalence

| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |
|---|---|---|---|---|---|---|
| same_definition | 51 | 57% (29) | 90% (26/29) | 3% | 0% (0/22) | 94% |
| performance | 51 | 53% (27) | 89% (24/27) | 11% | 92% (22/24) | 51% |

#### Family `structural` by counter

| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |
|---|---|---|---|---|---|---|
| Registers per thread | 19 | 89% (17) | 82% (14/17) | 6% | 0% (0/2) | 84% |
| Performance direction | 19 | 100% (19) | 95% (18/19) | 5% | n/a (0/0) | 95% |

#### Family `tuning` by counter

| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |
|---|---|---|---|---|---|---|
| Registers per thread | 32 | 38% (12) | 100% (12/12) | 0% | 0% (0/20) | 100% |
| Performance direction | 32 | 25% (8) | 75% (6/8) | 25% | 92% (22/24) | 25% |

## Declared signatures (rules)

#### By transformation family

| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |
|---|---|---|---|---|---|---|
| structural | 37 | 100% (37) | 89% (33/37) | 0% | n/a (0/0) | 89% |
| tuning | 58 | 100% (58) | 50% (29/58) | 43% | n/a (0/0) | 50% |
| rule_only | 8 | 100% (8) | 25% (2/8) | 38% | n/a (0/0) | 25% |

#### By counter

| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |
|---|---|---|---|---|---|---|
| Registers per thread | 30 | 100% (30) | 87% (26/30) | 0% | n/a (0/0) | 87% |
| Performance direction | 73 | 100% (73) | 52% (38/73) | 38% | n/a (0/0) | 52% |

#### By mapping equivalence

| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |
|---|---|---|---|---|---|---|
| same_definition | 30 | 100% (30) | 87% (26/30) | 0% | n/a (0/0) | 87% |
| performance | 73 | 100% (73) | 52% (38/73) | 38% | n/a (0/0) | 52% |

#### Family `structural` by counter

| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |
|---|---|---|---|---|---|---|
| Registers per thread | 18 | 100% (18) | 78% (14/18) | 0% | n/a (0/0) | 78% |
| Performance direction | 19 | 100% (19) | 100% (19/19) | 0% | n/a (0/0) | 100% |

#### Family `tuning` by counter

| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |
|---|---|---|---|---|---|---|
| Registers per thread | 12 | 100% (12) | 100% (12/12) | 0% | n/a (0/0) | 100% |
| Performance direction | 46 | 100% (46) | 37% (17/46) | 54% | n/a (0/0) | 37% |

#### Family `rule_only` by counter

| Group | Claims | Coverage | Conditional accuracy | Confidently wrong | Tie resolution | Overall accuracy |
|---|---|---|---|---|---|---|
| Performance direction | 8 | 100% (8) | 25% (2/8) | 38% | n/a (0/0) | 25% |

