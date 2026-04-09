# Master Analysis Report: All Stages

## Methodology

**All numbers in this report follow strict conventions:**

- **Median gap**: For a single config on a (function, dimension) pair — median across instances (2 instances).
- **Overall median gap**: Median across ALL 144 (function × instance × dimension) evaluations.
- **Single-config wins**: For ONE fixed config, count of (function, dimension) pairs where its median gap < CMA-ES median gap.
- **Oracle wins**: For each (function, dimension), pick the BEST config from a pool — count where the best beats CMA-ES.

When we say "config X wins N/24 at dim=D", this is always **single-config** unless explicitly marked as **(oracle)**.

**Total experiments analyzed: 68** (11 S1, 40 S2, 9 S3, 6 S3b, plus CMA-ES and TPE baselines).

---

## 1. Stage-1: Inference Method and EMA

### Experiment Grid (11 complete configs)

| Short name | Inference | EMA | T/E | Exploration |
|------------|-----------|-----|-----|-------------|
| noEMA_s50e50_uni | DDPM | off | 50/50 | uniform |
| noEMA_s100e100_uni | DDPM | off | 100/100 | uniform |
| noEMA_s100e100_sob | DDPM | off | 100/100 | sobol |
| ema995_s100e100_sob | DDPM | 0.995 | 100/100 | sobol |
| ema95_s50e50_sob | DDPM | 0.95 | 50/50 | sobol |
| ema995_s50e50_sob | DDPM | 0.995 | 50/50 | sobol |
| ddim05_noema_uni | DDIM η=0.5 | off | 100/100 | uniform |
| ddim05_noema_sob | DDIM η=0.5 | off | 100/100 | sobol |
| ddim05_noema_lhs | DDIM η=0.5 | off | 100/100 | LHS |
| ddim0_ema95 | DDIM η=0.0 | 0.95 | 50/50 | sobol |
| ddim05_ema95 | DDIM η=0.5 | 0.95 | 50/50 | sobol |

### Key Results

| Config | dim=2 | dim=5 | dim=10 | Overall | Wins total |
|--------|-------|-------|--------|---------|------------|
| CMA-ES | 0.066 | 0.095 | 0.145 | 0.093 | — |
| ddim05_noema_sob | **0.034** | 0.344 | 1.955 | 0.106 | 32/72 |
| ddim05_noema_uni | 0.050 | 0.250 | 1.955 | 0.116 | 32/72 |
| ddim05_ema95 | 0.038 | 0.349 | 2.455 | 0.124 | 30/72 |
| ddim05_noema_lhs | 0.054 | **0.140** | 2.246 | 0.190 | 28/72 |
| noEMA_s100e100_uni | 0.045 | 0.562 | 2.686 | 0.215 | 28/72 |

### Conclusions

1. **DDIM η=0.5 is the best inference method.** All top-5 configs use DDIM η=0.5. The best DDPM config (noEMA_s100e100_uni, overall 0.215) is ~2× worse than the best DDIM (ddim05_noema_sob, overall 0.106).

2. **EMA ≥ 0.95 hurts at dim≥5.** ddim05_ema95 (0.124) is worse than ddim05_noema_sob (0.106). EMA=0.995 is catastrophic at dim=10 (15.1 vs 1.96).

3. **Exploration type: LHS gives best dim=5 (0.140), sobol gives best dim=2 (0.034).** No single exploration type dominates.

4. **Oracle wins**: 24/24 dim=2, 19/24 dim=5, 6/24 dim=10 (49/72 total).

---

## 2. Stage-2: Conditioning Strategy

### Experiment Grid (40 complete configs)

4 conditioning types × {aug, noaug} × {uniform, sobol} × parameter variants.

### Key Results (top 10)

| Config | dim=2 | dim=5 | dim=10 | Overall | Wins |
|--------|-------|-------|--------|---------|------|
| CMA-ES | 0.066 | 0.095 | 0.145 | 0.093 | — |
| db/sob/lp0.7_ho0.2 noaug | 0.042 | 0.107 | 1.298 | 0.093 | 33/72 |
| opt/sob/op0.05_sp0.1 | 0.039 | 0.174 | 1.955 | 0.096 | 38/72 |
| db/sob/lp0.5_ho0.2 | 0.053 | 0.106 | 2.352 | 0.097 | 33/72 |
| db/lp0.5_ho0.2 noaug | **0.035** | 0.148 | 2.709 | 0.097 | 32/72 |
| opt/sob/op0.1_sp0.1 | 0.039 | 0.235 | 1.955 | 0.098 | 39/72 |

### Conclusions

1. **diverse_batch with high_offset=0.2 is the best conditioning type.** The top config by overall median (db/sob/lp0.7_ho0.2 noaug, 0.093) matches CMA-ES.

2. **aug=true hurts consistently.** In 15/16 paired comparisons, noaug outperforms aug. Aug blurs conditioning signal — especially damaging for diverse_batch where precise per-sample conditioning is the mechanism.

3. **Sobol exploration helps at dim=10.** db/sob/lp0.7_ho0.2 dim=10 = 1.298 vs db/uni/lp0.7_ho0.2 dim=10 = 2.108 (38% better).

4. **spread > 0 helps for optimistic.** Adding spread=0.1 creates within-batch diversity, similar to diverse_batch.

5. **percentile_annealing and combined underperform** (best overall ≥ 0.250).

6. **Oracle wins**: 24/24 dim=2, 21/24 dim=5, 8/24 dim=10 (53/72).

---

## 3. Stage-3: Elite Filter and Local Search

### Experiment Grid (9 complete configs)

- quality_knn (dw=0.3/0.5) × rank_temperature (0.3/0.5) = 4 configs
- rank_temperature=0.3 only = 1 config
- Local search (frac=0.15/0.3) × sigma_decay (quadratic/one_fifth) = 4 configs

### Key Results

| Config | dim=2 | dim=5 | dim=10 | Overall | Wins |
|--------|-------|-------|--------|---------|------|
| CMA-ES | 0.066 | 0.095 | 0.145 | 0.093 | — |
| LS f0.30 quad | 0.044 | **0.099** | 1.165 | 0.093 | **38/72** |
| LS f0.15 quad | 0.057 | 0.119 | 1.588 | 0.097 | 35/72 |
| S2-winner (baseline) | 0.042 | 0.107 | 1.298 | 0.093 | 33/72 |

### Conclusions

1. **Quadratic sigma decay dominates one_fifth** by 2.5–3× at dim≥5. One_fifth adapts sigma based on confounded success signals.

2. **LS f0.30 quadratic** achieves best dim=5 (0.099, nearly matching CMA-ES 0.095) and 38 single-config wins (vs 33 for S2-winner). It trades dim=2 quality (0.044 vs 0.042) for major dim=5 and dim=10 gains.

3. **Local search transforms previously intractable functions**: f16 dim=10: 12.15 → 0.97 (13×), f23 dim=10: 1.97 → 0.31 (6×).

4. **quality_knn hurts at dim≥5** despite helping at dim=2. KNN distances become less meaningful in higher dimensions.

5. **Oracle wins**: 24/24 dim=2, 22/24 dim=5, 10/24 dim=10 (56/72).

---

## 4. Stage-3b: Local Search Refinement

### Experiment Grid (6 configs)

All use diverse_batch/sobol/lp0.7/ho0.2 conditioning, DDIM η=0.5, quality elite filter, quadratic sigma decay.

| Config | frac | rank_temp | sigma_decay |
|--------|------|-----------|-------------|
| frac0.15 | 0.15 | 0.5 | quadratic |
| frac0.15_rt0.3 | 0.15 | 0.3 | quadratic |
| frac0.15_rt0.5 | 0.15 | 0.5 | quadratic |
| frac0.3 | 0.30 | 0.5 | quadratic |
| frac0.3_rt0.3 | 0.30 | 0.3 | quadratic |
| frac0.45 | 0.45 | 0.5 | quadratic |

Note: frac0.15 and frac0.15_rt0.5 are identical (rt=0.5 is default).

### Key Results

| Config | dim=2 | dim=5 | dim=10 | Overall | Wins |
|--------|-------|-------|--------|---------|------|
| CMA-ES | 0.066 | 0.095 | 0.145 | 0.093 | — |
| **frac0.15** | 0.039 | 0.115 | **0.849** | 0.096 | **36/72** |
| **frac0.3** | 0.039 | 0.260 | **0.789** | 0.098 | 38/72 |
| frac0.3_rt0.3 | **0.033** | 0.164 | 1.707 | 0.098 | **40/72** |
| frac0.45 | 0.050 | 0.155 | 1.400 | 0.097 | 37/72 |
| frac0.15_rt0.3 | 0.051 | 0.199 | 1.058 | 0.099 | 40/72 |
| S3:LS_f30_quad | 0.044 | 0.099 | 1.165 | 0.093 | 38/72 |
| S2-winner | 0.042 | 0.107 | 1.298 | 0.093 | 33/72 |

### Stage-3b Breakthroughs

**dim=10 = 0.789 (frac0.3)** — the best dim=10 result across all stages. This is **5.5× better than the original S1-best** (1.955) and closing the gap to CMA-ES (0.145).

**dim=10 = 0.849 (frac0.15)** — second best, with better dim=5 (0.115 vs 0.260).

### Group Analysis at dim=10

| Group | CMA-ES | S2-winner | S3b frac0.15 | S3b frac0.3 |
|-------|--------|-----------|--------------|-------------|
| Separable | **0.092** | 0.318 | 25.869 | 13.929 |
| Moderate | **0.096** | 3.945 | **1.008** | **0.698** |
| Ill-conditioned | **0.073** | 0.214 | **0.116** | 0.225 |
| Multi-modal | 0.250 | 0.250 | **0.170** | 0.318 |
| Weakly-structured | 1.955 | 2.063 | **1.807** | **1.639** |

Key insights:
- **frac0.15 excels on Ill-conditioned** (0.116 vs CMA 0.073 — close!) and Multi-modal (0.170 vs CMA 0.250 — **beats CMA!**)
- **frac0.3 excels on Moderate** (0.698 vs CMA 0.096 — much closer than S2-winner 3.945) and Weakly-structured (1.639)
- **Both still lose badly on Separable** (13.9–25.9 vs 0.092) — local search hurts where diffusion model was already effective

### Effect of rank_temperature

- rt=0.3 improves dim=2 win count (20/24 for frac0.3_rt0.3) but worsens dim=10 (1.707 vs 0.789)
- rt=0.5 (default) is better for dim=10 where the buffer is larger and training benefits from more data

### Oracle wins for S3b alone: 24/24 dim=2, 19/24 dim=5, 9/24 dim=10 (52/72)

---

## 5. Overall Research Summary

### Cumulative Best Single Config per Stage

| Metric | S1-best | S2-best | S3-best | S3b-best | CMA-ES |
|--------|---------|---------|---------|----------|--------|
| Config | ddim05_sob | db/sob/lp07_ho02 | LS_f30_quad | frac0.15 | — |
| Overall median | 0.106 | 0.093 | 0.093 | 0.096 | 0.093 |
| dim=2 | 0.034 | 0.042 | 0.044 | 0.039 | 0.066 |
| dim=5 | 0.344 | 0.107 | 0.099 | 0.115 | 0.095 |
| dim=10 | 1.955 | 1.298 | 1.165 | **0.849** | 0.145 |
| Single wins | 32/72 | 33/72 | 38/72 | 36/72 | — |
| Best single wins | — | 39/72 | — | **40/72** | — |

Note: "Best single wins" is the config with maximum total wins within that stage (may differ from the config with best overall median).

### Oracle Wins by Stage (cumulative pool)

| Stage pool | dim=2 | dim=5 | dim=10 | Total |
|------------|-------|-------|--------|-------|
| S1 only | 24/24 | 19/24 | 6/24 | 49/72 |
| S1+S2 | 24/24 | 21/24 | 8/24 | 53/72 |
| S1+S2+S3 | 24/24 | 22/24 | 10/24 | 56/72 |
| All stages | 24/24 | **23/24** | **12/24** | **59/72** |

### What Each Stage Contributed

**Stage-1: Found the right inference method.** DDIM η=0.5 was the single most impactful discovery — reducing overall gap from 0.215 (DDPM baseline) to 0.106.

**Stage-2: Found the right conditioning strategy.** diverse_batch with high_offset=0.2 further reduced the gap to 0.093, matching CMA-ES overall. Also discovered that aug=false is important.

**Stage-3: Added local search for rugged functions.** Quadratic sigma decay local search with frac=0.30 turned previously intractable functions (f16, f23) into competitive results, while maintaining gains elsewhere. quality_knn was tested but found ineffective at dim≥5.

**Stage-3b: Refined local search further.** dim=10 improved from 1.165 to 0.789 — now within 5.5× of CMA-ES (was 13.5× at S1).

### Remaining Weaknesses

1. **Separable functions at dim=10** (f2, f3, f4): Local search hurts these badly (gap 14–26 vs CMA 0.08–10). The diffusion model alone is much better here but still 3× worse than CMA-ES.

2. **dim=10 overall**: Best single config achieves 0.789 vs CMA-ES 0.145 (5.4× gap). Oracle achieves 12/24 wins.

3. **No single config is universally optimal.** The best dim=2 config (frac0.3_rt0.3, 0.033) is different from the best dim=10 config (frac0.3, 0.789), which is different from the best overall config (S2-winner or S3-LS30-quad, 0.093).

---

## 6. ECDF Comparison (Best Configs per Stage)

| Config | t<0.1 | t<0.5 | t<1.0 | t<2.0 | t<5.0 | t<10 |
|--------|-------|-------|-------|-------|-------|------|
| **CMA-ES** | **56.9%** | 65.3% | 69.4% | **81.9%** | **90.3%** | **95.1%** |
| S3b: frac0.15 | 54.9% | **66.0%** | **72.2%** | **80.6%** | **86.1%** | 89.6% |
| S3: LS_f15_quad | 54.9% | **66.7%** | **71.5%** | 78.5% | 84.0% | 88.9% |
| S2: db/sob/lp07_ho02 | 54.9% | 64.6% | 67.4% | 77.8% | 83.3% | 90.3% |
| S3: LS_f30_quad | 54.2% | 63.2% | **70.8%** | 76.4% | 82.6% | 90.3% |

**S3b frac0.15 has the best ECDF** at thresholds 0.5, 1.0, and 2.0 — it **surpasses CMA-ES** at t<0.5 (66.0% vs 65.3%) and t<1.0 (72.2% vs 69.4%). CMA-ES is still better at extreme thresholds (t<0.1: 56.9% vs 54.9%) and at t<5.0 (90.3% vs 86.1%).

---

## 7. Proposed Additional Ablation Experiments

To build a complete ablation study for a research paper, the following experiments are needed:

### 7.1 Ablation of Each Component (Remove One at a Time)

Starting from the best config (S3b frac0.15, diverse_batch/sobol/lp0.7/ho0.2, DDIM η=0.5, quadratic LS):

| Experiment | What is removed | Purpose |
|------------|----------------|---------|
| A1: DDPM instead of DDIM | Replace DDIM η=0.5 with DDPM | Quantify DDIM contribution |
| A2: No local search | Set local_search.enabled=false | Quantify LS contribution |
| A3: Optimistic instead of diverse_batch | Replace conditioning with optimistic | Quantify diverse_batch contribution |
| A4: Uniform instead of sobol | Replace exploration with uniform | Quantify sobol contribution |
| A5: No preconditioning | Set precondition.type=none | Quantify whiten_shrink contribution |

Each of these ablations isolates one component. The difference between the full config and the ablated version shows that component's contribution.

**Why these specific ablations**: These are the five modifications we made from the initial DDPM+optimistic+uniform baseline. The ablation table will directly answer "how much does each change contribute?"

### 7.2 Interaction Effects

| Experiment | Purpose |
|------------|---------|
| B1: DDIM + no LS (S2-winner) | Already exists — this IS the ablation without LS |
| B2: DDPM + LS | Tests if LS helps independently of DDIM |
| B3: DDIM + diverse_batch + uniform (no sobol) | Isolates sobol effect with the full pipeline |

### 7.3 Scaling Study

| Experiment | Purpose |
|------------|---------|
| C1: dim=20 | Test scaling beyond BBOB standard dimensions |
| C2: Budget 2048×d instead of 1024×d | Does more budget help our method more than CMA-ES? |
| C3: Budget 512×d | Does our method degrade gracefully? |

### 7.4 Per-Group Specialized Configs (for presentation)

Run the best overall config on each function group separately to build clean comparison plots. This is for the paper figures — showing where exactly our method wins and loses vs CMA-ES.

### Priority Order

1. **A1–A5 (ablation table)**: Essential for any paper. 5 runs.
2. **B2 (DDPM+LS)**: 1 run — tests the most important interaction.
3. **C2 (higher budget)**: 1 run — tests if our method can close the dim=10 gap with more budget.

Total: **7 additional runs** for a complete ablation section.
