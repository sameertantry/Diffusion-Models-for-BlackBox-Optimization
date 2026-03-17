# Design Benchmarks

These benchmarks expose real-world black-box optimisation tasks through the
unified `BaseBenchmark` interface.  Each benchmark is a *minimisation* problem,
implemented as the negative of the underlying property (so maximising the
property corresponds to minimising the returned value).

All six benchmarks follow the experimental setup from the **Diffusion-BBO**
paper (Wu et al., arXiv:2407.00610, 2024), which evaluates online BBO
algorithms on Design-Bench tasks (Trabucco et al., ICML 2022).

## Shared setup

1. Install `design-bench`:
   ```bash
   pip install design-bench
   ```
2. Download datasets:
   ```bash
   python scripts/download_data.py --benchmarks all
   ```
3. Datasets are stored under `data/<benchmark_name>/` in this repo.

Two evaluation modes are supported:

| Mode    | When                                 | Oracle source                   |
|---------|--------------------------------------|---------------------------------|
| oracle  | `design-bench` installed             | Exact / trained model from DB   |
| lookup  | Only local data in `data/<name>/`    | Nearest-neighbour on dataset    |

---

## TFBind8

- **Config name**: `tfbind8` (or `tfbind`)
- **Description**: DNA-binding protein sequence design (8-mer), optimising
  binding affinity.
- **Input dim**: 32 (8 positions × 4 nucleotides, one-hot)
- **Oracle**: Exact lookup table (4^8 = 65 536 sequences)
- **Dataset source**: Design-Bench `TFBind8-Exact-v0`
- **Download**:
  ```bash
  python scripts/download_data.py --benchmarks tfbind8
  ```

---

## TFBind10

- **Config name**: `tfbind10`
- **Description**: DNA-binding protein sequence design (10-mer), optimising
  binding affinity.
- **Input dim**: 40 (10 positions × 4 nucleotides, one-hot)
- **Oracle**: Exact lookup table (4^10 = 1 048 576 sequences)
- **Dataset source**: Design-Bench `TFBind10-Exact-v0`
- **Download**:
  ```bash
  python scripts/download_data.py --benchmarks tfbind10
  ```

---

## Superconductor

- **Config name**: `superconductor` (or `supercon`)
- **Description**: Materials discovery benchmark optimising superconducting
  critical temperature (Tc) from composition vectors.
- **Input dim**: 86
- **Oracle**: Random Forest (trained on UCI Superconductor dataset)
- **Dataset source**: Design-Bench `Superconductor-RandomForest-v0`
- **Download**:
  ```bash
  python scripts/download_data.py --benchmarks superconductor
  ```

---

## Ant

- **Config name**: `ant`
- **Description**: RL locomotion task (Ant). Optimises morphology parameters
  and evaluates episodic return via MuJoCo simulation.
- **Input dim**: 60
- **Oracle**: Exact (MuJoCo simulation)
- **Dataset source**: Design-Bench `AntMorphology-Exact-v0`
- **Download**:
  ```bash
  python scripts/download_data.py --benchmarks ant
  ```

---

## D'Kitty

- **Config name**: `dkitty`
- **Description**: RL locomotion task (D'Kitty). Optimises morphology
  parameters and evaluates episodic return via MuJoCo simulation.
- **Input dim**: 56
- **Oracle**: Exact (MuJoCo simulation)
- **Dataset source**: Design-Bench `DKittyMorphology-Exact-v0`
- **Download**:
  ```bash
  python scripts/download_data.py --benchmarks dkitty
  ```

---

## ChEMBL

- **Config name**: `chembl`
- **Description**: Drug discovery benchmark using ChEMBL data and Morgan
  fingerprints.  Optimises fingerprint vectors to maximise proxy scores
  for assay CHEMBL3885882.
- **Input dim**: varies (depends on Morgan fingerprint configuration)
- **Oracle**: Random Forest
- **Dataset source**: Design-Bench
  `ChEMBL_MCHC_CHEMBL3885882_MorganFingerprint-RandomForest-v0`
- **Download**:
  ```bash
  python scripts/download_data.py --benchmarks chembl
  ```

---

## Running experiments

All six benchmarks can be run together with the diffusion optimiser:

```bash
python main.py --config configs/design_bench_diffusion.json
```

Or use CMA-ES / TPE baselines (create a similar JSON config with
`"method": "cma_es"` or `"method": "tpe"`).

## References

- Wu et al., "Diffusion-BBO: Diffusion-Based Inverse Modeling for Online
  Black-Box Optimization", arXiv:2407.00610, 2024.
- Trabucco et al., "Design-Bench: Benchmarks for Data-Driven Offline
  Model-Based Optimization", ICML 2022.
