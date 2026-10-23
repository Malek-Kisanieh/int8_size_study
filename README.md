# INT8 post-training quantization and object size in aerial detection

Course project for DV2654 Research Methodology in Computer Science, Blekinge Institute of
Technology (BTH).

**Authors:** Ali Reza Hussaini, Mhd Malek Kisanieh

**Research question.** When YOLOv8n is quantized to INT8 with TensorRT post-training quantization,
does it lose relatively more accuracy (COCO AP) on small objects (box area below 32² px) than on
non-small objects in drone images (VisDrone2019-DET)?

The study was pre-registered. The frozen design and the deviation log are in
[`PREREGISTRATION.md`](PREREGISTRATION.md). All settings are in [`config.yaml`](config.yaml), and the
scripts read their settings only from that file.

## Status and outcome

The study is finished (October 2026).

- **Main statistic.** D is the relative AP drop for small objects minus the relative AP drop for
  non-small objects (FP32 vs INT8 seed 0, AP@[.5:.95], test-dev). It is estimated with a paired
  image-level bootstrap with 2,000 resamples. H1 is D > 0.
- **Result.** D = +0.96 percentage points, 95% CI [−0.81, +3.03]. The interval contains zero, so the
  pre-registered test is inconclusive and H1 is not supported.
- **Post-hoc check (deviation D4).** A cluster bootstrap over the 61 filename prefixes of test-dev
  gives a 95% CI of [−0.65, +4.14]. It is not used for the confirmatory decision.

## Paper

The graded report is [`report.tex`](report.tex) / [`report.pdf`](report.pdf) (IEEE conference
format). Build it from the project root with

```bash
latexmk -pdf report.tex
```

The figures are read from `results/results/main/analysis/test/figures/`.

## Project structure

```
int8_size_study/
├── PREREGISTRATION.md         frozen design, decisions and deviation log (§11)
├── config.yaml                all settings
├── report.tex, report.pdf     the paper
├── requirements/
│   ├── common.txt             pinned packages shared by both profiles
│   ├── trt11-cu12.txt         main profile: TensorRT 11, INT8 via NVIDIA ModelOpt
│   └── trt10-cu12.txt         pre-registered fallback: TensorRT 10
├── src/
│   ├── common.py              config, hashing, downloads, logging
│   └── evaluation.py          size-binned COCO evaluation and bootstrap AP
├── scripts/
│   ├── 01_environment_report.py   Stage 2  machine and package report
│   ├── 02_smoke_test.py           Stage 2  toolchain check on COCO8
│   ├── 03_prepare_dataset.py      Stage 3a download, verify and convert VisDrone
│   ├── 04_check_dataset.py        Stage 3b dataset checks C1-C8 (gate before modelling)
│   ├── 05_train_fp32.py           Stage 4  fine-tune YOLOv8n, freeze the FP32 checkpoint
│   ├── 06_build_engines.py        Stage 5  calibration sets, FP32/FP16/INT8 engines
│   ├── 07_evaluate.py             Stage 6  AP per size bin with pycocotools
│   ├── 08_analyze.py              Stage 8  paired bootstrap, decision, tables, figures
│   └── 09_cluster_bootstrap.py    Stage 9  post-hoc cluster bootstrap (D4)
├── tests/test_evaluation.py   checks our AP code against pycocotools on synthetic detections
└── results/results/           outputs of the final run (see below)
```

The scripts also create `data/`, `models/` and `.ultralytics/`. These folders are not committed.
Stage 7, the controlled latency benchmark, was planned but not implemented (deviation D2).

## Environment used for the final run

| Item | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 |
| Python | 3.11.9 |
| PyTorch | 2.10.0+cu128 |
| Ultralytics | 8.4.174 |
| TensorRT | 11.3.0.99, INT8 through NVIDIA ModelOpt |

ModelOpt is pinned to 0.47.0 in `requirements/trt11-cu12.txt`, but its version is not printed in
the logs. The run used Python 3.11.9 / PyTorch 2.10.0 instead of the planned 3.12 / 2.14.1
(deviation D1).

## Setup

You need Linux x86_64, an NVIDIA GPU with compute capability ≥ 7.5 for INT8, a driver that
supports CUDA 12, about 15 GiB of free disk, and access to pypi.org, pypi.nvidia.com,
download.pytorch.org and github.com. Run all commands from the project root.

1. Report the bare machine. This script uses only the standard library, so it runs before
   anything is installed. If it prints a note starting with BLOCKER, stop.

   ```bash
   python scripts/01_environment_report.py --label before_install
   ```

2. Create a Python 3.11 environment.
3. Install PyTorch built for CUDA 12 first. The default PyPI build may target CUDA 13, which does
   not work with the TensorRT CUDA 12 wheels and ModelOpt.

   ```bash
   pip install torch==2.10.0 torchvision==<matching version> --index-url https://download.pytorch.org/whl/cu128
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```

   The check must print a CUDA version starting with `12.` and `True`.
4. Install the TensorRT 11 profile:

   ```bash
   pip install -r requirements/trt11-cu12.txt --extra-index-url https://pypi.nvidia.com
   ```

5. Check the installed environment and the toolchain (Stage 2):

   ```bash
   python scripts/01_environment_report.py
   python scripts/02_smoke_test.py
   ```

   The smoke test uses COCO-pretrained YOLOv8n on COCO8 and must end with `OVERALL: PASS`. Its
   AP values are not results. The outputs of these two scripts (`results/environment/`,
   `results/smoke_test/`) are not part of the committed results.

   If the INT8 export fails on TensorRT 11, switch to the pre-registered fallback
   (`PREREGISTRATION.md` §9), re-run both checks and add a row to §11. We did not need it.

   ```bash
   pip uninstall -y tensorrt-cu12 tensorrt_cu12_libs tensorrt_cu12_bindings nvidia-modelopt
   pip install -r requirements/trt10-cu12.txt --extra-index-url https://pypi.nvidia.com
   ```

The scripts do not use `sudo`, keep the Ultralytics settings inside the project, and turn off
Ultralytics' automatic pip installs, so a missing package shows up as an error.

## Reproducing the study

**Stage 3: dataset.** `03_prepare_dataset.py` downloads about 1.9 GB of archives, verifies them by
SHA-256 and writes YOLO labels and COCO JSON from the same parsed boxes (`--splits` and
`--skip-download` are optional). `04_check_dataset.py` runs checks C1-C8 and exits with code 1 if
any check fails. Do not continue in that case.

```bash
python scripts/03_prepare_dataset.py
python scripts/04_check_dataset.py
python tests/test_evaluation.py
```

`tests/test_evaluation.py` needs the Stage 3 output (`data/VisDrone/annotations/coco_gt_val.json`)
and pycocotools. Both of its tests must pass. Because the archives are hash-verified, the counts
should match `results/results/dataset/size_bin_counts.csv`:

| Split | Images | Objects | Small | Medium | Large |
|---|---|---|---|---|---|
| train | 6,471 | 343,204 | 60.5% | 34.0% | 5.5% |
| val | 548 | 38,759 | 68.6% | 28.7% | 2.8% |
| test-dev | 1,610 | 75,102 | 67.7% | 29.1% | 3.2% |

**Development run on val.** The `dev` tag trains for 3 epochs on 10% of train and checks that every
stage runs. Its results are never reported.

```bash
python scripts/05_train_fp32.py    --tag dev 2>&1 | tee results/train_dev_console.log
python scripts/06_build_engines.py --tag dev
python scripts/07_evaluate.py      --tag dev --split val
python scripts/08_analyze.py       --tag dev --split val --reps 50
```

**Main run (Stages 4-6, evaluated on val).**

```bash
python scripts/05_train_fp32.py    --tag main 2>&1 | tee results/train_main_console.log
python scripts/06_build_engines.py --tag main 2>&1 | tee results/engines_main_console.log
python scripts/07_evaluate.py      --tag main --split val 2>&1 | tee results/eval_val_console.log
```

If training stops, continue it with `python scripts/05_train_fp32.py --tag main --resume`.
`06_build_engines.py` builds each engine once (FP32, FP16 and INT8 with calibration seeds 0, 1 and
2) and does not rebuild an existing engine unless `--force` is given. `07_evaluate.py` and
`06_build_engines.py` also accept `--conditions` to run a subset.

**Safeguards.** `05_train_fp32.py` refuses to run unless the Stage 3 checks passed, and it never
overwrites a frozen model. `models/<tag>/model_record.json` holds the SHA-256 of the checkpoint and
`engines_record.json` holds the engine settings and hashes. Stages 5 and 6 check these hashes
before they use a file. The PyTorch FP32 reference runs with TF32 disabled. In the COCO files,
`category_id` is the class index + 1, as in the Ultralytics prediction JSON.

**Final confirmatory run on test-dev.** Run this once, after everything else is finished.

```bash
python scripts/07_evaluate.py --tag main --split test --final
python scripts/08_analyze.py  --tag main --split test --final
```

**Stage 9: post-hoc cluster bootstrap (D4).**

```bash
python scripts/09_cluster_bootstrap.py --workers 6
```

This script was added after the confirmatory analysis. It needs the raw Stage 6 files
(`predictions.json` and `matches.npz` under `main/evaluation/test/`), which are not committed.

### Test-split guard and blinding

The pre-registration fixes the blinding rule: INT8 results on test-dev are produced only in the
final run, and all development uses val. The scripts enforce it. `07_evaluate.py` and
`08_analyze.py` refuse the test split unless both `--final` and `--tag main` are given, and
`08_analyze.py` does not allow `--reps` to change the 2,000 resamples of the final analysis.

## Results

The scripts write to `results/<tag>/`. The outputs of our run are committed in `results/results/`,
and every number in the paper comes from these files.

| Path in `results/results/` | Content |
|---|---|
| `dataset/` | conversion report, checks C1-C8 (`dataset_checks.json`), counts per size bin and class, filename-prefix groups, sample and size-distribution figures |
| `main/calibration/` | the 512 calibration images per seed and their size statistics |
| `main/evaluation/{val,test}/<condition>/` | `metrics.json` (AP@[.5:.95] and AP@0.5 per size bin) and the Ultralytics log for each condition |
| `main/analysis/test/table3_accuracy.csv` | AP per condition and size bin with 95% CIs |
| `main/analysis/test/table4_hypothesis.csv` | absolute and relative AP drops, D and D_abs with 95% CIs |
| `main/analysis/test/summary.json` | H1 decision, robustness over the three INT8 seeds, FP16 control |
| `main/analysis/test/validation.csv` | pipeline checks (our AP vs pycocotools, PyTorch FP32 vs TensorRT FP32) |
| `main/analysis/test/cluster_bootstrap.json` | post-hoc cluster bootstrap (D4) |
| `main/analysis/test/figures/` | figures used in the paper |
| `dev/` | development run on val, not reported (see D3) |
| `*_console.log` | console logs of training, engine builds and val evaluation |
| `train_main_terminal.txt` | terminal copy of the main training run, which shows the environment (evidence for D1) |

The raw files `predictions.json` and `*.npz` (about 1.3 GB in total) are not committed.

## Changing the setup

Change settings only in `config.yaml`, and log every change after the design freeze in
`PREREGISTRATION.md` §11.
