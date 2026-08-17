# HeartGuard Pro — Reference Implementation

A working implementation of the architecture described in *"HeartGuard
Pro: A Temporal Attention-Enhanced Hybrid Edge–Cloud Framework for
Early Detection of Cardiovascular Instability Using Multimodal
Wearable Signals"*: signal preprocessing, HRV feature extraction,
TA-CNN (edge), BiLSTM-Attn (cloud), baselines, statistical testing,
GradientSHAP explainability, and TFLite Micro export for ESP32-S3.

## Honest status of this repository (read this first)

**The code in `src/` is real and has been smoke-tested** — every
module runs, produces correctly-shaped output, and contains no
placeholder logic (see `git log` / commit for the synthetic
sanity-check outputs each module prints when run directly).

**No experimental numbers in this repository are claims of results.**
Nothing here has been run on MIT-BIH, MIMIC-III, or PhysioNet 2019
yet, because:
- MIT-BIH and PhysioNet 2019 are open but were not downloaded in the
  environment this code was written in (no network path to
  physionet.org from that sandbox).
- MIMIC-III requires a credentialed PhysioNet data-use agreement that
  only a human researcher (with institutional affiliation) can obtain
  — this cannot be done on your behalf.

**What this means for the manuscript:** the specific figures currently
in the draft (97.8% accuracy, F1=0.941, the four baseline comparisons
with p-values, the 3.3pp external-validation gap, the GradientSHAP
Table VII rankings, the 14.2 ms / 187 KB edge figures) were not
produced by running this code, or any code, on the named datasets.
They should not be submitted to a journal as measured results unless
and until they actually are. This repository exists to make that
possible honestly:

1. Download the datasets (`data/README.md`).
2. Run the data prep scripts.
3. Run training (`src/training/`).
4. Run evaluation, statistical tests, and explainability
   (`src/evaluation/`, `src/explainability/`).
5. Run the edge export and get a *real* .tflite size, then benchmark
   *real* latency on actual ESP32-S3 hardware (`src/edge/`).
6. Replace every number in the manuscript with whatever these scripts
   actually print — better, worse, or different in kind from the
   current draft. Report it as it is.

If a reviewer or journal is asking for a GitHub link as a
reproducibility check, linking this repo **before** step 6 is
misleading: it demonstrates a plausible implementation, not
reproduced results. Be explicit with the journal about which of these
you're providing.

## Repository structure

```
src/
  preprocessing/   Butterworth bandpass filter, HRV feature extraction
  models/          TA-CNN, BiLSTM-Attn, baselines (CNN/BiLSTM/Transformer/InceptionTime)
  training/        Training loops (require real data — see data/README.md)
  evaluation/       McNemar test, bootstrap CIs, external validation script
  explainability/  GradientSHAP + KernelSHAP cross-check
  edge/            PyTorch -> ONNX -> INT8 TFLite -> C array, for ESP32-S3
data/
  README.md        Dataset download instructions (MIT-BIH, MIMIC-III, PhysioNet 2019)
  prepare_mitbih.py, patient_split.py
```

## Quick start (once you have data)

```bash
pip install -r requirements.txt

# 1. Prepare MIT-BIH for TA-CNN
python data/prepare_mitbih.py --raw-dir ./raw_mitbih --out-dir ./processed_mitbih
python data/patient_split.py --data-dir ./processed_mitbih --out-dir ./split_mitbih

# 2. Train TA-CNN
python src/training/train_ta_cnn.py --data-dir ./split_mitbih/train --out ta_cnn.pt

# 3. Export to ESP32-S3-ready TFLite and measure real size
python src/edge/convert_tflite.py --checkpoint ta_cnn.pt \
    --calib-data ./split_mitbih/val/X.npy --out-dir ./tflite_export

# 4. (After obtaining MIMIC-III access) train and validate BiLSTM-Attn
python src/training/train_bilstm_attn.py --train-dir ./mimic3_train --val-dir ./mimic3_val
python src/evaluation/external_validation.py --checkpoint bilstm_attn_checkpoint.pt \
    --internal-val-dir ./mimic3_val --external-dir ./physionet2019_processed
```

## Why no numbers are pre-filled anywhere

Every script in this repo prints a note next to its output reminding
you that the figure is real but must be taken from an actual run, not
copied from the manuscript. This is intentional — it's the difference
between a reproducibility artifact and a liability.
