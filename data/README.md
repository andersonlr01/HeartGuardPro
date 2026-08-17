# Dataset Acquisition

All three datasets used in the manuscript are hosted on PhysioNet.
None of them can be redistributed in this repository (PhysioNet's
license terms prohibit rehosting), so you must download them
yourself.

## 1. MIT-BIH Arrhythmia Database (open, no credentialing required)

```bash
wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/ -P ./raw_mitbih/
# or:
python -c "import wfdb; wfdb.dl_database('mitdb', './raw_mitbih')"
```

Then:
```bash
python prepare_mitbih.py --raw-dir ./raw_mitbih --out-dir ./processed_mitbih
python patient_split.py --data-dir ./processed_mitbih --out-dir ./processed_mitbih_split
```

## 2. MIMIC-III Waveform Database (requires credentialed PhysioNet access)

MIMIC-III requires completing CITI human-subjects research training
and signing a data use agreement through PhysioNet
(https://physionet.org/content/mimic3wdb/1.0/). This is a real,
non-trivial approval process -- budget several days to weeks.

Once approved:
```bash
wget -r -N -c -np --user <physionet_username> --ask-password \
    https://physionet.org/files/mimic3wdb/1.0/ -P ./raw_mimic3/
```

Then write/adapt `prepare_mimic3.py` (not included here as a stub
because the exact waveform-to-HRV-sequence pipeline depends on which
MIMIC-III subset and label definition for "cardiovascular
instability" you settle on -- this must be specified precisely and
honestly in the Methods section, since the current draft does not
define it operationally).

## 3. PhysioNet/CinC Challenge 2019 (open, no credentialing required)

```bash
wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/ \
    -P ./raw_physionet2019/
```

## A note on scope

Realistically, MIMIC-III approval plus building and validating a
correct instability-labeling pipeline is itself a multi-week research
task -- it is the actual scientific content of the "cloud BiLSTM-Attn"
half of the paper, not a data-engineering formality. If your deadline
doesn't allow for it, that supports going with the "proposed
framework" honest-reframing path instead (see main README).
