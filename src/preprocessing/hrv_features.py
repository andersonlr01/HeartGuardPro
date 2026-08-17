"""
Heart Rate Variability (HRV) feature extraction.

Implements the five HRV metrics reported as the top GradientSHAP
predictors in the manuscript (Table VII): RMSSD, LF/HF ratio, SD2/SD1
(Poincare), SDNN, and SampEn. These are computed from a sequence of
R-R (beat-to-beat) intervals, typically derived from detected peaks in
the filtered PPG or ECG signal.

NOTE ON HONESTY: this module computes real, standard HRV metrics from
whatever R-R interval series you feed it. It makes no claim about what
values you *should* get -- those come only from running it on real
data (MIT-BIH / MIMIC-III / PhysioNet 2019).
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch


def rmssd(rr_intervals_ms: np.ndarray) -> float:
    """Root Mean Square of Successive Differences (time-domain HRV)."""
    diffs = np.diff(rr_intervals_ms)
    return float(np.sqrt(np.mean(diffs ** 2)))


def sdnn(rr_intervals_ms: np.ndarray) -> float:
    """Standard deviation of NN (R-R) intervals."""
    return float(np.std(rr_intervals_ms, ddof=1))


def poincare_sd1_sd2(rr_intervals_ms: np.ndarray) -> tuple:
    """Poincare plot descriptors SD1 (short-term) and SD2 (long-term)
    variability, per Brennan et al. (ref [13] in the manuscript)."""
    rr_n = rr_intervals_ms[:-1]
    rr_n1 = rr_intervals_ms[1:]
    diff = rr_n - rr_n1
    summ = rr_n + rr_n1
    sd1 = float(np.std(diff, ddof=1) / np.sqrt(2))
    sd2 = float(np.std(summ, ddof=1) / np.sqrt(2))
    return sd1, sd2


def lf_hf_ratio(rr_intervals_ms: np.ndarray, fs_interp: float = 4.0) -> dict:
    """Frequency-domain HRV via Lomb-free Welch PSD on an evenly
    resampled R-R tachogram.

    LF band: 0.04-0.15 Hz, HF band: 0.15-0.4 Hz (standard Task Force
    1996 definitions).
    """
    if len(rr_intervals_ms) < 4:
        raise ValueError("Need at least 4 RR intervals for spectral HRV.")

    t_rr = np.cumsum(rr_intervals_ms) / 1000.0  # seconds
    t_rr -= t_rr[0]
    f_interp = interp1d(t_rr, rr_intervals_ms, kind="cubic",
                         fill_value="extrapolate")
    t_uniform = np.arange(0, t_rr[-1], 1.0 / fs_interp)
    rr_uniform = f_interp(t_uniform)
    rr_uniform = rr_uniform - np.mean(rr_uniform)

    nperseg = min(256, len(rr_uniform))
    if nperseg < 8:
        raise ValueError("Signal too short for spectral estimation.")

    freqs, psd = welch(rr_uniform, fs=fs_interp, nperseg=nperseg)

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)

    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz
    lf_power = _trapz(psd[lf_mask], freqs[lf_mask]) if lf_mask.any() else 0.0
    hf_power = _trapz(psd[hf_mask], freqs[hf_mask]) if hf_mask.any() else 1e-8

    ratio = float(lf_power / hf_power) if hf_power > 1e-8 else float("nan")
    return {"lf_power": float(lf_power), "hf_power": float(hf_power),
            "lf_hf_ratio": ratio}


def sample_entropy(signal: np.ndarray, m: int = 2, r: float = None) -> float:
    """Sample Entropy (SampEn), a measure of HRV complexity/regularity.

    Straightforward O(N^2) reference implementation -- fine for
    windows of a few hundred beats; for very long series consider a
    KD-tree accelerated variant.
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if r is None:
        r = 0.2 * np.std(signal)
    if n <= m + 1:
        return float("nan")

    def _count_matches(template_len):
        templates = np.array([signal[i:i + template_len]
                               for i in range(n - template_len + 1)])
        count = 0
        for i in range(len(templates)):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dists <= r) - 1  # exclude self-match
        return count

    b = _count_matches(m)
    a = _count_matches(m + 1)
    if b == 0 or a == 0:
        return float("nan")
    return float(-np.log(a / b))


def extract_hrv_feature_vector(rr_intervals_ms: np.ndarray) -> dict:
    """Convenience wrapper returning all five features used in the
    manuscript's GradientSHAP ranking (Table VII), in one call."""
    sd1, sd2 = poincare_sd1_sd2(rr_intervals_ms)
    lf_hf = lf_hf_ratio(rr_intervals_ms)
    return {
        "RMSSD": rmssd(rr_intervals_ms),
        "SDNN": sdnn(rr_intervals_ms),
        "SD1": sd1,
        "SD2": sd2,
        "SD2_SD1": sd2 / sd1 if sd1 > 1e-8 else float("nan"),
        "LF_HF": lf_hf["lf_hf_ratio"],
        "SampEn": sample_entropy(rr_intervals_ms),
    }
