"""
XAI-HRV: GradientSHAP-based explainability module (manuscript
Section III / Table VII).

Wraps Captum's GradientShap to attribute the BiLSTM-Attn model's risk
prediction back to the input HRV features (RMSSD, SDNN, SD1, SD2,
SD2/SD1, LF/HF, SampEn), producing the feature-importance ranking
the paper reports. Also provides a KernelSHAP cross-check to validate
the GradientSHAP approximation quality (the manuscript claims 0.008
mean absolute deviation -- that number must come from actually running
both methods on the same inputs, not being restated).
"""
import numpy as np
import torch
from captum.attr import GradientShap


FEATURE_NAMES = ["RMSSD", "SDNN", "SD1", "SD2", "SD2_SD1", "LF_HF", "SampEn"]


def compute_gradient_shap(model, inputs: torch.Tensor,
                           baseline: torch.Tensor = None,
                           target_class: int = 1,
                           n_samples: int = 50) -> np.ndarray:
    """Compute GradientSHAP attributions for a batch of HRV feature
    sequences.

    Parameters
    ----------
    model : trained BiLSTMAttn (or any nn.Module returning logits)
    inputs : (B, T, F) tensor of HRV feature sequences
    baseline : reference distribution for SHAP (default: zeros)
    target_class : which output class to attribute (1 = "instability")

    Returns
    -------
    np.ndarray, shape (B, T, F): per-timestep, per-feature attributions
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    gs = GradientShap(model)
    attributions = gs.attribute(
        inputs, baselines=baseline, target=target_class,
        n_samples=n_samples, stdevs=0.09,
    )
    return attributions.detach().cpu().numpy()


def rank_feature_importance(attributions: np.ndarray,
                             feature_names: list = None) -> list:
    """Aggregate |attribution| across batch and time to rank features,
    reproducing the format of manuscript Table VII.

    Parameters
    ----------
    attributions : (B, T, F) array from compute_gradient_shap
    """
    feature_names = feature_names or FEATURE_NAMES
    mean_abs = np.mean(np.abs(attributions), axis=(0, 1))  # (F,)
    std_abs = np.std(np.abs(attributions), axis=(0, 1))
    order = np.argsort(-mean_abs)
    return [
        {
            "rank": i + 1,
            "feature": feature_names[j],
            "mean_abs_phi": float(mean_abs[j]),
            "std_abs_phi": float(std_abs[j]),
        }
        for i, j in enumerate(order)
    ]


def kernel_shap_crosscheck(model, inputs: torch.Tensor,
                            gradient_shap_attributions: np.ndarray,
                            target_class: int = 1,
                            n_samples: int = 200) -> float:
    """Compare GradientSHAP attributions against exact KernelSHAP on a
    small subsample, returning the mean absolute deviation between the
    two. This is what the manuscript's "0.008 deviation" claim must be
    backed by -- run this and report whatever number comes out.
    """
    from captum.attr import KernelShap

    model.eval()
    baseline = torch.zeros_like(inputs)
    ks = KernelShap(model)
    ks_attr = ks.attribute(
        inputs, baselines=baseline, target=target_class,
        n_samples=n_samples,
    ).detach().cpu().numpy()

    mad = np.mean(np.abs(ks_attr - gradient_shap_attributions))
    return float(mad)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.models.bilstm_attn import BiLSTMAttn

    torch.manual_seed(0)
    model = BiLSTMAttn(input_dim=7)
    dummy = torch.randn(16, 20, 7)  # synthetic sanity check only

    attrs = compute_gradient_shap(model, dummy)
    ranking = rank_feature_importance(attrs)
    print("Synthetic sanity check (random untrained model, NOT a "
          "manuscript result -- run on a real trained checkpoint and "
          "real data for the actual Table VII):")
    for entry in ranking:
        print(f"  #{entry['rank']} {entry['feature']}: "
              f"{entry['mean_abs_phi']:.4f} ± {entry['std_abs_phi']:.4f}")
