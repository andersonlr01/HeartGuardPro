"""
Convert a trained TA-CNN PyTorch checkpoint to an INT8-quantized
TFLite model suitable for TFLite Micro on ESP32-S3.

This produces the ARTIFACT (a .tflite file + a C byte array) needed
to actually measure the two claims in the manuscript that currently
have no measurement behind them:
  1. Model size in KB (paper claims 187 KB)
  2. On-device inference latency in ms (paper claims 14.2 ms)

Both numbers MUST be measured on real ESP32-S3 hardware after
flashing -- this script gets you to the point of having a flashable
model; it does not itself produce a latency number. See the printed
instructions at the end for the on-device benchmark step.

Requires: onnx, onnx-tf (or ai-edge-torch / torch>=2.x built-in ONNX
export), tensorflow. PyTorch -> TFLite has no single official path,
so this uses the common PyTorch -> ONNX -> TensorFlow -> TFLite route.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.ta_cnn import TACNN


def export_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, onnx_path: str):
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["input"], output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"Exported ONNX model to {onnx_path}")


def onnx_to_tflite_int8(onnx_path: str, tflite_path: str,
                         representative_data: np.ndarray):
    """Requires `onnx-tf` and `tensorflow` installed. Converts
    ONNX -> SavedModel -> INT8 TFLite with representative-dataset
    calibration (needed for full INT8 quantization, matching the
    manuscript's INT8-precision claim)."""
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    saved_model_dir = str(Path(tflite_path).parent / "saved_model_tmp")
    tf_rep.export_graph(saved_model_dir)

    def representative_dataset():
        for i in range(min(100, len(representative_data))):
            yield [representative_data[i:i + 1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    Path(tflite_path).write_bytes(tflite_model)
    print(f"Wrote INT8 TFLite model to {tflite_path} "
          f"({len(tflite_model) / 1024:.1f} KB -- this IS a real, "
          f"measured size, unlike a number copied into a draft)")
    return tflite_model


def tflite_to_c_array(tflite_path: str, out_c_path: str, var_name: str = "ta_cnn_model"):
    """xxd-style conversion to a C byte array for TFLite Micro on
    ESP32-S3 (Arduino/ESP-IDF projects include this file directly)."""
    data = Path(tflite_path).read_bytes()
    with open(out_c_path, "w") as f:
        f.write(f"#include <cstdint>\n\n")
        f.write(f"alignas(8) const unsigned char {var_name}[] = {{\n")
        for i in range(0, len(data), 12):
            chunk = data[i:i + 12]
            f.write("  " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n")
        f.write(f"const unsigned int {var_name}_len = {len(data)};\n")
    print(f"Wrote C byte array to {out_c_path} ({len(data)} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                         help="Trained TA-CNN state_dict (.pt)")
    parser.add_argument("--calib-data", required=True,
                         help=".npy file with representative beat windows "
                              "for INT8 calibration (use validation set)")
    parser.add_argument("--out-dir", default="./tflite_export")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TACNN(in_channels=1, num_classes=5)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    calib_data = np.load(args.calib_data)  # (N, T)
    dummy = torch.randn(1, 1, calib_data.shape[-1])

    onnx_path = out_dir / "ta_cnn.onnx"
    export_onnx(model, dummy, str(onnx_path))

    tflite_path = out_dir / "ta_cnn_int8.tflite"
    onnx_to_tflite_int8(str(onnx_path), str(tflite_path),
                         calib_data[:, np.newaxis, :])

    tflite_to_c_array(str(tflite_path), str(out_dir / "ta_cnn_model_data.h"))

    print("\n--- NEXT STEP: real on-device measurement ---")
    print("1. Flash ta_cnn_model_data.h into a TFLite Micro sketch on")
    print("   the ESP32-S3 (see espressif/esp-tflite-micro examples).")
    print("2. Wrap the invoke() call with esp_timer_get_time() before/after")
    print("   to measure real per-inference latency in microseconds.")
    print("3. Report the .tflite file size above and the measured latency")
    print("   -- both are now real numbers, not projections.")


if __name__ == "__main__":
    main()
