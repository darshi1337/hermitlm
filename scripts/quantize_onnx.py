"""Quantize ONNX model to uint8 for smaller web download."""
import argparse
from pathlib import Path


def main():
    from onnxruntime.quantization import QuantType, quantize_dynamic

    p = argparse.ArgumentParser()
    p.add_argument("--input", default="web/model.onnx")
    p.add_argument("--output", default="web/model.quant.onnx")
    a = p.parse_args()
    quantize_dynamic(Path(a.input), Path(a.output), weight_type=QuantType.QUInt8)
    inp, out = Path(a.input).stat().st_size, Path(a.output).stat().st_size
    print(f"{inp/1e6:.1f}MB -> {out/1e6:.1f}MB : {a.output}")

if __name__ == "__main__":
    main()
