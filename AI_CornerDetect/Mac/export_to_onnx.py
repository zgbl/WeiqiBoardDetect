"""
将 PatchClassifier (ResNet18, 4类) 从 PyTorch .pth 导出为 ONNX 格式
供 Android ONNX Runtime 使用

用法:
  python export_to_onnx.py                         # 使用默认路径
  python export_to_onnx.py --weights path/to.pth   # 指定权重
  python export_to_onnx.py --output path/to.onnx   # 指定输出

导出后将 .onnx 文件复制到 Android 项目:
  app/src/main/assets/models/patch_classifier.onnx
"""

import argparse
import sys
from pathlib import Path
import torch

# 确保能 import model.py
SCRIPT_DIR = Path(__file__).parent
CLASSIFIER_DIR = SCRIPT_DIR.parent / "Classifier_V1"
sys.path.insert(0, str(CLASSIFIER_DIR))
from model import PatchClassifier


def main():
    parser = argparse.ArgumentParser(description="Export PatchClassifier to ONNX")
    parser.add_argument("--weights", type=str,
                        default=str(SCRIPT_DIR.parent.parent / "data" / "Weights" / "4Classes-V5_Scratch.pth"))
    parser.add_argument("--output", type=str,
                        default=str(SCRIPT_DIR / "patch_classifier.onnx"))
    parser.add_argument("--input-size", type=int, default=128,
                        help="CNN input spatial size (default 128, matching hybrid_scanner_v5)")
    args = parser.parse_args()

    print(f"[1/4] Loading model from: {args.weights}")
    model = PatchClassifier(num_classes=4)
    ckpt = torch.load(args.weights, map_location="cpu")
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt

    # 处理 key 前缀不一致问题 (与 hybrid_scanner_v5.py 一致)
    new_sd = {}
    model_keys = model.state_dict().keys()
    for k, v in state_dict.items():
        if k in model_keys:
            new_sd[k] = v
        elif "backbone." + k in model_keys:
            new_sd["backbone." + k] = v
        elif k.replace("backbone.", "") in model_keys:
            new_sd[k.replace("backbone.", "")] = v

    model.load_state_dict(new_sd, strict=False)
    model.eval()
    print(f"   Loaded {len(new_sd)} parameters")

    print(f"[2/4] Creating dummy input ({args.input_size}x{args.input_size})")
    dummy = torch.randn(1, 3, args.input_size, args.input_size)

    print(f"[3/4] Exporting to ONNX: {args.output}")
    torch.onnx.export(
        model, dummy, args.output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13
    )

    # 验证
    print(f"[4/4] Verifying ONNX model...")
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(args.output)
    result = sess.run(None, {"input": dummy.numpy()})
    classes = ['Corner', 'Inner', 'Edge', 'Outer']
    probs = np.exp(result[0][0]) / np.exp(result[0][0]).sum()
    pred = classes[probs.argmax()]
    print(f"   Test inference OK: {pred} ({probs.max():.3f})")
    print(f"\n✅ Done! Copy to Android project:")
    print(f"   cp {args.output} <BRLiveStream>/app/src/main/assets/models/patch_classifier.onnx")


if __name__ == "__main__":
    main()
