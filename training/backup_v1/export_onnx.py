"""
export_onnx.py — 从已保存的 best_model.pt 导出 ONNX 模型
不需要重新训练。

使用方法:
  pip install onnx
  python3 export_onnx.py
  python3 export_onnx.py --model checkpoints/best_model.pt --output checkpoints/stone_classifier.onnx
"""
import torch
import argparse
from pathlib import Path
from model import build_model

def main():
    parser = argparse.ArgumentParser(description="导出 ONNX 模型")
    parser.add_argument("--model",  default="checkpoints/best_model.pt",
                        help="PyTorch 模型权重路径")
    parser.add_argument("--output", default="checkpoints/stone_classifier.onnx",
                        help="输出 ONNX 文件路径")
    parser.add_argument("--opset",  type=int, default=17, help="ONNX opset 版本")
    args = parser.parse_args()

    ckpt_path = Path(args.model)
    if not ckpt_path.exists():
        print(f"❌ 找不到模型文件: {ckpt_path}")
        return

    # 加载模型元数据
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch        = ckpt.get("arch",        "StoneCNN")
    patch_size  = ckpt.get("patch_size",  48)
    num_classes = ckpt.get("num_classes", 3)

    print(f"  模型: {arch}  |  Patch: {patch_size}px  |  类别数: {num_classes}")
    print(f"  Val Acc: {ckpt.get('val_acc', 'N/A')}")

    model = build_model(arch, num_classes, patch_size)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, 3, patch_size, patch_size)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
    )
    print(f"✅ ONNX 导出成功: {out_path}")
    print(f"   文件大小: {out_path.stat().st_size / 1024:.1f} KB")
    print(f"\n   用法示例（OpenCV DNN 加载推理）:")
    print(f"   net = cv2.dnn.readNetFromONNX('{out_path}')")
    print(f"   blob = cv2.dnn.blobFromImage(patch, 1/255.0, ({patch_size},{patch_size}))")
    print(f"   net.setInput(blob); pred = net.forward().argmax()  # 0=B 1=W 2=E")

if __name__ == "__main__":
    main()
