import argparse
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from hybrid_scanner_v4 import HybridScannerV4

# python /Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/test_hybrid-v4.py --img /Users/tuxy/Codes/AI/Data/EmptyBoard/EmptyBoard3.jpg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str,
                        default="/Users/tuxy/Codes/AI/Data/EmptyBoard/EmptyBoard3.jpg")
                        
    parser.add_argument("--weights", type=str,
                        #default="/Users/tuxy/Codes/AI/WeiqiBoardDetect/data/Weights/4Classes-V4.pth")
                        #default="/Users/tuxy/Codes/AI/WeiqiBoardDetect/data/Weights/4Classes-V3.pth")
                        default="/Users/tuxy/Codes/AI/WeiqiBoardDetect/data/Weights/4Classes-V5_Scratch.pth")
    args = parser.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        print("[-] 无法读取图片:", args.img)
        return

    scanner = HybridScannerV4(args.weights, debug_show=True)
    corners, sel_h, sel_v, edges, dist_dir1, dist_dir2 = scanner.detect(img)

    if len(corners) >= 4:
        print(f"\n[结果] 角点:")
        labels = ["TL", "TR", "BR", "BL"]
        for i, (x, y) in enumerate(corners):
            lb = labels[i] if i < 4 else f"C{i+1}"
            print(f"  {lb}: ({int(x)}, {int(y)})")
        print(f"[结果] 网格线: H={len(sel_h)}, V={len(sel_v)}")
    else:
        print("\n[失败] 未能检测到完整棋盘")

    print("\n(按任意键关闭窗口)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
