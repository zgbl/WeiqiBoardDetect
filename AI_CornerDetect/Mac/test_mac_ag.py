import argparse
import cv2
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from opencv_engine_ag import OpenCVDetector
#p 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="/Users/tuxy/Codes/AI/Data/EmptyBoard/EmptyBoard3.jpg")
    args = parser.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        print("[-] 无法读取图片：", args.img)
        return

    print("\n==================================")
    print("  AG 版: Antigravity 独立实现")
    print("==================================\n")

    print("[*] 正在执行模块 1: OpenCV 角点检测引擎 (AG)...")
    opencv_engine = OpenCVDetector(debug_show=True)
    opencv_corners = opencv_engine.find_corners(img)

    if len(opencv_corners) < 4:
        print("[-] 模块 1 未能找到足够的角点！流程终止。")
        return

    print(f"\n[+] 模块 1 (OpenCV AG) 返回了 {len(opencv_corners)} 个角点候选")
    for pt in opencv_corners:
        print(f"    - 候选点：(X:{int(pt[0])}, Y:{int(pt[1])})")

    disp_img = img.copy()
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, (x, y) in enumerate(opencv_corners):
        c = colors[i] if i < 4 else (255, 255, 255)
        lbl = labels[i] if i < 4 else f"C{i+1}"
        cv2.circle(disp_img, (int(x), int(y)), 20, c, 3)
        cv2.putText(disp_img, lbl, (int(x)-30, int(y)-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, c, 2)

    cv2.imshow("AG Result", disp_img)
    print("\n(正在调出结果显示窗口，请按键盘上任意键关闭以退出程序)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
