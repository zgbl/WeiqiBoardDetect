import argparse
import cv2
import sys
from pathlib import Path

# 保证能顺利导我们的独立 module
sys.path.append(str(Path(__file__).parent))
from opencv_engine1b import OpenCVDetector
#from opencv_engine_qw_V1 import OpenCVDetector
#python /Users/tuxy/Codes/AI/WeiqiBoardDetect/AI_CornerDetect/Mac/test_mac1b.py --img /Users/tuxy/Codes/AI/Data/EmptyBoard/EmptyBoard3.jpg


from cnn_engine import CNNVerifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="/Users/tuxy/Codes/AI/Data/EmptyBoard/EmptyBoard3.jpg")
    parser.add_argument("--weights", type=str, default="/Users/tuxy/Codes/AI/WeiqiBoardDetect/data/Weights/4Classes-V4.pth")
    args = parser.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        print("[-] 无法读取图片:", args.img)
        return

    print("\n==================================")
    print(" 混合系统: 模块组合调用流水线 ")
    print("==================================\n")

    # ---------------------------------------------------------
    # 模块 1: OpenCV 传统算法黑盒，只负责丢给它图，它吐出 4 个点
    # 此步骤在未来可任意被其他诸如 Heatmap 或纯网络检测组件替换
    # ---------------------------------------------------------
    print("[*] 正在执行模块 1: OpenCV 角点检测引擎...")
    opencv_engine = OpenCVDetector()
    opencv_corners = opencv_engine.find_corners(img)
    
    if len(opencv_corners) < 4:
        print("[-] 模块 1 未能找到足够的角点！流程终止。")
        return
        
    print(f"[+] 模块 1 (OpenCV) 强制返回了 4 个物理极值角点坐标候选")
    for pt in opencv_corners:
        print(f"    - 候选点: (X:{int(pt[0])}, Y:{int(pt[1])})")

    # ---------------------------------------------------------
    # 模块 2: CNN 真伪验证黑盒。只负责对坐标进行判定
    # ---------------------------------------------------------
    print("\n[*] 正在执行模块 2: CNN 交叉验证引擎...")
    cnn_engine = CNNVerifier(args.weights)
    verification_results = cnn_engine.verify_corners(img, opencv_corners)

    # ---------------------------------------------------------
    # 后处理决定逻辑：未来可以在此设计 Step 3
    # ---------------------------------------------------------
    print("\n[*] 最终评判结果:")
    disp_img = img.copy()
    
    all_passed = True
    for res in verification_results:
        pt = res["point"]
        label = res["label"]
        conf = res["confidence"]
        is_valid = res["is_corner"]
        
        x, y = int(pt[0]), int(pt[1])
        
        if is_valid:
            # CNN 确认这是真实的角点
            cv2.circle(disp_img, (x, y), 12, (0, 0, 255), -1)       # 红心
            cv2.circle(disp_img, (x, y), 14, (255, 255, 255), 2)    # 白边
            cv2.putText(disp_img, f"CONFIRMED:{conf:.2f}", (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"  [+] 验证通过 -> {pt} 是真实角点 (置信度:{conf:.2f})")
        else:
            # CNN 否决了 OpenCV 的找点
            all_passed = False
            cv2.circle(disp_img, (x, y), 12, (0, 255, 255), 2)      # 黄色虚框
            cv2.putText(disp_img, f"REJECTED({label})", (x+15, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            print(f"  [-] 验证失败 -> OpenCV 认为是角点，但 CNN 识别它为 '{label}'!")

    if not all_passed:
        print("\n[!] 结论: 部分角点验证失败。模块 1 (OpenCV) 找出的点并非全部正确。")
        print("    -> 留出切入点: 下一步将在此阶段设计新的逻辑(如在局部范围内重新搜索真实角点)。")
    else:
        print("\n[✓] 结论: 所有 4 个角点均顺利通过 CNN 验证！")

    cv2.imshow("Module 1 (OpenCV) + Module 2 (CNN)", disp_img)
    print("\n(正在调出结果显示窗口，请按键盘上任意键关闭以退出程序)")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
