"""
EXECUTION GUIDE

Before running this script, ensure you are in your custom Conda environment where OpenCV is installed.
You can activate your Conda environment and run the script as follows:

conda activate <your_environment_name>

Then execute the script with a test image:
python /Users/tuxy/Codes/AI/WeiqiBoardDetect/BoardCornerDetect/human_like_corner_detect.py --image <path_to_test_image.jpg> --output_dir <path_to_save_results>

Example:
python /Users/tuxy/Codes/AI/WeiqiBoardDetect/BoardCornerDetect/human_like_corner_detect.py --image /Users/tuxy/Codes/AI/OpenCVTest1/data/raw/Board1.jpg

This script will output progress to the console and save intermediate visualization images (raw lines, clustered lines, merged lines) to the specified output folder.
"""

import cv2
import numpy as np
import argparse
import os
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Human-like Go Board Corner Detection Algorithm")
    parser.add_argument("--image", type=str, required=True, help="Path to the input test image")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save intermediate visualizations")
    return parser.parse_args()

def extract_line_segments(gray_img):
    """
    Step 1: Extract Line Segments using LSD (Line Segment Detector).
    LSD is highly sensitive and extracts short local fragments, simulating human "fragment" perception.
    """
    # Create Line Segment Detector
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(gray_img)
    return lines

def line_length(line):
    return math.hypot(line[2] - line[0], line[3] - line[1])

def cluster_lines_by_direction(lines):
    """
    Step 2: Cluster lines into Horizontal and Vertical sets based on their angles.
    For strong perspective, a Vanishing Point estimation would be ideal.
    Here we use a simplified angle-based clustering as a baseline.
    """
    horizontal_lines = []
    vertical_lines = []

    if lines is None:
        return horizontal_lines, vertical_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Filter out extremely short lines that are just noise
        if math.hypot(x2 - x1, y2 - y1) < 10:
            continue
            
        # Calculate angle of the line
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Normalize angle to be between 0 and 180
        if angle < 0:
            angle += 180
        
        # Simple heuristic: angles near 0/180 are horizontal, near 90 are vertical.
        # This assumes the board isn't rotated by exactly 45 degrees.
        # Let's tighten the angle range slightly assuming the camera is somewhat upright
        if (angle < 45) or (angle > 135):
            horizontal_lines.append([x1, y1, x2, y2])
        elif (50 < angle < 130):
            vertical_lines.append([x1, y1, x2, y2])
            
    return horizontal_lines, vertical_lines

def merge_collinear_lines(lines, distance_thresh=10, angle_thresh=3, gap_thresh=50):
    """
    Step 3: Line Merging (joining broken collinear fragments).
    Improved merging using distance, angle, and gap along the line.
    """
    merged_lines = []
    if not lines:
        return merged_lines

    # Sort lines by length (longest first)
    lines.sort(key=line_length, reverse=True)
    
    used = [False] * len(lines)
    
    for i in range(len(lines)):
        if used[i]:
            continue
            
        current_cluster = [lines[i]]
        used[i] = True
        
        # Keep track of the expanding bounding line
        base_line = list(lines[i])
        
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
                
            x1, y1, x2, y2 = base_line
            x3, y3, x4, y4 = lines[j]
            
            base_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            test_angle = math.degrees(math.atan2(y4 - y3, x4 - x3))
            
            # 1. Check angle difference
            angle_diff = abs(base_angle - test_angle)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
                
            if angle_diff > angle_thresh:
                continue
                
            # 2. Check orthogonal distance (is the test line collinear roughly?)
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            denom = math.sqrt(a*a + b*b)
            if denom == 0:
                continue
                
            dist3 = abs(a * x3 + b * y3 + c) / denom
            dist4 = abs(a * x4 + b * y4 + c) / denom
            
            if dist3 > distance_thresh or dist4 > distance_thresh:
                continue
                
            # 3. Check gap along the line direction
            # Project all points onto the base line direction unit vector
            vx, vy = x2 - x1, y2 - y1
            length = math.hypot(vx, vy)
            ux, uy = vx / length, vy / length
            
            # Projections
            p1 = 0 # base point 1 is origin
            p2 = length
            p3 = (x3 - x1) * ux + (y3 - y1) * uy
            p4 = (x4 - x1) * ux + (y4 - y1) * uy
            
            # Sort projections to find the range
            base_min_p, base_max_p = min(p1, p2), max(p1, p2)
            test_min_p, test_max_p = min(p3, p4), max(p3, p4)
            
            # Check overlap or gap
            if (test_max_p < base_min_p - gap_thresh) or (test_min_p > base_max_p + gap_thresh):
                # Too far apart along the line
                continue
                
            # It's a match! Merge it.
            current_cluster.append(lines[j])
            used[j] = True
            
            # Update base line to encompass the new line
            all_points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            # Find the two points that are furthest apart (the real endpoints of the merged line)
            max_dist = -1
            best_p1, best_p2 = None, None
            for p_a in all_points:
                for p_b in all_points:
                    d = math.hypot(p_a[0] - p_b[0], p_a[1] - p_b[1])
                    if d > max_dist:
                        max_dist = d
                        best_p1, best_p2 = p_a, p_b
            
            base_line = [best_p1[0], best_p1[1], best_p2[0], best_p2[1]]
                
        # After going through all candidates, if the final merged line is long enough, keep it
        if line_length(base_line) > 30:
            merged_lines.append(base_line)
            
    return merged_lines

def visualize_lines(img_shape, lines, color, thickness=1):
    vis_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        cv2.line(vis_img, (x1, y1), (x2, y2), color, thickness)
    return vis_img

def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Target image '{args.image}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.image))[0]

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Failed to load image '{args.image}'.")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # For speed and robustness, optionally resize down if image is huge
    # (Keeping it original size for now to maintain accuracy)

    print("Step 1: Extracting raw line segments...")
    lines = extract_line_segments(gray)
    print(f" -> Found {len(lines) if lines is not None else 0} raw line segments.")

    # Visualize raw lines
    raw_lines_vis = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line[0])
            cv2.line(raw_lines_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_01_raw_lines.jpg"), raw_lines_vis)

    print("Step 2: Directional Clustering (Horizontal vs Vertical)...")
    h_lines, v_lines = cluster_lines_by_direction(lines)
    print(f" -> {len(h_lines)} Horizontal lines, {len(v_lines)} Vertical lines.")

    # Visualize Clustered Lines
    cluster_vis = img.copy()
    for line in h_lines:
        cv2.line(cluster_vis, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (255, 0, 0), 2) # Blue horizontal
    for line in v_lines:
        cv2.line(cluster_vis, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 255, 0), 2) # Green vertical
    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_02_clustered_lines.jpg"), cluster_vis)

    print("Step 3: Merging collinear broken fragments...")
    merged_h_lines = merge_collinear_lines(h_lines)
    merged_v_lines = merge_collinear_lines(v_lines)
    print(f" -> Merged into {len(merged_h_lines)} Horizontal lines, {len(merged_v_lines)} Vertical lines.")

    # Visualize Merged Lines
    merged_vis = img.copy()
    for line in merged_h_lines:
        cv2.line(merged_vis, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (255, 255, 0), 2) # Cyan horizontal
    for line in merged_v_lines:
        cv2.line(merged_vis, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 255, 255), 2) # Yellow vertical
    cv2.imwrite(os.path.join(args.output_dir, f"{basename}_03_merged_lines.jpg"), merged_vis)

    # TODO: Step 4: 19x19 Grid Graph Reconstruction
    # TODO: Step 5: Corner localization and perspective correction

    print(f"\nProcessing complete. Check the '{args.output_dir}' directory for visualizations.")
    print("Next step: Use these merged lines to compute intersections and reconstruct the 19x19 grid graph.")

    # Display the results using cv2.imshow
    cv2.namedWindow('01 Raw Lines', cv2.WINDOW_NORMAL)
    cv2.namedWindow('02 Clustered Lines', cv2.WINDOW_NORMAL)
    cv2.namedWindow('03 Merged Lines', cv2.WINDOW_NORMAL)
    
    # Resize windows to a reasonable size for viewing
    height, width = img.shape[:2]
    display_width = 800
    display_height = int(height * (display_width / width))
    
    cv2.resizeWindow('01 Raw Lines', display_width, display_height)
    cv2.resizeWindow('02 Clustered Lines', display_width, display_height)
    cv2.resizeWindow('03 Merged Lines', display_width, display_height)
    
    cv2.imshow('01 Raw Lines', raw_lines_vis)
    cv2.imshow('02 Clustered Lines', cluster_vis)
    cv2.imshow('03 Merged Lines', merged_vis)
    
    print("\nImages are displayed in external windows. Press any key to close them...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
