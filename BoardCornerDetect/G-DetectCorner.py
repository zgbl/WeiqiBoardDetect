import cv2
import numpy as np
import argparse
import math
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="./output.jpg")
    return parser.parse_args()


# ------------------------------------------------------------
# Step 1: Edge Detection
# ------------------------------------------------------------

def detect_edges(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    return edges


# ------------------------------------------------------------
# Step 2: Hough Infinite Lines
# ------------------------------------------------------------

def detect_lines(edges):

    lines = cv2.HoughLines(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=150
    )

    result = []

    if lines is not None:
        for l in lines:
            rho, theta = l[0]
            result.append((rho, theta))

    return result


# ------------------------------------------------------------
# Step 3: Split into 2 angle clusters
# ------------------------------------------------------------

def cluster_angles(lines):

    group1 = []
    group2 = []

    if not lines:
        return group1, group2

    base = lines[0][1]

    for rho, theta in lines:

        diff = abs(theta - base)

        diff = min(diff, np.pi - diff)

        if diff < np.pi/4:
            group1.append((rho, theta))
        else:
            group2.append((rho, theta))

    return group1, group2


# ------------------------------------------------------------
# Step 4: Cluster by rho (distance between parallel lines)
# ------------------------------------------------------------

def cluster_rho(lines, tolerance=15):

    lines = sorted(lines, key=lambda x: x[0])

    clusters = []

    for rho, theta in lines:

        placed = False

        for c in clusters:

            if abs(c[0][0] - rho) < tolerance:

                c.append((rho, theta))
                placed = True
                break

        if not placed:
            clusters.append([(rho, theta)])

    # average cluster
    result = []

    for c in clusters:

        rho_avg = np.mean([x[0] for x in c])
        theta_avg = np.mean([x[1] for x in c])

        result.append((rho_avg, theta_avg))

    return result


# ------------------------------------------------------------
# Step 5: pick 19 most evenly spaced lines
# ------------------------------------------------------------

def select_19(lines):

    if len(lines) <= 19:
        return lines

    rhos = np.array([l[0] for l in lines])

    best = None
    best_score = 1e9

    for i in range(len(lines)-18):

        subset = rhos[i:i+19]

        spacing = np.diff(subset)

        score = np.std(spacing)

        if score < best_score:

            best_score = score
            best = lines[i:i+19]

    return best


# ------------------------------------------------------------
# Step 6: intersection
# ------------------------------------------------------------

def intersect(l1, l2):

    rho1, th1 = l1
    rho2, th2 = l2

    A = np.array([
        [np.cos(th1), np.sin(th1)],
        [np.cos(th2), np.sin(th2)]
    ])

    b = np.array([rho1, rho2])

    x, y = np.linalg.solve(A, b)

    return int(x), int(y)


# ------------------------------------------------------------
# Step 7: draw results
# ------------------------------------------------------------

def draw_lines(img, lines, color):

    h, w = img.shape[:2]

    for rho, theta in lines:

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*rho
        y0 = b*rho

        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))

        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))

        cv2.line(img, (x1,y1), (x2,y2), color, 2)


def main():

    args = parse_args()

    img = cv2.imread(args.image)

    edges = detect_edges(img)

    lines = detect_lines(edges)

    g1, g2 = cluster_angles(lines)

    g1 = cluster_rho(g1)
    g2 = cluster_rho(g2)

    g1 = select_19(g1)
    g2 = select_19(g2)

    vis = img.copy()

    draw_lines(vis, g1, (0,255,0))
    draw_lines(vis, g2, (255,0,0))

    # intersections
    for l1 in g1:
        for l2 in g2:

            try:
                x,y = intersect(l1,l2)

                if 0 < x < img.shape[1] and 0 < y < img.shape[0]:

                    cv2.circle(vis,(x,y),3,(0,0,255),-1)

            except:
                pass

    cv2.imwrite(args.output, vis)
    print("Saved to", args.output)

    cv2.imshow("result", vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()