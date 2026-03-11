import cv2
import numpy as np
import argparse


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------

def preprocess(gray):

    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Unsharp mask
    sharp = cv2.addWeighted(gray,1.5,blur,-0.5,0)

    # CLAHE contrast
    clahe = cv2.createCLAHE(2.0,(8,8))
    enhanced = clahe.apply(sharp)

    return enhanced


# ---------------------------------------------------------
# Tile Shi-Tomasi (avoid corner density bias)
# ---------------------------------------------------------

def detect_corners(gray):

    h,w = gray.shape

    tile = 200

    pts = []

    for y in range(0,h,tile):
        for x in range(0,w,tile):

            patch = gray[y:y+tile,x:x+tile]

            corners = cv2.goodFeaturesToTrack(
                patch,
                maxCorners=80,
                qualityLevel=0.01,
                minDistance=5,
                blockSize=3
            )

            if corners is None:
                continue

            for c in corners:

                px = int(c[0][0] + x)
                py = int(c[0][1] + y)

                pts.append([px,py])

    return np.array(pts)


# ---------------------------------------------------------
# Fit line using PCA
# ---------------------------------------------------------

def fit_line(points):

    mean = np.mean(points,axis=0)

    centered = points - mean

    _,_,vt = np.linalg.svd(centered)

    direction = vt[0]

    return mean,direction


# ---------------------------------------------------------
# RANSAC line detection
# ---------------------------------------------------------

def ransac_lines(points,iterations=200):

    lines = []

    pts = points.copy()

    for _ in range(iterations):

        if len(pts) < 30:
            break

        idx = np.random.choice(len(pts),2,replace=False)

        p1,p2 = pts[idx]

        v = p2 - p1
        v = v/np.linalg.norm(v)

        normal = np.array([-v[1],v[0]])

        d = np.abs((pts-p1)@normal)

        inliers = pts[d<5]

        if len(inliers) > 40:

            mean,dir = fit_line(inliers)

            lines.append((mean,dir))

            mask = d>=5
            pts = pts[mask]

    return lines


# ---------------------------------------------------------
# Cluster lines by direction
# ---------------------------------------------------------

def cluster_directions(lines):

    dirs = np.array([l[1] for l in lines])

    angles = np.arctan2(dirs[:,1],dirs[:,0])

    angles = angles.reshape(-1,1)

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2).fit(angles)

    g1=[]
    g2=[]

    for i,l in enumerate(lines):

        if kmeans.labels_[i]==0:
            g1.append(l)
        else:
            g2.append(l)

    return g1,g2


# ---------------------------------------------------------
# Intersection
# ---------------------------------------------------------

def intersect(l1,l2):

    p1,d1 = l1
    p2,d2 = l2

    A = np.array([d1,-d2]).T

    if abs(np.linalg.det(A)) < 1e-3:
        return None

    t = np.linalg.solve(A,p2-p1)

    return p1 + d1*t[0]


# ---------------------------------------------------------
# Build grid intersections
# ---------------------------------------------------------

def build_grid(h_lines,v_lines):

    pts=[]

    for h in h_lines:
        for v in v_lines:

            p = intersect(h,v)

            if p is not None:
                pts.append(p)

    return np.array(pts)


# ---------------------------------------------------------
# Select 19 evenly spaced lines
# ---------------------------------------------------------

def select_19(lines):

    if len(lines)<=19:
        return lines

    coords = np.array([l[0] for l in lines])

    xs = coords[:,0]

    order = np.argsort(xs)

    lines = [lines[i] for i in order]

    best=None
    best_std=1e9

    for i in range(len(lines)-18):

        subset = lines[i:i+19]

        centers = np.array([l[0][0] for l in subset])

        spacing = np.diff(centers)

        std = np.std(spacing)

        if std < best_std:

            best_std = std
            best = subset

    return best


# ---------------------------------------------------------
# Draw results
# ---------------------------------------------------------

def draw(img,corners,grid):

    vis = img.copy()

    for p in corners:

        cv2.circle(vis,(int(p[0]),int(p[1])),2,(0,255,255),-1)

    for p in grid:

        x,y = int(p[0]),int(p[1])

        if 0<x<img.shape[1] and 0<y<img.shape[0]:

            cv2.circle(vis,(x,y),3,(0,0,255),-1)

    return vis


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image",required=True)

    args = parser.parse_args()

    img = cv2.imread(args.image)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = preprocess(gray)

    corners = detect_corners(gray)

    print("corners:",len(corners))

    lines = ransac_lines(corners)

    print("lines:",len(lines))

    g1,g2 = cluster_directions(lines)

    h_lines = select_19(g1)
    v_lines = select_19(g2)

    grid = build_grid(h_lines,v_lines)

    vis = draw(img,corners,grid)

    cv2.imshow("result",vis)

    cv2.waitKey(0)


if __name__ == "__main__":

    main()