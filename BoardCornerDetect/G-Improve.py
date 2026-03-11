import cv2
import numpy as np
import argparse


def find_board_contour(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best = None

    for c in contours:

        area = cv2.contourArea(c)

        if area < 5000:
            continue

        peri = cv2.arcLength(c,True)

        approx = cv2.approxPolyDP(c,0.02*peri,True)

        if len(approx)==4 and area>max_area:

            best = approx
            max_area = area

    return best


def order_points(pts):

    pts = pts.reshape(4,2)

    s = pts.sum(axis=1)

    diff = np.diff(pts,axis=1)

    rect = np.zeros((4,2),dtype="float32")

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def perspective(img, pts):

    rect = order_points(pts)

    (tl,tr,br,bl) = rect

    width = int(max(
        np.linalg.norm(br-bl),
        np.linalg.norm(tr-tl)
    ))

    height = int(max(
        np.linalg.norm(tr-br),
        np.linalg.norm(tl-bl)
    ))

    dst = np.array([
        [0,0],
        [width-1,0],
        [width-1,height-1],
        [0,height-1]
    ],dtype="float32")

    M = cv2.getPerspectiveTransform(rect,dst)

    warp = cv2.warpPerspective(img,M,(width,height))

    return warp


def projection_lines(board):

    gray = cv2.cvtColor(board,cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

    vx = np.sum(np.abs(sobelx),axis=0)
    vy = np.sum(np.abs(sobely),axis=1)

    vx = (vx - vx.min())/(vx.max()-vx.min())
    vy = (vy - vy.min())/(vy.max()-vy.min())

    cols = np.argsort(vx)[-19:]
    rows = np.argsort(vy)[-19:]

    return sorted(cols),sorted(rows)


def draw_grid(board,cols,rows):

    vis = board.copy()

    for c in cols:
        cv2.line(vis,(c,0),(c,vis.shape[0]),(0,255,0),1)

    for r in rows:
        cv2.line(vis,(0,r),(vis.shape[1],r),(255,0,0),1)

    for c in cols:
        for r in rows:
            cv2.circle(vis,(c,r),3,(0,0,255),-1)

    return vis


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--image",required=True)

    args = parser.parse_args()

    img = cv2.imread(args.image)

    contour = find_board_contour(img)

    if contour is None:
        print("Board not found")
        return

    board = perspective(img,contour)

    cols,rows = projection_lines(board)

    vis = draw_grid(board,cols,rows)

    cv2.imshow("board",board)
    cv2.imshow("grid",vis)

    cv2.waitKey(0)


if __name__=="__main__":
    main()