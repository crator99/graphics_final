# -*- coding: utf-8 -*-

from __future__ import division
import sys
sys.path.append('C:\\users\\윤민성\\appdata\\local\\programs\\python\\python311\\lib\\site-packages')
import cv2
import numpy as np
import os
import sys
import argparse
from math import exp, pow
from augmentingPath import augmentingPath
from pushRelabel import pushRelabel
from boykovKolmogorov import boykovKolmogorov

# np.set_printoptions(threshold=np.inf)
graphCutAlgo = {"ap": augmentingPath, 
                "pr": pushRelabel, 
                "bk": boykovKolmogorov}
SIGMA = 30
# LAMBDA = 1
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (0, 0, 255)

SOURCE, SINK = -2, -1
SF = 10
LOADSEEDS = False
# drawing = False

def Transform(src, dst):
    A = [] #A와 B라는 두 개의 빈 리스트 생성
    B = []
    for i in range(4):
        x, y = src[i][0], src[i][1] #src에 대한 x, y좌표 반환
        u, v = dst[i][0], dst[i][1] #변경해야 하는 목표에 대한 x, y 좌표 반환
        A.append([x, y, 1, 0, 0, 0, -x*u, -y*u])
        A.append([0, 0, 0, x, y, 1, -x*v, -y*v])
        B.append(u)
        B.append(v)
    
    A = np.array(A)
    B = np.array(B)
    
    h = np.linalg.lstsq(A, B, rcond=None)[0] #A * h = B일 때, h 구하기
    
    H = np.append(h, 1).reshape((3, 3)) #h를 3*3 행렬로 변환
    return H

def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def selectROI(image):
    points = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(points)
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            cv2.imshow("Select ROI", image)

    cv2.imshow("Select ROI", image)
    cv2.setMouseCallback("Select ROI", onMouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        raise ValueError("네 점을 찍지 않았습니다")
    
    return np.array(points, dtype="float32")


def crop_image(image, points):
    (tl, tr, br, bl) = points
    points = np.float32([tl, tr, br, bl])
    
    im_tl = [0, 0]
    im_tr = [256, 0]
    im_br = [256, 256]
    im_bl = [0, 256]
    
    image_params = np.float32([im_tl, im_tr, im_br, im_bl])

    M = Transform(points, image_params)
    warped = cv2.warpPerspective(image, M, (256, 256))

    image = cv2.resize(warped, (256, 256))

    return image
    
def plantSeed(image):
    def drawLines(x, y, pixelType):
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
        else:
            color, code = BKGCOLOR, BKGCODE
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def onMouse(event, x, y, flags, pixelType):
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paintSeeds(pixelType):
        print("Planting", pixelType, "seeds")
        global drawing
        drawing = False
        windowname = "Plant " + pixelType + " seeds"
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowname, onMouse, pixelType)
        while (1):
            cv2.imshow(windowname, image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    
    
    seeds = np.zeros(image.shape, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)

    radius = 5
    thickness = -1 # fill the whole circle
    global drawing
    drawing = False

    paintSeeds(OBJ)
    paintSeeds(BKG)
    return seeds, image


# Large when ip - iq < sigma, and small otherwise
def boundaryPenalty(ip, iq):
    bp = 100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return bp

def buildGraph(image):
    V = image.size + 2
    graph = np.zeros((V, V), dtype='int32')
    K = makeNLinks(graph, image)
    seeds, seededImage = plantSeed(image)
    makeTLinks(graph, seeds, K)
    return graph, seededImage

def makeNLinks(graph, image):
    K = -float("inf")
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if i + 1 < r: # pixel below
                y = (i + 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
            if j + 1 < c: # pixel to the right
                y = i * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
    return K



def makeTLinks(graph, seeds, K):
    r, c = seeds.shape

    for i in range(r):
        for j in range(c):
            x = i * c + j
            if seeds[i][j] == OBJCODE:
                # graph[x][source] = K
                graph[SOURCE][x] = K
            elif seeds[i][j] == BKGCODE:
                graph[x][SINK] = K
                # graph[sink][x] = K
            # else:
            #     graph[x][source] = LAMBDA * regionalPenalty(image[i][j], BKG)
            #     graph[x][sink]   = LAMBDA * regionalPenalty(image[i][j], OBJ)



def displayCut(image, cuts):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    return image
    


def imageSegmentation(imagefile, size=(256, 256), algo="ff"):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, size)

    roi = selectROI(image)
    image = crop_image(image, roi)

    image = cv2.resize(image, size)

    graph, seededImage = buildGraph(image)
    cv2.imwrite(pathname + "seeded.jpg", seededImage)

    global SOURCE, SINK
    SOURCE += len(graph) 
    SINK   += len(graph)
    
    cuts = graphCutAlgo[algo](graph, SOURCE, SINK)
    print("cuts:")
    print(cuts)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (256, 256), fx=1, fy=1)
    show_image(image)
    savename = pathname + "cut.jpg"
    cv2.imwrite(savename, image)
    print("Saved image as", savename)
    

def parseArgs():
    def algorithm(string):
        if string in graphCutAlgo:
            return string
        raise argparse.ArgumentTypeError(
            "Algorithm should be one of the following:", graphCutAlgo.keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    parser.add_argument("--size", "-s", 
                        default=30, type=int,
                        help="Defaults to 30x30")
    parser.add_argument("--algo", "-a", default="ap", type=algorithm)
    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()
    imageSegmentation(args.imagefile, (args.size, args.size), args.algo)