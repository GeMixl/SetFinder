import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from itertools import combinations

print("opencv version: ", cv2.__version__)
img = cv2.imread("./samples/sample_1.jpg")
# convert to grayscale and blur
sep_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sep_gray_blur = cv2.medianBlur(sep_gray, ksize=25)
# threshold image to binary
ret, sep_blur_bin = cv2.threshold(sep_gray_blur, 170, 255, cv2.THRESH_BINARY_INV)
# find contours
contours, hierarchy = cv2.findContours(sep_blur_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# find external contours
external = np.zeros(img.shape)

for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(external, contours, i, 1., 5)

index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)

# Otherwise, initialize empty sorted contour and hierarchy lists
cnts_sort = []
hier_sort = []
cnt_is_card = np.zeros(len(contours),dtype=int)

# Fill empty lists with sorted contour and sorted hierarchy. Now,
# the indices of the contour list still correspond with those of
# the hierarchy list. The hierarchy array can be used to check if
# the contours have parents or not.
for i in index_sort:
    cnts_sort.append(contours[i])
    hier_sort.append(hierarchy[0][i])

size = cv2.contourArea(cnts_sort[1])

peri = cv2.arcLength(cnts_sort[8], True)
approx = cv2.approxPolyDP(cnts_sort[8], 0.075*peri, True)
approx = np.squeeze(approx, axis = 1)

s = np.sum(approx, axis=1)
topLeft = approx[np.argmin(s)]
bottomRight = approx[np.argmax(s)]

d = np.diff(approx, axis = -1)
topRight = approx[np.argmax(d)]
bottomLeft = approx[np.argmin(d)]

# UpperLeft, UpperRight, LowerRight, LowerLeft
approx_sorted = np.array([topLeft, bottomLeft, bottomRight, topRight], np.int32)

# Determine which of the contours are cards by applying the
# following criteria: 1) Smaller area than the maximum card size,
# 2), bigger area than the minimum card size, 3) have no parents,
# and 4) have four corners

boundingBox = []
Card = []
maxWidth = 200
maxHeight = 300

for i in range(len(cnts_sort)):
    size = cv2.contourArea(cnts_sort[i])
    peri = cv2.arcLength(cnts_sort[i], True)
    approx = cv2.approxPolyDP(cnts_sort[i], 0.075 * peri, True)

    if ((size < 1000000) and (size > 250000)
            and (hier_sort[i][3] != -1) and (len(approx) == 4)):
        cnt_is_card[i] = 1
        s = np.sum(approx, axis=2)
        topLeft = approx[np.argmin(s)]
        bottomRight = approx[np.argmax(s)]
        d = np.diff(approx, axis=-1)
        topRight = approx[np.argmax(d)]
        bottomLeft = approx[np.argmin(d)]
        src = np.array([topLeft, bottomLeft, bottomRight, topRight], np.float32)
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        Card.append(cv2.warpPerspective(img, M, (maxWidth, maxHeight)))

# x, y = 3, 4
# figure, axis = plt.subplots(y, x)
# for i in range(y):
#     for j in range(x):
#         axis[i, j].imshow(Card[i*x+j])
#
# plt.savefig("./samples/result_1.jpg")


def detect_shape_in_card(card):
    # Convert the image to grayscale
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    # Apply a binary threshold to the grayscale image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Iterate through the contours
    for contour in contours:
        # Approximate the shape
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # If the approximated contour has 4 vertices, it might be a diamond
        if len(approx) == 4:
            # Calculate the angles to confirm it's a diamond
            angles = []
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % 4][0]
                p3 = approx[(i + 2) % 4][0]
                v1 = p1 - p2
                v2 = p3 - p2
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angles.append(np.degrees(angle))

            # Check if all angles are approximately 90 degrees
            if all(30 < angle < 120 for angle in angles):
                cv2.drawContours(img, [approx], 0, (0, 255, 0), 5)

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)

    # Display the result
    plt.imshow(img_rgb)
    plt.title("Detected Diamond Shapes")
    plt.axis('off')  # Hide the axis
    plt.savefig("./samples/result_2.jpg")


def find_set_from_deck(deck):
    deck = [{'form': 'pille', 'anzahl': 3, 'farbe': 'grun', 'fullung': 'voll'},
            {'form': 'schlange', 'anzahl': 3, 'farbe': 'grun', 'fullung': 'voll'},
            {'form': 'diamant', 'anzahl': 3, 'farbe': 'grun', 'fullung': 'halb'},
            {'form': 'pille', 'anzahl': 2, 'farbe': 'rot', 'fullung': 'voll'},
            {'form': 'pille', 'anzahl': 3, 'farbe': 'lila', 'fullung': 'voll'},
            {'form': 'pille', 'anzahl': 1, 'farbe': 'lila', 'fullung': 'voll'},
            ]

    def check_if_set(cards, feature):
        return len(set([c[feature] for c in cards])) in (1, 3)

    [[i, j, k]
     for i, j, k
     in combinations(deck, 3)
     if check_if_set((i, j, k), 'form') and
     check_if_set((i, j, k), 'anzahl') and
     check_if_set((i, j, k), 'farbe') and
     check_if_set((i, j, k), 'fullung')
     ]

detect_shape_in_card(Card[0])
