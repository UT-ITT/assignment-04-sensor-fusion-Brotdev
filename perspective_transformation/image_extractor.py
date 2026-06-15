import cv2
import argparse
import numpy as np
from time import sleep

def parse_resolution(x):
    try:
        w, h = x.lower().split('x')
        return int(w), int(h)
    except:
        raise argparse.ArgumentTypeError('Format must be WIDTHxHEIGHT, e.g. 800x600')

# Parse arguments
parser = argparse.ArgumentParser(prog='Image Extractor')
parser.add_argument('in_file')
parser.add_argument('out_file')
parser.add_argument('resolution', type=parse_resolution)
args = parser.parse_args()

# Setup Input, Working and Output image
img = cv2.imread(args.in_file)
temp = img.copy()
out = None

# Clock-wise corner mapping (starting with top left)
mapping = np.float32([
    [0,0], [args.resolution[0],0],
    args.resolution, [0,args.resolution[1]]
])

# List for point storage
points = []

WINDOW_NAME = 'Image Extractor Window'

cv2.namedWindow(WINDOW_NAME)

def mouse_callback(event, x, y, flags, param):
    global img, temp, out

    # If mouse is not left mouse button or we have enough points
    if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
        return

    # Add point
    points.append((x,y))
    
    match len(points):
        case 4:
            # Warp Perspective of image
            M = cv2.getPerspectiveTransform(np.float32(points), mapping)
            out = cv2.warpPerspective(img, M, args.resolution)
            cv2.imshow(WINDOW_NAME, out)
        case _:
            # Draw point onto image
            temp = cv2.circle(temp, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow(WINDOW_NAME, temp)

cv2.imshow(WINDOW_NAME, img)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

while 1:
    match cv2.waitKey(0):
        case 27:  # Keycode ESC
            temp = img.copy()
            points.clear()
            out = None
            cv2.imshow(WINDOW_NAME, img)
        case 115: # Keycode s
            if out is not None:
                cv2.imwrite(args.out_file, out)
                print(f'Image has been written to {args.out_file}')
            else:
                print('No image to save')
        case 113: # Keycode q
            break
        case _:
            sleep(0.1)
