import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from PIL import Image
import sys

import flock_sim

video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# converts OpenCV image to PIL image and then to pyglet texture
# https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
def cv2glet(img,fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
      rows, cols = img.shape
      channels = 1
    else:
      rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)

# Create frame resolution mapping
f_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
f_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Technically flipped
mapping = np.float32([
    [f_width-1,0], [0,0],
    [0,f_height-1], [f_width-1,f_height-1]
])

# Setup marker detector
detector = aruco.ArucoDetector(
    aruco.getPredefinedDictionary(aruco.DICT_6X6_250),
    aruco.DetectorParameters())

# Setup flock sim
particles = flock_sim.setup(f_width, f_height)

def transform_board(frame, pos):
    # Find Marker closest to corners (using projection)
    s = np.sum(pos, axis=1)
    d = np.diff(pos, axis=1).ravel()
    tl = pos[np.argmin(s)]
    br = pos[np.argmax(s)]
    tr = pos[np.argmin(d)]
    bl = pos[np.argmax(d)]

    M = cv2.getPerspectiveTransform(np.float32([tl,tr,br,bl]), mapping)
    return cv2.warpPerspective(frame, M, (f_width, f_height))

def calculate_avoid_points(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV mask
    lower = np.array([0, 30, 60])
    upper = np.array([20, 150, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Filter mask
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask = cv2.erode(mask, None,  iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)

    # Find Contours
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnt:
        return []
    pts = max(cnt, key=cv2.contourArea)[:, 0, :]


    # Calculate Arc-length distance
    deltas = np.diff(pts, axis=0)
    distances = np.insert(np.cumsum(np.hypot(deltas[:, 0], deltas[:, 1])), 0, 0)
    total = distances[-1]

    # Select point along contour
    sample_space = np.linspace(0, total, 20)
    samples = []
    for d in sample_space:
        i = np.searchsorted(distances, d)
        i = len(pts) - 1 if i >= len(pts) else i
        samples.append(tuple(pts[i]))

    return samples

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

@window.event
def on_draw():
    window.clear()
    ret, frame = cap.read()
    
    label = None

    # Get marker corners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        if len(ids) == 4:
            #aruco.drawDetectedMarkers(frame, corners, ids)

            # Transform board to camera
            frame = transform_board(frame, [np.mean(c[0], axis=0) for c in corners])

            # Get points to avoid
            points = calculate_avoid_points(frame)

            for p in points:
                cv2.circle(frame, p, 5, (0, 255, 255), -1)

            # Draw particles
            for pos in flock_sim.step(particles, points):
                frame = cv2.circle(frame, pos, 5, (255, 0, 0), -1)

            # Draw label
            label = pyglet.text.Label(f'{len(particles)} particles left' if len(particles) else 'You won',
                font_name='Times New Roman',
                font_size=24,
                x=WINDOW_WIDTH//2, y= WINDOW_HEIGHT - 50,
                anchor_x='center', anchor_y='top',
                color=(0,0,0)
            )
    
    img = cv2glet(frame, 'BGR')
    img.blit(0, 0, 0)

    if label:
        label.draw()

@window.event
def on_key_press(symbol, modifier):
    if symbol == pyglet.window.key.Q:
        pyglet.app.exit()

pyglet.app.run()
