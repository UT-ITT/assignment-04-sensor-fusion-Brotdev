import cv2
import cv2.aruco as aruco
import numpy as np
import pyglet
from PIL import Image
from DIPPID import SensorUDP
import sys

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

# Global vars for passthrough
m_pos = None
m_vel = np.array([0.0,0.0])
p_pred = None
alpha = 0.0
ACCL_GAIN = 100000
"""
Depending on the chosen weights (alpha & ACCL_GAIN) the prediction results
vastly differ. If the ACCL_GAIN is set to low, then the prediction "cursor"
barely moves as expected. On the other hand setting it to high causes the
cursor to rapidly jump and sometimes leave the plane.
The weight (alpha) controls how fast/slow the cursor responds to
marker movement. Low alpha values cause the prediction to "lag" behind
due to the high camera latency, while high values also cause jittering and
unpredictable movement.
With a correct values, we can strike a good balance between latency and
accuracy.
"""

# Setup marker detector
detector = aruco.ArucoDetector(
    aruco.getPredefinedDictionary(aruco.DICT_6X6_250),
    aruco.DetectorParameters())

# use UPD (via WiFi) for communication
PORT = 5700
CAPTURE_TIMEOUT = 10

sensor = SensorUDP(PORT)

def transform_board(frame, pos):
    # Find Marker closest to corners (using projection)
    s = np.sum(pos, axis=1)
    d = np.diff(pos, axis=1).ravel()
    tl = pos[np.argmin(s)]
    br = pos[np.argmax(s)]
    tr = pos[np.argmin(d)]
    bl = pos[np.argmax(d)]

    M = cv2.getPerspectiveTransform(np.float32([tl,tr,br,bl]), mapping)
    return cv2.warpPerspective(frame, M, (f_width, f_height)), M

def predict_pos(frame, m_pos):
    global p_pred, m_vel

    if p_pred is None:
        p_pred = m_pos

    print(m_vel)

    # Draw prediction dot
    p_pred += m_vel * 1/60.0
    p_pred = alpha * p_pred + (1 - alpha) * m_pos
    frame = cv2.circle(frame, [int(p) for p in p_pred], 10, (0, 255, 0), -1)

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)

@window.event
def on_draw():
    global p_pred, m_vel, m_pos, alpha
    window.clear()
    ret, frame = cap.read()

    label = None

    # Get marker corners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        label = pyglet.text.Label(f'Alpha: {alpha}',
            font_name='Times New Roman',
            font_size=24,
            x=WINDOW_WIDTH//2, y= WINDOW_HEIGHT - 50,
            anchor_x='center', anchor_y='top',
            color=(0,0,0)
        )

        markers = [np.mean(c[0], axis=0) for i, c in enumerate(corners) if ids[i] != [5]]
        if len(markers) == 4:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Transform board to camera
            frame, M = transform_board(frame, markers)

            # Find marker index
            i = np.where(ids == 5)[0]
            i = i[0] if i.size > 0 else None

            if i is not None:
                # Draw marker pos
                base_m_pos = np.array([[np.mean(corners[i][0], axis=0)]])
                m_pos = cv2.perspectiveTransform(base_m_pos, M)[0,0,:]
                frame = cv2.circle(frame, [int(p) for p in m_pos], 10, (0, 0, 255), -1)

                # Draw marker prediction
                predict_pos(frame, np.array(m_pos))
            else:
                p_pred = None
                m_vel = np.array([0.0,0.0])

    img = cv2glet(frame, 'BGR')
    img.blit(0, 0, 0)

    if label is not None:
        label.draw()

def update(dt):
    global m_vel
    accl = sensor.get_value('accelerometer')

    if accl is not None:
        accl = np.array([accl['x'], accl['z']]) * ACCL_GAIN
        m_vel = accl * dt
    else:
        m_vel = np.array([0.0,0.0])

@window.event
def on_key_press(symbol, modifier):
    global alpha
    match symbol:
        case pyglet.window.key.UP:
            alpha = min(1.0, alpha + 0.01)
            print(alpha)
        case pyglet.window.key.RIGHT:
            alpha = min(1.0, alpha + 0.1)
            print(alpha)
        case pyglet.window.key.DOWN:
            alpha = max(0.0, alpha - 0.01)
            print(alpha)
        case pyglet.window.key.LEFT:
            alpha = max(0.0, alpha - 0.1)
            print(alpha)
        case pyglet.window.key.Q:
            pyglet.app.exit()

def reset(pressed):
    global p_pred, m_pos
    if pressed != 1:
        return

    if m_pos is not None:
        p_pred = m_pos
        print("Prediction reset")
    else:
        print("Prediction not reset - no marker position")

pyglet.clock.schedule_interval(update, 1/60.0)
sensor.register_callback('button_1', reset)

pyglet.app.run()