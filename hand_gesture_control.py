import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math

# ------------------------------
# Create Hand Landmarker
# ------------------------------
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.HandLandmarker.create_from_options(options)

# ------------------------------
# Hand Connections
# ------------------------------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ------------------------------
# Screen Size
# ------------------------------
screen_width, screen_height = pyautogui.size()

pyautogui.FAILSAFE = True

# Right Click Flag
right_clicked = False
left_clicked=False

# ------------------------------
# Start Webcam
# ------------------------------
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    h, w, _ = frame.shape

    if result.hand_landmarks:

     for hand in result.hand_landmarks:

        points = []

        # Draw landmarks
        for landmark in hand:
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Draw hand skeleton
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 0, 0), 2)

        # -------------------------
        # Finger Tips
        # -------------------------
        index_tip = hand[8]
        thumb_tip = hand[4]
        pinky_tip = hand[20]

        ix = int(index_tip.x * w)
        iy = int(index_tip.y * h)

        tx = int(thumb_tip.x * w)
        ty = int(thumb_tip.y * h)

        px = int(pinky_tip.x * w)
        py = int(pinky_tip.y * h)

        cv2.circle(frame, (ix, iy), 10, (0, 0, 255), -1)
        cv2.circle(frame, (tx, ty), 10, (255, 0, 255), -1)
        cv2.circle(frame, (px, py), 10, (255, 255, 0), -1)

        # -------------------------
        # Move Mouse
        # -------------------------
        mouse_x = int(index_tip.x * screen_width)
        mouse_y = int(index_tip.y * screen_height)

        pyautogui.moveTo(mouse_x, mouse_y)

        # -------------------------
        # Distances
        # -------------------------
        thumb_index = math.hypot(ix - tx, iy - ty)
        thumb_pinky = math.hypot(tx - px, ty - py)

        # -------------------------
        # RIGHT CLICK
        # Thumb + Index
        # -------------------------
        if thumb_index < 40:

            cv2.putText(frame, "RIGHT CLICK", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

            if not right_clicked:
                pyautogui.rightClick()
                right_clicked = True

        else:
            right_clicked = False
            left_clicked = False

        # -------------------------
        # LEFT CLICK
        # Index + Pinky
        # -------------------------
        if thumb_pinky < 40:

            cv2.putText(frame, "LEFT CLICK", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

            if not left_clicked:
                pyautogui.leftClick()
                left_clicked = True

    else:
        left_clicked = False
    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()