import cv2
import mediapipe as mp
import numpy as np

# -----------------------
# MediaPipe Hand Setup
# -----------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,          # Track only one hand for drawing
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------
# Canvas and Colors Setup
# -----------------------
canvas = None
pen_color = (0, 0, 255)  # Default red
pen_thickness = 5
eraser_thickness = 50
drawing = False

# Track previous finger position
prev_x, prev_y = 0, 0

# -----------------------
# Video Capture
# -----------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

print("Controls:")
print("Press 'r' for Red pen")
print("Press 'g' for Green pen")
print("Press 'b' for Blue pen")
print("Press 'e' for Eraser")
print("Press 'c' to Clear Canvas")
print("Press 'q' to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Index finger tip landmark
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        h, w, c = frame.shape
        x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

        if prev_x == 0 and prev_y == 0:
            prev_x, prev_y = x, y

        # Draw line on canvas
        cv2.line(canvas, (prev_x, prev_y), (x, y), pen_color, pen_thickness)

        prev_x, prev_y = x, y

        # Optional: Draw hand landmarks for visualization
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # Reset previous positions when no hand is detected
        prev_x, prev_y = 0, 0

    # Overlay canvas on frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("Virtual Drawing Pad", frame)

    # -----------------------
    # Key Controls
    # -----------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('r'):
        pen_color = (0, 0, 255)
        pen_thickness = 5
    elif key == ord('g'):
        pen_color = (0, 255, 0)
        pen_thickness = 5
    elif key == ord('b'):
        pen_color = (255, 0, 0)
        pen_thickness = 5
    elif key == ord('e'):
        pen_color = (0, 0, 0)
        pen_thickness = eraser_thickness

cap.release()
cv2.destroyAllWindows()