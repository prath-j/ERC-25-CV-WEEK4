import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Configure the hand detector
hands = mp_hands.Hands(
    static_image_mode=False,      # Use video feed (False)
    max_num_hands=2,              # Detect up to 2 hands
    min_detection_confidence=0.7, # Confidence threshold for detection
    min_tracking_confidence=0.7   # Confidence threshold for tracking
)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks and connections
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
    
    # Display the frame
    cv2.imshow("Hand Tracking", frame)
    
    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
