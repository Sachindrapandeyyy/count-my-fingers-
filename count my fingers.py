import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  
    thumb_tip = 4

    fingers = []
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Check the thumb
    if hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_tip - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers.count(1)

def label_fingers(frame, hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20] 

    h, w, _ = frame.shape
    for i, tip in enumerate(finger_tips):
        cx, cy = int(hand_landmarks.landmark[tip].x * w), int(hand_landmarks.landmark[tip].y * h)

        cv2.putText(frame, finger_names[i], (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (11, 25, 55), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    finger_counts = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(hand_landmarks)
            finger_counts.append(finger_count)

            h, w, _ = frame.shape
            cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
            cv2.putText(frame, f'Fingers: {finger_count}', (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            label_fingers(frame, hand_landmarks)

    total_fingers = sum(finger_counts)
    cv2.putText(frame, f'Total Fingers: {total_fingers}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    frame = cv2.resize(frame, (700, 700), fx=3, fy=3)

    cv2.imshow('Finger Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
