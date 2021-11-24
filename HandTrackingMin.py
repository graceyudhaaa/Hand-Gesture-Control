import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

prev_time = 0
current_time = 0

with mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # if results.multi_hand_landmarks:
        #     for hand_landmarks in results.multi_hand_landmarks:
        #         for id, landmark in enumerate(hand_landmarks.landmark):
        #             h, w, c = image.shape
        #             cx, cy = int(landmark.x * w), int(landmark.y * h)
        #             print(id, cx, cy)

        #         mp_drawing.draw_landmarks(image, landmark, mp_hands.HAND_CONNECTIONS)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _ = image.shape
                    pos_x, pos_y = int(landmark.x * w), int(landmark.y * h)
                    
                    print(id, pos_x, pos_y)

                    if id == 8:
                        cv2.circle(image, (pos_x, pos_y), 25, (255,0, 255), cv2.FILLED)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        current_time = time.time()
        fps = 1/(current_time - prev_time)
        prev_time = current_time

        cv2.putText(image, str(int(fps)), (10, 78), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("bruh", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

