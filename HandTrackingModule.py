import cv2
import mediapipe as mp
import time

class Hand_Detector():
    def __init__(self, mode=False, max_num_hands = 2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence)

    def find_hands(self, frame, draw=True):
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        self.results = self.hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return image

    
    def find_position(self, image, hand_id=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            hand_ = self.results.multi_hand_landmarks[hand_id]
            for id, landmark in enumerate(hand_.landmark):
                h, w, _ = image.shape
                pos_x, pos_y = int(landmark.x * w), int(landmark.y * h)
                
                landmark_list.append([id, pos_x, pos_y])

                if draw:
                    cv2.circle(image, (pos_x, pos_y), 7, (255,0, 255), cv2.FILLED)
        
        return landmark_list

def main():
    cap = cv2.VideoCapture(0)
    detector = Hand_Detector(min_detection_confidence=0.1)
    
    prev_time = 0
    current_time = 0

    # with mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        image = detector.find_hands(frame)
        landmark_list = detector.find_position(image)
        if len(landmark_list) != 0:
            print(landmark_list[4])
        current_time = time.time()
        fps = 1/(current_time - prev_time)
        prev_time = current_time

        cv2.putText(image, str(int(fps)), (10, 78), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("bruh", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()