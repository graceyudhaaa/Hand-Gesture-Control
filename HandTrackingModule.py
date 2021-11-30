import cv2
import math
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
        self.tip_ids = [4, 8, 12, 16, 20]

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
        x_list = []
        y_list = []
        bbox = []
        self.landmark_list = []
        if self.results.multi_hand_landmarks:
            hand_ = self.results.multi_hand_landmarks[hand_id]
            for id, landmark in enumerate(hand_.landmark):
                h, w, _ = image.shape
                pos_x, pos_y = int(landmark.x * w), int(landmark.y * h)
                
                x_list.append(pos_x)
                y_list.append(pos_y)
                self.landmark_list.append([id, pos_x, pos_y])
                
                if draw:
                    cv2.circle(image, (pos_x, pos_y), 7, (255,0, 255), cv2.FILLED)

            x_min, x_max, y_min, y_max = min(x_list), max(x_list), min(y_list), max(y_list)
            bbox = x_min, y_min, x_max, y_max

            if draw:
                cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
        
        return self.landmark_list, bbox

    def fingers_up(self):
        fingers = []
        # Thumb
        if self.landmark_list[self.tip_ids[0]][1] > self.landmark_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.landmark_list[self.tip_ids[id]][2] < self.landmark_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def find_distance(self, p1, p2, image, draw=True,r=15, t=3):
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
        if draw:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(image, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(image, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
 
        return length, image, [x1, y1, x2, y2, cx, cy]

def main():
    cap = cv2.VideoCapture(0)
    detector = Hand_Detector(min_detection_confidence=0.1)
    
    prev_time = 0
    current_time = 0

    # with mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1) as hands:
    while cap.isOpened():
        _, frame = cap.read()
        image = detector.find_hands(frame)
        landmark_list, bbox = detector.find_position(image)
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