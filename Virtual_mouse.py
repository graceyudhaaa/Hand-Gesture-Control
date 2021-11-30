from math import degrees
import cv2
import numpy as np
from numpy.lib.type_check import imag
import HandTrackingModule as htm
import time
import autopy


cap = cv2.VideoCapture(0)

cam_width, cam_height = 640, 480
frame_reduction = 100
screen_width, screen_height = autopy.screen.size()

smoothening = 3
prev_loc_x, prev_loc_y = 0, 0
current_loc_x, current_loc_y = 0, 0

cap.set(3, cam_width)
cap.set(4, cam_height)

prev_time = 0
current_time = 0

detector = htm.Hand_Detector(max_num_hands=1, min_detection_confidence=0.1)

while cap.isOpened():
    _, frame = cap.read()

    # mencari landmark di telapak tangan
    image = detector.find_hands(frame)
    landmark_list, bbox = detector.find_position(image)

    # mencari ujung jari tengah dan telunjuk
    if len(landmark_list) != 0:
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]

        # mengecek jari apa yang sedang terangkat
        fingers = detector.fingers_up()
        cv2.rectangle(
            image,
            (frame_reduction, frame_reduction),
            (cam_width - frame_reduction,
             cam_height - frame_reduction),
            (255, 0, 255), 2)

        # jika hanya telnjuk yang terangkat maka sedang dalam mode gerak
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frame_reduction, cam_width -
                           frame_reduction), (0, screen_width))
            y3 = np.interp(y1, (frame_reduction, cam_height -
                           frame_reduction), (0, screen_height))

            # memperhalus
            current_loc_x = prev_loc_x + (x3 - prev_loc_x) / smoothening
            current_loc_y = prev_loc_y + (y3 - prev_loc_y) / smoothening

            # menggerakan mouse
            # autopy.mouse.move(screen_width - x3, y3)
            autopy.mouse.move(current_loc_x, current_loc_y)
            cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            prev_loc_x, prev_loc_y = current_loc_x, current_loc_y

        # clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, image, line_info = detector.find_distance(8, 12, image)
            if length < 30:
                cv2.circle(
                    image, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, str(int(fps)), (10, 78),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow("bruhee", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
