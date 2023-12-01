import os
import numpy as np
import cv2
from utils import get_empty_or_not, get_parking_spot_bounding_boxes, calc_diff


video_path = './data/parking_1920_1080.mp4'
mask_path = './data/mask_1920_1080.png'
cap = cv2.VideoCapture(video_path)
mask = cv2.imread(mask_path, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spot_bounding_boxes(connected_components)

spots_status = [None for i in range(len(spots))]
diffs = [1. for i in range(len(spots))]
step = 35
frame_number = 0
previous_frame = None
arr_ = []

ret, frame = cap.read()
while ret:
    if frame_number % step == 0:
        for spot_idx, spot in enumerate(spots):
            x, y, w, h = spot
            spot_image = frame[y:y+h, x:x+w, :]
            if previous_frame is not None:
                diffs[spot_idx] = calc_diff(spot_image, previous_frame[y:y + h, x:x + w, :])
        arr_ = [j for j in np.argsort(diffs)[::-1] if diffs[j] / np.amax(diffs)>= 0.4]
        for idx in arr_:
            x, y, w, h = spots[idx]
            spots_status[idx] = get_empty_or_not(frame[y:y+h, x:x+w, :])

        previous_frame = frame.copy()

    for spot_idx, spot in enumerate(spots):
        x, y, w, h = spot
        if spots_status[spot_idx]:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Available spots: {sum(spots_status)}/{len(spots)}", (100,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()
    frame_number += 1

cap.release()
cv2.destroyAllWindows()
