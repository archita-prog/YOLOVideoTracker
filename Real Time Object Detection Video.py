#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', 'install ultralytics')



# In[2]:


import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO


# In[3]:


model = YOLO('yolov8s.pt')


# In[4]:


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        colorsBGR = [x,y]
        print(colorsBGR)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

CAP = cv2.VideoCapture('vidyolov8.mp4')


# In[5]:


get_ipython().system('jupyter nbconvert --to script tracker.ipynb')


# In[ ]:


import cv2
from ultralytics import YOLO
import numpy as np  
import importlib
import tracker
importlib.reload(tracker)

# Store unique car IDs that entered the polygon
area_c = set()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Load COCO class names
with open("coco.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# Load video
video_path = r"C:\Users\Archita Shrivastava\Downloads\vidyolov8\vidyolov8.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    box_width = 700
    box_height = 180
    center_x = frame_width // 2
    center_y = frame_height // 2

    # Define central polygon (ROI)
    area = [
        [center_x - box_width // 2, center_y - box_height // 2],
        [center_x + box_width // 2, center_y - box_height // 2],
        [center_x + box_width // 2, center_y + box_height // 2],
        [center_x - box_width // 2, center_y + box_height // 2]
    ]

    # Run YOLOv8 detection
    results = model.predict(frame, verbose=False)
    car_boxes = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, class_ids, confidences):
            class_name = CLASS_NAMES[int(cls_id)]

            # Keep only cars
            if class_name == "car":
                x1, y1, x2, y2 = map(int, box)
                car_boxes.append([x1, y1, x2, y2])

    # Track cars using your tracker
    tracked_cars = tracker.update(car_boxes)

    for bbox in tracked_cars:
        x1, y1, x2, y2, car_id = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Check if center of the car is inside the polygon
        if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
            # Draw center dot
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f'Car ID: {car_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

            # Add to set of counted cars
            area_c.add(car_id)

    # Draw the ROI
    cv2.polylines(frame, [np.array(area, np.int32)], True, (0, 0, 255), 3)

    # Show count of unique cars inside
    count = len(area_c)
    cv2.putText(frame, f"Cars in area: {count}", (30, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)

    # Display
    cv2.imshow("Car Detection in Region", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




