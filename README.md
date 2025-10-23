# 🚗 Vehicle Detection & Tracking (YOLOv8)

This project implements a **real-time vehicle detection, tracking, and counting system** using the **YOLOv8 deep learning model**.  
It can process both **live webcam feeds** and **recorded video files**, enabling you to identify, track, and count vehicles efficiently — all with dynamic visualization overlays.

---

## 🧠 Overview

The system leverages **Ultralytics YOLOv8**, one of the most advanced real-time object detection architectures, to detect multiple vehicle types (cars, trucks, buses, motorcycles, etc.) in video streams.  
It then applies a **tracking algorithm** to assign unique IDs to each detected vehicle, ensuring consistent identification across frames.  
Additionally, it performs **vehicle counting** when objects cross predefined zones, making it ideal for **traffic analytics**, **smart surveillance**, and **transportation research**.

---

## ✨ Features

- 🚘 **Real-Time Vehicle Detection** → Detects multiple classes of vehicles using YOLOv8 with high accuracy.  
- 🎯 **Object Tracking** → Assigns unique and persistent IDs to vehicles across video frames.  
- 🔢 **Vehicle Counting** → Counts vehicles crossing a user-defined line or region of interest (ROI).  
- 📹 **Video Source Support** → Compatible with **live webcam feeds**, **MP4**, **AVI**, and other video formats.  
- ⚙️ **Optimized Performance** → Adjustable detection confidence, frame size, and tracking parameters for improved speed and accuracy.  
- 🖼️ **Live Visualization** → Displays bounding boxes, class labels, tracking IDs, and live vehicle counts in real-time.  
- 📊 **Modular Design** → Easy to extend for additional analytics like speed detection or density estimation.  

---

## 🧰 Tech Stack

- **Model:** [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics  
- **Frameworks/Libraries:**  
  - Python 3.8+  
  - OpenCV (cv2)  
  - NumPy  
  - Ultralytics YOLO  
- **Input Sources:**  
  - Live webcam feed (`cv2.VideoCapture(0)`)  
  - Pre-recorded video files (`.mp4`, `.avi`, etc.)  

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vehicle-detection-tracking-yolov8.git
   cd vehicle-detection-tracking-yolov8
   
2. **Install dependencies**
pip install -r requirements.txt

3. **Run the program**
   python vehicle_tracking.py --source path/to/video.mp4
4.  **for live webcam feed:**
    python vehicle_tracking.py --source 0
---
**Use Cases**

Traffic management and congestion analysis
Smart city monitoring systems
Vehicle flow analytics for road design
Toll plaza automation and entry-exit counting systems

---

**Future Improvements**

 Multi-zone counting → Track vehicles across multiple predefined areas
 Speed estimation → Approximate vehicle speed using frame rate and distance calibration
 Data export → Export vehicle logs (time, ID, type, speed) to CSV/Excel for analytics
 Direction detection → Identify entry/exit directions for advanced reporting
 Integration with dashboards → Build real-time dashboards using Streamlit or Dash

 ---





