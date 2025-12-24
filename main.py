import cv2
import numpy as np
import winsound
import threading
import os
from datetime import datetime
import time

# --- Parameters ---
# HOG/SVM detection is computationally more intensive than simple motion detection.
# We will use a time-based cooldown to prevent excessive processing/saving.
COOLDOWN_SECONDS = 3
SAVE_DIR = "human_captures"
FRAME_WIDTH = 640 # Resizing helps speed up processing

os.makedirs(SAVE_DIR, exist_ok=True)

def play_alarm():
    # Frequency 1000Hz, duration 500ms
    try:
        winsound.Beep(1200, 700) # Slightly different beep for human detection
    except:
        print("Alarm beep failed.")

# 1. Initialize the HOG Descriptor/Detector
# This object is pre-configured to detect humans (pedestrians).
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

last_detection_time = 0

print("Human Detection Active - Press Q to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, int(frame.shape[0] * FRAME_WIDTH / frame.shape[1])))
    
    # 2. Perform Human Detection
    # `winStride`: How much to move the detection window (smaller = slower but more accurate)
    # `padding`: Padding added to the image (helps detect humans near the edges)
    # `scale`: Controls the search window size (smaller = detects farther/smaller objects, but slower)
    (rects, weights) = hog.detectMultiScale(frame, 
                                            winStride=(4, 4),
                                            padding=(8, 8), 
                                            scale=1.05)

    human_detected = False

    # 3. Process Detections
    for (x, y, w, h) in rects:
        # Draw the bounding box around the detected human
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        human_detected = True

    now = time.time()

    # 4. Save and Alarm logic
    if human_detected and (now - last_detection_time) > COOLDOWN_SECONDS:
        last_detection_time = now
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"human_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print("🧍 Human Detected → Image Saved:", filename)
        threading.Thread(target=play_alarm, daemon=True).start()

    # 5. Display Status
    cv2.putText(frame, "STATUS: HUMAN DETECTED" if human_detected else "STATUS: NO HUMAN",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if human_detected else (255, 0, 0), 2)

    cv2.imshow("Human Detector (HOG/SVM) - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
