import subprocess
import threading
import time

import cv2
from ultralytics import YOLO


# =========================================================
# Settings
# =========================================================

MODEL_PATH = "best.pt"
CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.55

PHONE_CONFIRM_FRAMES = 8
ALERT_COOLDOWN_SECONDS = 4

PHONE_MESSAGE = "Please do not use your phone while driving."


# =========================================================
# Alert variables
# =========================================================

phone_frames = 0
last_alert_time = 0
speech_running = False
speech_lock = threading.Lock()


# =========================================================
# Voice Alert (macOS)
# =========================================================

def speak_message(message):
    global speech_running

    with speech_lock:
        if speech_running:
            return

        speech_running = True

    def run_speech():
        global speech_running

        try:
            subprocess.run(["say", message], check=False)
        finally:
            with speech_lock:
                speech_running = False

    threading.Thread(target=run_speech, daemon=True).start()


# =========================================================
# Load model
# =========================================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit


print("Phone detector started.")
print("Press Q to quit.")


# =========================================================
# Camera loop
# =========================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    phone_detected = False

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name == "Phone":

            phone_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"Phone {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    if phone_detected:

        phone_frames += 1

        cv2.putText(
            frame,
            "Phone detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        current_time = time.time()

        if (
            phone_frames >= PHONE_CONFIRM_FRAMES
            and current_time - last_alert_time >= ALERT_COOLDOWN_SECONDS
        ):

            speak_message(PHONE_MESSAGE)
            last_alert_time = current_time

    else:

        phone_frames = 0

        cv2.putText(
            frame,
            "No phone detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Phone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()