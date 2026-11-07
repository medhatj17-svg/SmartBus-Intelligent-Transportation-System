import subprocess
import threading
import time

import cv2
from ultralytics import YOLO


# =========================================================
# الإعدادات
# =========================================================

MODEL_PATH = "best.pt"
CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.55

# عدد الفريمات المتتالية قبل إعطاء التنبيه
NO_SEATBELT_CONFIRM_FRAMES = 10

# المدة بين كل تنبيه صوتي والذي يليه
ALERT_COOLDOWN_SECONDS = 4

SEATBELT_MESSAGE = "Please wear your seatbelt."


# =========================================================
# متغيرات التنبيه
# =========================================================

no_seatbelt_frames = 0
last_alert_time = 0
speech_running = False
speech_lock = threading.Lock()


# =========================================================
# تشغيل الصوت على macOS
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
            # أمر say موجود تلقائيًا على أجهزة Mac
            subprocess.run(
                ["say", message],
                check=False
            )
        finally:
            with speech_lock:
                speech_running = False

    threading.Thread(
        target=run_speech,
        daemon=True
    ).start()


# =========================================================
# تحميل موديل الحزام
# =========================================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    raise SystemExit


print("Seatbelt detector started.")
print("Press Q to close the program.")


# =========================================================
# تشغيل الكاميرا
# =========================================================

while True:
    success, frame = cap.read()

    if not success:
        print("Error: Could not read a camera frame.")
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    seatbelt_detected = False

    # نرسم فقط مربعات الحزام، وليس العين أو الهاتف أو السيجارة
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name == "Seatbelt":
            seatbelt_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Seatbelt {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # =====================================================
    # منطق التنبيه
    # =====================================================

    if seatbelt_detected:
        no_seatbelt_frames = 0

        cv2.putText(
            frame,
            "Seatbelt detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    else:
        no_seatbelt_frames += 1

        cv2.putText(
            frame,
            "Please wear your seatbelt",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        current_time = time.time()

        if (
            no_seatbelt_frames >= NO_SEATBELT_CONFIRM_FRAMES
            and current_time - last_alert_time >= ALERT_COOLDOWN_SECONDS
        ):
            speak_message(SEATBELT_MESSAGE)
            last_alert_time = current_time

    cv2.imshow("Seatbelt Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()