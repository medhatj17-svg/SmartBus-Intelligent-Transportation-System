import subprocess
import threading
import time
import os
import sys
import cv2
from ultralytics import YOLO

from camera_blocked import CameraBlockedDetector
from drowsiness_detector import DrowsinessDetector


# =========================================================
# Settings
# =========================================================

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)

    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        relative_path
    )


MODEL_PATH = resource_path("best.pt")
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.55

CONFIRM_FRAMES = 8
ALERT_COOLDOWN_SECONDS = 5

SEATBELT_MESSAGE = "Please wear your seatbelt."
PHONE_MESSAGE = "Please do not use your phone while driving."
SMOKING_MESSAGE = "Please do not smoke while driving."


# =========================================================
# Driving mode
# =========================================================

driving_mode = True


# =========================================================
# Counters
# =========================================================

no_seatbelt_frames = 0
phone_frames = 0
cigarette_frames = 0

last_seatbelt_alert = 0
last_phone_alert = 0
last_smoking_alert = 0


# =========================================================
# Voice system
# =========================================================

speaking_messages = set()
speech_lock = threading.Lock()


def speak_message(message):

    with speech_lock:

        if message in speaking_messages:
            return

        speaking_messages.add(message)

    def run_speech():

        try:

            subprocess.run(
                ["say", message],
                check=False
            )

        finally:

            with speech_lock:
                speaking_messages.discard(message)

    threading.Thread(
        target=run_speech,
        daemon=True
    ).start()


# =========================================================
# Load model and detectors
# =========================================================

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

print("Loading drowsiness detector...")
drowsiness_detector = DrowsinessDetector()

print("Loading camera blocked detector...")
camera_blocked_detector = CameraBlockedDetector()


# =========================================================
# Open camera
# =========================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    raise SystemExit


print("Driver Monitoring System started.")
print("Press D to turn driving mode ON or OFF.")
print("Press Q to close the program.")


# =========================================================
# Main loop
# =========================================================

while True:

    # -----------------------------------------------------
    # Read camera frame
    # -----------------------------------------------------

    success, frame = cap.read()

    if not success:
        print("Error: Could not read camera frame.")
        break

    frame = cv2.flip(frame, 1)


    # =====================================================
    # YOLO prediction
    # =====================================================

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    # =====================================================
    # Detection states
    # =====================================================

    seatbelt_detected = False
    phone_detected = False
    cigarette_detected = False

    open_eye_detected = False
    closed_eye_detected = False


    # =====================================================
    # Read YOLO detections
    # =====================================================

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        # =================================================
        # Seatbelt
        # =================================================

        if class_name == "Seatbelt":

            seatbelt_detected = True
            box_color = (0, 255, 0)


        # =================================================
        # Phone
        # =================================================

        elif class_name == "Phone":

            phone_detected = True
            box_color = (0, 0, 255)


        # =================================================
        # Cigarette
        # =================================================

        elif class_name == "Cigarette":

            cigarette_detected = True
            box_color = (0, 0, 255)


        # =================================================
        # Open Eye
        # =================================================

        elif class_name == "Open Eye":

            open_eye_detected = True
            box_color = (0, 255, 0)


        # =================================================
        # Closed Eye
        # =================================================

        elif class_name == "Closed Eye":

            closed_eye_detected = True
            box_color = (0, 165, 255)


        else:

            continue


        # =================================================
        # Draw detection box
        # =================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        cv2.putText(
            frame,
            f"{class_name} {confidence:.2f}",
            (
                x1,
                max(y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2
        )


    current_time = time.time()


    # =====================================================
    # Alerts while vehicle is running
    # =====================================================

    if driving_mode:

        # =================================================
        # Seatbelt
        # =================================================

        if seatbelt_detected:

            no_seatbelt_frames = 0

        else:

            no_seatbelt_frames += 1

            if (
                no_seatbelt_frames >= CONFIRM_FRAMES
                and
                current_time - last_seatbelt_alert
                >= ALERT_COOLDOWN_SECONDS
            ):

                speak_message(
                    SEATBELT_MESSAGE
                )

                last_seatbelt_alert = current_time


        # =================================================
        # Phone
        # =================================================

        if phone_detected:

            phone_frames += 1

            if (
                phone_frames >= CONFIRM_FRAMES
                and
                current_time - last_phone_alert
                >= ALERT_COOLDOWN_SECONDS
            ):

                speak_message(
                    PHONE_MESSAGE
                )

                last_phone_alert = current_time

        else:

            phone_frames = 0


        # =================================================
        # Cigarette
        # =================================================

        if cigarette_detected:

            cigarette_frames += 1

            if (
                cigarette_frames >= CONFIRM_FRAMES
                and
                current_time - last_smoking_alert
                >= ALERT_COOLDOWN_SECONDS
            ):

                speak_message(
                    SMOKING_MESSAGE
                )

                last_smoking_alert = current_time

        else:

            cigarette_frames = 0


    # =====================================================
    # Vehicle OFF
    # =====================================================

    else:

        no_seatbelt_frames = 0
        phone_frames = 0
        cigarette_frames = 0


    # =====================================================
    # Vehicle status
    # =====================================================

    if driving_mode:

        mode_text = "Vehicle Status: Running"
        mode_color = (0, 255, 0)

    else:

        mode_text = "Vehicle Status: Off"
        mode_color = (0, 0, 255)


    cv2.putText(
        frame,
        mode_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        mode_color,
        2
    )


    # =====================================================
    # Display alerts
    # =====================================================

    if driving_mode and not seatbelt_detected:

        cv2.putText(
            frame,
            "Seatbelt not detected",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    if driving_mode and phone_detected:

        cv2.putText(
            frame,
            "Phone detected",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    if driving_mode and cigarette_detected:

        cv2.putText(
            frame,
            "Smoking detected",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    # =====================================================
    # Drowsiness detector
    # =====================================================

    frame = drowsiness_detector.process_frame(
        frame,
        open_eye_detected=open_eye_detected,
        closed_eye_detected=closed_eye_detected,
        alerts_enabled=driving_mode
    )


    # =====================================================
    # Camera blocked detector
    # =====================================================

    if driving_mode:

        frame, camera_blocked = (
            camera_blocked_detector.detect(
                frame
            )
        )


    # =====================================================
    # Show final frame
    # =====================================================

    cv2.imshow(
        "Driver Monitoring System",
        frame
    )


    # =====================================================
    # Keyboard controls
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    # Q = Quit
    if key == ord("q"):
        break


    # D = Driving mode ON/OFF
    if key == ord("d"):

        driving_mode = not driving_mode

        if driving_mode:

            print("Driving mode is ON.")

        else:

            print("Driving mode is OFF.")


# =========================================================
# Cleanup
# =========================================================

drowsiness_detector.close()

camera_blocked_detector.close()

cap.release()

cv2.destroyAllWindows()