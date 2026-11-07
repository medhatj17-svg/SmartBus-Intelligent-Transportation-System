import cv2
import numpy as np
import time
import subprocess


class CameraBlockedDetector:

    def __init__(self):

        self.blocked_start = None
        self.camera_blocked = False
        self.speech_process = None

        # إذا الصورة صارت مظلمة جدًا
        self.BRIGHTNESS_THRESHOLD = 25

        # إذا الصورة فقدت تفاصيلها
        self.VARIANCE_THRESHOLD = 15

        # بعد كم ثانية نعتبر الكاميرا مغطاة
        self.BLOCK_TIME = 2.0

        self.ALERT_MESSAGE = "Please do not cover the camera."


    def stop_speech(self):

        if self.speech_process is not None:

            if self.speech_process.poll() is None:
                self.speech_process.terminate()

                try:
                    self.speech_process.wait(timeout=0.5)

                except subprocess.TimeoutExpired:
                    self.speech_process.kill()

            self.speech_process = None


    def play_alert(self):

        if (
            self.speech_process is None
            or self.speech_process.poll() is not None
        ):

            self.speech_process = subprocess.Popen(
                ["say", self.ALERT_MESSAGE]
            )


    def detect(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        brightness = np.mean(gray)
        variance = np.var(gray)

        looks_blocked = (
            brightness < self.BRIGHTNESS_THRESHOLD
            or
            variance < self.VARIANCE_THRESHOLD
        )

        blocked_time = 0.0


        if looks_blocked:

            if self.blocked_start is None:
                self.blocked_start = time.monotonic()

            blocked_time = (
                time.monotonic()
                - self.blocked_start
            )


            # العداد
            cv2.putText(
                frame,
                f"Camera covered: {blocked_time:.1f}s",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


            if blocked_time >= self.BLOCK_TIME:
                self.camera_blocked = True


        else:

            self.blocked_start = None
            self.camera_blocked = False

            self.stop_speech()


        if self.camera_blocked:

            cv2.putText(
                frame,
                "CAMERA BLOCKED!",
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            height, width = frame.shape[:2]

            cv2.rectangle(
                frame,
                (5, 5),
                (width - 5, height - 5),
                (0, 0, 255),
                7
            )

            self.play_alert()


        return frame, self.camera_blocked


    def close(self):

        self.stop_speech()


# =========================================================
# تشغيل الملف لحاله
# =========================================================

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera.")
        raise SystemExit


    detector = CameraBlockedDetector()

    print("Camera Blocked Detector started.")
    print("Press Q to close.")


    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read camera frame.")
            break


        frame = cv2.flip(
            frame,
            1
        )


        frame, blocked = detector.detect(
            frame
        )


        cv2.imshow(
            "Camera Blocked Test",
            frame
        )


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    detector.close()
    cap.release()
    cv2.destroyAllWindows()