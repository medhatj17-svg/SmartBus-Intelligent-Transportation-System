import cv2
import time
import subprocess


# =========================================================
# Settings
# =========================================================

DROWSINESS_SECONDS = 4.0

CLOSED_CONFIRM_FRAMES = 5
OPEN_CONFIRM_FRAMES = 5

ALERT_MESSAGE = "Wake up! Wake up!"


# =========================================================
# Drowsiness Detector
# =========================================================

class DrowsinessDetector:

    def __init__(self):

        self.closed_frames = 0
        self.open_frames = 0

        self.eyes_closed_start = None

        self.eyes_are_closed = False
        self.alarm_active = False

        self.speech_process = None


    # =====================================================
    # Stop voice
    # =====================================================

    def stop_speech(self):

        if self.speech_process is not None:

            if self.speech_process.poll() is None:

                self.speech_process.terminate()

                try:
                    self.speech_process.wait(timeout=0.5)

                except subprocess.TimeoutExpired:
                    self.speech_process.kill()

        self.speech_process = None


    # =====================================================
    # Play alert
    # =====================================================

    def play_alert(self):

        if (
            self.speech_process is None
            or self.speech_process.poll() is not None
        ):

            self.speech_process = subprocess.Popen(
                ["say", ALERT_MESSAGE]
            )


    # =====================================================
    # Reset
    # =====================================================

    def reset(self):

        self.closed_frames = 0
        self.open_frames = 0

        self.eyes_closed_start = None

        self.eyes_are_closed = False
        self.alarm_active = False

        self.stop_speech()


    # =====================================================
    # Process detections
    # =====================================================

    def process_frame(
        self,
        frame,
        open_eye_detected,
        closed_eye_detected,
        alerts_enabled=True
    ):

        # إذا السيارة طافية
        if not alerts_enabled:

            self.reset()

            return frame


        closed_time = 0.0

        status = "Eyes Not Detected"


        # =================================================
        # Closed Eye
        # =================================================

        if closed_eye_detected:

            self.closed_frames += 1
            self.open_frames = 0


            if self.closed_frames >= CLOSED_CONFIRM_FRAMES:

                if not self.eyes_are_closed:

                    self.eyes_are_closed = True

                    self.eyes_closed_start = time.monotonic()


            if (
                self.eyes_are_closed
                and
                self.eyes_closed_start is not None
            ):

                closed_time = (
                    time.monotonic()
                    - self.eyes_closed_start
                )

                status = "Eyes Closed"


                if closed_time >= DROWSINESS_SECONDS:

                    self.alarm_active = True

                    status = "DROWSINESS DETECTED!"


        # =================================================
        # Open Eye
        # =================================================

        elif open_eye_detected:

            self.open_frames += 1
            self.closed_frames = 0


            if self.open_frames >= OPEN_CONFIRM_FRAMES:

                self.eyes_are_closed = False

                self.eyes_closed_start = None

                self.alarm_active = False

                self.stop_speech()


            status = "Driver Awake"


        # =================================================
        # Eyes not detected
        # =================================================

        else:

            self.closed_frames = 0
            self.open_frames = 0

            status = "Eyes Not Detected"


        # =================================================
        # Alarm
        # =================================================

        if self.alarm_active:

            self.play_alert()


            cv2.putText(
                frame,
                "WAKE UP!",
                (40, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                4
            )


            height, width = frame.shape[:2]


            cv2.rectangle(
                frame,
                (5, 5),
                (width - 5, height - 5),
                (0, 0, 255),
                8
            )


        # =================================================
        # Display
        # =================================================

        cv2.putText(
            frame,
            f"Drowsiness: {status}",
            (30, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Eyes closed: {closed_time:.1f}s",
            (30, frame.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        return frame


    # =====================================================
    # Close
    # =====================================================

    def close(self):

        self.stop_speech()