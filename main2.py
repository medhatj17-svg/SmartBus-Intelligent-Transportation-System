import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import subprocess
import threading
import time
import os
import sys
import traceback


# =========================================================
# PATHS / LOGGING
# =========================================================

def resource_path(relative_path):
    """Return the correct path both in normal Python and PyInstaller app."""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def get_log_path():
    """Write crash details somewhere writable on macOS."""
    home = os.path.expanduser("~")
    log_dir = os.path.join(home, "Library", "Logs", "SmartBus")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "passenger_monitor.log")


def write_log(message):
    try:
        with open(get_log_path(), "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except Exception:
        pass


# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = resource_path("yolo11n.pt")
CAMERA_ID = 0
BUS_CAPACITY = 30

# المسافة بين خطي العد
LINE_GAP = 80

# حماية من إعادة عد نفس الشخص
COOLDOWN = 1.5


# =========================================================
# GLOBAL DATA
# =========================================================

passengers = 0
tracks = {}
last_count_time = {}

alarm_active = False
alarm_thread = None
cap = None
model = None


# =========================================================
# ALARM
# =========================================================

def alarm_sound():
    while alarm_active:
        try:
            subprocess.run(
                ["/usr/bin/afplay", "/System/Library/Sounds/Glass.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception as e:
            write_log(f"Alarm error: {e}")

        time.sleep(0.3)


def start_alarm():
    global alarm_active, alarm_thread

    if alarm_active:
        return

    alarm_active = True
    alarm_thread = threading.Thread(target=alarm_sound, daemon=True)
    alarm_thread.start()


def stop_alarm():
    global alarm_active
    alarm_active = False


# =========================================================
# TKINTER WINDOW
# =========================================================

root = tk.Tk()
root.title("Bus Passenger Monitor")
root.geometry("1200x700")
root.configure(bg="#0f172a")


# =========================================================
# HEADER
# =========================================================

header = tk.Frame(root, bg="#111827", height=70)
header.pack(fill="x")
header.pack_propagate(False)

title = tk.Label(
    header,
    text="BUS PASSENGER MONITOR",
    font=("Segoe UI", 22, "bold"),
    fg="white",
    bg="#111827",
)
title.pack(side="left", padx=30)

online = tk.Label(
    header,
    text="● SYSTEM STARTING",
    font=("Segoe UI", 12, "bold"),
    fg="#f59e0b",
    bg="#111827",
)
online.pack(side="right", padx=30)


# =========================================================
# MAIN AREA
# =========================================================

main_frame = tk.Frame(root, bg="#0f172a")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)


# =========================================================
# CAMERA PANEL
# =========================================================

camera_frame = tk.Frame(main_frame, bg="#111827")
camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 15))

camera_title = tk.Label(
    camera_frame,
    text="LIVE CAMERA",
    font=("Segoe UI", 14, "bold"),
    fg="white",
    bg="#111827",
)
camera_title.pack(anchor="w", padx=20, pady=15)

camera_label = tk.Label(camera_frame, bg="#020617")
camera_label.pack(fill="both", expand=True, padx=15, pady=(0, 15))


# =========================================================
# DASHBOARD
# =========================================================

dashboard = tk.Frame(main_frame, bg="#111827", width=350)
dashboard.pack(side="right", fill="y")
dashboard.pack_propagate(False)

dashboard_title = tk.Label(
    dashboard,
    text="BUS STATUS",
    font=("Segoe UI", 16, "bold"),
    fg="white",
    bg="#111827",
)
dashboard_title.pack(pady=(25, 30))

passenger_title = tk.Label(
    dashboard,
    text="CURRENT PASSENGERS",
    font=("Segoe UI", 11, "bold"),
    fg="#94a3b8",
    bg="#111827",
)
passenger_title.pack()

passenger_value = tk.Label(
    dashboard,
    text=f"0 / {BUS_CAPACITY}",
    font=("Segoe UI", 38, "bold"),
    fg="white",
    bg="#111827",
)
passenger_value.pack(pady=(5, 30))

available_title = tk.Label(
    dashboard,
    text="AVAILABLE",
    font=("Segoe UI", 11, "bold"),
    fg="#94a3b8",
    bg="#111827",
)
available_title.pack()

available_value = tk.Label(
    dashboard,
    text=str(BUS_CAPACITY),
    font=("Segoe UI", 32, "bold"),
    fg="white",
    bg="#111827",
)
available_value.pack(pady=(5, 30))

status_title = tk.Label(
    dashboard,
    text="STATUS",
    font=("Segoe UI", 11, "bold"),
    fg="#94a3b8",
    bg="#111827",
)
status_title.pack()

status_value = tk.Label(
    dashboard,
    text="NOT FULL",
    font=("Segoe UI", 22, "bold"),
    fg="#22c55e",
    bg="#111827",
)
status_value.pack(pady=15)


# =========================================================
# DASHBOARD UPDATE
# =========================================================

def update_dashboard():
    available = max(BUS_CAPACITY - passengers, 0)

    if passengers > BUS_CAPACITY:
        status = "OVER CAPACITY"
        status_color = "#ef4444"
        start_alarm()
    elif passengers == BUS_CAPACITY:
        status = "FULL"
        status_color = "#f59e0b"
        stop_alarm()
    elif passengers >= BUS_CAPACITY * 0.9:
        status = "ALMOST FULL"
        status_color = "#f59e0b"
        stop_alarm()
    else:
        status = "NOT FULL"
        status_color = "#22c55e"
        stop_alarm()

    passenger_value.config(text=f"{passengers} / {BUS_CAPACITY}")
    available_value.config(text=str(available))
    status_value.config(text=status, fg=status_color)


# =========================================================
# COUNTING
# =========================================================

def count_person(person_id, direction):
    global passengers

    now = time.time()
    last_time = last_count_time.get(person_id, 0)

    if now - last_time < COOLDOWN:
        return

    if direction == "IN":
        passengers += 1
        write_log(f"Person {person_id} ENTERED. Passengers={passengers}")
    elif direction == "OUT":
        passengers = max(passengers - 1, 0)
        write_log(f"Person {person_id} EXITED. Passengers={passengers}")

    last_count_time[person_id] = now


# =========================================================
# INITIALIZE MODEL + CAMERA
# =========================================================

def initialize_system():
    global model, cap

    try:
        write_log(f"Starting. Model path: {MODEL_NAME}")
        write_log(f"Model exists: {os.path.exists(MODEL_NAME)}")

        if not os.path.exists(MODEL_NAME):
            raise FileNotFoundError(f"Model file not found: {MODEL_NAME}")

        model = YOLO(MODEL_NAME)

        # On macOS, AVFoundation is the native camera backend.
        if sys.platform == "darwin":
            cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(CAMERA_ID)

        # Fallback if the native backend did not open the camera.
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(CAMERA_ID)

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open camera. Check macOS System Settings > "
                "Privacy & Security > Camera."
            )

        online.config(text="● SYSTEM ONLINE", fg="#22c55e")
        update_camera()

    except Exception as e:
        online.config(text="● SYSTEM ERROR", fg="#ef4444")
        details = traceback.format_exc()
        write_log(details)
        messagebox.showerror(
            "SmartBus Passenger Monitor",
            f"Could not start the system.\n\n{e}\n\n"
            f"Log file:\n{get_log_path()}",
        )


# =========================================================
# CAMERA LOOP
# =========================================================

def update_camera():
    global cap

    if cap is None or not cap.isOpened():
        return

    ret, frame = cap.read()

    if not ret:
        root.after(30, update_camera)
        return

    height, width = frame.shape[:2]

    line1_x = width // 2 - LINE_GAP // 2
    line2_x = width // 2 + LINE_GAP // 2

    cv2.line(frame, (line1_x, 0), (line1_x, height), (0, 255, 255), 3)
    cv2.line(frame, (line2_x, 0), (line2_x, height), (255, 0, 255), 3)

    cv2.putText(
        frame, "OUTSIDE", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    cv2.putText(
        frame, "DOOR", (width // 2 - 25, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )

    cv2.putText(
        frame, "INSIDE", (width - 120, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )

    try:
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False,
        )

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, person_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                cv2.putText(
                    frame,
                    f"ID {person_id}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                if center_x < line1_x:
                    current_side = "LEFT"
                elif center_x > line2_x:
                    current_side = "RIGHT"
                else:
                    current_side = "ZONE"

                if person_id not in tracks:
                    tracks[person_id] = {
                        "last_side": current_side,
                        "history": [current_side],
                    }
                    continue

                track = tracks[person_id]

                if not track["history"] or track["history"][-1] != current_side:
                    track["history"].append(current_side)

                if len(track["history"]) > 10:
                    track["history"].pop(0)

                history = track["history"]

                if (
                    "LEFT" in history
                    and "ZONE" in history
                    and current_side == "RIGHT"
                ):
                    count_person(person_id, "IN")
                    track["history"] = ["RIGHT"]

                    cv2.putText(
                        frame,
                        "ENTERED +1",
                        (x1, min(y2 + 30, height - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                elif (
                    "RIGHT" in history
                    and "ZONE" in history
                    and current_side == "LEFT"
                ):
                    count_person(person_id, "OUT")
                    track["history"] = ["LEFT"]

                    cv2.putText(
                        frame,
                        "EXITED -1",
                        (x1, min(y2 + 30, height - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                track["last_side"] = current_side

    except Exception:
        write_log(traceback.format_exc())

    update_dashboard()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image.thumbnail((780, 540))

    photo = ImageTk.PhotoImage(image=image)
    camera_label.config(image=photo)
    camera_label.image = photo

    root.after(15, update_camera)


# =========================================================
# CLOSE
# =========================================================

def close_program():
    stop_alarm()

    if cap is not None and cap.isOpened():
        cap.release()

    root.destroy()


root.protocol("WM_DELETE_WINDOW", close_program)


# Start after Tkinter enters its event loop.
root.after(100, initialize_system)
root.mainloop()
