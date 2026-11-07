import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("CAMERA DID NOT OPEN")
    raise SystemExit

print("CAMERA OPENED")

while True:
    success, frame = cap.read()

    if not success:
        print("COULD NOT READ FRAME")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()