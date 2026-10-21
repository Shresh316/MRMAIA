
import cv2
cap = cv2.VideoCapture(0)            # 0 = first camera, 1 = second, or a video file path
if not cap.isOpened():               # camera busy, wrong index or no permission
    raise SystemExit("Camera not found")

while True:
    ok, frame = cap.read()           # ok: True if a frame arrived; frame: BGR array (h, w, 3)
    if not ok:
        break
    frame = cv2.flip(frame, 1)       # 1 = mirror left-right, so moving right on screen feels natural

    # ... your processing and drawing goes here ...

    cv2.imshow("Window", frame)
    key = cv2.waitKey(1) & 0xFF      # wait 1 ms; returns the key code, or 255 if none
    if key == ord('q'):              # ord('q') turns the character into its number
        break

cap.release()
cv2.destroyAllWindows()