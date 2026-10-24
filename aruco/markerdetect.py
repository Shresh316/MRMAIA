import math
import cv2
import numpy as np

MARKER_LENGTH = 0.05      # printed black-square side in METRES. Measure yours and change this.

# --- 1. One detector for each dictionary ---------------------------------
DICTS = {
    "4x4": cv2.aruco.DICT_4X4_50,     # 4x4 grid, IDs 0..49
    "5x5": cv2.aruco.DICT_5X5_50,
    "6x6": cv2.aruco.DICT_6X6_50,
}
detectors = {}
for name, dict_id in DICTS.items():
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    detectors[name] = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

# --- 2. Marker corners in 3D (centre at origin, flat in z=0) ---------------
h = MARKER_LENGTH / 2
obj_points = np.array([[-h,  h, 0], [ h,  h, 0],
                       [ h, -h, 0], [-h, -h, 0]], dtype=np.float32)

# --- 3. Rotation matrix -> roll, pitch, yaw in degrees ---------------------
def rotation_to_euler(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0
    return [math.degrees(a) for a in (roll, pitch, yaw)]

# --- 4. Open camera and read one frame to learn its size -------------------
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
if not ok:
    raise SystemExit("Camera not found")
H, W = frame.shape[:2]                 # height, width

# --- 5. Camera matrix: use calibration file if you have one ----------------
try:
    data = np.load("calibration.npz")  # saved earlier by cv2.calibrateCamera
    K, dist = data["K"], data["dist"]
except FileNotFoundError:              # otherwise a rough guess (depth will be approximate)
    f = 0.9 * W
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1))

# --- 6. Main loop ----------------------------------------------------------
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for name, detector in detectors.items():              # try 4x4, then 5x5, then 6x6
        corners, ids, _ = detector.detectMarkers(gray)    # corners: list of (1,4,2) arrays
        if ids is None:                                   # nothing found for this dictionary
            continue
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)  # green outline + ID

        for c, marker_id in zip(corners, ids.flatten()):  # one marker at a time
            img_points = c.reshape(4, 2).astype(np.float32)   # (1,4,2) -> (4,2)
            found, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist,
                                             flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not found:
                continue

            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, MARKER_LENGTH * 0.5)
            # drawFrameAxes: X axis red, Y green, Z blue, pointing out of the marker

            R, _ = cv2.Rodrigues(rvec)                    # rotation vector -> 3x3 matrix
            roll, pitch, yaw = rotation_to_euler(R)
            x, y, z = tvec.flatten() * 100                # metres -> centimetres
            dist_total = float(np.linalg.norm(tvec)) * 100

            org = tuple(img_points[0].astype(int))        # text starts near top-left corner
            lines = [f"{name} ID {marker_id}",
                     f"Depth Z: {z:.1f} cm  Dist: {dist_total:.1f} cm",
                     f"X:{x:.1f} Y:{y:.1f} cm",
                     f"R:{roll:.0f} P:{pitch:.0f} Y:{yaw:.0f} deg"]
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (org[0], org[1] - 10 - 18 * (len(lines) - 1 - i)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("ArUco Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
