import cv2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)  # must match the printed marker
params = cv2.aruco.DetectorParameters()                     # default detection settings
detector = cv2.aruco.ArucoDetector(dictionary, params)      # build the detector object

cap = cv2.VideoCapture(0)                                   # open the webcam

while True:
    ok, frame = cap.read()                                  # grab one frame
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # detection works on grayscale
    corners, ids, rejected = detector.detectMarkers(gray)   # find markers in the frame
    # corners  = list of 4 corner points per marker
    # ids      = array of marker IDs (None if nothing found)
    # rejected = square shapes that looked like markers but failed

    if ids is not None:                                     # only draw if something was found
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)  # outline markers and print their IDs
        print("Found IDs:", ids.flatten())                  # flatten turns [[7],[3]] into [7 3]

    cv2.imshow("ArUco", frame)                              # show the result
    if cv2.waitKey(1) & 0xFF == ord('q'):                   # press q to quit
        break

cap.release()
cv2.destroyAllWindows()