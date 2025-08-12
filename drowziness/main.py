import cv2
import mediapipe as mp
import threading
import winsound
import time

EAR_THRESHOLD = 0.25
FRAME_CHECK = 20

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Indices for eye landmarks in Mediapipe Face Mesh
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]  # 6 points for better EAR calc
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

frame_counter = 0
alert_playing = False
alert_thread = None
alert_lock = threading.Lock()

def calculate_ear(landmarks, eye_indices, shape):
    h, w = shape
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]

    # Compute euclidean distances
    def dist(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

    A = dist(points[1], points[5])
    B = dist(points[2], points[4])
    C = dist(points[0], points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def beep_alert():
    global alert_playing
    while True:
        with alert_lock:
            if not alert_playing:
                break
        winsound.Beep(1000, 500)  # beep 1000 Hz for 0.5 sec
        time.sleep(0.1)  # short pause between beeps

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            landmarks = face.landmark
            h, w, _ = frame.shape

            left_ear = calculate_ear(landmarks, LEFT_EYE_IDX, (h, w))
            right_ear = calculate_ear(landmarks, RIGHT_EYE_IDX, (h, w))
            ear = (left_ear + right_ear) / 2.0

            # Draw eye landmarks
            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                pt = landmarks[idx]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 0), -1)

            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= FRAME_CHECK:
                    if not alert_playing:
                        with alert_lock:
                            alert_playing = True
                        alert_thread = threading.Thread(target=beep_alert, daemon=True)
                        alert_thread.start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                frame_counter = 0
                with alert_lock:
                    alert_playing = False

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()


