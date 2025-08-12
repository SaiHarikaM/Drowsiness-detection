import cv2
import time
import sys

print("Starting webcam check...")
sys.stdout.flush()

for i in range(5):
    print(f"Trying camera index {i}...")
    sys.stdout.flush()
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"[SUCCESS] Webcam opened at index {i}")
        sys.stdout.flush()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break
            cv2.imshow("Test Webcam", frame)
            if cv2.waitKey(1) == 27:  # ESC to exit
                break
        cap.release()
        cv2.destroyAllWindows()
        break
    else:
        print(f"[FAIL] Webcam not accessible at index {i}")
        sys.stdout.flush()

print("Done checking webcams. Waiting 5 seconds before exit...")
sys.stdout.flush()
time.sleep(5)

