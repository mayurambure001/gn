from ultralytics import YOLO
import cv2

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully")

cap = cv2.VideoCapture(0)  # try 0, if not works try 1

if not cap.isOpened():
    print("ERROR: Webcam not accessible")
    exit()

print("Webcam started. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    results = model(frame)
    frame = results[0].plot()

    cv2.imshow("Live Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
print("Program closed")
