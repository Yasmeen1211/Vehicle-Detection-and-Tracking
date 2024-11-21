import cv2
import imutils
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the COCO class names file, which includes vehicles
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Specify layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Vehicle types to detect
vehicle_types = ["car", "motorbike", "bus", "truck"]

# Select input source
input_type = input("Enter 'video' for video input or 'image' for image input: ").strip().lower()
if input_type == "video":
    input_path = input("Enter the path to the video file (or 0 for webcam): ")
    input_path = 0 if input_path == "0" else input_path
    cap = cv2.VideoCapture(input_path)
elif input_type == "image":
    input_path = input("Enter the path to the image file: ")
    cap = None
    img = cv2.imread(input_path)
    if img is None:
        print("Error: Unable to load image.")
        exit()
else:
    print("Invalid input type.")
    exit()

while True:
    # Read frame from video or use the single image
    if cap:
        ret, img = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
    else:
        # Process single image only once
        if img is None:
            break

    # Resize frame or image
    img = imutils.resize(img, width=800)
    height, width, channels = img.shape

    # Prepare image for YOLO
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    # Dictionary to count each vehicle type
    vehicle_count = {v_type: 0 for v_type in vehicle_types}

    # Process each detection
    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if detected object is one of our vehicle types
            if confidence > 0.5 and classes[class_id] in vehicle_types:
                vehicle_type = classes[class_id]
                vehicle_count[vehicle_type] += 1

                # Get coordinates of bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box and label
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{vehicle_type}: {int(confidence * 100)}%"
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display vehicle counts in the console
    print("Vehicle counts:", vehicle_count)

    # Display the image with detections
    cv2.imshow("Vehicle Detection", img)

    # If it's a single image, break after one display
    if not cap:
        cv2.waitKey(0)
        break

    # Exit on pressing 'ESC'
    if cv2.waitKey(33) == 27:
        break

# Release resources
if cap:
    cap.release()
cv2.destroyAllWindows()
