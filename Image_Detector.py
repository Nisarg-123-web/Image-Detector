import cv2
import os
import time
from ultralytics import YOLO
import numpy as np

video_file = "C:/python programs/video.mp4"
cleaned_images_dir = "C:/python programs/cleaned_images1"
yolo_model_path = "yolov8m.pt"

yolo_model = YOLO(yolo_model_path)

def load_cleaned_images_and_features(cleaned_images_dir):
    orb = cv2.ORB_create(nfeatures=10000, WTA_K=2)
    cleaned_images = []
    cleaned_image_features = []
    cleaned_image_names = []

    if not os.path.exists(cleaned_images_dir):
        return cleaned_images, cleaned_image_features, cleaned_image_names

    for file in os.listdir(cleaned_images_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(cleaned_images_dir, file)
            image = cv2.imread(image_path, 0)
            if image is not None:
                image = apply_clahe(image)
                resized_image = cv2.resize(image, (416, 416))
                keypoints, descriptors = orb.detectAndCompute(resized_image, None)
                if descriptors is not None:
                    cleaned_images.append(resized_image)
                    cleaned_image_features.append((keypoints, descriptors))
                    cleaned_image_names.append(file)
    return cleaned_images, cleaned_image_features, cleaned_image_names

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def detect_objects_in_frame(frame):
    # results = yolo_model(frame, conf=0.7, iou=0.4, device='cuda')
    results = yolo_model(frame, conf=0.7, iou=0.4, device='cpu')

    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = yolo_model.names[class_id]
            detections.append((label, confidence, (x1, y1, x2, y2)))
    return detections

def compare_frames_with_cleaned_images(video_file, cleaned_image_features, cleaned_image_names, epochs=1):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_FPS, 30)
    orb = cv2.ORB_create(nfeatures=5000, WTA_K=2)
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=30)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for epoch in range(1, epochs + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        epoch_start_time = time.time()

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = apply_clahe(gray_frame)
            resized_frame = cv2.resize(gray_frame, (416, 416))
            frame_keypoints, frame_descriptors = orb.detectAndCompute(resized_frame, None)

            if frame_descriptors is None:
                continue

            for i, (keypoints, descriptors) in enumerate(cleaned_image_features):
                matches = flann.knnMatch(frame_descriptors, descriptors, k=2)
                good_matches = [m_n[0] for m_n in matches if len(m_n) > 1 and m_n[0].distance < 0.7 * m_n[1].distance]
                if len(good_matches) > 20:
                    print(f"Match found with cleaned image: {cleaned_image_names[i]}")
                    cv2.putText(frame, f"Matched: {cleaned_image_names[i]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    break

            detections = detect_objects_in_frame(frame)
            for label, confidence, (x1, y1, x2, y2) in detections:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Video Processing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        epoch_end_time = time.time()
    cap.release()
    cv2.destroyAllWindows()

if not os.path.exists(video_file):
    print(f"Error: Video file {video_file} does not exist.")
else:
    cleaned_images, cleaned_image_features, cleaned_image_names = load_cleaned_images_and_features(cleaned_images_dir)
    if len(cleaned_images) > 0:
        compare_frames_with_cleaned_images(video_file, cleaned_image_features, cleaned_image_names, epochs=3)