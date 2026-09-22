"""
Train Dataset Model Script
Extracts keypoints and 8 joint angles from the user's Kaggle Yoga Posture Dataset
(c:\\Users\\HP\\.cache\\kagglehub\\datasets\\tr1gg3rtrash\\yoga-posture-dataset\\versions\\1)

Trains and saves:
  1. models/yoga_pose_classifier.pkl (Random Forest Classifier for pose identification)
  2. models/dataset_pose_references.json (Reference angle profiles per dataset pose class)
"""

import os
import sys
import json
import pickle
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import RandomForestClassifier

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r"C:\Users\HP\.cache\kagglehub\datasets\tr1gg3rtrash\yoga-posture-dataset\versions\1"
MODEL_DIR = os.path.join(APP_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TASK_PATH = r"C:\Users\HP\fitness\pose_landmarker_heavy.task"

print("Initializing PoseLandmarker detector...")
base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def calculate_angles(keypoints):
    if not keypoints:
        return []
    left_elbow    = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
    right_elbow   = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
    left_shoulder  = calculate_angle(keypoints[23], keypoints[11], keypoints[13])
    right_shoulder = calculate_angle(keypoints[24], keypoints[12], keypoints[14])
    left_hip       = calculate_angle(keypoints[25], keypoints[23], keypoints[11])
    right_hip      = calculate_angle(keypoints[26], keypoints[24], keypoints[12])
    left_knee      = calculate_angle(keypoints[23], keypoints[25], keypoints[27])
    right_knee     = calculate_angle(keypoints[24], keypoints[26], keypoints[28])
    return [left_elbow, right_elbow, left_shoulder, right_shoulder,
            left_hip, right_hip, left_knee, right_knee]


def process_image(img_path):
    try:
        bgr = cv2.imread(img_path)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)
        if res and res.pose_landmarks and len(res.pose_landmarks) > 0:
            return res.pose_landmarks[0]
    except Exception as e:
        pass
    return None


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path not found: {DATASET_PATH}")
        return

    X = []
    y = []
    reference_angles = {}
    class_images = {}

    folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
    print(f"Found {len(folders)} pose categories in dataset.")

    for idx, folder in enumerate(folders):
        folder_path = os.path.join(DATASET_PATH, folder)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"[{idx+1}/{len(folders)}] Processing category '{folder}' ({len(images)} images)...")

        cat_angles = []
        # Process up to 15 images per folder for fast & accurate reference profiles
        for img_name in images[:15]:
            img_path = os.path.join(folder_path, img_name)
            keypoints = process_image(img_path)
            if keypoints:
                angles = calculate_angles(keypoints)
                if len(angles) == 8:
                    X.append(angles)
                    y.append(folder)
                    cat_angles.append(angles)
                    if folder not in class_images:
                        class_images[folder] = img_path

        if cat_angles:
            # Store median angle vector for reference pose
            median_ref = np.median(cat_angles, axis=0).tolist()
            reference_angles[folder] = median_ref

    print(f"\nExtracted total {len(X)} valid pose angle vectors from dataset.")

    if not X:
        print("No valid keypoints extracted.")
        return

    # Train Random Forest Classifier on dataset angles
    print("Training Random Forest Classifier on dataset features...")
    classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    classifier.fit(X, y)

    # Save model and reference JSON
    model_save_path = os.path.join(MODEL_DIR, "yoga_pose_classifier.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(classifier, f)
    print(f"Saved trained classifier to: {model_save_path}")

    ref_json_path = os.path.join(MODEL_DIR, "dataset_pose_references.json")
    with open(ref_json_path, "w") as f:
        json.dump(reference_angles, f, indent=2)
    print(f"Saved reference posture profiles to: {ref_json_path}")

    # Copy one representative image per class into assets if available
    asset_dir = os.path.join(APP_DIR, "assets")
    os.makedirs(asset_dir, exist_ok=True)
    for cls_name, sample_img in class_images.items():
        safe_name = cls_name.replace(" ", "_").lower() + ".jpg"
        target_path = os.path.join(asset_dir, safe_name)
        try:
            bgr = cv2.imread(sample_img)
            if bgr is not None:
                cv2.imwrite(target_path, bgr)
        except Exception:
            pass

    print("Model training & dataset extraction completed successfully!")


if __name__ == "__main__":
    main()
