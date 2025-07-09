import cv2
import torch
import mediapipe as mp
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import time

# parameters
depth_model_id = "depth-anything/Depth-Anything-V2-Small-hf"

ROI_DEPTH_THRESHOLD_MIN = 0.5
ROI_DEPTH_THRESHOLD_MAX = 5.0
ROI_PADDING = 20

PATCH_RADIUS = 5
VISIBILITY_THRESHOLD = 0.6
SMOOTHING_FACTOR = 0.3
Z_SMOOTHING_FACTOR = 0.85
Z_JUMP_THRESHOLD = 0.3

print("Loading Depth-Anything-V2 model...")
try:
    image_processor = AutoImageProcessor.from_pretrained(depth_model_id)
    depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_id)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    depth_model.to(device)
    depth_model.eval()
    print(f"Depth-Anything-V2 model loaded to {device}.")
except Exception as e:
    print(f"Error loading Depth-Anything model: {e}")
    exit()

print("Loading MediaPipe Pose model...")
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
print("MediaPipe Pose model loaded.")

smoothed_positions = {}
last_frame_time = time.time()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("\nStarting AI Kinect Lite. Press 'q' to quit.")


def is_valid_landmark(landmark, depth_value, visibility_threshold):
    if landmark.visibility < visibility_threshold:
        return False
    if np.isnan(depth_value) or depth_value <= 0 or depth_value > 100:
        return False
    return True


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame_time = time.time()
    delta_time = current_frame_time - last_frame_time
    if delta_time == 0:
        delta_time = 1e-6
    fps = 1 / delta_time
    last_frame_time = current_frame_time

    original_h, original_w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(img_rgb)
    inputs = image_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth
        prediction_resized = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()
    depth_map = prediction_resized.cpu().numpy()

    person_mask = ((depth_map > ROI_DEPTH_THRESHOLD_MIN) & (depth_map < ROI_DEPTH_THRESHOLD_MAX)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, original_w, original_h
    cropped_img_rgb = img_rgb

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:
            x, y, w_roi, h_roi = cv2.boundingRect(largest_contour)
            roi_x1 = max(0, x - ROI_PADDING)
            roi_y1 = max(0, y - ROI_PADDING)
            roi_x2 = min(original_w, x + w_roi + ROI_PADDING)
            roi_y2 = min(original_h, y + h_roi + ROI_PADDING)
            cropped_img_rgb = img_rgb[roi_y1:roi_y2, roi_x1:roi_x2]

    if cropped_img_rgb.size > 0:
        cropped_img_rgb.flags.writeable = False
        results = pose.process(cropped_img_rgb)
        cropped_img_rgb.flags.writeable = True
    else:
        results = None

    frame_display = frame.copy()
    final_keypoints_3d = {}

    if results and results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            joint_name = mp_pose.PoseLandmark(idx).name

            px_cropped = int(landmark.x * cropped_img_rgb.shape[1])
            py_cropped = int(landmark.y * cropped_img_rgb.shape[0])
            px_original = px_cropped + roi_x1
            py_original = py_cropped + roi_y1

            raw_depth = -1.0
            if 0 <= px_original < original_w and 0 <= py_original < original_h:
                y_start, y_end = max(0, py_original - PATCH_RADIUS), min(original_h, py_original + PATCH_RADIUS)
                x_start, x_end = max(0, px_original - PATCH_RADIUS), min(original_w, px_original + PATCH_RADIUS)
                patch = depth_map[y_start:y_end, x_start:x_end]
                valid_depths = patch[np.isfinite(patch) & (patch > 0)]
                if valid_depths.size > 0:
                    raw_depth = np.median(valid_depths)

            if not is_valid_landmark(landmark, raw_depth, VISIBILITY_THRESHOLD):
                continue

            current_pos = np.array([px_original, py_original, raw_depth])
            smoothed_pos = current_pos

            if joint_name in smoothed_positions:
                prev_pos = smoothed_positions[joint_name]
                prev_z = prev_pos[2]
                z_change = abs(raw_depth - prev_z)
                if z_change > Z_JUMP_THRESHOLD:
                    corrected_z = prev_z + np.sign(raw_depth - prev_z) * Z_JUMP_THRESHOLD
                else:
                    corrected_z = raw_depth

                smoothed_x = SMOOTHING_FACTOR * prev_pos[0] + (1 - SMOOTHING_FACTOR) * px_original
                smoothed_y = SMOOTHING_FACTOR * prev_pos[1] + (1 - SMOOTHING_FACTOR) * py_original
                smoothed_z = Z_SMOOTHING_FACTOR * prev_z + (1 - Z_SMOOTHING_FACTOR) * corrected_z
                smoothed_pos = np.array([smoothed_x, smoothed_y, smoothed_z])

            smoothed_positions[joint_name] = smoothed_pos
            final_keypoints_3d[joint_name] = {
                "x_pixel": smoothed_pos[0],
                "y_pixel": smoothed_pos[1],
                "z_meters": float(smoothed_pos[2]),
                "visibility": landmark.visibility
            }

        connections = mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start_joint = mp_pose.PoseLandmark(connection[0]).name
            end_joint = mp_pose.PoseLandmark(connection[1]).name

            if start_joint in final_keypoints_3d and end_joint in final_keypoints_3d:
                start_pt = final_keypoints_3d[start_joint]
                end_pt = final_keypoints_3d[end_joint]
                start_coords = (int(start_pt["x_pixel"]), int(start_pt["y_pixel"]))
                end_coords = (int(end_pt["x_pixel"]), int(end_pt["y_pixel"]))

                cv2.line(frame_display, start_coords, end_coords, (0, 255, 0), 2)
                cv2.circle(frame_display, start_coords, 4, (0, 0, 255), -1)
                cv2.circle(frame_display, end_coords, 4, (0, 0, 255), -1)

        if "LEFT_WRIST" in final_keypoints_3d:
            lw = final_keypoints_3d["LEFT_WRIST"]
            text = f"LW Depth: {lw['z_meters']:.2f}m"
            cv2.putText(frame_display, text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    depth_display = cv2.normalize(depth_map, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(frame_display, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
    cv2.putText(frame_display, f"FPS: {fps:.1f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    combined_display = np.concatenate((frame_display, depth_display), axis=1)
    cv2.imshow('AI Kinect Lite | Pose & Depth', combined_display)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

print("Stopping AI Kinect Lite...")
cap.release()
cv2.destroyAllWindows()
pose.close()

if 'depth_model' in locals() and device == torch.device("cuda"):
    depth_model.cpu()
    del depth_model, image_processor
    torch.cuda.empty_cache()
    print("CUDA memory cleared.")

print("Tracker stopped.")
