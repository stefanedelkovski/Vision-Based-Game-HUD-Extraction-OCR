import torch
import cv2
import numpy as np
import logging
import os


def detect_gameplay_areas(frames, model, confidence_threshold=0.8, save_images=False, player_name=None):
    """
    Detect gameplay areas in a batch of frames using the YOLOv5 model.

    Args:
        frames (list): List of frames (numpy arrays) to process.
        model (torch.nn.Module): The YOLOv5 model.
        confidence_threshold (float): Confidence threshold to filter detections.

    Returns:
        dict: Dictionary containing detected gameplay areas and their mappings.
    """
    gameplay_areas = {}
    mappings = {}

    for i, frame in enumerate(frames):
        # Pad the frame to make dimensions multiples of 32
        height, width, _ = frame.shape
        new_height = int(np.ceil(height / 32) * 32)
        new_width = int(np.ceil(width / 32) * 32)
        padded_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        padded_frame[:height, :width, :] = frame

        # Perform inference
        results = model(padded_frame)

        # Extract bounding boxes
        detections = results.xyxy[0].cpu().numpy()

        frame_areas = []
        frame_mapping = []

        for x1, y1, x2, y2, confidence, cls in detections:
            if confidence > confidence_threshold:  # Confidence threshold
                cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]
                frame_areas.append(cropped_frame)
                frame_mapping.append((i, (int(x1), int(y1), int(x2), int(y2)), confidence))
                # if save_images and player_name:
                #     save_cropped_frame(cropped_frame, i, player_name)

        gameplay_areas[i] = frame_areas
        mappings[i] = frame_mapping

    return gameplay_areas, mappings


def save_cropped_frame(frame, frame_idx, player_name):
    try:
        output_dir = os.path.join("output", player_name, "frames")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Get the latest image number
        existing_files = [f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.jpg')]
        if existing_files:
            latest_file = sorted(existing_files, key=lambda x: int(x.split('_')[1].split('.')[0]))[-1]
            latest_num = int(latest_file.split('_')[1].split('.')[0])
        else:
            latest_num = 0

        filename = os.path.join(output_dir, f"frame_{latest_num + 1}.jpg")
        # filename = os.path.join(output_dir, f"frame_{frame_idx + 1}.jpg")
        cv2.imwrite(filename, frame)
        logging.info(f"Cropped frame saved successfully at {filename}.")
    except Exception as e:
        logging.error(f"Error in save_cropped_frame: {e}")

