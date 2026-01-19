import logging
import time

import cv2
from pytesseract import image_to_string
import numpy as np

# Tesseract setup
tesseract_languages = 'eng+deu+fra+rus'


def detect_credit_bet_win(batch, model, reader, confidence_threshold=0.8, win_confidence_threshold=0.7):
    results = []
    class_mapping = {0: 'credit', 1: 'bet', 2: 'win'}  # Assuming class indices
    for frame in batch:
        try:
            detections = model(frame).pred[0]  # Ensure the model output is accessed correctly
            bboxes = detections[:, :4]
            confidences = detections[:, 4]
            classes = detections[:, 5]

            # Initialize dictionaries to store the highest confidence detections for each class
            highest_confidence_detections = {'credit': None, 'bet': None, 'win': None}

            for bbox, confidence, cls in zip(bboxes, confidences, classes):
                class_name = class_mapping.get(int(cls), 'unknown')
                logging.info(f"Detected bbox: {bbox.tolist()}, Confidence: {confidence}, Class: {class_name}")

                # Set the appropriate confidence threshold
                current_threshold = confidence_threshold if class_name != 'win' else win_confidence_threshold

                # Update the highest confidence detection for the class
                if confidence >= current_threshold:
                    if highest_confidence_detections[class_name] is None or confidence > highest_confidence_detections[class_name]['confidence']:
                        highest_confidence_detections[class_name] = {
                            'bbox': bbox.tolist(),
                            'confidence': confidence.item()
                        }

            # Process the highest confidence detections
            for class_name, detection in highest_confidence_detections.items():
                if detection:
                    bbox = detection['bbox']
                    confidence = detection['confidence']
                    now = round(time.time() * 1000)
                    preprocessed_image = preprocess_bbox(frame, np.array(bbox))
                    text = extract_text(preprocessed_image, tesseract_languages)
                    logging.info(f"SINGLE IMAGE PROCESSING TIME: {round(time.time() * 1000) - now}")
                    logging.info(f'Extracted text: {text}')
                    results.append({
                        'class': class_name,
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence
                    })
        except Exception as e:
            logging.error(f"Error in detect_credit_bet_win for frame: {e}")
    return results


def preprocess_bbox(image, bbox):
    try:
        x1, y1, x2, y2 = map(int, bbox[:4])
        roi = image[y1:y2, x1:x2]
        height, width = roi.shape[:2]
        resized_roi = cv2.resize(roi, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)
        preprocessed_image = preprocess_image(resized_roi)
        if preprocessed_image is not None:
            padded_image = add_padding(preprocessed_image, padding_color=(0, 0, 0))
            logging.info("Bounding box preprocessed successfully.")
            return padded_image
        else:
            logging.error("Preprocessing of image failed.")
            return None
    except Exception as e:
        logging.error(f"Error in preprocess_bbox: {e}")
        return None


def preprocess_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray)
        denoised_image = cv2.fastNlMeansDenoising(enhanced_image, None, 30, 7, 21)
        _, binarized_image = cv2.threshold(denoised_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        cleaned_image = cv2.morphologyEx(binarized_image, cv2.MORPH_CLOSE, kernel)
        logging.info("Image preprocessed successfully.")
        return cleaned_image
    except Exception as e:
        logging.error(f"Error in preprocess_image: {e}")
        return None


def add_padding(image, padding=20, padding_color=(0, 0, 0)):
    try:
        padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=padding_color)
        logging.info("Padding added successfully.")
        return padded_image
    except Exception as e:
        logging.error(f"Error in add_padding: {e}")
        return None


def extract_text(image, languages):
    try:
        config = f'--psm 6 -l {languages}'
        text = image_to_string(image, config=config)
        logging.info(f"Text extracted successfully: {text}")
        return text.strip().replace('\n', ' ')
    except Exception as e:
        logging.error(f"Error in extract_text: {e}")
        return ""
