import logging
import time

import cv2
from pytesseract import image_to_string
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Assuming 'image' is a NumPy array coming from OpenCV
def visualize_image(image):
    # Convert image from BGR to RGB for correct display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image using matplotlib
    plt.imshow(image_rgb)
    plt.axis('off')  # Hide axis
    plt.show(block=True)


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
                    text = extract_text(preprocessed_image, reader)
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
        if roi is not None:
            return roi
        else:
            logging.error("Preprocessing of image failed.")
            return None
    except Exception as e:
        logging.error(f"Error in preprocess_bbox: {e}")
        return None


def preprocess_image(image):
    try:
        denoised_image = cv2.fastNlMeansDenoising(image, None, h=30, templateWindowSize=7, searchWindowSize=21)
        gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((2, 2), np.uint8)
        processed_image = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        return processed_image
    except Exception as e:
        logging.error(f"Error in preprocess_image: {e}")
        return image


def extract_text(image, reader):
    try:
        image = preprocess_image(image)
        visualize_image(image)
        result = reader.readtext(image, detail=0, paragraph=True, x_ths=1000, allowlist='')
        result = (''.join([i + ' ' for i in result]).rstrip(' ').replace('  ', ' ')
                  .replace(', ', ' ').upper())
        logging.info(f"Text extracted successfully: {result}")
        return result.strip().replace('\n', ' ')
    except Exception as e:
        logging.error(f"Error in extract_text: {e}")
        return ""
