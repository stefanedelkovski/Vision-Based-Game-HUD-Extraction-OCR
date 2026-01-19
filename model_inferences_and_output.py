import logging
import time
from datetime import datetime

import cv2
from model_inferences.gameplay_area import detect_gameplay_areas
from model_inferences.game_detection2 import detect_game
from model_inferences.label_extraction2 import detect_credit_bet_win

# Set up logging
logging.getLogger(__name__)


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def process_batch(frames_with_timestamps, gameplay_area_model, game_detection_model, credit_bet_win_model, reader,
                  interval_time, start_filename, start_timestamp=0.0, save_images=False, player_name=None):
    try:
        logging.info("Starting batch processing")
        batch = [frame for frame, _ in frames_with_timestamps]
        timestamps = [round(start_timestamp + idx * (interval_time / 1000), 3) for idx in range(len(batch))]
        filenames = [start_filename + i + 1 for i in range(len(batch))]
        logging.info(f"Batch size: {len(batch)}, Timestamps: {timestamps}")

        # Detect gameplay areas
        now = round(time.time() * 1000)
        gameplay_areas, gameplay_mappings = detect_gameplay_areas(frames=batch, model=gameplay_area_model, save_images=save_images, player_name=player_name)
        logging.info(f"GA PROCESSING TIME: {round(time.time() * 1000) - now}")
        logging.info(f"Gameplay areas detected: {sum(len(areas) for areas in gameplay_areas.values())}")

        results = {}
        gameplay_area_count = 0

        for frame_idx, frame_areas in gameplay_areas.items():
            if not frame_areas:
                continue

            frame_mappings = gameplay_mappings[frame_idx]

            for area_idx, frame_area in enumerate(frame_areas):
                mapping = frame_mappings[area_idx]

                if area_idx >= len(results):
                    gameplay_area_count += 1
                    results[f"Screen {gameplay_area_count}"] = []

                logging.info("Detecting games")
                now = round(time.time() * 1000)
                game_results = detect_game([frame_area], [mapping], game_detection_model)
                logging.info(f"GD PROCESSING TIME: {round(time.time() * 1000) - now}")
                logging.info(f"Games detected: {len(game_results)}")

                logging.info("Detecting credit, bet, win")
                now = round(time.time() * 1000)
                credit_bet_win_results = detect_credit_bet_win([frame_area], credit_bet_win_model, reader)
                logging.info(f"LD PROCESSING TIME: {round(time.time() * 1000) - now}")
                logging.info(f"Credit/Bet/Win detections: {len(credit_bet_win_results)}")

                logging.info("Detecting the labels and extracting text")
                now = round(time.time() * 1000)
                for game_result in game_results:
                    credit = "Unknown"
                    bet = "Unknown"
                    win = "Unknown"

                    for result in credit_bet_win_results:
                        iou = calculate_iou(result['bbox'], game_result['bbox'])
                        if iou > 0.3:
                            logging.info(f'Bounding boxes overlap with IoU: {iou}')
                        credit = result['text'] if result['class'] == 'credit' and result['confidence'] >= 0.9 else credit
                        bet = result['text'] if result['class'] == 'bet' and result['confidence'] >= 0.9 else bet
                        win = result['text'] if result['class'] == 'win' and result['confidence'] >= 0.9 else win

                    results[f"Screen {gameplay_area_count}"].append({
                        'filename': f'frame_{filenames[frame_idx]}',
                        'timestamp': timestamps[frame_idx],
                        'game': game_result['game'],
                        'credit': credit,
                        'bet': bet,
                        'win': win
                    })
                logging.info(f"LE PROCESSING TIME: {round(time.time() * 1000) - now}")

        logging.info("Batch processing complete")
        return results, timestamps[-1], filenames[-1]

    except Exception as e:
        logging.error(f"Error in process_batch: {e}")
        return {
            f"Screen 1": [{
                'timestamp': "N/A",
                'game': "Error processing batch",
                'credit': "N/A",
                'bet': "N/A",
                'win': "N/A"
            }]
        }, start_timestamp
