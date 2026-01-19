from datetime import datetime

from flask import Flask, request, jsonify
import uuid
from queue import Queue
from stream_worker4 import StreamWorker, stop_all_workers
import os
import logging
import torch
from torchvision import transforms, models
import pathlib
import h5py
import json
import tensorflow as tf

import easyocr
from game_detection_efficientnet_inference import load_model

pathlib.PosixPath = pathlib.WindowsPath

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

app = Flask(__name__)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reader = easyocr.Reader(lang_list=['en', 'de', 'nl'], gpu=True)

# Load models
gameplay_area_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                     path='C:\\Users\\User\\Desktop\\Work\\captureGambling2\\yolov5\\runs\\train\\gameplayarea_model_2.0\\weights\\last.pt',
                                     force_reload=False).to(device)
credit_bet_win_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                      path='yolov5/runs/train/credit_bet_win_2.0/weights/best.pt',
                                      force_reload=False).to(device)

model_path = 'models/best_game_classifier_efficientnet_b0.h5'
game_detection_model = load_model(model_path)

# # Load class indices mapping to determine number of classes
# with open('models/class_indices.json', 'r') as f:
#     class_indices = json.load(f)
# num_classes = len(class_indices)
#
# # Load the new ResNet model
# model = models.resnet50(weights=None)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, num_classes)
#
#
# # Load the weights from the best model
# def load_model_weights(model, filepath):
#     with h5py.File(filepath, 'r') as f:
#         for name, param in model.state_dict().items():
#             if isinstance(f[name], h5py.Dataset) and len(f[name].shape) > 0:
#                 param.copy_(torch.tensor(f[name][:]))
#             else:
#                 param.copy_(torch.tensor(f[name][()]))
#     return model
#
#
# game_detection_model = load_model_weights(model, 'models/best_model_game_detection.h5')
# game_detection_model = game_detection_model.to(device)
# game_detection_model.eval()

# Setup queues and logging
stream_queue = Queue()
active_streams = {}
inactive_streams = {}
stream_status = {}
stream_threads = {}
workers = []

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'logs/run_{timestamp}.log'

# Set up logging
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

active_streams.clear()
inactive_streams.clear()
stream_status.clear()
stream_threads.clear()


@app.route('/add', methods=['POST'])
def add_stream():
    data = request.get_json()
    url = data.get('url')
    save_images = data.get('save_images', False)

    # Check if stream is already active
    if url in active_streams.values():
        return jsonify({'message': 'Stream is already active and cannot be added again.'}), 400

    # Check if stream is inactive and reactivate it
    for stream_id, stream_url in inactive_streams.items():
        if stream_url == url:
            active_streams[stream_id] = inactive_streams.pop(stream_id)
            stream_status[stream_id]['status'] = 'waiting'
            stream_queue.put((stream_id, url, save_images))
            logging.info(f"Reactivated stream {stream_id} and added it to queue with save_images={save_images}.")
            return jsonify({'message': 'Stream reactivated successfully', 'id': stream_id, 'url': url}), 200

    # If not, add as new stream
    if url:
        stream_id = str(uuid.uuid4())
        active_streams[stream_id] = url
        stream_status[stream_id] = {'status': 'waiting', 'save_images': save_images}
        stream_queue.put((stream_id, url, save_images))
        logging.info(f"Added new stream {stream_id} to queue with save_images={save_images}.")
        return jsonify({'message': 'Stream added successfully', 'id': stream_id, 'url': url}), 200
    else:
        return jsonify({'message': 'URL is required'}), 400


@app.route('/interrupt', methods=['POST'])
def interrupt_stream():
    data = request.get_json()
    stream_id = data.get('id')
    if stream_id in stream_threads:
        worker = stream_threads[stream_id]
        worker.interrupt()
        stream_status[stream_id]['status'] = 'interrupted'

        # Move the stream from active to inactive
        inactive_streams[stream_id] = active_streams.pop(stream_id)
        logging.info(f"Stream {stream_id} interrupted and moved to inactive streams.")
        return jsonify({'message': 'Stream interrupted successfully', 'id': stream_id}), 200
    else:
        return jsonify({'message': 'Stream not found or already completed'}), 404


@app.route('/delete', methods=['POST'])
def delete_stream():
    data = request.get_json()
    stream_id = data.get('id')
    if stream_id in stream_status and (stream_status[stream_id]['status'] == 'finished'
                                       or stream_status[stream_id]['status'] == 'interrupted'):
        if os.path.exists(f"{stream_id}.txt"):
            os.remove(f"{stream_id}.txt")
            logging.info(f"Deleted file for stream {stream_id}.")

        # Remove the stream from both active and inactive
        active_streams.pop(stream_id, None)
        inactive_streams.pop(stream_id, None)
        stream_status.pop(stream_id, None)
        logging.info(f"Stream {stream_id} deleted.")
        return jsonify({'message': 'Stream deleted successfully', 'id': stream_id}), 200
    else:
        return jsonify({'message': 'Stream not finished or not found'}), 404


@app.route('/status', methods=['GET'])
def check_status():
    stream_id = request.args.get('id')
    if stream_id in stream_status:
        worker = stream_threads.get(stream_id)
        if worker:
            processed_batches = worker.processed_batches
            total_batches = worker.total_batches
            return jsonify({
                'id': stream_id,
                'status': stream_status[stream_id]['status'],
                'processed_batches': processed_batches,
                'total_batches': total_batches[0]
            }), 200
        else:
            return jsonify({
                'id': stream_id,
                'status': stream_status[stream_id]['status'],
                'message': 'Worker not found'
            }), 200
    else:
        return jsonify({'message': 'Stream ID not found'}), 404


@app.route('/list', methods=['GET'])
def list_streams():
    return jsonify({
        'active_streams': list(active_streams.items()),
        'inactive_streams': list(inactive_streams.items())
    }), 200


def initialize_workers(num_workers):
    for i in range(num_workers):
        worker = StreamWorker(i, stream_queue, active_streams, stream_status, stream_threads, gameplay_area_model,
                              game_detection_model, credit_bet_win_model, reader, batch_size=16)
        worker.start()
        workers.append(worker)
    logging.info("All workers initialized.")


initialize_workers(8)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, use_reloader=False)
