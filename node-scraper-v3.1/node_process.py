import subprocess
import struct
import threading
import time
import os
import base64
import logging
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt


class NodeProcess:
    def __init__(self, streamer, output_dir, interval_ms):
        self.process = None
        self.streamer = streamer
        self.output_dir = output_dir
        self.interval_ms = interval_ms
        self.stop_event = threading.Event()

        # Set up logging
        logging.basicConfig(filename='logs/python_node_process.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def start_node_script(self):
        # Start the node process
        basedir = os.path.abspath(os.path.dirname(__file__))
        self.process = subprocess.Popen(
            [
                'node',
                os.path.join(basedir, 'index.js'),
                f"--streamer={self.streamer}",
                f"--output={self.output_dir}",
                f"--interval-ms={self.interval_ms}"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False  # Important for binary output
        )
        logging.info(f"Started Node.js process for streamer {self.streamer}")

    def read_images(self):
        while not self.stop_event.is_set():
            length_bytes = self.read_exactly(self.process.stdout, 4)

            if length_bytes is None:
                if self.process.poll() is not None:
                    logging.info("No more data, process has ended.")
                    break
                else:
                    logging.warning("No length_bytes received, skipping this loop iteration.")
                    continue

            try:
                image_size = struct.unpack('>I', length_bytes)[0]
                logging.info(f"Expected image size: {image_size} bytes")
            except struct.error as e:
                logging.error(f"Error unpacking length_bytes: {e}")
                continue

            image_data = self.read_exactly(self.process.stdout, image_size)

            if image_data is None:
                logging.warning("No image data received, skipping this loop iteration.")
                continue

            yield image_data

    def stop_node_script(self):
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                logging.info("Node.js process terminated.")
            except ProcessLookupError as e:
                logging.error(f"Process termination error: {e}")

    def read_exactly(self, data_stream, n):
        data = b''
        while len(data) < n:
            packet = data_stream.read(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def log_errors(self):
        # Log errors from the Node.js stderr
        for line in self.process.stderr:
            logging.error(line.decode())


if __name__ == "__main__":
    streamer = "roshtein"
    output_dir = "stdout"  # You can switch between "stdout" or a directory
    interval_ms = 100
    node_process = NodeProcess(streamer, output_dir, interval_ms)

    try:
        node_process.start_node_script()

        # Start a separate thread to log errors
        error_thread = threading.Thread(target=node_process.log_errors)
        error_thread.start()

        # Wait for Node.js process to start producing output
        time.sleep(2)

        # Read images from the Node.js output
        for image_data in node_process.read_images():
            decoded_data = base64.b64decode(image_data)
            nparr = np.frombuffer(decoded_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        node_process.stop_node_script()
