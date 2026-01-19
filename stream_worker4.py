import os
import mmap
import threading
from datetime import datetime

import cv2
import logging
import os
import time
import subprocess
import numpy as np
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue, Empty
from model_inferences_and_output import process_batch

# Base directory setup
basedir = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(basedir, 'output')

# Set up logger
logger = logging.getLogger(__name__)


class NodeProcess:
    def __init__(self, streamer, output_dir, interval_ms, stream_worker):
        self.stream_worker = stream_worker
        self.streamer = streamer
        self.output_dir = output_dir
        self.interval_ms = interval_ms
        self.process = None
        self.initialized = threading.Event()

    def start_node_script(self):
        """Starts the Node.js script to capture images."""
        try:
            self.process = subprocess.Popen(
                [
                    '/usr/bin/node',
                    os.path.join(basedir, 'node-scraper-v2', 'index.js'),
                    f"--streamer={self.streamer}",
                    f"--output={self.output_dir}",
                    f"--interval-ms={self.interval_ms}"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False
            )
            self.initialized.set()
            logger.info(f"Node.js script started for streamer {self.streamer}")
            threading.Thread(target=self.log_stderr, daemon=True).start()

        except Exception as e:
            logger.error(f"Error starting Node.js process: {e}")
            raise RuntimeError(f"Failed to start Node.js script: {e}")

    def log_stderr(self):
        """Logs stderr from the Node.js process."""
        for line in iter(self.process.stderr.readline, b''):
            logger.error(f"Node.js stderr: {line.decode('utf-8')}")

    def stop_node_script(self):
        """Stops the Node.js process."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            time.sleep(1)
            if self.process.poll() is None:
                self.process.kill()

        # Log Node.js process exit code
        exit_code = self.process.poll()
        if exit_code is not None:
            logger.info(f"Node.js script terminated for streamer {self.streamer} with exit code {exit_code}")
        else:
            logger.info(f"Node.js script for streamer {self.streamer} terminated successfully.")

        # Also log the stderr output for more details on errors
        _, stderr = self.process.communicate()
        if stderr:
            logger.error(f"Node.js script for streamer {self.streamer} exited with error: {stderr}")
        else:
            logger.info(f"Node.js script for streamer {self.streamer} exited without errors.")


class MMapImageHandler(FileSystemEventHandler):
    def __init__(self, batch, frame_queue, batch_size, save_images, first_batch_ready, leftover_batch_ready, processed_files, total_batches, capture_done, worker):
        self.batch = batch
        self.frame_queue = frame_queue
        self.batch_size = batch_size
        self.save_images = save_images
        self.first_batch_ready = first_batch_ready
        self.leftover_batch_ready = leftover_batch_ready
        self.processed_files = processed_files
        self.processed_files_lock = threading.Lock()
        self.total_batches = total_batches
        self.capture_done = capture_done
        self.worker = worker
        self.last_image_time = time.time()

    def on_created(self, event):
        """Event handler for created image files."""
        if not event.is_directory and event.src_path.endswith('.png'):
            image_path = event.src_path

            self.leftover_batch_ready.clear()
            self.last_image_time = time.time()

            with self.processed_files_lock:
                if image_path in self.processed_files:
                    logger.warning(f"Duplicate image detected and skipped: {image_path}")
                    return
                self.processed_files.add(image_path)

            logger.info(f"New image detected: {image_path}")

            stable_size_check_interval = 0.1
            max_stable_checks = 10
            initial_size = os.path.getsize(image_path)
            stable_checks = 0

            while stable_checks < max_stable_checks:
                time.sleep(stable_size_check_interval)
                current_size = os.path.getsize(image_path)
                if current_size == initial_size:
                    stable_checks += 1
                else:
                    stable_checks = 0
                initial_size = current_size

            if initial_size == 0:
                logger.error(f"File is empty, skipping: {image_path}")
                return

            try:
                if os.path.getsize(image_path) == 0:
                    logger.error(f"File is empty, skipping: {image_path}")
                    return
                with open(image_path, "rb") as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                        image_data = mmapped_file.read()

                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is None:
                        logger.error(f"Failed to decode image {image_path}. Skipping.")
                        return

                    timestamp = time.time()
                    self.batch.append((frame, timestamp))
                    self.worker.added_images += 1

                    if not self.save_images:
                        os.remove(image_path)

                    logger.info(f"Added: {self.worker.added_images}, processed: {self.worker.processed_images}")
                    if len(self.batch) >= self.batch_size:
                        logger.info(f"Batch of size {len(self.batch)} is ready, adding to queue.")
                        self.frame_queue.put(self.batch.copy())
                        with threading.Lock():
                            self.total_batches[0] += 1
                        self.batch.clear()

                        if not self.first_batch_ready.is_set():
                            self.first_batch_ready.set()

            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")

    def check_for_no_new_images(self):
        if self.capture_done.is_set() and (time.time() - self.last_image_time > 5):
            logger.info("No new images detected for 5 seconds. Setting leftover batch ready flag.")
            self.leftover_batch_ready.set()


class StreamWorker(threading.Thread):
    def __init__(self,
                 worker_id,
                 stream_queue,
                 active_streams,
                 stream_status,
                 stream_threads,
                 gameplay_area_model,
                 game_detection_model,
                 credit_bet_win_model,
                 reader,
                 batch_size=32):
        super().__init__()
        self.worker_id = worker_id
        self.stream_queue = stream_queue
        self.active_streams = active_streams
        self.stream_status = stream_status
        self.stream_threads = stream_threads
        self.gameplay_area_model = gameplay_area_model
        self.game_detection_model = game_detection_model
        self.credit_bet_win_model = credit_bet_win_model
        self.reader = reader
        self.batch_size = batch_size
        self.interrupted = False
        self.capture_done = threading.Event()
        self.frame_queue = Queue()
        self.total_batches = [0]
        self.processed_batches = 0
        self.first_batch_ready = threading.Event()
        self.leftover_batch_ready = threading.Event()
        self.node_process = None
        self.interval_time = 300
        self.current_batch = []
        self.processed_files = set()
        self.stream_output_dir = ''
        self.added_images = 0
        self.processed_images = 0

    def run(self):
        """Main worker function that starts Node.js process and monitors the directory."""
        while not self.interrupted:
            try:
                stream_id, url, save_images = self.stream_queue.get(timeout=3)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stream_name = self.get_filename_from_url(url)
                if not stream_name:
                    stream_name = stream_id
                base_filename = f"{stream_name}_{timestamp}"

                self.capture_done.clear()

                with threading.Lock():
                    if stream_id not in self.stream_status:
                        self.stream_status[stream_id] = {'status': 'processing', 'save_images': save_images}
                    else:
                        self.stream_status[stream_id]['status'] = 'processing'
                    self.stream_threads[stream_id] = self

                self.stream_output_dir = os.path.join(output_dir, f"{stream_name}_{timestamp}")
                os.makedirs(self.stream_output_dir, exist_ok=True)

                stream_frames_dir = os.path.join(output_dir, f"{stream_name}_{timestamp}", datetime.now().strftime("%Y-%m-%d"))
                os.makedirs(stream_frames_dir, exist_ok=True)

                # Start the Node.js process
                self.node_process = NodeProcess(stream_name, self.stream_output_dir, self.interval_time, self)
                self.node_process.start_node_script()

                # Start monitoring the directory for new images
                observer = Observer()
                event_handler = MMapImageHandler(self.current_batch, self.frame_queue, self.batch_size, save_images,
                                                 self.first_batch_ready, self.leftover_batch_ready, self.processed_files, self.total_batches, self.capture_done, self)
                observer.schedule(event_handler, stream_frames_dir, recursive=False)
                observer.start()

                logger.info(f"Started monitoring directory: {stream_frames_dir}")

                # capture_thread = threading.Thread(target=self.process_batches())
                process_thread = threading.Thread(target=self.process_batches,
                                                  args=(stream_id, base_filename, save_images))

                process_thread.start()
                process_thread.join()

                observer.stop()
                observer.join()

                if self.interrupted:
                    with threading.Lock():
                        self.stream_status[stream_id]['status'] = 'interrupted'
                else:
                    with threading.Lock():
                        self.stream_status[stream_id]['status'] = 'finished'
                self.stream_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                with threading.Lock():
                    self.stream_status[stream_id]['status'] = 'error'
                logger.error(f"Error processing stream {stream_id}: {e}")

    def interrupt(self):
        """Interrupts the current stream processing."""
        self.interrupted = True
        logger.info(f"Interrupt received for worker {self.worker_id}.")
        self.capture_done.set()
        if hasattr(self, 'node_process') and self.node_process:
            self.node_process.stop_node_script()
        time.sleep(3)

    def process_batches(self, stream_id, base_filename, save_images):
        """Process the batches from the queue."""
        screen_results = {}
        last_timestamp = 0.0
        last_filename = 0
        screen_results = {}
        waiting_counter = 0

        logger.info("Waiting for the first batch to be ready for processing.")
        self.first_batch_ready.wait()

        logger.info(f'Added images total: {self.added_images}')
        logger.info(f'Processed images total: {self.processed_images}')
        while not (self.capture_done.is_set() and self.added_images == self.processed_images):
            try:
                batch = self.frame_queue.get(timeout=1)
                if batch is None:
                    continue

                logger.info(f"Processing batch of size {len(batch)}")

                try:
                    results, last_timestamp, last_filename = process_batch(
                        batch,
                        self.gameplay_area_model,
                        self.game_detection_model,
                        self.credit_bet_win_model,
                        self.reader,
                        interval_time=self.interval_time,
                        start_filename=last_filename,
                        start_timestamp=last_timestamp,
                        save_images=save_images,
                        player_name=base_filename
                    )
                    last_timestamp += self.interval_time / 1000

                    self.processed_images += len(batch)

                except Exception as batch_process_error:
                    logger.error(f"Error in process_batch: {batch_process_error}")
                    continue

                for screen, screen_result in results.items():
                    if screen not in screen_results:
                        screen_results[screen] = []
                    screen_results[screen].extend(screen_result)

                with threading.Lock():
                    self.processed_batches += 1

                waiting_counter = 0
                logger.info("Completed batch.")

            except Empty:
                if waiting_counter == int(self.batch_size) * 3:
                    logger.info(f"Timeout occurred: batch not ready after {waiting_counter} retries. Saving results.")
                    if screen_results:
                        self.save_results(base_filename, screen_results, base_filename, self.stream_output_dir)
                    else:
                        logger.info("No results.")
                logger.info(f"Waiting for next batch...")
                waiting_counter += 1
                continue
            except Exception as e:
                logger.error(f"Error processing batches: {e}")

        logger.info("Waiting for leftover batch to be ready...")
        self.leftover_batch_ready.wait()

        if len(self.current_batch) > 0:
            logger.info(f"Processing leftover batch of size {len(self.current_batch)}")
            try:
                results, last_timestamp, last_filename = process_batch(
                    self.current_batch,
                    self.gameplay_area_model,
                    self.game_detection_model,
                    self.credit_bet_win_model,
                    self.reader,
                    interval_time=self.interval_time,
                    start_filename=last_filename,
                    start_timestamp=last_timestamp,
                    save_images=save_images,
                    player_name=base_filename
                )
                self.current_batch.clear()

                # Merge leftover batch results
                for screen, screen_result in results.items():
                    if screen not in screen_results:
                        screen_results[screen] = []
                    screen_results[screen].extend(screen_result)
            except Exception as leftover_process_error:
                logger.error(f"Error in processing leftover batch: {leftover_process_error}")

        # Save final results after processing everything
        logger.info(f"Saving results to {self.stream_output_dir}")
        self.save_results(base_filename, screen_results, self.stream_output_dir)
        logger.info(f"Finished processing all batches for stream {stream_id}")

    @staticmethod
    def save_results(base_filename, screen_results, player_name, output_dir):
        logger.info(f"Output dir for save results: {output_dir}")
        streamer_dir = os.path.join(os.getcwd(), output_dir)
        if not os.path.exists(streamer_dir):
            os.makedirs(streamer_dir)

        text_filename = f"{base_filename}.txt"
        with open(os.path.join(streamer_dir, text_filename), 'w', encoding='utf-8') as f:
            for screen, results in screen_results.items():
                f.write(f"{screen}:\n")
                for result in results:
                    f.write(
                        f"{result['filename']} | {result['timestamp']} | {result['game']} | {result['credit']} | {result['bet']} | {result['win']}\n")
                f.write("\n")
        logger.info(f"Results saved to text file at {text_filename}.")

        excel_path = os.path.join(streamer_dir, f"{base_filename}.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            has_visible_sheets = False  # Track if there are any visible sheets
            for screen, results in screen_results.items():
                df = pd.DataFrame(results)
                if not df.empty:
                    df.to_excel(writer, sheet_name=screen, index=False)
                    has_visible_sheets = True
                else:
                    df_empty = pd.DataFrame({"Info": ["No results to display"]})
                    df_empty.to_excel(writer, sheet_name=f"{screen}_Info", index=False)
                    has_visible_sheets = True

            # If no visible sheets were added, add a default visible sheet
            if not has_visible_sheets:
                df_default = pd.DataFrame({"Info": ["No results to display"]})
                df_default.to_excel(writer, sheet_name='Info', index=False)

    @staticmethod
    def get_filename_from_url(url):
        if '/video' in url:
            return None
        else:
            return url.rstrip('/').split('/')[-1]


def stop_all_workers(workers):
    """Helper function to stop all workers."""
    for worker in workers:
        worker.interrupt()
        worker.join()
    logger.info("All workers stopped.")