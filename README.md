# Vision-Based Game HUD Extraction OCR

Vision-Based Game HUD Extraction OCR is a computer vision project focused on extracting structured information from video game HUDs (Heads-Up Displays) using deep learning and OCR techniques. The system processes game frames or streams, detects HUD regions, performs text recognition, and exposes results through scripts or a Flask-based inference API.

---

## Repository Structure


| Path | Description |
|-----|-------------|
| evaluation_images/ | Sample images for testing and evaluation |
| inference_flask_scripts/ | Flask inference preprocessing and postprocessing scripts |
| model_inferences/ | Stored inference outputs and logs |
| models/ | Pretrained models and configs |
| node-scraper-v3.1/ | Node.js scraper for frame extraction |
| yolov5/ | YOLOv5 detection framework |
| app.py | Flask API entry point |
| model_inferences_and_output.py | Output aggregation and postprocessing |
| stream_worker4.py | Video stream/batch processing worker |


---

## Environment & Setup

env.yaml  
Defines Conda environment (Python + dependencies).

requirements_og.txt  
Original pip requirements list.

node-scraper-v3.1/  
Supports scraping video frames or resources via Node.js.

---

## Installation

Clone the repository and install dependencies:

```git clone https://github.com/stefanedelkovski/Vision-Based-Game-HUD-Extraction-OCR.git```

```cd Vision-Based-Game-HUD-Extraction-OCR```

If using Conda:
```
conda env create -f env.yaml
conda activate vision-hud-ocr
```

Or using pip:
```pip install -r requirements_og.txt```

---

## Usage

### Start the Flask Inference Server

```python app.py```

The server exposes endpoints for submitting game frames and receiving extracted HUD information.

API endpoints (verify in app.py):
POST /infer    - send a game frame and receive HUD extraction results
GET  /status   - optional health check endpoint

---

### Run Inference on a Single Image

```python game_detection_efficientnet_inference.py --input path/to/image.png```

This script runs the detection model on a single frame and outputs HUD-related predictions.

---

### Stream or Batch Processing

```python stream_worker4.py --video path/to/video.mp4```

Continuously processes video frames, applies detection + OCR, and aggregates outputs.

---

## Purpose & Use Cases

- Automated game analytics
- Esports data extraction
- HUD reverse engineering
- Vision-based telemetry where APIs are unavailable
- Dataset creation for game-related ML research

---

## Notes

- Models and thresholds may require tuning for new games or HUD layout.
- Some scripts are experimental and intended for research use.
