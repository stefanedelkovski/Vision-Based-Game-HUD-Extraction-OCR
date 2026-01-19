import logging
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms, models
from PIL import Image
import json
import torch.nn.functional as F

# Set up logging
logging.getLogger(__name__)

# Load class indices mapping
with open('models/class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the class indices mapping
game_name_mapping = {v: k for k, v in class_indices.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Load the new EfficientNet model
def load_model(model_path):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 79)  # Adjust to your number of classes
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Detect games in the cropped images using the new model
def detect_game(cropped_images, mapping, model):
    results = []
    confidence_threshold = 0.8

    for idx, image in enumerate(cropped_images):
        try:
            image = Image.fromarray(image.astype('uint8'), 'RGB')
            image = transform(image).unsqueeze(0)  # Add batch dimension
            image = image.to(device)

            with torch.no_grad():
                outputs = model(image)
                probabilities = F.softmax(outputs, dim=1)
                confidence, preds = torch.max(probabilities, 1)

            game_confidence = confidence.item()
            game_index = preds.item()
            game_name = game_name_mapping.get(game_index, "Unknown")

            if game_confidence < confidence_threshold:
                game_name = "Unknown"

            logging.info(f"Frame {mapping[idx][0]}, Detection {idx}: Game={game_name}, Confidence={game_confidence}")

            results.append({
                'frame_index': mapping[idx][0],
                'bbox': mapping[idx][1],
                'confidence': game_confidence,
                'game': game_name
            })
        except Exception as e:
            logging.error(f"Error in detect_game for frame {mapping[idx][0]}, detection {idx}: {e}")

    return results
