import logging
import torch
from torchvision import transforms
from PIL import Image
import json

# # Set up logging
# logging.basicConfig(filename='logs/game_detection.log', level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

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
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
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
