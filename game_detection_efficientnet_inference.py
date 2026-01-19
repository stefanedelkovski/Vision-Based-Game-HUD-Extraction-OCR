import torch
import torch.nn as nn
from PIL import Image
import json
from torchvision import datasets, transforms, models
import torch.nn.functional as F


# Load the trained model
def load_model(model_path):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 79)  # Adjust to your number of classes
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model.eval()
    return model.to(torch.device('cuda'))


# Preprocess the image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img


# Predict the class of the image and return the confidence
def predict_image(model, img):
    with torch.no_grad():
        outputs = model(img)
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        # Get the predicted class and confidence
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()


# Map the prediction to the class label
def load_class_mapping(mapping_path):
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    return {int(v): k for k, v in class_mapping.items()}


# Main function
def main(image_path, model_path, mapping_path):
    model = load_model(model_path)
    img = preprocess_image(image_path)
    predicted_class, confidence = predict_image(model, img)
    class_mapping = load_class_mapping(mapping_path)
    print(f"Predicted class: {class_mapping[predicted_class]}, Confidence: {confidence:.4f}")


if __name__ == '__main__':
    # Replace these paths with your actual file paths
    image_path = 'test_imgs/frame_1.jpg'  # Path to the image you want to classify
    model_path = 'best_game_classifier_efficientnet_b0.h5'  # Path to your trained model
    mapping_path = 'class_indices.json'  # Path to the class mapping JSON file

    main(image_path, model_path, mapping_path)
