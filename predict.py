import torch
from torchvision import transforms
from PIL import Image

import config
from model import build_model

predict_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================
# Load Trained Model
# =====================================

model = build_model()

model.load_state_dict(
    torch.load(
        config.MODEL_PATH,
        map_location=config.DEVICE
    )
)

model.eval()

print("Model Loaded Successfully!")

# =====================================
# Prediction Function
# =====================================

def predict_image(image_path):

    # Open image
    image = Image.open(image_path).convert("RGB")

    # Apply preprocessing
    image_tensor = predict_transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to device
    image_tensor = image_tensor.to(config.DEVICE)

    # Disable gradients
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)

    predicted_class = config.CLASS_NAMES[predicted.item()]

    return (
        predicted_class,
        confidence.item(),
        probabilities.squeeze()
    )

# =====================================
# Test Prediction
# =====================================

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    predicted_class, confidence, probabilities = predict_image(image_path)

    print("\n" + "=" * 60)
    print("Prediction Result")
    print("=" * 60)

    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence      : {confidence * 100:.2f}%")

    print("\nClass Probabilities")
    print("-" * 60)

    for class_name, probability in zip(
        config.CLASS_NAMES,
        probabilities
    ):
        print(
            f"{class_name:<12} : {probability.item()*100:.2f}%"
        )