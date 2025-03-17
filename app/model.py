import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class StrokeCNN(nn.Module):
    def __init__(self):
        super(StrokeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3 channels, 64 filters
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 26 * 37, 128)  # Adjust based on the feature map size
        self.fc2 = nn.Linear(128, 1)  # 1 output instead of 2

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 26 * 37)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model():
    model_path = "app/stroke_cnn (1).pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = StrokeCNN()  # Ensure this matches the trained architecture
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)  # Allow partial loading
    model.to(device)
    model.eval()
    
    return model

def predict(image_path, model, hypertension=0, bmi=0, heart_disease=0):
    """Predict stroke probability with optional additional factors."""
    base_transform = transforms.Compose([
        transforms.Resize((215, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    aug_transform = transforms.Compose([
        transforms.Resize((215, 300)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    
    original_image = base_transform(image).unsqueeze(0)
    augmented_image = aug_transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_image = original_image.to(device)
    augmented_image = augmented_image.to(device)

    with torch.no_grad():
        output1 = model(original_image)
        output2 = model(augmented_image)
        
        prob1 = torch.sigmoid(output1).item()
        prob2 = torch.sigmoid(output2).item()
        average_prob = (prob1 + prob2) / 2

    # **Modify this part later to use hypertension, bmi, heart_disease**
    return average_prob * 100



def analyze_images(image_paths):
    """Process multiple images and return predictions."""
    model = load_model()
    results = {}
    for image_path in image_paths:
        results[image_path] = predict(image_path, model)
    return results

if __name__ == "__main__":
    model = load_model()
    test_images = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]  # Replace with actual paths
    results = analyze_images(test_images)
    print(results)
