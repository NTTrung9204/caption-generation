import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

class InceptionV3Model():
    def __init__(self, device: str) -> None:
        self.device = device
        self.model = self.get_feature_extractor()

    def get_feature_extractor(self) -> torch.nn.Module:
        model = models.inception_v3(pretrained=True)
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        return feature_extractor

    def preprocess_image(self, image_path) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize((299, 299)),   
            transforms.ToTensor(),        
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)

    def extract_features(self, image_list) -> dict[str, torch.Tensor]:
        features = {}
        self.model = self.model.to(self.device)
        for image_path in image_list:
            image_name = os.path.basename(image_path)
            input_tensor = self.preprocess_image(image_path).to(self.device)
            with torch.no_grad():
                feature = self.model(input_tensor)
            features[image_name] = feature.squeeze().cpu().numpy()
        return features
