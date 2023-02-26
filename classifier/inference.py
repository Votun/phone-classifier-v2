# This py script contains
import torch
from torchvision import transforms

import preprocess as prep
from preprocess import PhoneDataset


def judge_image(image_file, model):
    image = PhoneDataset.load_sample(image_file)
    image = PhoneDataset.prepare_sample(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    logits = model(image)
    result = torch.argmax(logits, dim=1)
    print(result)
    judgement = bool(result)
    return judgement


if __name__ == "__main__":
    pretrained_model = torch.load('./models/model_ft.pth', map_location=torch.device('cpu'))
    image_path = "./"
    print(judge_image(image_path, pretrained_model))
