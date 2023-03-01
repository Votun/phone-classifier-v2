# This py script contains
import os

import torch
from torchvision import transforms
from classifier.preprocess import PhoneDataset
from classifier.models.simple import ConvBlock, SimpleCNN
TEST_MODE = False

def judge_image(image_file):
    """
        Accepts image file and processes it in predefined and pretrained model.
        Baseline model simple_Cnn checks whether it is a phone on the image or not.
        params:
            image_file: path to image
        output: bool judgement
    """
    # Image preprocess repeats steps from preprocess PhoneDataset class.
    image = PhoneDataset.load_sample(image_file)
    image = PhoneDataset.prepare_sample(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    # Models work with data batches, thus single image tensor demands reshaping via torch.view.
    # Models receive image batches of shape: [batch_size, channels, img_size, img_size]
    image = image.view(1, 3, 224, 224)

    # Loading pretrained model.
    if TEST_MODE:
        model_path = "trained_models/model_simple.pth"
    else:
        model_path = "./classifier/trained_models/model_simple.pth"
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()

    #Results.
    logits = model(image)
    result = torch.argmax(logits, dim=1)
    print(result, logits)
    judgement = bool(result)
    return judgement


if __name__ == "__main__":
    TEST_MODE = True
    print(judge_image("../test_inference/IPhone.jpg"))
    print(judge_image("../test_inference/cat.jpg"))
    print(judge_image("../test_inference/IPhone2.jpg")) # from train set
    print(judge_image("../test_inference/iPhone-3.jpg"))
    print(judge_image("../test_inference/IPhone4.jpg"))
    print(judge_image("../test_inference/randomPhone.jpg"))
    print(judge_image("../test_inference/test.jpg"))
