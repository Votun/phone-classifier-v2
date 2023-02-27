# This py script contains
import torch
from torch import nn
from torchvision import transforms

from classifier.preprocess import PhoneDataset


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=8)
        self.conv2 = ConvBlock(in_channels=8, out_channels=16)
        self.conv3 = ConvBlock(in_channels=16, out_channels=32)
        self.conv4 = ConvBlock(in_channels=32, out_channels=64)
        self.conv5 = ConvBlock(in_channels=64, out_channels=96)
        self.out = nn.Linear(96 * 5 * 5, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)
        logits = self.out(x)
        return logits

def judge_image(image_file):
    image = PhoneDataset.load_sample(image_file)
    image = PhoneDataset.prepare_sample(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image)
    model = SimpleCNN(2)
    checkpoint = torch.load("classifier/models/model_simple.pth")
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    logits = model(image)
    result = torch.argmax(logits, dim=1)
    print(result)
    judgement = bool(result)
    return judgement


if __name__ == "__main__":
    pretrained_model = torch.load('./models/model_ft.pth', map_location=torch.device('cpu'))
    image_path = "./"
    print(judge_image(image_path, pretrained_model))
