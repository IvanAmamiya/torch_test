import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Net(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Net, self).__init__()
        self.model = models.resnet18(pretrained=False)
        # 修改最后的全连接层以适应CIFAR-10类别数
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(weights=None):
    model = ResNet18Net()
    if weights:
        model.load_state_dict(torch.load(weights))
    return model

def predict(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return predicted.item()