# Importing the requirements
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Allowed file formats
ALLOWED_FILE_FORMATS = {'png', 'jpg', 'jpeg'}

# Lenet Model
class Lenet5(nn.Module):
    def __init__(self, in_channels=1, out_classes=10):
        super(Lenet5, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(120*1*1, 84),
            nn.Linear(84, out_classes)
        )

    def forward(self, x):
        x = self.network(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

# Model Object
model = Lenet5(in_channels=1, out_classes=10)

# Get model state_dicts(weights)
weight_path = "app/lenet_mnist_weights.pth"
model.load_state_dict(torch.load(weight_path))
model.eval()

# Data normalizing
def get_data_normalized(imageBytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)),
    ])

    image = Image.open(io.BytesIO(imageBytes))
    
    return transform(image).unsqueeze(0)

def get_prediction(image_data):

    pred = model(image_data)

    _, prediction = torch.max(pred.data, 1)
    
    return prediction
    
def check_file_format(filename):

    return '.' in filename and filename.split('.')[1] in ALLOWED_FILE_FORMATS
