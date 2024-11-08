import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

# GPU 가용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1024 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 5)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 1024 * 16 * 16)
        x = self.fc_layers(x)
        return x

# 모델과 옵티마이저 초기화
model = CNN().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 모델 불러오기
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 이미지 예측
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
    
    return outputs


image_path = 'test.png'
prob = predict_image(image_path)
prob = [f'{p:.5f}' for p in prob[0]]
dict = ['차은우', '송강', '정국', '원빈', '뷔']
for p in prob:
    print(dict[prob.index(p)] + ': ' + p)
image_path = 'test1.png'
prob = predict_image(image_path)
prob = [f'{p:.5f}' for p in prob[0]]
for p in prob:
    print(dict[prob.index(p)] + ': ' + p)