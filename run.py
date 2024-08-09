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
            nn.Conv2d(3, 32, kernel_size=3, padding=1),     # RGB로 분리
            nn.ReLU(),                                      # 활성화
            nn.MaxPool2d(kernel_size=2, stride=2),          # 풀링

            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # 컨볼루션
            nn.ReLU(),                                      # 활성화
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # 컨볼루션
            nn.ReLU(),                                      # 활성화
            nn.MaxPool2d(kernel_size=2, stride=2),          # 풀링
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 컨볼루션
            nn.ReLU(),                                      # 활성화
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 컨볼루션
            nn.ReLU(),                                      # 활성화
            nn.MaxPool2d(kernel_size=2, stride=2),          # 풀링

            nn.Conv2d(512, 1024, kernel_size=3, padding=1), # 컨볼루션
            nn.ReLU(),                                      # 활성화
            nn.Conv2d(1024, 2048, kernel_size=3, padding=1),# 컨볼루션
            nn.ReLU(),                                      # 활성화
            nn.MaxPool2d(kernel_size=2, stride=2),          # 풀링
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(2048 * 8 * 8, 2048),
            nn.ReLU(),
            nn.Linear(2048, 5)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 2048 * 8 * 8)
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
        probabilities = torch.sigmoid(outputs)
    
    return probabilities


image_path = 'test.png'
prob = predict_image(image_path)
prob = [f'{p:.5f}' for p in prob[0]]
print(prob)