import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# GPU 가용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, subdir in enumerate(os.listdir(root_dir)):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for img_name in os.listdir(subdir_path):
                    img_path = os.path.join(subdir_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 데이터셋 경로
root_dir = './dataset'

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 데이터셋 생성
dataset = Dataset(root_dir=root_dir, transform=transform)

# 데이터로더 설정
batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

model = CNN().to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 학습 함수
def train_model(model, data_loader, criterion, optimizer, num_epochs=30):
    for epoch in range(num_epochs):
        with tqdm(data_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item())

    # 모델 저장
    torch.save(model.state_dict(), 'model.pth')
    print("model saved.")
    
if __name__ == '__main__':
    train_model(model, data_loader, criterion, optimizer)
