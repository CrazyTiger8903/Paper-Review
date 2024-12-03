"""
boundary unlearning: rapid forgetting of deep networks via shifting the decision boundary(CVPR 2023)

Retrain 모델(Dr, 즉 forgetting data를 제외한 데이터세트로 학습한 모델) 학습 및 시각화 코드
Train Remainset Accuracy (others): 99.99%
Train Forgetset Accuracy (deer): 0.00%
Test Remainset Accuracy (others): 83.93%
Test Forgetset Accuracy (deer): 0.00%

to do : 모듈화
"""
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

save_dir = "./Result"
os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 데이터셋 분리 함수
def split_datasets(dataset, target_class):
    forget_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
    remain_indices = [i for i, (_, label) in enumerate(dataset) if label != target_class]
    return Subset(dataset, forget_indices), Subset(dataset, remain_indices)

# 삭제 대상 클래스 - deer(4)
target_class = 4

train_forgetset, train_remainset = split_datasets(train_dataset, target_class)
test_forgetset, test_remainset = split_datasets(test_dataset, target_class)

train_forget_loader = DataLoader(train_forgetset, batch_size=64, shuffle=True)
train_remain_loader = DataLoader(train_remainset, batch_size=64, shuffle=True)
test_forget_loader = DataLoader(test_forgetset, batch_size=64, shuffle=False)
test_remain_loader = DataLoader(test_remainset, batch_size=64, shuffle=False)

# ResNet18 모델 초기화
def create_model(num_classes=10):
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  
    return model

# 학습 함수
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 평가 함수
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 학습
best_accuracy = 0.0 
epoch = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrain_model = create_model(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(retrain_model.parameters(), lr=0.001, momentum=0.9)

# 학습 루프
for i in range(epoch):
    train_model(retrain_model, train_remain_loader, criterion, optimizer, device)
    accuracy = evaluate_model(retrain_model, test_remain_loader, device)
    print(f"Epoch {i+1}, Test Accuracy: {accuracy * 100:.2f}%")

    # 최고 성능 저장
    if accuracy > best_accuracy:
        best_accuracy = accuracy

        best_weight_path = os.path.join(save_dir, "retrain_model.pth")
        torch.save(retrain_model.state_dict(), best_weight_path)
        print(f"New best accuracy: {accuracy * 100:.2f}% - Weights saved at {best_weight_path}")

# 학습 완료
print("Training complete. Best weights saved as 'best_model.pth'")


# TSNE 시각화 함수
def plot_tsne(model, remain_loader, forget_loader, device, title, classes, save_path):
    model.eval()  # 모델을 평가 모드로 전환
    features = []  # 추출된 특징 저장 리스트
    predictions = []  # 예측 클래스 저장 리스트
    dataset_type = []  # "Remain" 또는 "Forget" 구분 저장 리스트

    # Remainset 데이터에서 특징 및 예측 결과 추출
    with torch.no_grad():
        for inputs, _ in remain_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())  # 모델 출력 특징 저장
            predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())  # 예측 결과 저장
            dataset_type.extend(["Remain"] * inputs.size(0))  # Remainset 데이터 구분 추가

    # Forgetset 데이터에서 특징 및 예측 결과 추출
    with torch.no_grad():
        for inputs, _ in forget_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs.cpu().numpy())  # 모델 출력 특징 저장
            predictions.append(torch.argmax(outputs, dim=1).cpu().numpy())  # 예측 결과 저장
            dataset_type.extend(["Forget"] * inputs.size(0))  # Forgetset 데이터 구분 추가

    # 데이터를 합침
    features = np.concatenate(features, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    dataset_type = np.array(dataset_type)

    # TSNE로 차원 축소
    # tsne = TSNE(n_components=2, perplexity=40, learning_rate=300, n_iter=3000, random_state=42)
    tsne = TSNE(n_components=2, perplexity=50, learning_rate=500, n_iter=4000, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # 시각화
    plt.figure(figsize=(12, 10))
    colors = plt.cm.Paired(np.linspace(0, 1, len(classes))) 

    # Remainset 데이터 시각화
    remain_features = reduced_features[dataset_type == "Remain"]
    remain_predictions = predictions[dataset_type == "Remain"]
    for i, class_name in enumerate(classes):
        indices_remain = remain_predictions == i 
        plt.scatter(
            remain_features[indices_remain, 0],
            remain_features[indices_remain, 1],
            label=f"Class {i+1} ({class_name})",
            alpha=0.5, 
            s=30,  
            color=colors[i]
        )
    
    # Forgetset 데이터 시각화
    forget_features = reduced_features[dataset_type == "Forget"]
    forget_predictions = predictions[dataset_type == "Forget"]
    for i, class_name in enumerate(classes):
        indices_forget = forget_predictions == i 
        plt.scatter(
            forget_features[indices_forget, 0],
            forget_features[indices_forget, 1],
            alpha=0.6,
            s=30,
            edgecolor="black",
            color=colors[i]
        )

    plt.title(title, fontsize=20, fontweight="bold")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small", markerscale=1.0)
    plt.xticks([]) 
    plt.yticks([])  
    plt.grid(False)  
    plt.tight_layout()

    # 시각화 이미지 저장
    plt.savefig(save_path, dpi=300) 
    plt.close()
    print(f"t-SNE visualization saved at {save_path}")

# 'Retrain' 모델 로드 및 평가
final_model = create_model(num_classes=10).to(device)
final_model.load_state_dict(torch.load(best_weight_path))
final_model.eval()  
print(f"Loaded best weights from {best_weight_path}")

train_forget_acc = evaluate_model(final_model, train_forget_loader, device)
train_remain_acc = evaluate_model(final_model, train_remain_loader, device)
test_forget_acc = evaluate_model(final_model, test_forget_loader, device)
test_remain_acc = evaluate_model(final_model, test_remain_loader, device)

# 결과 출력
print(f"Train Remainset Accuracy (others): {train_remain_acc * 100:.2f}%")
print(f"Train Forgetset Accuracy (deer): {train_forget_acc * 100:.2f}%")
print(f"Test Remainset Accuracy (others): {test_remain_acc * 100:.2f}%")
print(f"Test Forgetset Accuracy (deer): {test_forget_acc * 100:.2f}%")

# TSNE 시각화
tsne_predictions_path = os.path.join(save_dir, "retrain.png")
print("\nGenerating TSNE visualization with Forgetset predictions...")

plot_tsne(
    final_model,
    test_remain_loader,
    test_forget_loader,
    device,
    "TSNE Visualization(Retrain Model)",
    classes,
    tsne_predictions_path
)
