"""
boundary unlearning: rapid forgetting of deep networks via shifting the decision boundary(CVPR 2023)

Random Labels 모델(랜덤으로 라벨링한 forgetting data를 사용해 original 모델을 파인튜닝) 학습 및 시각화 코드

Train Remainset Accuracy (others): 89.39%
Train Forgetset Accuracy (deer): 8.88%
Test Remainset Accuracy (others): 73.58%
Test Forgetset Accuracy (deer): 6.70%

to do :
- 모듈화
- ASR 추가
"""

"""
Selective forgetting of deep networks at a finer level than samples(CoRR 2020)

[랜덤 라벨 디스틸레이션(Random Label Distillation, RLD)]
특정 입력 데이터가 네트워크에 무작위 라벨을 학습하도록 강제하여 기존 학습된 정보를 "망각" 시킴

[학습 목표]
Df(잊어야할 데이터)에 대한 성능은 떨어뜨리고, Dr(나머지 데이터)의 성능은 유지하는 것이 목표
이를 위해 2가지의 손실 함수를 결합하여 사용

1. 망각 손실(Lf)
Df 데이터에 대한 성능을 낮추기 위해 사용
랜덤 라벨 디스틸레이션(Random Label Distillation, RLD)을 사용하여 Df를 망각 시킴

2. 기억 손실(R)
Dr 데이터에 대한 성능을 유지하기 위해 사용
Elastic Weight Consolidation(EWC) 방식을 사용하여 모델이 기존 데이터의 중요한 특징을 유지하도록 정규화
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
import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 랜덤 시드 고정
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 데이터셋 분리 함수
def split_datasets(dataset, target_class):
    forget_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
    remain_indices = [i for i, (_, label) in enumerate(dataset) if label != target_class]
    return Subset(dataset, forget_indices), Subset(dataset, remain_indices)

target_class = 4  # deer

train_forgetset, train_remainset = split_datasets(train_dataset, target_class)
test_forgetset, test_remainset = split_datasets(test_dataset, target_class)

train_forget_loader = DataLoader(train_forgetset, batch_size=64, shuffle=True)
train_remain_loader = DataLoader(train_remainset, batch_size=64, shuffle=True)
test_forget_loader = DataLoader(test_forgetset, batch_size=64, shuffle=False)
test_remain_loader = DataLoader(test_remainset, batch_size=64, shuffle=False)

def create_model(num_classes=10):
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 랜덤 라벨 생성 함수
def generate_random_labels(labels, num_classes):
    random_labels = torch.randint(0, num_classes, labels.size(), device=labels.device)
    while (random_labels == labels).any():
        random_labels = torch.randint(0, num_classes, labels.size(), device=labels.device)
    return random_labels

# Random Label Distillation(RLD) Loss : CrossEntropyLoss 사용
# 망각 손실(Lf) 참고
def random_label_distillation_loss(outputs, random_labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, random_labels)

# Elastic Weight Consolidation (EWC) Loss
# 기억 손실(R) 참고
def ewc_loss(model, fisher_matrix, old_weights, lambda_ewc):
    ewc_loss_value = 0
    for name, param in model.named_parameters():
        if name in fisher_matrix:
            fisher = fisher_matrix[name]   # fisher 정보(파라미터 중요도)
            old_weight = old_weights[name] # original 모델의 파라미터 값
            # (현재 파라미터 - 과거 파라미터)^2에 Fisher 정보로 가중치 부여 후 합산
            ewc_loss_value += (fisher * (param - old_weight).pow(2)).sum()
    # EWC 손실에 가중치 부여
    return lambda_ewc * ewc_loss_value

# Fisher 정보 계산 함수
# original 모델로부터 파라미터 중요도 계산
def calculate_fisher_information(model, dataloader, device):
    # fisher행렬 초기화(모델의 각 파라미터에 대해 0으로 초기화) 
    fisher_matrix = {name: torch.zeros_like(param) for name, param in model.named_parameters() if param.requires_grad}
    model.eval()

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        # fθ(x) 
        outputs = model(inputs)
        # Lcls(fθ(x),l)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        # ∂Lcls(fθ(x),l) / ∂θi
        loss.backward()

        # 최종 fisher 정보 행렬
        # 손실 함수의 기울기 제곱의 평균으로 정의
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_matrix[name] += param.grad.pow(2) / len(dataloader)

    return fisher_matrix

# 학습 함수
def train_model(model, dataloader, optimizer, fisher_matrix, old_weights, lambda_ewc, device):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 랜덤 라벨 생성
        random_labels = generate_random_labels(labels, len(classes))
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # 망각 손실 (RLD)
        rld_loss = random_label_distillation_loss(outputs, random_labels)
        
        # 기억 손실 (EWC)
        ewc_loss_value = ewc_loss(model, fisher_matrix, old_weights, lambda_ewc)
        
        # 총 손실
        loss = rld_loss + ewc_loss_value
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

# 모델 초기화 및 학습
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Random_Labels_Model = create_model(num_classes=len(classes)).to(device)
original_model = create_model(num_classes=len(classes)).to(device)

original_weights_path = "./Result/Org_Model.pth"
original_model.load_state_dict(torch.load(original_weights_path))
print(f"Loaded original model weights from {original_weights_path}")

Random_Labels_Model.load_state_dict(copy.deepcopy(original_model.state_dict()))

# original_model 가중치
old_weights = {name: param.clone().detach() for name, param in original_model.named_parameters()}
# 전체 데이터에 대해 Fisher 정보 행렬 계산
fisher_matrix = calculate_fisher_information(original_model, train_loader, device)

# We set λKL = 10^5, lr = 10^−5 throughout experiments.
optimizer = optim.SGD(Random_Labels_Model.parameters(), lr=1e-5)
lambda_ewc = 1e5
best_tradeoff_score = -float('inf')  

for epoch in range(10):
    print(f"Epoch {epoch + 1}/10")
    train_model(Random_Labels_Model, train_forget_loader, optimizer, fisher_matrix, old_weights, lambda_ewc, device)

    #train_remain_acc = evaluate_model(Random_Labels_Model, train_remain_loader, device)
    test_remain_acc = evaluate_model(Random_Labels_Model, test_remain_loader, device)
    #train_forget_acc = evaluate_model(Random_Labels_Model, train_forget_loader, device)
    test_forget_acc = evaluate_model(Random_Labels_Model, test_forget_loader, device)

    # print(f"Train Remainset Accuracy: {train_remain_acc * 100:.2f}%")
    # print(f"Train Forgetset Accuracy: {train_forget_acc * 100:.2f}%")
    # print(f"Test Remainset Accuracy: {test_remain_acc * 100:.2f}%")
    # print(f"Test Forgetset Accuracy: {test_forget_acc * 100:.2f}%")

    tradeoff_score = test_remain_acc - test_forget_acc

    if tradeoff_score > best_tradeoff_score:
        best_tradeoff_score = tradeoff_score
        best_weight_path = os.path.join(save_dir, "Random_Labels_Model.pth")
        torch.save(Random_Labels_Model.state_dict(), best_weight_path)
        print(f"New best trade-off score: {tradeoff_score:.4f} - Model saved to {best_weight_path}")

print("Training complete.")

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


final_model = create_model(num_classes=10).to(device)
final_model.load_state_dict(torch.load(best_weight_path))
final_model.eval()  
print(f"Loaded best weights from {best_weight_path}")

test_acc = evaluate_model(final_model, test_loader, device)
train_forget_acc = evaluate_model(final_model, train_forget_loader, device)
train_remain_acc = evaluate_model(final_model, train_remain_loader, device)
test_forget_acc = evaluate_model(final_model, test_forget_loader, device)
test_remain_acc = evaluate_model(final_model, test_remain_loader, device)

# 결과 출력
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Train Remainset Accuracy (others): {train_remain_acc * 100:.2f}%")
print(f"Train Forgetset Accuracy (deer): {train_forget_acc * 100:.2f}%")
print(f"Test Remainset Accuracy (others): {test_remain_acc * 100:.2f}%")
print(f"Test Forgetset Accuracy (deer): {test_forget_acc * 100:.2f}%")

# TSNE 시각화
tsne_predictions_path = os.path.join(save_dir, "Random_Labels_Model.png")
print("\nGenerating TSNE visualization with Forgetset predictions...")

plot_tsne(
    final_model,
    test_remain_loader,
    test_forget_loader,
    device,
    "TSNE Visualization(Random Labels Model)",
    classes,
    tsne_predictions_path
)