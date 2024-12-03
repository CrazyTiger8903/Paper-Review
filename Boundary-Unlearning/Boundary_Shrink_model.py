"""
boundary unlearning: rapid forgetting of deep networks via shifting the decision boundary(CVPR 2023)

Boundary Shrink 모델 학습 및 시각화(https://www.dropbox.com/scl/fi/j3hgtrvp1vjptk9qz5ck6/Boundary-Unlearning-Code.zip?rlkey=h32gro8ysi4umtolzmi54gbdo&e=1&dl=0)

Train Remainset Accuracy (others): 94.02%
Train Forgetset Accuracy (deer): 3.22%
Test Remainset Accuracy (others): 77.32%
Test Forgetset Accuracy (deer): 2.50%

to do :
- 모듈화
- ASR 추가
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
from tqdm import tqdm  
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.distributions as distributions
import torch
from torch import nn
from tqdm import tqdm

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def split_datasets(dataset, target_class):
    forget_indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
    remain_indices = [i for i, (_, label) in enumerate(dataset) if label != target_class]
    return Subset(dataset, forget_indices), Subset(dataset, remain_indices)

target_class = 4 
train_forgetset, train_remainset = split_datasets(train_dataset, target_class)
test_forgetset, test_remainset = split_datasets(test_dataset, target_class)

train_forget_loader = DataLoader(train_forgetset, batch_size=64, shuffle=True)
train_remain_loader = DataLoader(train_remainset, batch_size=64, shuffle=True)
test_forget_loader = DataLoader(test_forgetset, batch_size=64, shuffle=False)
test_remain_loader = DataLoader(test_remainset, batch_size=64, shuffle=False)

def create_model(num_classes=10):
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

"""
adversarial attack이란? : perturbation(사람의 눈으로는 구별 불가능하지만, 모델의 예측에 영향을 주도록 하는 노이즈)를 추가하여 모델의 오분류를 유도하는 공격 기법
대표적인 기법으로는 FGSM과 PGD가 존재한다.

1. FGSM(Fast Gradient Sign Method)
    Explaining and Harnessing Adversarial Examples(ICLR 2015)
    오분류를 일으킬 수 있는 최적의 perturbation(노이즈, 델타)를 찾는 방법
    - 손실함수 L의 입력 x에 대한 gradient를 계산.
    - sign을 취하여 부호(방향)만을 추출
    - 일정한 값(엡실론)을 추가
    한번의 기울기 계산만 수행하므로 빠르고 계산 비용이 낮다.

2. PGD(Projected Gradient Descent)
    Towards Deep Learning Models Resistant to Adversarial Attacks(ICLR 2018)
    FGSM을 반복 수행하여 gradient를 업데이트
    각 과정에서  step size(=lr) 지정
    -> 더 강력한 adversarial sample을 생성 가능하다.
    각 과정에서 projection을 통해 perturbation(델타)을 일정 범위(L-infinity norm)내로 제한한다.
"""
class LinfPGD:
    def __init__(self, model=None, bound=None, step=None, iters=None, norm=False, random_start=False, device=None):
        """
        Args:
            model : 공격 대상 모델 - test 모델
            bound (float): 허용 범위 (L-infinity 기준 epsilon 값) - 0.1
            step (float): 한 번의 업데이트에서 변경할 크기 (step size) - 2 / 255
            iters (int): 적대적 샘플 생성을 반복할 횟수 - 5
            norm (bool): 입력 데이터를 정규화할지 여부 - True
            random_start (bool): 초기 샘플에 무작위 변동을 추가할지 여부 - True
            device
        """       
        self.model = model 
        self.bound = bound
        self.step = step
        self.iter = iters
        self.norm = norm
        if self.norm:
            # CIFAR10 데이터셋 기준 정규화 평균 및 표준편차
            self.mean = (0.4914, 0.4822, 0.2265)
            self.std = (0.2023, 0.1994, 0.2010)
        self.rand = random_start
        self.device = device
        # CrossEntropyLoss 사용
        self.criterion = nn.CrossEntropyLoss().to(device)

    def perturb(self, x, y, model=None, bound=None, step=None, iters=None, x_nat=None, device=None):
        """
        PGD(Projected Gradient Descent) attack을 수행하여 적대적 샘플 생성

        Args:
            x : 원본 입력 데이터
            y : 원본 입력 데이터의 정답 라벨
            model : 공격 대상 모델 - test 모델
            bound (float): 허용 범위 (L-infinity 기준 epsilon 값) - 0.1
            step (float): 한 번의 업데이트에서 변경할 크기 (step size) - 2 / 255
            iters (int): 적대적 샘플 생성을 반복할 횟수 - 5
            x_nat (torch.Tensor): 정규화되지 않은 원본 입력 데이터 - None
            device
        """
        criterion = self.criterion   # CrossEntropyLoss
        model = model or self.model  
        bound = bound or self.bound
        step = step or self.step
        iters = iters or self.iter
        device = device or self.device

        # 모델 그레디언트 초기화
        model.zero_grad()

        # 정규화되지 않은 원본 데이터를 생성 (기본 설정).
        # torch.detach().clone() : 그래디언트 계산 그래프에서 분리 후 복사하여 새로운 메모리 공간에 저장
        if x_nat is None:
            x_nat = self.inverse_normalize(x.detach().clone().to(device))
        else:
            x_nat = self.inverse_normalize(x_nat.detach().clone().to(device))
        
        # 적대적 샘플 초기화
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        
        # 적대적 샘플에 노이즈 추가한 상태로 시작
        if self.rand:
            # 허용 범위 내에서 랜덤한 노이즈를 생성
            rand_perturb_dist = distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            # 생성한 노이즈 추가 및 범위 제한
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat),
                                 bound=bound, inverse_normalized=True)
            # 노이즈 추가한 초기 x_adv 정규화 및 이산화
            x_adv = self.normalize(self.discretize(x_adv)).detach().clone().requires_grad_(True)

        # PGD attack 반복 -> 최적의 적대적 샘플 반환
        for i in range(iters):
            # 현재 적대적 샘플에 대한 모델(original 모델) 출력
            adv_pred = model(x_adv)
            # loss 계산. 모델 예측값과 실제 정답 간의 손실 계산
            loss = criterion(adv_pred, y)
            # 손실함수의 그래디언트 계산 - 수식(2)
            loss.backward()
            # 그래디언트의 부호 계산 - 수식(2)
            grad_sign = x_adv.grad.data.detach().sign()
            # x_adv를 업데이트: x_adv에 step size만큼 perturbation 추가 - 수식(2)
            x_adv = self.inverse_normalize(x_adv) + grad_sign * step
            # 범위 제한
            x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)
            model.zero_grad()

        # 최종적으로 적대적 샘플 반환(노이즈가 추가된 x)
        return x_adv.detach().to(device)
    
    # 입력 데이터 정규화
    def normalize(self, x):
        # x shape : (batch_size, channel, height, width)
        # norm이 True일 경우에는 정규화 후 반환
        # norm이 False일 경우에는 그대로 반환
        if self.norm:
            y = x.clone().to(x.device)
            # 각 체널별로 정규화 (x - 평균)/표준편차       
            y[:, 0, :, :] = (y[:, 0, :, :] - self.mean[0]) / self.std[0]  # R체널
            y[:, 1, :, :] = (y[:, 1, :, :] - self.mean[1]) / self.std[1]  # G체널
            y[:, 2, :, :] = (y[:, 2, :, :] - self.mean[2]) / self.std[2]  # B체널
            return y
        return x

    # 입력 데이터 역정규화
    def inverse_normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            y[:, 0, :, :] = y[:, 0, :, :] * self.std[0] + self.mean[0]
            y[:, 1, :, :] = y[:, 1, :, :] * self.std[1] + self.mean[1]
            y[:, 2, :, :] = y[:, 2, :, :] * self.std[2] + self.mean[2]
            return y
        return x
    
    # 데이터 이산화 -> 계산 이득?
    def discretize(self, x):
        return torch.round(x * 255) / 255
    
    # 적대적 샘플(x_adv)이 원본 샘플(x_nat)의 허용 범위(bound)를 넘지 않도록 제한(clamp)하는 함수
    # torch.clamp(x, a, b) : x의 값을 a, b 범위 안으로 조정해주는 함수
    def clamper(self, x_adv, x_nat, bound=None, inverse_normalized=False):
        # 역정규화 안되어 있으면 역정규화 진행
        if not inverse_normalized:
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)

        # 픽셀 값 차이를 bound로 제한
        clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)

        # 노이즈 더해주기
        x_adv = x_nat + clamp_delta
        # 범위 제한(0~1)사이로
        x_adv = torch.clamp(x_adv, 0., 1.)
        # 이산화(discretize) 및 정규화(normalize) 후 반환, 이후 추가 그래디언트 계산 가능하도록 requires_grad 설정
        return self.normalize(self.discretize(x_adv)).clone().detach().requires_grad_(True)


def inf_generator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

###################################################################################################################################################

# boundary_shrink
def boundary_shrink(ori_model, train_forget_loader, dt, dv, test_loader, device, evaluate,
                    bound=0.1, step=2 / 255, iter=5, poison_epoch=10, forget_class=0, path='./'):
    
    # 정규화, 무작위 시작
    norm = True 
    random_start = True

    # test_model : 적대적 샘플 생성에 사용
    # unlearn_model : 학습(unlearning)에 사용
    test_model = copy.deepcopy(ori_model).to(device)
    unlearn_model = copy.deepcopy(ori_model).to(device)

    # LinfPGD 인스턴스 생성(적대적 샘플 생성을 위한)
    adv = LinfPGD(test_model, bound, step, iter, norm, random_start, device)

    # loss / 옵티마이저 설정 
    # 논문 : For the fine-tune process in Boundary Unlearning we use a learning rate of 10−5 for 10 epochs.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(unlearn_model.parameters(), lr=1e-5, momentum=0.9)
    
    # num_hits = 0
    # num_sum = 0
    best_tradeoff_score = -float('inf')  
    best_weight_path = os.path.join(path, "shrink_model.pth")

    # boundary shrink 학습 과정
    for epoch in range(poison_epoch):
        print(f"Epoch {epoch + 1}/{poison_epoch}")

        for x, y in tqdm(train_forget_loader, desc=f"Epoch {epoch + 1} Batches"):
            x, y = x.to(device), y.to(device)

            # test모델(적대적 샘플 생성을 위한 모델)을 사용해 적대적 샘플을 생성
            # 적대적 샘플 : 노이즈 추가된 x
            test_model.eval()
            x_adv = adv.perturb(x, y, model=test_model, device=device)

            # 적대적 샘플에 대한 예측값 출력 및 예측 라벨 계산
            adv_logits = test_model(x_adv)
            pred_label = torch.argmax(adv_logits, dim=1)

            # num_hits += (y != pred_label).float().sum()
            # num_sum += y.shape[0]

            # unlearn모델 학습
            unlearn_model.train()
            optimizer.zero_grad()
            # unlearn모델에 적대적 샘플을 넣어 모델의 예측 출력
            ori_logits = unlearn_model(x)
            # loss 계산 - 모델의 예측값과 적대적 샘플의 라벨(바뀐 라벨) 손실 계산
            loss = criterion(ori_logits, pred_label)
            # 손실함수 그래디언트 계산 및 업데이트
            loss.backward()
            optimizer.step()

        test_remain_acc = evaluate_model(unlearn_model, test_remain_loader, device)
        test_forget_acc = evaluate_model(unlearn_model, test_forget_loader, device)
        
        print(f"Test Remainset Accuracy: {test_remain_acc * 100:.2f}%")
        print(f"Test Forgetset Accuracy: {test_forget_acc * 100:.2f}%")
        tradeoff_score = test_remain_acc - test_forget_acc

        if tradeoff_score > best_tradeoff_score:
            best_tradeoff_score = tradeoff_score
            torch.save(unlearn_model.state_dict(), best_weight_path)
            print(f"New best trade-off score: {tradeoff_score:.4f} - Model saved to {best_weight_path}")

    # asr = (num_hits / num_sum).float()
    # print('Attack Success Ratio (ASR):', asr)

    return unlearn_model

###################################################################################################################################################


# 정확도 평가 함수
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

# 모델 생성 및 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ori_model = create_model(num_classes=10).to(device)
original_weights_path = "./Result/Org_Model.pth"
ori_model.load_state_dict(torch.load(original_weights_path))
print(f"Original model loaded from {original_weights_path}")

save_dir = "./Result"
os.makedirs(save_dir, exist_ok=True)
evaluation = ''

unlearn_model = boundary_shrink(ori_model, train_forget_loader, train_dataset, test_dataset,
                                                    test_loader, device, evaluation,
                                                    forget_class=target_class, path=save_dir)



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

# 디바이스 설정 및 모델 로드
best_weight_path = os.path.join(save_dir, "shrink_model.pth")
final_model = create_model(num_classes=10).to(device)
final_model.load_state_dict(torch.load(best_weight_path))

# 각 데이터셋에서 정확도 평가
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
tsne_predictions_path = os.path.join(save_dir, "Boundary_Shrink_model.png")
print("\nGenerating TSNE visualization with Forgetset predictions...")

plot_tsne(
    final_model,
    test_remain_loader,
    test_forget_loader,
    device,
    "TSNE Visualization(Boundary Shrink model)",
    classes,
    tsne_predictions_path
)
