import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from DataLoader import get_dataloaders
from model import get_resnet_model

def train_for_ablation(name, pretrained, num_epochs, device, train_loader, val_loader):
    print(f"\n实验启动: {name}")
    model = get_resnet_model(num_classes=37, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()

    # ====== 核心逻辑：根据是否预训练，分配不同的优化器配置 ======
    if pretrained:
        # 1. 继承 4_High_WD 最优配置 (新fc LR: 0.01, 旧nn LR: 0.001, WD: 0.01)
        base_params = [param for name, param in model.named_parameters() if 'fc' not in name]
        fc_params = model.fc.parameters()
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 1e-3}, # 旧身体微调 (0.001)
            {'params': fc_params, 'lr': 1e-2}    # 新嘴巴猛学 (0.01)
        ], momentum=0.9, weight_decay=0.01)      # 权重衰减修改为 0.01
    else:
        # 2. 纯净婴儿版：全身都是新的，统一用大学习率。为了控制变量，WD 同样设为 0.01
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.01) # ⭐️ 权重衰减修改为 0.01

    history = {'val_acc': [], 'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        # 验证过程
        model.eval()
        correct, total, running_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        history['val_acc'].append(acc)
        history['val_loss'].append(running_loss / total)
        print(f"  [{name}] Epoch {epoch+1}/{num_epochs} | Val Acc: {acc:.4f} | Val Loss: {(running_loss / total):.4f}")

    return history

def main():
    # 检测设备：支持 Mac 的 mps 苹果芯片加速
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders(data_dir='data', batch_size=32)

    # 第一场：预训练模型（继承最优超参数 4_High_WD，跑 15 轮）
    hist_pretrained = train_for_ablation("With_Pretrained", pretrained=True, num_epochs=15, device=device, train_loader=train_loader, val_loader=val_loader)

    # 第二场：随机初始化模型（给双倍时间，跑 30 轮）
    hist_random = train_for_ablation("Random_Init", pretrained=False, num_epochs=30, device=device, train_loader=train_loader, val_loader=val_loader)

    # ================= 画图环节 =================
    plt.figure(figsize=(12, 5))
    
    # 准确率对比
    plt.subplot(1, 2, 1)
    plt.plot(hist_pretrained['val_acc'], label='With Pretrained (15 Epochs)', color='#1f77b4', marker='x')
    plt.plot(hist_random['val_acc'], label='Random Init (30 Epochs)', color='#d62728', marker='o')
    plt.title('Ablation Study: Pretrained vs Random (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # 损失率对比
    plt.subplot(1, 2, 2)
    plt.plot(hist_pretrained['val_loss'], label='With Pretrained', color='#1f77b4', marker='x')
    plt.plot(hist_random['val_loss'], label='Random Init', color='#d62728', marker='o')
    plt.title('Ablation Study: Pretrained vs Random (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('ablation_study_results.png', dpi=300)
    print("\n消融实验结果图已保存")
    plt.show()

if __name__ == '__main__':
    main()