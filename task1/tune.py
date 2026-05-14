import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from DataLoader import get_dataloaders
from model import get_resnet_model

# ================= 1. 定义一个通用的训练工厂 =================
def train_experiment(config_name, fc_lr, base_lr, num_epochs, weight_decay, train_loader, val_loader, device):
    print(f"\n正在启动实验: {config_name}")
    print(f"参数配置 -> 新fc LR: {fc_lr}, 旧nn LR: {base_lr}, 轮数: {num_epochs}, 权重衰减(WD): {weight_decay}")
    
    model = get_resnet_model(num_classes=37).to(device)
    criterion = nn.CrossEntropyLoss()
    
    base_params = [param for name, param in model.named_parameters() if 'fc' not in name]
    fc_params = model.fc.parameters()
    
    # 注意这里加入了 weight_decay，这是抗过拟合的核心参数！
    # 修改为 SGD 优化器
    optimizer = optim.SGD([
        {'params': base_params, 'lr': base_lr},
        {'params': fc_params, 'lr': fc_lr}
        ], momentum=0.9, weight_decay=weight_decay) # 建议加上 momentum

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0 

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # --- 验证阶段 ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        
        # 记录成绩
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"  Epoch {epoch+1}/{num_epochs} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        # 保存本实验的最好模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{config_name}.pth')

    print(f" {config_name} 实验结束，最高验证准确率: {best_val_acc:.4f}")
    return history

# ================= 2. 实验总指挥部 =================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, val_loader, _ = get_dataloaders(data_dir='data', batch_size=32)

    # 在这里列出你想测试的所有参数组合！(字典列表)
        # 在这里列出你想测试的所有参数组合！(遵循控制变量法)
    experiments = [
        # 1. 基准组 (Control Group): 经典的 SGD 配置，作为所有其他实验的参考系
        {"name": "1_Baseline", "fc_lr": 1e-2, "base_lr": 1e-3, "epochs": 15, "wd": 1e-4},

        # 2. 激进组 (仅改变 LR): 把学习率调大 5 倍。测试模型是否会“步子迈太大”导致梯度爆炸或震荡。
        {"name": "2_High_LR", "fc_lr": 5e-2, "base_lr": 5e-3, "epochs": 15, "wd": 1e-4},

        # 3. 龟速组 (改变 LR 和 Epoch): 学习率缩小 100 倍。因为步子太小，必须把轮数加到 30 才能看出它在缓慢爬坡。
        {"name": "3_Low_LR_Long", "fc_lr": 1e-4, "base_lr": 1e-5, "epochs": 30, "wd": 1e-4},

        # 4. 强力防作弊组 (仅改变 WD): 把权重衰减调大 100 倍。观察模型是不是因为“管得太严”而变成欠拟合。
        {"name": "4_High_WD", "fc_lr": 1e-2, "base_lr": 1e-3, "epochs": 15, "wd": 1e-2},

        # 5. 真正欠拟合组 (仅改变 Epoch): 真正的“短时间训练”。只跑 3 轮就交卷，看它没学透时是什么惨状。
        {"name": "5_Short_Epoch", "fc_lr": 1e-2, "base_lr": 1e-3, "epochs": 3, "wd": 1e-4}
    ]
    # 用一个大字典把所有实验的结果存起来
    all_results = {}

    for config in experiments:
        # 挨个调用训练工厂
        hist = train_experiment(
            config_name=config["name"],
            fc_lr=config["fc_lr"],
            base_lr=config["base_lr"],
            num_epochs=config["epochs"],
            weight_decay=config["wd"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        all_results[config["name"]] = hist

    # ================= 3. 画出终极大比拼图 =================
    plt.figure(figsize=(12, 5))

    # 画 Accuracy 对比
    plt.subplot(1, 2, 1)
    for name, hist in all_results.items():
        # 我们只看最关键的 Validation Accuracy 就可以了
        plt.plot(hist['val_acc'], label=name, marker='x')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 画 Loss 对比
    plt.subplot(1, 2, 2)
    for name, hist in all_results.items():
        plt.plot(hist['val_loss'], label=name, marker='o')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('hyperparameter_comparison.png', dpi=300)
    print("\n对比图已保存为 'hyperparameter_comparison.png'")
    plt.show()

if __name__ == '__main__':
    main()