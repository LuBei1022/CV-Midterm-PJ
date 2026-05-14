import torch
import torch.nn as nn
import torch.optim as optim
from DataLoader import get_dataloaders
from model import get_resnet_model
import os 
import matplotlib.pyplot as plt

def train_model():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    train_loader, val_loader, idx_to_class = get_dataloaders(data_dir = 'data', batch_size=32)
    model = get_resnet_model(num_classes=37)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    base_params = [param for name, param in model.named_parameters() if 'fc' not in name]
    fc_params = model.fc.parameters()

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 1e-4},
        {'params': fc_params, 'lr': 1e-2}],
        momentum = 0.9
    )

    num_epochs = 10 # 既然要观察曲线，可以稍微跑多几个 epoch，比如 10 个

    # 💥新增 1：准备一个笔记本，记录每轮的成绩
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # 💥新增 2：记录历史最高分，方便“保存最好”
    best_val_acc = 0.0 

    for epoch in range(num_epochs):
        print(f'\n--- Epoch {epoch+1}/{num_epochs} ---')

        # ----------------- A. 练习时间 (Train) -----------------
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
        print(f'练习成绩 -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f}')

        # ----------------- B. 考试时间 (Validation) -----------------
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
        print(f'考试成绩 -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f}')

        # 新增 3：把这轮的成绩写进笔记本
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 新增 4：如果这次考试破纪录了，赶紧把 AI 的脑子存下来！
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # state_dict() 就是模型所有参数的字典
            save_path = 'baseline_resnet18_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"保存当前最佳模型到 {save_path} (准确率: {best_val_acc:.4f})")

    print('\n训练结束！')
    
    # 💥新增 5：考试结束，画出成绩走势图！
    plot_history(history)

# 新增 6：画图的专属小函数
def plot_history(history):
    # 画 Accuracy 曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(history['val_acc'], label='Validation Accuracy', marker='x')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 画 Loss 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='x')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 保存图片，这可是你写报告和论文最重要的素材！
    plt.tight_layout()
    plt.savefig('baseline_training_curve.png', dpi=300)
    print("训练曲线图已保存为 'baseline_training_curve.png'")
    plt.show()

if __name__ == '__main__':
    train_model()