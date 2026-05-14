import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import types
import swanlab

from DataLoader import get_dataloaders
from torchvision import models

# ================= 1. 定义 SE 注意力模块 =================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze 操作：全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation 操作：两层全连接网络学习通道权重
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) # 将学习到的权重乘回原特征图

# ================= 2. 模型构建 =================

def get_resnet_baseline(num_classes=37):
    """传统的 CNN 架构 (Baseline)"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def se_forward(self, x):
    """自定义的 ResNet BasicBlock 前向传播，加入了 SE 模块"""
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    # 在残差相加之前，经过注意力模块进行特征重标定
    if hasattr(self, 'se'):
        out = self.se(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu(out)
    return out

def get_se_resnet(num_classes=37):
    """加入注意力机制的 SE-ResNet-18"""
    # 1. 先加载完整的预训练 ResNet-18，继承强大的基础特征提取能力
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. 遍历网络，找到所有的 BasicBlock，将 SE 模块“缝合”进去
    for name, module in model.named_modules():
        if isinstance(module, models.resnet.BasicBlock):
            channels = module.conv2.out_channels
            module.se = SEBlock(channels)
            # 替换原本的 forward 函数
            module.forward = types.MethodType(se_forward, module)

    # 3. 换上新的分类头
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

# ================= 3. 训练引擎 =================

def train_model(model, name, num_epochs, device, train_loader, val_loader):
    print(f"\n正在训练模型: {name}")
    swanlab.init(
        project="CV-Midterm-Attention_Study", 
        experiment_name=name,           
        config={
            "learning_rate_head": 1e-2,
            "learning_rate_base": 1e-3,
            "weight_decay": 0.01,
            "batch_size": 32,           
            "architecture": "SE-ResNet" if "SE" in name else "ResNet Baseline"
        }
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 动态识别分类头和SE模块，应用最优分层学习率
    head_params = []
    base_params = []

    for param_name, param in model.named_parameters():
        # ⭐️ 新增：把 se 模块也识别出来，放入 head_params 给大学习率
        if 'fc' in param_name or 'se' in param_name: 
            head_params.append(param)
        else:
            base_params.append(param) 

    # 这样 SE 模块和分类头用 1e-2 快速学习，原本的 ResNet 权重用 1e-3 慢慢微调
    optimizer = optim.SGD([
        {'params': base_params, 'lr': 1e-3}, 
        {'params': head_params, 'lr': 1e-2}  
    ], momentum=0.9, weight_decay=0.01)    

    history = {'val_acc': [], 'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        total_train = 0
        running_train_correct = 0  # ➕ 新增：用来记录训练集预测正确的数量
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            total_train += labels.size(0)

            # ➕ 新增：计算这个 Batch 里猜对了多少个
            _, predicted = torch.max(outputs, 1)
            running_train_correct += (predicted == labels).sum().item()

            if i % 10 == 0:
                swanlab.log({"Train/Batch_Loss": loss.item()})

        train_loss = running_train_loss / total_train
        train_acc = running_train_correct / total_train # ➕ 新增：计算当前 Epoch 的训练准确率

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
        current_val_loss = running_loss / total 
        
        history['val_acc'].append(acc)
        history['val_loss'].append(current_val_loss)

        swanlab.log({
            "Train/Epoch_Loss": train_loss,
            "Train/Accuracy": train_acc,
            "Val/Loss": current_val_loss,
            "Val/Accuracy": acc,
            "Epoch": epoch + 1
        })
        
        print(f"  [{name}] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val Loss: {current_val_loss:.4f}")
    swanlab.finish()
    return history

# ================= 4. compare =================

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备: {device}")
    

    train_loader, val_loader, _ = get_dataloaders(data_dir='data', batch_size=32)
    
    num_epochs = 15 # 按照你最优 Baseline 的配置，跑 15 轮即可

    # 第一场：纯净版 ResNet-18 (Control Group)
    resnet_model = get_resnet_baseline(num_classes=37)
    hist_resnet = train_model(resnet_model, "Baseline (ResNet-18)", num_epochs, device, train_loader, val_loader)

    # 第二场：注意力机制版 SE-ResNet-18 (Experimental Group)
    se_resnet_model = get_se_resnet(num_classes=37)
    hist_se = train_model(se_resnet_model, "Attention (SE-ResNet-18)", num_epochs, device, train_loader, val_loader)

    # ================= 5. 成绩发榜 =================
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(hist_resnet['val_acc'], label='Baseline (ResNet-18)', color='#1f77b4', marker='x')
    plt.plot(hist_se['val_acc'], label='Attention (SE-ResNet-18)', color='#ff7f0e', marker='o')
    plt.title('Accuracy: Baseline vs SE-Attention')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(hist_resnet['val_loss'], label='Baseline (ResNet-18)', color='#1f77b4', marker='x')
    plt.plot(hist_se['val_loss'], label='Attention (SE-ResNet-18)', color='#ff7f0e', marker='o')
    plt.title('Loss: Baseline vs SE-Attention')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('se_attention_comparison.png', dpi=300)
    print("\n注意力机制对比实验完成，图表已保存为 'se_attention_comparison.png'")
    plt.show()

if __name__ == '__main__':
    main()