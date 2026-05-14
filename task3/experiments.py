import torch
import torch.optim as optim
import torch.nn as nn
import swanlab
import numpy as np
from DataLoader import get_dataloaders
from model import UNet
from losses import DiceLoss

# --- 全局硬件配置 ---
# 自动检测 Mac 的 MPS 或 CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4

# 数据路径
IMG_DIR = "data/images"
MASK_DIR = "data/annotations/trimaps"

#计算mIoU
def calculate_miou(preds, masks, num_classes=3):
    """
    preds: [Batch, H, W] - 模型预测的类别索引
    masks: [Batch, H, W] - 真实标签
    """
    ious = []
    # 转换为 numpy 方便计算
    preds = preds.cpu().numpy()
    masks = masks.cpu().numpy()
    
    for cls in range(num_classes):
        intersection = np.logical_and(preds == cls, masks == cls).sum()
        union = np.logical_or(preds == cls, masks == cls).sum()
        if union == 0:
            ious.append(1.0)  # 如果图中完全没有这个类别，默认 IoU 为 1
        else:
            ious.append(intersection / union)
    return np.mean(ious)

#验证函数
def validate(model, loader):
    model.eval()
    mious = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1) # 取概率最大的类别
            
            # 计算这一批次的 mIoU
            miou = calculate_miou(preds, masks)
            mious.append(miou)
    return np.mean(mious)

def visualize_progress(model, loader, epoch, loss_type):
    """可视化函数：将预测结果上传至 SwanLab"""
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(loader))
        img, mask = images[0:1].to(DEVICE), masks[0].cpu().numpy()
        
        output = model(img)
        pred = torch.argmax(output[0], dim=0).cpu().numpy()
        raw_image = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        class_labels = {0: "Pet", 1: "Background", 2: "Border"}
        
        swanlab.log({
            f"Sample_{loss_type}": swanlab.Image(
                raw_image,
                caption=f"Epoch {epoch+1}",
                masks={
                    "prediction": {"mask_data": pred, "class_labels": class_labels},
                    "ground_truth": {"mask_data": mask, "class_labels": class_labels},
                },
            )
        })

def run_experiment(loss_type):
    print(f"\n启动实验: {loss_type} ")
    
    swanlab.init(
        project="Pet-Segmentation-Comparison",
        experiment_name=f"Unet-{loss_type}",
        config={"loss_type": loss_type, "lr": LR}
    )

    # 假设你已经处理好了 train_loader 和 val_loader
    # 简单起见，你可以从数据集中分出一部分做验证
    train_loader, val_loader = get_dataloaders(IMG_DIR, MASK_DIR, batch_size=BATCH_SIZE)

    model = UNet(n_channels=3, n_classes=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ce_crit = nn.CrossEntropyLoss()
    dice_crit = DiceLoss()

    for epoch in range(EPOCHS):
        # 1. 训练阶段
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            
            if loss_type == "ce": loss = ce_crit(outputs, masks)
            elif loss_type == "dice": loss = dice_crit(outputs, masks)
            else: loss = ce_crit(outputs, masks) + dice_crit(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 2. 验证阶段 (体现 mIoU 的地方！)
        val_miou = validate(model, val_loader)
        
        # 3. 记录到 SwanLab
        # 你可以在图表中同时看到 Loss 下降和 mIoU 上升
        swanlab.log({
            "val/mIoU": val_miou,
            "epoch": epoch + 1
        })
        
        print(f"[{loss_type}] Epoch {epoch+1} - val_mIoU: {val_miou:.4f}")

    swanlab.finish()

if __name__ == "__main__":
    # --- 任务(3)的核心：循环跑三种配置 ---
    experiments = ["ce", "dice", "combined"]
    
    for exp_type in experiments:
        run_experiment(exp_type)
        
    print("\n所有实验已完成")