import torch
import torch.optim as optim
from DataLoader import get_dataloaders
from model import UNet
from losses import DiceLoss
import torch.nn as nn
import swanlab
import numpy as np

# --- 配置区 ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
LOSS_TYPE = "combined" # 可选: 'ce', 'dice', 'combined'

# 数据路径（请根据你实际存放位置修改）
IMG_DIR = "data/images"
MASK_DIR = "data/annotations/trimaps"

def train():
    swanlab.init(
        project="Pet-Segmentation",
        experiment_name=f"Unet-{LOSS_TYPE}",
        config={
            "learning_rate": LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "loss_type": LOSS_TYPE,
            "device": str(DEVICE),
        },
    )

    # 1. 加载数据
    train_loader = get_dataloaders(IMG_DIR, MASK_DIR, batch_size=BATCH_SIZE)
    
    # 2. 初始化模型、损失函数和优化器
    model = UNet(n_channels=3, n_classes=3).to(DEVICE)
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"开始训练，模式: {LOSS_TYPE}, 设备: {DEVICE}")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # 前向传播
            outputs = model(images)
            
            # 3. 根据任务要求选择损失函数
            if LOSS_TYPE == "ce":
                loss = ce_loss(outputs, masks)
            elif LOSS_TYPE == "dice":
                loss = dice_loss(outputs, masks)
            else: # combined
                loss = ce_loss(outputs, masks) + dice_loss(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # 每 10 个 batch 记录一次当前的 Loss
            if batch_idx % 10 == 0:
                swanlab.log({"train/batch_loss": loss.item()})
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {batch_idx} Loss: {loss.item():.4f}")

        # 2. 记录每个 Epoch 的平均 Loss
        avg_loss = epoch_loss / len(train_loader)
        swanlab.log({"train/epoch_loss": avg_loss, "epoch": epoch + 1})

        # 3. 每训练完一个 Epoch，可视化一张图片看效果
        visualize_progress(model, train_loader, epoch)

    torch.save(model.state_dict(), f"unet_{LOSS_TYPE}.pth")
    print("训练完成！")

def visualize_progress(model, loader, epoch):
    """
    选取一张图片，将预测结果上传到 SwanLab
    """
    model.eval()
    with torch.no_grad():
        # 取出一个 batch 的第一张图
        images, masks = next(iter(loader))
        img, mask = images[0:1].to(DEVICE), masks[0].cpu().numpy()
        
        output = model(img)
        # 获取预测类索引 [H, W]
        pred = torch.argmax(output[0], dim=0).cpu().numpy()
        
        # 转换原始图片用于展示 (Tensor -> Numpy [H, W, C])
        raw_image = img[0].permute(1, 2, 0).cpu().numpy()
        # 归一化到 0-255 方便显示
        raw_image = (raw_image * 255).astype(np.uint8)

        # 4. 使用 swanlab.Image 记录分割结果
        # 我们定义 0:前景, 1:背景, 2:边缘 的标签名称
        class_labels = {0: "Pet", 1: "Background", 2: "Border"}
        
        swanlab.log({
            "Prediction_Sample": swanlab.Image(
                raw_image,
                caption=f"Epoch {epoch+1} Result",
                masks={
                    "prediction": {"mask_data": pred, "class_labels": class_labels},
                    "ground_truth": {"mask_data": mask, "class_labels": class_labels},
                },
            )
        })

if __name__ == "__main__":
    train()