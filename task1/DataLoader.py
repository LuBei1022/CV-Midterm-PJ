import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class PetDataset(Dataset):
    """自定义宠物数据集"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 强制转换为 RGB，防止有些灰度图报错
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def prepare_data(data_dir='data'):
    """从 images 文件夹提取图片路径和标签"""
    images_dir = os.path.join(data_dir, 'images')
    
    # 过滤掉 macOS 的 .DS_Store 等隐藏文件，只取 jpg
    all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    image_paths = []
    labels = []
    class_to_idx = {}
    current_idx = 0

    for img_name in all_images:
        # Oxford-IIIT Pet 文件名格式：类名_数字.jpg (例如: Abyssinian_1.jpg 或 american_bulldog_100.jpg)
        # 用 '_' 分割并去掉最后一部分，剩下的拼起来就是类名
        class_name = "_".join(img_name.split('_')[:-1])

        if class_name not in class_to_idx:
            class_to_idx[class_name] = current_idx
            current_idx += 1

        image_paths.append(os.path.join(images_dir, img_name))
        labels.append(class_to_idx[class_name])

    return image_paths, labels, class_to_idx

def get_dataloaders(data_dir='data', batch_size=32, test_split=0.2):
    """生成训练和验证的 DataLoader"""
    image_paths, labels, class_to_idx = prepare_data(data_dir)

    # 按照标签比例进行分层拆分 (Stratified Split)，保证训练集和验证集类别分布一致
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=test_split, random_state=42, stratify=labels
    )

    # 训练集数据增强：随机裁剪、随机水平翻转 + ImageNet 标准归一化
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证集数据处理：中心裁剪 + ImageNet 标准归一化
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PetDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PetDataset(val_paths, val_labels, transform=val_transform)

    # macOS 环境下 num_workers 建议设为 0 或 2，设太大容易报错
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, class_to_idx

# ================= 测试代码 =================
# 你可以直接运行 python DataLoader.py 来验证环境是否调通
if __name__ == '__main__':
    print("正在初始化 DataLoader...")
    
    # 假设你的终端当前处于 task1 目录下
    train_loader, val_loader, class_mapping = get_dataloaders(data_dir='data', batch_size=16)
    
    print(f"成功找到 {len(class_mapping)} 个宠物类别！")
    print(f"训练集 batch 数量: {len(train_loader)}")
    print(f"验证集 batch 数量: {len(val_loader)}")
    
    # 抽一个 batch 出来看看维度
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape} (应为: [batch_size, 3, 224, 224])")
        print(f"Label batch shape: {labels.shape} (应为: [batch_size])")
        break # 看一个就够了