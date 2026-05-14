import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split # 需要安装: pip install scikit-learn

class PetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_names, img_size=128):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = img_names # 这里接收分好的文件名列表

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '.png')
        
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")

        image = image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        image_tensor = transforms.ToTensor()(image)
        # 1,2,3 -> 0,1,2
        mask_np = np.array(mask).astype(np.int64) - 1
        mask_tensor = torch.from_numpy(mask_np)

        return image_tensor, mask_tensor

def get_dataloaders(image_dir, mask_dir, batch_size=8, img_size=224, split_ratio=0.2):
    # 1. 先拿到所有合法的图片文件名
    all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    # 2. 随机切分文件名 (80% 训练, 20% 验证)
    train_names, val_names = train_test_split(all_images, test_size=split_ratio, random_state=42)
    
    # 3. 创建两个 Dataset 实例
    train_ds = PetDataset(image_dir, mask_dir, train_names, img_size)
    val_ds = PetDataset(image_dir, mask_dir, val_names, img_size)
    
    # 4. 创建 Loader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader