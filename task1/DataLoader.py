import os #处理文件路径
from PIL import Image #pillow库 读取图片文件
import torch
from torch.utils.data import Dataset, DataLoader #数据处理
from torchvision import transforms #图像预处理（比如变成224x224)
from sklearn.model_selection import train_test_split #随机切分训练集&测试集

#定义petdataset 类【初始化+图片数量+取图片getitem】
class PetDataset(Dataset): #继承了pytorch里的dataset类，在这个基础上加入init里新定义的3个
    def __init__(self, image_paths, labels, transform = None):
        self.image_paths = image_paths 
        self.labels = labels 
        self.transform = transform 
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')#打开图片，转为RGB三色模式
        label= self.labels[idx]

        if self.transform: #transform != None,就是有预处理的意思
            image = self.transform(image)
        
        return image, label

def prepare_data_from_list(data_dir = 'data'):
    list_file = os.path.join(data_dir, 'annotations', 'list.txt')
    images_dir = os.path.join(data_dir, 'images')
    image_paths = []
    labels = []
    idx_to_class = {}

    with open(list_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        img_name = parts[0] + '.jpg'
        label = int(parts[1]) - 1 #要-1，是因为label的作用是下标，从0开始
        class_name = parts[0].rsplit('_', 1)[0]
        full_path = os.path.join(images_dir, img_name)
        if label not in idx_to_class:
                idx_to_class[label] = class_name
                

        if os.path.exists(full_path):
            image_paths.append(full_path)
            labels.append(label)
        
    return image_paths, labels, idx_to_class



def get_dataloaders(data_dir = 'data', batch_size = 32):
    all_paths, all_labels, idx_to_class = prepare_data_from_list(data_dir)
    train_paths, val_paths, train_labels, val_labels = train_test_split(all_paths, all_labels, test_size=0.2, stratify=all_labels, random_state = 42)
    train_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip()]
    )
    val_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = PetDataset(train_paths, train_labels, transform=train_transform)
    val_ds = PetDataset(val_paths, val_labels, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False, num_workers=2)

    return train_loader, val_loader, idx_to_class

