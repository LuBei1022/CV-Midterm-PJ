import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 基础模块：连续两次卷积
# U-Net 的每个阶段都会连续做两次卷积，我们把它封装起来
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # 归一化，让训练更稳定
            nn.ReLU(inplace=True),        # 激活函数
            # 第二层卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# 2. 下采样模块 (Encoder)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # 尺寸缩小一半
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# 3. 上采样模块 (Decoder) + 特征拼接 (Skip Connection)
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用双线性插值进行上采样，也可以用 nn.ConvTranspose2d
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 是来自深层的数据 (需要放大)
        # x2 是来自浅层编码器的跳跃连接数据 (Skip Connection)
        x1 = self.up(x1)
        
        # 拼接两个特征图
        # x1: [Batch, Channels, H, W]
        # 在 Channels 维度进行拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 4. U-Net 总架构组装
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- 编码器 (左半边) ---
        self.inc = DoubleConv(n_channels, 64)      # 输入层
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)               # 最底层

        # --- 解码器 (右半边) ---
        # 注意：Up 模块的输入通道数是左边的两倍，因为有 Skip Connection 拼接了特征
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1) # 1x1 卷积输出 3 分类

    def forward(self, x):
        # 记录下采样每一层的结果，用于 Skip Connection
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 开始上采样并拼接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return logits

# 测试模型
if __name__ == "__main__":
    # 输入：1 张图片, 3 通道 (RGB), 224x224 尺寸
    # 任务：3 分类
    model = UNet(n_channels=3, n_classes=3)
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}") # 预期: [1, 3, 224, 224]