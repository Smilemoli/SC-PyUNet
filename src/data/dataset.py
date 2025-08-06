import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class IronSpectrumDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512, is_train=True, indices=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train

        all_images = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        all_masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

        if indices is not None:
            self.images = [all_images[i] for i in indices]
            self.masks = [all_masks[i] for i in indices]
        else:
            self.images = all_images
            self.masks = all_masks

        # 图像预处理和归一化
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 标签预处理
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        # 数据增强
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ]) if is_train else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 应用数据增强
        if self.is_train and self.augmentation:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.augmentation(image)
            torch.manual_seed(seed)
            mask = self.augmentation(mask)

        # 转换为tensor并归一化
        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # 确保标签为二值
        mask = (mask > 0.5).float()

        return {
            'image': image,
            'mask': mask,
            'filename': self.images[idx]
        }

class MixedDataset(Dataset):
    """混合真实数据和增强数据的数据集"""
    def __init__(self, real_dataset, aug_dataset, real_ratio=0.75):
        """
        Args:
            real_dataset: 真实数据集
            aug_dataset: 增强数据集  
            real_ratio: 真实数据在混合中的比例 (0.0-1.0)
        """
        self.real_dataset = real_dataset
        self.aug_dataset = aug_dataset
        self.real_ratio = real_ratio
        
        # 计算总长度，确保每个epoch都能遍历到所有真实数据
        self.total_length = max(len(real_dataset), int(len(real_dataset) / real_ratio))
        
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # 根据比例决定采样来源
        if random.random() < self.real_ratio:
            # 采样真实数据
            real_idx = idx % len(self.real_dataset)
            return self.real_dataset[real_idx]
        else:
            # 采样增强数据
            aug_idx = random.randint(0, len(self.aug_dataset) - 1)
            return self.aug_dataset[aug_idx]

def create_mixed_dataloaders(real_img_dir, real_mask_dir, aug_img_dir=None, aug_mask_dir=None, 
                            batch_size=16, img_size=224, num_workers=4, real_ratio=0.75):
    """
    创建混合训练数据加载器和纯净验证数据加载器
    
    Args:
        real_img_dir: 真实图像目录
        real_mask_dir: 真实标签目录
        aug_img_dir: 增强图像目录 (可选)
        aug_mask_dir: 增强标签目录 (可选)
        batch_size: 批次大小
        img_size: 图像尺寸
        num_workers: 工作线程数
        real_ratio: 真实数据在训练批次中的比例
    """
    
    # Step 1: 划分真实数据为训练集和验证集
    all_real_images = sorted([f for f in os.listdir(real_img_dir) if f.endswith('.png')])
    np.random.seed(42)
    train_idx = np.random.choice(
        len(all_real_images), int(0.8 * len(all_real_images)), replace=False
    )
    val_idx = np.array(list(set(range(len(all_real_images))) - set(train_idx)))

    # Step 2: 创建真实数据的训练集和验证集
    real_train_dataset = IronSpectrumDataset(
        img_dir=real_img_dir, mask_dir=real_mask_dir, 
        img_size=img_size, is_train=True, indices=train_idx
    )
    
    # 验证集只使用真实数据，不使用增强
    val_dataset = IronSpectrumDataset(
        img_dir=real_img_dir, mask_dir=real_mask_dir, 
        img_size=img_size, is_train=False, indices=val_idx
    )

    # Step 3: 根据是否提供增强数据路径，决定训练策略
    if aug_img_dir and aug_mask_dir and os.path.exists(aug_img_dir) and os.path.exists(aug_mask_dir):
        print("🔥 检测到增强数据，启用混合训练策略！")
        
        # 创建增强数据集
        aug_dataset = IronSpectrumDataset(
            img_dir=aug_img_dir, mask_dir=aug_mask_dir,
            img_size=img_size, is_train=True, indices=None  # 使用所有增强数据
        )
        
        # 创建混合数据集
        mixed_train_dataset = MixedDataset(
            real_dataset=real_train_dataset,
            aug_dataset=aug_dataset,
            real_ratio=real_ratio
        )
        
        print(f"📊 训练集构成:")
        print(f"   - 真实训练数据: {len(real_train_dataset)} 张")
        print(f"   - 增强数据: {len(aug_dataset)} 张") 
        print(f"   - 混合比例: {real_ratio:.1%} 真实数据 + {1-real_ratio:.1%} 增强数据")
        print(f"   - 每个批次期望构成: {int(batch_size * real_ratio)} 张真实 + {batch_size - int(batch_size * real_ratio)} 张增强")
        
        train_dataset = mixed_train_dataset
    else:
        print("📝 未检测到增强数据，使用纯真实数据训练")
        train_dataset = real_train_dataset

    # Step 4: 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"✅ 数据加载器创建完成:")
    print(f"   - 训练集大小: {len(train_dataset)}")
    print(f"   - 验证集大小: {len(val_dataset)} (纯真实数据)")
    
    return train_loader, val_loader

# 保持向后兼容性
def create_dataloaders(img_dir, mask_dir, batch_size=16, img_size=224, num_workers=4):
    """原有的数据加载器创建函数，保持向后兼容性"""
    return create_mixed_dataloaders(
        real_img_dir=img_dir,
        real_mask_dir=mask_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers
    )