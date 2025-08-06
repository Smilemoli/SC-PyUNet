import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import logging
from src.models.swin_convnext_unet import SwinConvNextUNet
from src.data.dataset import create_mixed_dataloaders
from src.models.loss import CombinedLoss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.current_epoch = 0

        # 创建保存目录
        self.exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(config.save_dir, self.exp_name)
        self.pred_dir = os.path.join(self.save_dir, "predictions")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)

        # 初始化数据加载器
        self.train_loader, self.val_loader = create_mixed_dataloaders(
            real_img_dir=config.train_img_dir,
            real_mask_dir=config.train_mask_dir,
            aug_img_dir="./data/AugData/images",  # 您的增强数据路径
            aug_mask_dir="./data/AugData/labels", # 您的增强标签路径
            batch_size=config.batch_size,
            img_size=config.img_size,
            num_workers=config.num_workers,
            real_ratio=0.75  # 75% 真实数据 + 25% 增强数据
        )

        # 初始化模型
        self.model = SwinConvNextUNet(
            img_size=config.img_size,
            in_chans=3,
            num_classes=1,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.0,
            drop_path_rate=0.1,
        ).to(self.device)
        self._init_weights()

        # 使用组合损失
        self.criterion = CombinedLoss()

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=5e-5, weight_decay=1e-4, betas=(0.9, 0.999)
        )

        # 学习率调度器
        total_steps = len(self.train_loader) * config.epochs
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=5e-4,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4,
        )

        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.save_dir, "train.log")),
                logging.StreamHandler(),
            ],
        )

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def save_predictions(self, images, masks, outputs, epoch, phase="train"):
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        masks = masks.cpu().numpy()
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

        save_folder = os.path.join(self.pred_dir, f"epoch_{epoch}")
        os.makedirs(save_folder, exist_ok=True)

        for i in range(min(2, len(images))):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(images[i])
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(masks[i, 0], cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(preds[i, 0], cmap="gray")
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, f"{phase}_{i}.png"))
            plt.close()

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        epoch_dice = 0

        with tqdm(
            self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}"
        ) as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                masks = (masks > 0.5).float()

                self.optimizer.zero_grad()
                outputs = self.model(images)

                loss = self.criterion(outputs, masks)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                # 计算Dice分数
                with torch.no_grad():
                    dice_loss = self.criterion.dice(torch.sigmoid(outputs), masks)
                    dice_score = 1 - dice_loss.item()

                epoch_loss += loss.item()
                epoch_dice += dice_score

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "dice": f"{dice_score:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
                    }
                )

                if batch_idx == len(self.train_loader) - 1:
                    self.save_predictions(images, masks, outputs, epoch, "train")

        return epoch_loss / len(self.train_loader), epoch_dice / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0
        val_dice = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                masks = (masks > 0.5).float()

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice_loss = self.criterion.dice(torch.sigmoid(outputs), masks)

                val_loss += loss.item()
                val_dice += 1 - dice_loss.item()

                if batch_idx == len(self.val_loader) - 1:
                    self.save_predictions(
                        images, masks, outputs, self.current_epoch, "val"
                    )

        return val_loss / len(self.val_loader), val_dice / len(self.val_loader)

    def train(self):
        best_dice = 0
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            train_loss, train_dice = self.train_epoch(epoch)
            val_loss, val_dice = self.validate()

            logging.info(
                f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Dice={train_dice:.4f}, "
                f"Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}"
            )

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "val_dice": val_dice,
                    },
                    os.path.join(self.save_dir, "best_model.pth"),
                )


if __name__ == "__main__":

    class Config:
        img_size = 512
        batch_size = 4
        epochs = 150  # 增加训练轮数
        train_img_dir = "./data/RealData/trainO/images"
        train_mask_dir = "./data/RealData/trainO/labels"
        save_dir = "./checkpoints/SwinConvNextUNet/"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_workers = 4
        pin_memory = True

    os.makedirs(Config.save_dir, exist_ok=True)
    trainer = Trainer(Config())
    trainer.train()

