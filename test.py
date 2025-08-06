import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.models.swin_convnext_unet import SwinConvNextUNet
from tqdm import tqdm
import os
from scipy.ndimage import distance_transform_edt


class Tester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        os.makedirs(config.save_dir, exist_ok=True)

        # 加载模型
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

        # 加载训练好的权重
        checkpoint = torch.load(config.model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"模型已从 {config.model_path} 加载")

    def preprocess_image(self, image):
        """预处理图像"""
        # 转换为tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        image_tensor = image_tensor / 255.0

        # 添加归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std

        return image_tensor.unsqueeze(0)
    
    def _get_edges(self, img):
        """提取二值图像的边缘点"""
        if img.sum() == 0:  # 全黑图像
            return np.zeros_like(img)
            
        # 找到所有非零点坐标
        edges = cv2.Canny(img.astype(np.uint8) * 255, 100, 200)
        return edges > 0

    def calculate_hd95(self, pred, target):
        """计算95%豪斯多夫距离 (HD95)，越小越好"""
        pred_bin = (pred > 0.5).astype(np.uint8)
        target_bin = (target > 0.5).astype(np.uint8)
        
        if pred_bin.sum() == 0 and target_bin.sum() == 0:
            # 如果两个掩码都是空的，返回0
            return 0.0
        
        if pred_bin.sum() == 0 or target_bin.sum() == 0:
            # 如果其中一个掩码是空的，返回一个很大的值
            return 100.0  # 设置一个较大的惩罚值
        
        # 获取边缘
        pred_edges = self._get_edges(pred_bin)
        target_edges = self._get_edges(target_bin)
        
        # 如果没有边缘，返回0
        if not np.any(pred_edges) or not np.any(target_edges):
            return 0.0
        
        # 计算距离变换
        pred_distances = distance_transform_edt(~pred_edges)
        target_distances = distance_transform_edt(~target_edges)
        
        # 获取真实边界到预测边界的距离
        dt_pred = pred_distances[target_edges]
        # 获取预测边界到真实边界的距离
        dt_target = target_distances[pred_edges]
        
        # 计算95%分位数的Hausdorff距离
        hd95_pred = np.percentile(dt_pred, 95)
        hd95_target = np.percentile(dt_target, 95)
        
        # 两个方向的95%分位数的最大值
        hd95 = max(hd95_pred, hd95_target)
        
        return hd95

    def calculate_metrics(self, pred, target):
        """计算准确率、HD95、Dice和IoU四个指标"""
        pred_bin = pred > 0.5
        target_bin = target > 0.5

        # 计算基本指标：TP, TN, FP, FN
        TP = np.logical_and(pred_bin, target_bin).sum()
        TN = np.logical_and(~pred_bin, ~target_bin).sum()
        FP = np.logical_and(pred_bin, ~target_bin).sum()
        FN = np.logical_and(~pred_bin, target_bin).sum()
        
        # 计算总像素数
        N = pred.size
        
        # 1. Dice系数 (DSC)
        dice = 2 * TP / (2 * TP + FP + FN + 1e-6)
        
        # 2. IoU (交并比)
        iou = TP / (TP + FP + FN + 1e-6)
        
        # 3. 准确率 (ACC)
        acc = (TP + TN) / (N + 1e-6)
        
        # 4. 95%豪斯多夫距离 (HD95)
        hd95 = self.calculate_hd95(pred, target)
        
        return {
            'dice': dice,
            'iou': iou,
            'accuracy': acc,
            'hd95': hd95
        }

    def draw_contours(self, image, mask):
        """在原图上绘制分割边缘，包括内部轮廓"""
        # 二值化处理
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        # 寻找所有轮廓，包括内部
        contours, hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_TREE,  # 检索所有轮廓，包括内部的
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # 在原图上绘制所有轮廓
        result = image.copy()
        cv2.drawContours(
            result,
            contours,
            -1,  # -1表示绘制所有轮廓
            (0, 255, 0),  # 红色
            2,  # 线条粗细
        )

        return result, binary_mask

    def save_image_without_border(self, img, save_path, is_gray=False):
        """直接使用OpenCV保存图像，避免matplotlib的边框"""
        if is_gray:
            # 确保灰度图像有正确的形状
            if len(img.shape) == 3 and img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(save_path, img)
        else:
            # RGB图像需要转换为BGR (OpenCV格式)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, img)

    def test(self):
        print(f"开始测试SwinConvNextUNet模型...")
        
        # 初始化四个指标的累加器
        metrics_sum = {
            'dice': 0, 
            'iou': 0, 
            'accuracy': 0, 
            'hd95': 0
        }
        
        # 记录每张图片的HD95值和其他指标
        image_metrics = {}
        
        test_images = sorted(
            [f for f in os.listdir(self.config.test_img_dir) if f.endswith(".png")]
        )
        
        print(f"测试集包含 {len(test_images)} 张图像")

        # 创建子目录用于保存不同类型的图像
        original_dir = os.path.join(self.config.save_dir, "original")
        binary_dir = os.path.join(self.config.save_dir, "binary")
        contour_dir = os.path.join(self.config.save_dir, "contour")
        groundtruth_dir = os.path.join(self.config.save_dir, "groundtruth")
        
        # 创建子目录
        for dir_path in [original_dir, binary_dir, contour_dir, groundtruth_dir]:
            os.makedirs(dir_path, exist_ok=True)

        for image_name in tqdm(test_images, desc="处理测试图像"):
            # 读取图像
            image_path = os.path.join(self.config.test_img_dir, image_name)
            mask_path = os.path.join(self.config.test_mask_dir, image_name)

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # 预处理图像
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.to(self.device)

            with torch.no_grad():
                # 调整大小为模型输入尺寸
                input_tensor = F.interpolate(
                    image_tensor,
                    size=(self.config.img_size, self.config.img_size),
                    mode="bilinear",
                    align_corners=True,
                )

                # 模型预测
                output = self.model(input_tensor)
                pred = torch.sigmoid(output)

                # 还原到原始尺寸
                pred = F.interpolate(
                    pred,
                    size=(image.shape[0], image.shape[1]),
                    mode="bilinear",
                    align_corners=True,
                )

                # 转换预测结果
                pred_np = pred[0, 0].cpu().numpy()
                mask_np = mask / 255.0

                # 计算评价指标
                metrics = self.calculate_metrics(pred_np, mask_np)
                
                # 记录图片指标
                image_metrics[image_name] = metrics
                
                # 累加各指标
                for key in metrics_sum:
                    metrics_sum[key] += metrics[key]

                # 绘制分割边缘
                contour_image, binary_mask = self.draw_contours(image, pred_np)

                # 直接使用OpenCV保存图像，不使用matplotlib，避免边框问题
                
                # 1. 原始图像
                self.save_image_without_border(
                    image, 
                    os.path.join(original_dir, f"{image_name}")
                )
                
                # 2. 分割结果 - 显示二值化结果
                self.save_image_without_border(
                    binary_mask,
                    os.path.join(binary_dir, f"{image_name}"),
                    is_gray=True
                )
                
                # 3. 边缘叠加
                self.save_image_without_border(
                    contour_image,
                    os.path.join(contour_dir, f"{image_name}")
                )
                
                # 4. 标签图像
                self.save_image_without_border(
                    mask,
                    os.path.join(groundtruth_dir, f"{image_name}"),
                    is_gray=True
                )

        # 计算平均指标
        num_images = len(test_images)
        avg_metrics = {k: v / num_images for k, v in metrics_sum.items()}

        # 找出HD95最高的5张图片
        hd95_sorted = sorted(image_metrics.items(), key=lambda x: x[1]['hd95'], reverse=True)[:5]

        # 打印结果
        print("\n===== SwinConvNextUNet模型分割性能评估结果 =====")
        print(f"Dice系数 (DSC): {avg_metrics['dice']:.4f}")
        print(f"交并比 (IoU): {avg_metrics['iou']:.4f}")
        print(f"准确率 (ACC): {avg_metrics['accuracy']:.4f}")
        print(f"95%豪斯多夫距离 (HD95): {avg_metrics['hd95']:.4f} 像素")
        print("==========================")

        # 打印HD95最高的图片
        print("\n===== HD95最高的5张图片 =====")
        for i, (img_name, metrics) in enumerate(hd95_sorted):
            print(f"{i+1}. {img_name}: HD95={metrics['hd95']:.4f}, Dice={metrics['dice']:.4f}")
        print("==========================")

        # 保存评价指标
        with open(os.path.join(self.config.save_dir, "metrics.txt"), "w") as f:
            f.write("===== SwinConvNextUNet模型分割性能评估结果 =====\n")
            f.write(f"Dice系数 (DSC): {avg_metrics['dice']:.4f}\n")
            f.write(f"交并比 (IoU): {avg_metrics['iou']:.4f}\n")
            f.write(f"准确率 (ACC): {avg_metrics['accuracy']:.4f}\n")
            f.write(f"95%豪斯多夫距离 (HD95): {avg_metrics['hd95']:.4f} 像素\n")
            f.write("==========================\n\n")
            
            # 保存HD95最高的图片信息
            f.write("===== HD95最高的5张图片 =====\n")
            for i, (img_name, metrics) in enumerate(hd95_sorted):
                f.write(f"{i+1}. {img_name}: HD95={metrics['hd95']:.4f}, Dice={metrics['dice']:.4f}\n")
            f.write("==========================\n")
        
        print(f"\n结果已保存至 {self.config.save_dir}")
        return avg_metrics


if __name__ == "__main__":

    class Config:
        img_size = 512
        model_path = (
            "./checkpoints/SwinConvNextUNet/20250716_175750/best_model.pth"  # 修改为实际模型路径
        )
        test_img_dir = "./data/RealData/testO/images/"
        test_mask_dir = "./data/RealData/testO/labels/"
        save_dir = "./results/swinConvNextUNet/"
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tester = Tester(Config())
    tester.test()