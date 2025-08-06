import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer import SwinTransformerEncoder
from .convnext import ConvNeXtDecoder
import numpy as np


class PyramidPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 使用 LayerNorm 替换 BatchNorm
        self.pools = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(s),
                    nn.Conv2d(dim, dim // 4, 1),
                    nn.GroupNorm(8, dim // 4),  # 替换 BatchNorm
                    nn.ReLU(inplace=True),
                )
                for s in [1, 2, 4, 8]
            ]
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(dim + dim // 4 * 4, dim, 1),
            nn.GroupNorm(8, dim),  # 替换 BatchNorm
            nn.ReLU(inplace=True),
        )
        self.out_channels = dim

    def forward(self, x):
        h, w = x.shape[2:]
        if h <= 1 or w <= 1:
            return x

        feats = [x]
        for pool in self.pools:
            feat = pool(x)
            if feat.size(2) != h or feat.size(3) != w:
                feat = F.interpolate(feat, (h, w), mode="bilinear", align_corners=True)
            feats.append(feat)

        x = self.fuse(torch.cat(feats, dim=1))
        return x


class FeatureEnhancement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.GroupNorm(8, dim)  # 替换 BatchNorm
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.norm2 = nn.GroupNorm(8, dim)  # 替换 BatchNorm
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + identity


class FusionAttention(nn.Module):
    """
    Fusion Attention Module (CBAM-like) to refine fused features.
    """
    def __init__(self, dim, ratio=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim // ratio, dim, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x_att = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x_att, dim=1, keepdim=True)
        max_out, _ = torch.max(x_att, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x_att = x_att * spatial_att
        return x_att


class SwinConvNextUNet(nn.Module):
    def __init__(
        self,
        img_size,
        in_chans,
        num_classes,
        embed_dim,
        depths,
        num_heads,
        window_size,
        mlp_ratio,
        drop_path_rate,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)

        # 特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, padding=1),
            nn.GroupNorm(8, embed_dim),  # 替换 BatchNorm
            nn.ReLU(inplace=True),
        )

        # 特征增强
        self.enhance = nn.ModuleList(
            [FeatureEnhancement(embed_dim * 2**i) for i in range(len(depths))]
        )

        # 特征融合
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(embed_dim * 2**i, embed_dim, 1) for i in range(len(depths))]
        )

        self.fpn_convs = nn.ModuleList(
            [nn.Conv2d(embed_dim, embed_dim, 3, padding=1) for i in range(len(depths))]
        )

        # 金字塔池化
        self.ppm = PyramidPooling(embed_dim)

        # Swin Transformer编码器
        self.encoder = SwinTransformerEncoder(
            pretrain_img_size=img_size,
            patch_size=4,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

        # ConvNeXt解码器
        decoder_dims = [embed_dim * 2**i for i in range(len(depths))][::-1]
        self.decoder = ConvNeXtDecoder(dims=decoder_dims, drop_path_rate=drop_path_rate)

        # 创新的融合模块
        fusion_in_channels = embed_dim * (1 + len(depths)) # 1 for decoder, N for FPN
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fusion_in_channels, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.ReLU(inplace=True)
        )
        self.fusion_attention = FusionAttention(embed_dim)

        # 输出头
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GroupNorm(8, embed_dim),  # 替换 BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, 1),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, _, H, W = x.shape

        # 特征提取
        x = self.stem(x)

        # 编码器特征
        encoder_features = self.encoder(x)

        # 特征转换和增强
        transformed_features = []
        laterals = []

        for i, feat in enumerate(encoder_features):
            # 维度转换
            B, L, C = feat.shape
            h = w = int(np.sqrt(L))
            feat = feat.transpose(1, 2).reshape(B, C, h, w)

            # 特征增强
            feat = self.enhance[i](feat)
            transformed_features.append(feat)

            # 特征融合准备
            laterals.append(self.lateral_convs[i](feat))

        # 自顶向下路径
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

        # FPN输出
        fpn_outs = []
        for i in range(len(laterals)):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))

        # 解码器特征
        x_decoder = self.decoder(transformed_features)

        # 融合FPN和解码器输出
        # 1. 对解码器输出应用PPM
        base_features = self.ppm(x_decoder)
        base_size = base_features.shape[-2:]

        # 2. 准备所有待融合的特征
        all_features = [base_features]
        for fpn_feat in fpn_outs:
            all_features.append(F.interpolate(fpn_feat, size=base_size, mode="bilinear", align_corners=False))

        # 3. 拼接并通过融合卷积
        fused_x = torch.cat(all_features, dim=1)
        fused_x = self.fusion_conv(fused_x)

        # 4. 应用注意力机制并添加残差连接
        x = fused_x + self.fusion_attention(fused_x)

        # 输出
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
        x = self.head(x)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}
