import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape
    
    def forward(self, x):
        if x.dim() == 4:
            # 对于4D输入，先调整维度
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        else:
            # 对于2D或3D输入保持原有逻辑
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight.unsqueeze(-1) * x + self.bias.unsqueeze(-1)
        return x

class GRN(nn.Module):
    """Global Response Normalization"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNextBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)  # 改用Conv2d替代Linear
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)  # 改用Conv2d替代Linear
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return input + self.gamma * x

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # 1x1卷积用于调整拼接后的通道数
        self.conv_adjust = nn.Conv2d(out_channels * 2, out_channels * 2, 1)

        # 多尺度特征提取 - 使用扩张卷积
        self.multiscale_conv = nn.ModuleList([
            nn.Conv2d(
                out_channels*2, out_channels*2, 
                kernel_size=3, padding=r, dilation=r, 
                groups=out_channels*2
            ) for r in [1, 2, 4]
        ])
        
        # 特征重标定
        self.grn = GRN(out_channels*2*4)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels*2*4, out_channels*2, 1),
            LayerNorm(out_channels*2),
            nn.GELU()
        )
        
        # 使用修改后的ConvNextBlock
        self.conv = nn.Sequential(
            ConvNextBlock(out_channels * 2),
            ConvNextBlock(out_channels * 2)
        )
        
        self.conv_out = nn.Conv2d(out_channels * 2, out_channels, 1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape[-2:] != x2.shape[-2:]:
            x1 = F.interpolate(x1, size=x2.shape[-2:], mode='bilinear', align_corners=True)
            
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_adjust(x) # 调整通道数
        
        # 多尺度特征
        multi_feats = [x]
        for conv in self.multiscale_conv:
            multi_feats.append(conv(x))
        
        # 特征融合
        x = torch.cat(multi_feats, dim=1)
        x = self.grn(x)
        x = self.fusion(x)
        x = self.conv(x)
        x = self.conv_out(x)
        
        return x

class AttentionGate(nn.Module):
    """
    Attention Gate module to selectively focus on relevant features from skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # Gating signal processing
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm(F_int)  # Using LayerNorm for consistency
        )
        
        # Skip connection feature processing
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm(F_int)
        )

        # Attention coefficient generation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # Process gating signal and skip connection features
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match x1's spatial dimensions if they differ
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=True)

        # Combine and generate attention map
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention to the skip connection
        return x * psi


class AttentionGate(nn.Module):
    """
    Attention Gate module to selectively focus on relevant features from skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # Gating signal processing
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm(F_int)  # Using LayerNorm for consistency
        )
        
        # Skip connection feature processing
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm(F_int)
        )

        # Attention coefficient generation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # Process gating signal and skip connection features
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match x1's spatial dimensions if they differ
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode='bilinear', align_corners=True)

        # Combine and generate attention map
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # Apply attention to the skip connection
        return x * psi

class ConvNeXtDecoder(nn.Module):
    def __init__(self, dims=[768, 384, 192, 96], drop_path_rate=0.):
        super().__init__()
        self.dims = dims
        
        # 添加初始特征适配层
        self.init_conv = nn.Conv2d(dims[0] * 2, dims[0], 1)  # 用于处理金字塔池化后的特征
        
        # 特征增强
        self.enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                LayerNorm(dim),
                nn.GELU(),
                GRN(dim)
            ) for dim in dims
        ])
        
        # 特征自适应
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim//4, 1),
                nn.GELU(),
                nn.Conv2d(dim//4, dim, 1),
                nn.Sigmoid()
            ) for dim in dims
        ])
        
        # Attention Gates for skip connections
        self.attention_gates = nn.ModuleList([
            AttentionGate(F_g=dims[i], F_l=dims[i+1], F_int=dims[i+1] // 2)
            for i in range(len(dims)-1)
        ])

        # 上采样块
        self.up_blocks = nn.ModuleList([
            UpSampleBlock(dims[i], dims[i+1])
            for i in range(len(dims)-1)
        ])
        
        # 特征金字塔池化
        self.pyramid_pooling = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(dims[0], dims[0]//4, 1),
                LayerNorm(dims[0]//4),
                nn.GELU()
            ) for s in [1, 2, 4, 8]
        ])
            
    def forward(self, encoder_features):
        x = encoder_features[-1]
        
        # 添加金字塔池化特征
        pyramid_feats = [x]
        h, w = x.shape[2:]
        for pool in self.pyramid_pooling:
            feat = pool(x)
            feat = F.interpolate(feat, (h, w), mode='bilinear', align_corners=True)
            pyramid_feats.append(feat)
        
        # 合并并调整特征通道数
        x = torch.cat(pyramid_feats, dim=1)
        x = self.init_conv(x)  # 将通道数调整回原始维度
        
        # 解码过程
        for i, up_block in enumerate(self.up_blocks):
            # 特征增强和自适应
            g = self.enhance[i](x) * self.adapters[i](x)
            skip = self.enhance[i+1](encoder_features[-(i+2)]) * self.adapters[i+1](encoder_features[-(i+2)])
            
            # 应用注意力门控
            gated_skip = self.attention_gates[i](g=g, x=skip)

            # 上采样和融合
            x = up_block(g, gated_skip)
        
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output