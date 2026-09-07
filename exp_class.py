import torch
import torch.nn as nn

class BaselineTactileTransformer(nn.Module):
    def __init__(self, input_channels=3, num_classes=4, d_model=64, num_heads=4, num_layers=2, max_seq_len=150):
        super(BaselineTactileTransformer, self).__init__()
        
        # 1. 时序预处理阶段 (Temporal Preprocessing / Token Embedding)
        # 将原始的 3 维合力 (Fx, Fy, Fz) 投影到高维特征空间，同时利用卷积核捕捉局部时间依赖
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        
        # 可学习的位置编码 (Learnable Positional Encoding)
        # 长度设为 150 帧，维度与 d_model 一致
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # 2. Transformer 主体 (Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. 分类头 (Classification Head)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        x: 经过标准化对齐的极简触觉序列，形状 [Batch, Time_steps(150), Channels(3)]
        """
        # Conv1d 需要的输入形状是 [Batch, Channels, Time_steps]
        x = x.transpose(1, 2)
        
        # 时序预处理投影
        x = self.input_proj(x)  # 形状变为: [Batch, d_model, 150]
        
        # Transformer 需要的输入形状是 [Batch, Time_steps, d_model]
        x = x.transpose(1, 2)   # 形状变为: [Batch, 150, d_model]
        
        # 注入位置编码
        # 注意：这里切片是为了防止实际输入序列不足 150 帧时报错
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Transformer 提取全局上下文关联
        x = self.transformer_encoder(x) # 形状保持: [Batch, 150, d_model]
        
        # 全局平均池化 (Global Average Pooling) 将整个时间序列压缩为一个特征向量
        x = x.mean(dim=1)               # 形状变为: [Batch, d_model]
        
        # 分类器输出
        out = self.classifier(x)        # 形状变为: [Batch, num_classes(4)]
        return out