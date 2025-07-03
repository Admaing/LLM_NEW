from transformers import PretrainedConfig
import torch.nn as nn
import torch
from Attention import Attention
from ModelConfig import ModelConfig
from common import precompute_freqs_cis, apply_rotary_emb
from RMSNorm import RMSNorm
from MLP import MLP

if __name__ == "__main__":
    config = ModelConfig()
    norm = RMSNorm(config.dim, config.norm_eps)
    x = torch.randn(1, 50, config.dim)
    output = norm(x)
    print(output.shape)
    xq = torch.randn(1, 50, 6, 48)  # bs, seq_len, dim//n_head, n_head_dim
    xk = torch.randn(1, 50, 6, 48)  # bs, seq_len, dim//n_head, n_head_dim
    # 使用 precompute_freqs_cis 函数获取 sin和cos
    cos, sin = precompute_freqs_cis(288 // 6, 50)
    print(cos.shape, sin.shape)
    xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)
    xq_out.shape, xk_out.shape
    print(xq_out.shape, xk_out.shape)
    # 创建Attention实例
    attention_model = Attention(config)

    # 模拟输入数据
    batch_size = 1
    seq_len = 50  # 假设实际使用的序列长度为50
    dim = config.dim
    x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量
    # freqs_cos = torch.rand(seq_len, dim // 2)  # 模拟cos频率，用于RoPE
    # freqs_sin = torch.rand(seq_len, dim // 2)  # 模拟sin频率，用于RoPE

    freqs_cos, freqs_sin = precompute_freqs_cis(dim // config.n_heads, seq_len)

    # 运行Attention模型
    output = attention_model(x, freqs_cos, freqs_sin)

    # attention出来之后的形状 依然是[batch_size, seq_len, dim]
    print("Output shape:", output.shape)
    # 创建MLP实例
    mlp = MLP(config.dim, config.hidden_dim, config.multiple_of, config.dropout)
    # 随机生成数据
    x = torch.randn(1, 50, config.dim)
    # 运行MLP模型
    output = mlp(x)
    print(output.shape)
