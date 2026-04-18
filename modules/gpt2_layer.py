import math

import torch
from torch import nn

from modules.attention import CausalSelfAttention


def gpt2_gelu(x):
  return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class GPT2Layer(nn.Module):
  """
    在原始 Transformer 中，层归一化（LayerNorm）是放在残差连接之后的（Post-LN）；
    但在 GPT-2 中，LayerNorm 被移到了每个子层（注意力层或 MLP 层）的输入端 。
    因此被称为 Pre-LN Transformer。
    对照Figure 2: GPT-2 transformer layer. 
    12 of these layers are stacked, one after the other, to create the (small)
    version of GPT-2 that you will implement.
    数据流：
    hidden_states [B, T, D]
      ↓
    LayerNorm: attention_layer_norm
      ↓
    masked multi-head self-attention: self_attention
      ↓
    attention_dense
      ↓
    attention_dropout
      ↓
    residual add: + 原始 hidden_states
      ↓
    LayerNorm: out_layer_norm
      ↓
    FFN第一层: interm_dense  [D -> intermediate_size]
      ↓
    GELU
      ↓
    FFN第二层: out_dense  [intermediate_size -> D]
      ↓
    out_dropout
      ↓
    residual add
      ↓
    output [B, T, D]
  """

  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = gpt2_gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """残差连接 + dropout 的 helper function.
    TODO: Implement this helper method for the forward function.
      - This function is applied after the multi-head attention layer as well as after the feed forward layer.
      - GPT-2 layer applies dropout to the transformed output of each sub-layer,
        before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
        IN THIS FUNCTION.
      input: [B, T, D] 子层的输入，包括 multi-head attention layer 和 feed forward layer 的输入。
      output: [B, T, D] 子层的输出，包括 multi-head attention layer 和 feed forward layer 的输出。
      dense_layer: 线性变换层 (nn.Linear)，用来对 output 进行线性变换。
      dropout: dropout 层 (nn.Dropout)，用来对线性变换后的 output 进行 dropout。
    """
    output = dropout(dense_layer(output)) + input
    return output
    ### YOUR CODE HERE


  def forward(self, hidden_states, attention_mask):
    """
    TODO: Implement the forward pass. Some key points to consider:
           - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
           - Layer normalization applied *before* the attention layer and feed-forward layer.
           - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
           - A feed-forward layer that applies transformations to further refine the hidden states.
          实现一个标准 GPT-2 block，顺序是：
          - LayerNorm
          - Self-Attention
          - Residual Add
          - LayerNorm
          - FFN
          - Residual Add
    input:
    - hidden_states: 残差连接的原始输入，形状为 [B, T, D]，attention部分D是hidden_size，FFN部分D是intermediate_size。
    - attention_mask: [B, 1, 1, T] 用于 self-attention 的 padding mask    
    """
    attention_input = self.attention_layer_norm(hidden_states)
    attention_output = self.self_attention(attention_input, attention_mask)

    hidden_states = self.add(hidden_states, attention_output, self.attention_dense, self.attention_dropout)
    ffn_input = self.out_layer_norm(hidden_states)
    ffn_output = self.interm_af(self.interm_dense(ffn_input))
    layer_output = self.add(hidden_states, ffn_output, self.out_dense, self.out_dropout)
    return layer_output
    ### YOUR CODE HERE
