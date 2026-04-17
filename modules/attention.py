import torch

from einops import rearrange
from torch import nn

class CausalSelfAttention(nn.Module):
  """“x -> scores -> context”的完整形状流：
    给定:
    B = batch size
    T = seq_len
    D = hidden_size
    H = num_heads
    Dh = D / H
    输入:
    x: [B, T, D]
    1) 线性投影 (分别对 Q/K/V)
    Q_lin = Wq(x): [B, T, D]
    K_lin = Wk(x): [B, T, D]
    V_lin = Wv(x): [B, T, D]
    2) 拆多头 + 调维 (transform 做的事)
    Q: [B, T, D] --rearrange 'b t (h d) -> b t h d'--> [B, T, H, Dh]
      --rearrange 'b t h d -> b h t d'--> [B, H, T, Dh]
    K: 同上 -> [B, H, T, Dh]
    V: 同上 -> [B, H, T, Dh]
    3) 注意力分数 scores
    K^T (最后两维转置): [B, H, Dh, T]
    scores = Q @ K^T: [B, H, T, T]
    scores = scores / sqrt(Dh): [B, H, T, T]
    4) 加 mask
    - causal mask: [1, 1, T, T] (上三角未来位置 = -inf)
    - padding mask: [B, 1, 1, T] (pad位置 = -10000)
    scores += causal_mask
    scores += attention_mask
    结果仍是: [B, H, T, T]
    5) softmax + dropout
    attn_probs = softmax(scores, dim=-1): [B, H, T, T]
    attn_probs = dropout(attn_probs): [B, H, T, T]
    6) 加权求和得到 context
    context_heads = attn_probs @ V: [B, H, T, Dh]
    7) 合并多头
    context_heads: [B, H, T, Dh]
    --rearrange 'b h t d -> b t h d'--> [B, T, H, Dh]
    --rearrange 'b t h d -> b t (h d)'--> [B, T, D]
    输出:
    context: [B, T, D]
    一行记忆版：
    - scores 是 QK^T，形状一定是 [B, H, T, T]
    - context 是 softmax(scores)V，先 [B,H,T,Dh]，再合并成 [B,T,D]。
  """

  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size # This is just config.hidden_size, but we write it in this way for better readability.

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    # shape of x is [Batch_size, seq_len, hidden_size], and the shape of proj is [Batch_size, seq_len, all_head_size] , where all_head_size = num_attention_heads * attention_head_size = hidden_size.
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    key, query, value: [bs, num_attention_heads, seq_len, attention_head_size]
    attention_mask: [bs, 1, 1, seq_len]

    return: [bs, seq_len, hidden_size]
    """
    ### YOUR CODE HERE
    
    # scores = Q@K^T / sqrt(Dh): [B, H, T, Dh] @ [B, H, Dh, T] -> [B, H, T, T]
    scores = torch.matmul(query, torch.transpose(key, -2, -1))
    scores = scores / (self.attention_head_size ** 0.5)

    # casual mask: [1, 1, T, T] (上三角未来位置 = -inf)
    T = query.size(2)
    causal = torch.triu(torch.ones(T, T, device=query.device), diagonal=1).bool()
    scores = scores.masked_fill(causal, float('-inf'))

    # padding mask: [B, 1, 1, T] (pad位置 = -10000)
    scores = scores + attention_mask

    # attn_probs = softmax(scores, dim=-1): [B, H, T, T]
    attn_probs = torch.softmax(scores, dim = -1)
    attn_probs = self.dropout(attn_probs)

    # context_heads: attn_probs @ V: [B, H, T, T] @ [B, H, T, Dh] -> [B, H, T, Dh]
    context = torch.matmul(attn_probs, value)

    # rearrange: 'b h t d -> b t h d'-> 'b t (h d)' -> [B, T, D]
    context = rearrange(context, 'b h t d -> b t (h d)')
    
    return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value