import flashinfer
import torch
from ..nested_tensor import NestedTensor, nested_view

workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device="cuda:0")
prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
    workspace_buffer, "NHD"
)
@torch.compiler.disable
def self_attention(q, k, v, num_qo_heads,
        num_kv_heads,
        head_dim):
    # q : B, L_ragged, C
    # k : B, L_ragged, C
    # v : B, L_ragged, C

    qo_indptr = q.offsets()
    kv_indptr = k.offsets()
    assert q.shape[-1] == num_qo_heads * head_dim
    assert k.shape[-1] == num_kv_heads * head_dim
    assert v.shape[-1] == num_kv_heads * head_dim
    q = nested_view(q, num_qo_heads, head_dim)
    k = nested_view(k, num_qo_heads, head_dim)
    v = nested_view(v, num_qo_heads, head_dim)
    # k's offsets == v's offsets()
    prefill_wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        causal=False,
        non_blocking=False
    )
    # we must wait for the data
    output = prefill_wrapper.run(q.values(), k.values(), v.values())
    output = output.view(-1, num_qo_heads * head_dim)
    output = torch.nested.nested_tensor_from_jagged(output, qo_indptr)
    return output
# import torch.nn.functional as F
# def convert(x):
#     return x.values().view(x.shape[0], -1, *x.shape[2:])
# import math
# def self_attention(q, k, v, num_qo_heads, num_kv_heads, head_dim):
#     """
#     Naive self-attention layer.

#     Args:
#         q (torch.Tensor): Queries tensor of shape [batch_size, seq_len, num_qo_heads * head_dim].
#         k (torch.Tensor): Keys tensor of shape [batch_size, seq_len, num_kv_heads * head_dim].
#         v (torch.Tensor): Values tensor of shape [batch_size, seq_len, num_kv_heads * head_dim].
#         num_qo_heads (int): Number of query/output heads.
#         num_kv_heads (int): Number of key/value heads.
#         head_dim (int): The dimension of each attention head.

#     Returns:
#         torch.Tensor: Output of the self-attention layer.
#     """
#     old_q = q
#     q = convert(q)
#     k = convert(k)
#     v = convert(v)
#     batch_size, seq_len, _ = q.shape

#     # Reshape queries, keys, and values into multiple heads
#     q = q.view(batch_size, seq_len, num_qo_heads, head_dim)
#     k = k.view(batch_size, seq_len, num_qo_heads, head_dim)
#     v = v.view(batch_size, seq_len, num_qo_heads, head_dim)

#     # Transpose for batch matrix multiplication
#     q = q.transpose(1, 2)  # [batch_size, num_qo_heads, seq_len, head_dim]
#     k = k.transpose(1, 2)  # [batch_size, num_qo_heads, seq_len, head_dim]
#     v = v.transpose(1, 2)  # [batch_size, num_qo_heads, seq_len, head_dim]

#     # Calculate attention scores
#     # Scaled dot-product attention: (Q * K^T) / sqrt(head_dim)
#     # [batch_size, num_qo_heads, seq_len, head_dim] @ [batch_size, num_qo_heads, head_dim, seq_len] -> [batch_size, num_qo_heads, seq_len, seq_len]
#     attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

#     # Apply softmax to get attention weights
#     attention_weights = F.softmax(attention_scores, dim=-1)

#     # Apply attention to values
#     # [batch_size, num_qo_heads, seq_len, seq_len] @ [batch_size, num_qo_heads, seq_len, head_dim] -> [batch_size, num_qo_heads, seq_len, head_dim]
#     output = torch.matmul(attention_weights, v)

#     # Concatenate heads and reshape back to original shape
#     output = output.transpose(1, 2).contiguous()
#     output = output.view(batch_size, seq_len, num_qo_heads * head_dim)

#     return output
