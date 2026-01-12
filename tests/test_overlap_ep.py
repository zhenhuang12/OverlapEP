from overlap_ep.pipelined import pipelined_async_ep
import torch

from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig

def create_grouped_scores(
    scores: torch.Tensor, group_idx: torch.Tensor, num_groups: int
):
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)
    mask = torch.zeros((num_tokens, num_groups), dtype=torch.bool, device=scores.device)
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)
    return (scores * mask).view(num_tokens, num_experts)

def test_overlap_ep(num_tokens,
                    hidden,
                    num_experts,
                    num_topk,      
                    num_nodes: int,
                    num_split, 
                    group,
                    num_topk_groups=4):
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    scores = (
        torch.randn((num_tokens, num_experts),
                    dtype=torch.float32, device="cuda").abs()
        + 1
    )
    group_scores = scores.view(num_tokens, num_nodes, -1).amax(dim=-1)
    group_idx = torch.topk(
        group_scores, k=num_topk_groups, dim=-1, sorted=False
    ).indices
    masked_scores = create_grouped_scores(scores, group_idx, num_nodes)
    topk_idx = torch.topk(masked_scores, num_topk, dim=-1, largest=True, sorted=False)[
        1
    ]
    topk_weights = (
        torch.ones((num_tokens, num_topk),
                   dtype=torch.float32, device="cuda")
    )
    
    mlp_module = PrimusTurboGroupedMLP(num_experts, hidden, num_topk, group)
    pipelined_async_ep(x, topk_idx, topk_weights, num_experts, num_topk, group, mlp_module, num_split)
