import torch
import primus_turbo.pytorch as turbo


def _chunked_async_ep(x: torch.Tensor,
                      topk_idx: torch.Tensor,
                      probs: torch.Tensor,
                      mlp_module,
                      num_experts: int,
                      router_topk: int,
                      group):
    # dispatch -> permute -> groupmlp -> unpermute-> combine
    assert x.ndim == 2
    x_shape = x.shape
    num_tokens = x.shape[0]
    num_local_experts = num_experts // group.size()
    num_worst_tokens = num_tokens * group.size()
    num_out_tokens = num_worst_tokens * router_topk
    rev_x, recv_topk_idx, recv_probs, tokens_per_expert, handle = turbo.ops.moe.moe_dispatch(x, topk_idx, probs, num_experts, group, async_finish=True,
                                                                                             allocate_on_comm_stream=True,
                                                                                             num_worst_tokens=num_worst_tokens)

    routing_map, dispatched_probs = turbo.ops.indices_to_multihot(
        recv_topk_idx, recv_probs, num_local_experts, fused=True
    )

    hidden_shape_before_permute = rev_x.shape

    hidden_states, permuted_probs, reversed_mapping_for_combine, tokens_per_expert = (
        turbo.ops.token_permute(
            hidden_states,
            num_out_tokens=num_out_tokens,
            routing_map=routing_map,
            probs=dispatched_probs,
            fused=True,
            return_tokens_per_expert=True,
        )
    )

    hidden_states = mlp_module(
        hidden_states, tokens_per_expert, permuted_probs)
    hidden_states = turbo.ops.token_unpermute(
        hidden_states,
        reversed_mapping_for_combine,
        restore_shape=hidden_shape_before_permute,
        routing_map=routing_map,
        fused=True,
    )
    hidden_states = turbo.ops.moe_combine(
        hidden_states,
        group,
        handle,
        async_finish=True,
        allocate_on_comm_stream=True,
    )

    return hidden_states.view(x_shape)


def pipelined_async_ep(x: torch.Tensor,
                       topk_idx: torch.Tensor,
                       probs: torch.Tensor,
                       num_experts: int,
                       router_topk: int,
                       group,
                       mlp_module,
                       num_splits: int = 4):

    x_chunks = torch.chunk(x, num_splits, dim=0)
    topk_idx_chunks = torch.chunk(topk_idx, num_splits, dim=0)
    probs_chunks = torch.chunk(probs, num_splits, dim=0)
    for i in range(num_splits):
        _chunked_async_ep(x_chunks[i], topk_idx_chunks[i],
                          probs_chunks[i], mlp_module, num_experts, router_topk, group)
