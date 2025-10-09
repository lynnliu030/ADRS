# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
"""

# EVOLVE-BLOCK-START

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Sort groups by weight in descending order.
    # 'weight' is already float and on its original device (e.g., GPU)
    # from rebalance_experts, so no explicit device transfer is needed here.
    _, sorted_indices = weight.sort(-1, descending=True)

    # Initialize pack_index and rank_in_pack.
    pack_index = torch.empty_like(weight, dtype=torch.int64)
    rank_in_pack = torch.empty_like(pack_index)
    
    # Generate the assignment pattern for items in sorted order (k = 0 to num_groups-1)
    k_indices = torch.arange(num_groups, dtype=torch.int64, device=weight.device)

    # Calculate rank_in_pack for the sorted items.
    # The rank is based on how many items have already been assigned to a pack.
    # Since each pack receives `groups_per_pack` items, and we are assigning
    # items in sorted order, the `k`-th sorted item will get rank `k // num_packs`.
    ranks_for_sorted_items = k_indices // num_packs

    # Calculate pack_id for the sorted items using the "snake" pattern.
    # This ensures that the heaviest items are distributed among packs,
    # and subsequent items fill them in a way that keeps the sums balanced.
    idx_in_block = k_indices % num_packs
    block_num = k_indices // num_packs
    is_even_block = (block_num % 2 == 0)
    
    packs_for_sorted_items = torch.where(
        is_even_block,
        idx_in_block,
        num_packs - 1 - idx_in_block
    )

    # Expand the assignments to all layers
    expanded_packs = packs_for_sorted_items.unsqueeze(0).expand(num_layers, -1)
    expanded_ranks = ranks_for_sorted_items.unsqueeze(0).expand(num_layers, -1)

    # Use scatter_ to populate the output tensors based on original group IDs.
    # For each layer, `sorted_indices[i, k]` gives the original group ID
    # for the item at sorted position `k`.
    # We want to assign `expanded_packs[i, k]` to `pack_index[i, original_group_id]`
    # And `expanded_ranks[i, k]` to `rank_in_pack[i, original_group_id]`
    pack_index.scatter_(1, sorted_indices, expanded_packs)
    rank_in_pack.scatter_(1, sorted_indices, expanded_ranks)

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized. Optimized using a greedy approach
    with vectorized PyTorch operations.

    Parameters:
        weight: [X, num_log]  (e.g., [num_layers * num_nodes, num_logical_experts_per_node])
        num_phy: total number of experts after replication (e.g., num_physical_experts_per_node)

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank (0-indexed)
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    num_layers, num_log = weight.shape
    assert num_phy >= num_log
    device = weight.device

    # Initialize output tensors
    phy2log = torch.empty((num_layers, num_phy), dtype=torch.int64, device=device)
    rank = torch.empty((num_layers, num_phy), dtype=torch.int64, device=device)
    
    # current_logcnt: tracks the number of replicas assigned to each logical expert.
    # Initialized to 0, will be incremented as replicas are assigned.
    current_logcnt = torch.zeros(num_layers, num_log, dtype=torch.int64, device=device)
    
    # current_replica_rank_to_assign: For each logical expert, this stores the next 0-indexed
    # replica rank that will be assigned if this expert is chosen.
    current_replica_rank_to_assign = torch.zeros(num_layers, num_log, dtype=torch.int64, device=device)

    # next_avg_load_score: This tensor stores the 'score' for each logical expert.
    # The score is `weight[i,j] / (current_logcnt[i,j] + 1.0)`.
    # We want to pick the expert that, if it receives its next replica,
    # would have the *highest* average load among all experts.
    # This is equivalent to picking the expert that currently has the highest average load,
    # and then assigning it a replica to reduce its load.
    # A small epsilon is added to the denominator to prevent division by zero,
    # especially if `weight` is 0 and `current_logcnt` is 0.
    # `weight` is already float from `rebalance_experts` caller.
    next_avg_load_score = weight / (current_logcnt.float() + 1.0 + 1e-6)

    # Pre-allocate ones tensor for scatter_add_ operations, as its shape is constant.
    ones_for_scatter = torch.ones(num_layers, dtype=torch.int64, device=device)

    # Iterate num_phy times to fill all physical expert slots.
    # In each iteration, one physical expert slot is assigned for each layer.
    for k_phy_idx in range(num_phy):
        # For each layer, find the logical expert with the maximum `next_avg_load_score`.
        # `max_scores` will be [num_layers], `best_log_indices` will be [num_layers].
        max_scores, best_log_indices = torch.max(next_avg_load_score, dim=-1)

        # Assign the selected logical expert and its rank to the current physical slot (`k_phy_idx`).
        # `best_log_indices` contains the logical expert ID for each layer.
        # `current_replica_rank_to_assign` (gathered using `best_log_indices`) provides the
        # 0-indexed rank for that logical expert's current replica being assigned.
        phy2log[:, k_phy_idx] = best_log_indices
        rank[:, k_phy_idx] = current_replica_rank_to_assign.gather(
            1, best_log_indices.unsqueeze(1)).squeeze(1)

        # Update `current_logcnt` for the selected experts (increment by 1).
        current_logcnt.scatter_add_(1, best_log_indices.unsqueeze(1), ones_for_scatter.unsqueeze(1))
        
        # Update `current_replica_rank_to_assign` for the selected experts (increment by 1).
        # This prepares the rank for the *next* replica assignment for these experts.
        current_replica_rank_to_assign.scatter_add_(
            1, best_log_indices.unsqueeze(1), ones_for_scatter.unsqueeze(1))
        
        # Calculate the new `next_avg_load_score` for the selected logical experts.
        # This is `weight[i, selected_expert] / (new_current_logcnt[i, selected_expert] + 1.0)`.
        # The `current_logcnt` for these experts has just been incremented.
        
        # Gather the weights of the selected experts. `weight` is already float.
        selected_weights = weight.gather(1, best_log_indices.unsqueeze(1)).squeeze(1)
        # Gather the new (incremented) counts for the selected experts.
        updated_counts = current_logcnt.gather(1, best_log_indices.unsqueeze(1)).squeeze(1).float()
        
        # Calculate the new scores for these selected experts.
        # This is the average load if they receive one more replica (i.e., (updated_counts + 1)).
        new_scores = selected_weights / (updated_counts + 1.0 + 1e-6)
        
        # Scatter the new scores back into `next_avg_load_score`.
        next_avg_load_score.scatter_(1, best_log_indices.unsqueeze(1), new_scores.unsqueeze(1))
        
    # The `current_logcnt` tensor now holds the final number of replicas for each logical expert.
    return phy2log, rank, current_logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        # More concise way to get values 0 to perm.size(1)-1 for scattering
        values = torch.arange(perm.size(1), dtype=torch.int64, device=perm.device)
        inv.scatter_(
            1,
            perm,
            values.unsqueeze(0).expand_as(perm), # Expand values to match perm's shape
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
    ).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
            (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of
            each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica
            indices for each expert
        expert_count: [layers, num_logical_experts], number of physical
            replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float() # Keep on original device (likely GPU)
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
    # Determine the maximum number of replicas any single logical expert received.
    # This ensures log2phy is sized precisely to avoid excessive memory allocation.
    max_log_replicas = logcnt.max().item()
    # Ensure max_log_replicas is at least 1 to prevent zero-sized dimensions
    # in case logcnt.max() is 0 (e.g., if no experts were assigned replicas,
    # though this should not happen given num_phy >= num_log assertion).
    if max_log_replicas == 0:
        max_log_replicas = 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, max_log_replicas),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    # The scatter_ operation needs a flat index.
    # The index is logical_expert_id * max_log_replicas + phyrank.
    # The maximum index will be (num_logical_experts - 1) * max_log_replicas + (max_log_replicas - 1)
    # which equals num_logical_experts * max_log_replicas - 1.
    # So the flattened size of log2phy must be num_logical_experts * max_log_replicas.
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * max_log_replicas + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=log2phy.device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

