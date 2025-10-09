GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Heuristic placement that explicitly minimises the *future* maximum KVPR.

    1.  Sort heavy / important models first.
    2.  For each model, evaluate every feasible GPU and pick the assignment
        that gives the smallest possible *resulting* maximum-KVPR.
    3.  Run a very small local-search (single-model moves) to polish the result.
    """
    # --- helpers ------------------------------------------------------------
    def kvpr(weight, mem):
        """Return KVPR for one GPU (∞ if no memory left)."""
        return weight / mem if mem > 0 else float("inf")

    def recompute_max_kvpr(w, m):
        return max(kvpr(w[i], m[i]) for i in range(len(w)))

    # ------------------------------------------------------------------------
    # Step-0: order models – place the "hard" ones first
    sorted_models = sorted(
        models,
        key=lambda mdl: (mdl.req_rate / mdl.slo, mdl.model_size),
        reverse=True,
    )

    # GPU state arrays
    placement = {i: [] for i in range(gpu_num)}
    rem_mem = [GPU_MEM_SIZE] * gpu_num
    weight = [0.0] * gpu_num  # Σ r_j / s_j

    # ------------------------------------------------------------------------
    # Step-1: greedy assignment that looks *ahead*
    for mdl in sorted_models:
        contrib = mdl.req_rate / mdl.slo
        best_gpu, best_future_max, best_remaining_mem = None, float("inf"), -1.0

        for gid in range(gpu_num):
            if mdl.model_size > rem_mem[gid]:
                continue  # does not fit

            # imagine we place the model on this GPU
            new_mem = rem_mem[gid] - mdl.model_size
            new_weight = weight[gid] + contrib

            future_weights = weight.copy()
            future_mems = rem_mem.copy()
            future_weights[gid] = new_weight
            future_mems[gid] = new_mem

            future_max = recompute_max_kvpr(future_weights, future_mems)

            # pick GPU with smallest future max-KVPR; break ties in favour of
            # more remaining memory to keep denominators large
            if (future_max < best_future_max - 1e-9) or (
                abs(future_max - best_future_max) <= 1e-9
                and new_mem > best_remaining_mem
            ):
                best_future_max = future_max
                best_gpu = gid
                best_remaining_mem = new_mem

        # If nothing fits we really cannot proceed ➜ error
        if best_gpu is None:
            raise ValueError(
                f"Model {mdl} (size={mdl.model_size}) does not fit on any GPU"
            )

        # commit
        placement[best_gpu].append(mdl)
        weight[best_gpu] += contrib
        rem_mem[best_gpu] -= mdl.model_size

    # ------------------------------------------------------------------------
    # Step-2: local improvement loop
    #
    # We repeatedly look for either:
    #   1) a single-model move that lowers the global max-KVPR, or
    #   2) a pairwise swap that lowers it, if no move helps.
    # This is still quick for the typical problem sizes (≲ hundreds of models)
    # but removes a lot more pressure than the previous “move only from worst
    # GPU” heuristic.
    def global_max():
        return recompute_max_kvpr(weight, rem_mem)

    improved = True
    while improved:
        improved = False
        current_max = global_max()

        # ---------- 1) single-model moves ----------
        for src in sorted(range(gpu_num),
                          key=lambda i: kvpr(weight[i], rem_mem[i]),
                          reverse=True):          # start from the most loaded
            for mdl in list(placement[src]):      # copy – we might mutate list
                contrib = mdl.req_rate / mdl.slo
                for tgt in range(gpu_num):
                    if tgt == src or mdl.model_size > rem_mem[tgt]:
                        continue
                    # simulate the move
                    tmp_w, tmp_m = weight.copy(), rem_mem.copy()
                    tmp_w[src] -= contrib;  tmp_m[src] += mdl.model_size
                    tmp_w[tgt] += contrib; tmp_m[tgt] -= mdl.model_size
                    if recompute_max_kvpr(tmp_w, tmp_m) < current_max:
                        # accept
                        placement[src].remove(mdl)
                        placement[tgt].append(mdl)
                        weight, rem_mem = tmp_w, tmp_m
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue  # start new iteration – we already improved

        # ---------- 2) pairwise swaps ----------
        for g1 in range(gpu_num):
            for g2 in range(g1 + 1, gpu_num):
                for m1 in placement[g1]:
                    c1 = m1.req_rate / m1.slo
                    for m2 in placement[g2]:
                        c2 = m2.req_rate / m2.slo
                        new_mem_g1 = rem_mem[g1] + m1.model_size - m2.model_size
                        new_mem_g2 = rem_mem[g2] + m2.model_size - m1.model_size
                        if new_mem_g1 < 0 or new_mem_g2 < 0:
                            continue  # infeasible
                        tmp_w, tmp_m = weight.copy(), rem_mem.copy()
                        tmp_w[g1] = tmp_w[g1] - c1 + c2
                        tmp_w[g2] = tmp_w[g2] - c2 + c1
                        tmp_m[g1] = new_mem_g1
                        tmp_m[g2] = new_mem_g2
                        if recompute_max_kvpr(tmp_w, tmp_m) < current_max:
                            # accept swap
                            placement[g1].remove(m1)
                            placement[g2].remove(m2)
                            placement[g1].append(m2)
                            placement[g2].append(m1)
                            weight, rem_mem = tmp_w, tmp_m
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return placement

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Test the algorithm

    from evaluator import generate_test_gpu_models
    from evaluator import calculate_kvcache_pressure
    from evaluator import safe_float
    import numpy as np

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):

        results = compute_model_placement(gpu_num, gpu_models)
        max_kvpr = calculate_kvcache_pressure(results)
        all_kvpr.append(safe_float(max_kvpr))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr


    print(f"Max KVPR: {avg_kvpr:.3f}")
