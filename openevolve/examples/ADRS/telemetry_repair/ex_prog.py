"""
Network-telemetry repair – “v4 (lean-confidence)” edition
========================================================

Goals of this revision
----------------------
1. **Keep the very solid counter-repair logic** that was already giving
   ≈ 0.95 accuracy.
2. **Improve confidence calibration** – the previous code was still a bit
   *too* optimistic whenever we did *not* touch a counter.  We now:
      • replicate the *larger* side of an asymmetric link more aggressively
        (ratio < 0.67 instead of 0.50) – this increases actual accuracy,
        so we can stay confident where it really matters;
      • cap *all* confidences at **0.90** and introduce a small “baseline
        uncertainty” so that we are never 100 % sure;
      • make the router-imbalance penalty harsher (0.7 instead of 0.8).

Keeping the public interface unchanged
---------------------------------------
`repair_network_telemetry()` still consumes the same *telemetry* and
*topology* dictionaries and returns the *triple* form
`(original, repaired, confidence)` for every mutable field.
"""

from typing import Dict, Any, Tuple, List

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
EPS        = 1e-9   # numerics
TAU_LINK   = 0.02   # 2 % for link-symmetry
RATIO_BAD  = 0.67   # below → assume the smaller value is wrong
TAU_ROUTER = 0.05   # 5 % flow-conservation
CONF_MAX   = 0.90   # never claim more than 0.90 certainty

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _rel_diff(a: float, b: float) -> float:
    """Symmetric relative difference."""
    return abs(a - b) / max(abs(a), abs(b), EPS)


def _put(store: Dict, if_id: str, field: str, new_val: float) -> None:
    """Write the <repaired> slot for a given field."""
    store[if_id][field][1] = new_val


# --------------------------------------------------------------------------- #
# Main work horse
# --------------------------------------------------------------------------- #
def repair_network_telemetry(
    telemetry: Dict[str, Dict[str, Any]],
    topology: Dict[str, List[str]],
) -> Dict[str, Dict[str, Tuple]]:
    # ------------------------------------------------------------------ #
    # 0)  Prepare working copy – keep both *orig* and *repaired* values
    # ------------------------------------------------------------------ #
    work: Dict[str, Dict[str, List[Any]]] = {}
    for if_id, d in telemetry.items():
        work[if_id] = {
            'rx_rate'          : [float(d['rx_rate']), float(d['rx_rate'])],
            'tx_rate'          : [float(d['tx_rate']), float(d['tx_rate'])],
            'interface_status' : [d['interface_status'], d['interface_status']],
            # meta (immutable – simple copy)
            'connected_to'     : d.get('connected_to'),
            'local_router'     : d.get('local_router'),
            'remote_router'    : d.get('remote_router'),
        }

    # ------------------------------------------------------------------ #
    # 1)  DOWN interfaces cannot transport traffic – zero them
    # ------------------------------------------------------------------ #
    for if_id, d in work.items():
        if d['interface_status'][1] == 'down':
            _put(work, if_id, 'rx_rate', 0.0)
            _put(work, if_id, 'tx_rate', 0.0)

    # ------------------------------------------------------------------ #
    # 2)  Link-symmetry hardening
    # ------------------------------------------------------------------ #
    handled = set()     # avoid double processing
    for if_id, d in work.items():
        peer = d['connected_to']
        if not peer or peer not in work or peer in handled:
            continue
        handled.update((if_id, peer))

        def _heal(a: str, fld_a: str, b: str, fld_b: str) -> None:
            va = work[a][fld_a][1]
            vb = work[b][fld_b][1]
            if _rel_diff(va, vb) <= TAU_LINK:
                return  # ok
            # decide how to fix
            hi, lo = (va, vb) if va >= vb else (vb, va)
            ratio = lo / max(hi, EPS)
            if ratio < RATIO_BAD:
                # one side almost vanished – copy the higher value
                if va < vb:
                    _put(work, a, fld_a, hi)
                else:
                    _put(work, b, fld_b, hi)
            else:
                # both non-trivial – take the mean
                avg = 0.5 * (va + vb)
                _put(work, a, fld_a, avg)
                _put(work, b, fld_b, avg)

        # direction A.tx  ↔ B.rx
        _heal(if_id, 'tx_rate', peer, 'rx_rate')
        # direction B.tx  ↔ A.rx
        _heal(peer, 'tx_rate', if_id, 'rx_rate')

        # status mismatch → safest mark both DOWN + zero
        st_a = work[if_id]['interface_status'][1]
        st_b = work[peer]['interface_status'][1]
        if st_a != st_b:
            for uid in (if_id, peer):
                work[uid]['interface_status'][1] = 'down'
                _put(work, uid, 'rx_rate', 0.0)
                _put(work, uid, 'tx_rate', 0.0)

    # ------------------------------------------------------------------ #
    # 3)  Flow-conservation check (only to estimate confidence)
    # ------------------------------------------------------------------ #
    router_ok: Dict[str, bool] = {}
    for rtr, ifs in topology.items():
        tot_tx = sum(work[i]['tx_rate'][1] for i in ifs if i in work)
        tot_rx = sum(work[i]['rx_rate'][1] for i in ifs if i in work)
        router_ok[rtr] = (
            max(tot_tx, tot_rx) < EPS
            or _rel_diff(tot_tx, tot_rx) <= TAU_ROUTER
        )

    # ------------------------------------------------------------------ #
    # 4)  Compile final answer with calibrated confidences
    # ------------------------------------------------------------------ #
    def _conf_edit(orig: float, rep: float) -> float:
        """1→no change, 0→≥20 % change (linear)."""
        d = _rel_diff(orig, rep)
        return max(0.0, 1.0 - 5.0 * d)

    out: Dict[str, Dict[str, Tuple]] = {}
    for if_id, d in work.items():
        peer = d['connected_to']
        rtr_bal = router_ok.get(d['local_router'], True)
        router_factor = 1.0 if rtr_bal else 0.7   # harsher penalty

        def _sym_factor(is_tx: bool) -> float:
            if not peer or peer not in work:
                return 0.8   # lack peer info
            mine  = d['tx_rate'][1] if is_tx else d['rx_rate'][1]
            other = work[peer]['rx_rate'][1] if is_tx else work[peer]['tx_rate'][1]
            delta = _rel_diff(mine, other)
            if delta <= TAU_LINK:
                return 1.0
            elif delta <= 0.10:
                return 0.6
            else:
                return 0.3

        # package results
        out_if = {
            'connected_to'  : d['connected_to'],
            'local_router'  : d['local_router'],
            'remote_router' : d['remote_router'],
        }

        for fld, is_tx in (('rx_rate', False), ('tx_rate', True)):
            orig, repaired = d[fld]
            conf = (
                0.9               # baseline “never sure”
                * _conf_edit(orig, repaired)
                * _sym_factor(is_tx)
                * router_factor
            )
            conf = max(0.0, min(CONF_MAX, conf))
            out_if[fld] = (orig, repaired, conf)

        # status confidence
        orig_s, rep_s = d['interface_status']
        if orig_s == rep_s:
            base = 0.85 if rep_s == 'up' else 0.90
        else:
            base = 0.60
        conf_s = min(CONF_MAX, base * router_factor)
        out_if['interface_status'] = (orig_s, rep_s, conf_s)

        out[if_id] = out_if

    return out


# --------------------------------------------------------------------------- #
# Thin evaluator wrapper
# --------------------------------------------------------------------------- #
def run_repair(
    telemetry: Dict[str, Dict[str, Any]],
    topology: Dict[str, List[str]],
) -> Dict[str, Any]:
    return repair_network_telemetry(telemetry, topology)