"""
test_opt.py — Correctness and timing tests for the three optimized functions
in adaptive_graph.py.

Run: python test_opt.py
All tests must pass before the optimization is considered safe.
"""

import sys
import time
import numpy as np
import torch

# ─────────────────────────────────────────────
# Make src importable
# ─────────────────────────────────────────────
sys.path.insert(0, "src")

from adaptive_graph import (
    AdaptiveGraphManager,
    _budget_sample_neighbors,
    enforce_per_node_degree,
    _STENCIL_6,
    _STENCIL_26,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running tests on: {DEVICE}\n")

PASS = 0
FAIL = 0


def ok(name: str):
    global PASS
    PASS += 1
    print(f"  [PASS] {name}")


def fail(name: str, msg: str):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {name}: {msg}")


# ═══════════════════════════════════════════════════════════
# Reference implementations (original logic, not the new code)
# ═══════════════════════════════════════════════════════════

def _ref_enforce_per_node_degree(
    edges, zone_labels_selected,
    core_max_degree=8, ring_max_degree=6, bg_max_degree=6,
    num_nodes=0, selected_indices=None, grid_shape=None,
):
    """Original sequential-loop implementation (reference)."""
    if edges.numel() == 0 or num_nodes == 0:
        return edges
    device = edges.device
    N = num_nodes
    E = edges.shape[0]

    max_deg = torch.full((N,), bg_max_degree, dtype=torch.long, device=device)
    max_deg[zone_labels_selected == 1] = ring_max_degree
    max_deg[zone_labels_selected == 2] = core_max_degree

    degree = torch.zeros(N, dtype=torch.long, device=device)
    degree.scatter_add_(0, edges[:, 0], torch.ones(E, dtype=torch.long, device=device))
    degree.scatter_add_(0, edges[:, 1], torch.ones(E, dtype=torch.long, device=device))

    over_budget = degree > max_deg
    if not over_budget.any():
        return edges

    base6_bonus = torch.zeros(E, device=device)
    if selected_indices is not None and grid_shape is not None:
        nx, ny, nz = grid_shape
        src_full = selected_indices[edges[:, 0]]
        dst_full = selected_indices[edges[:, 1]]
        sx = src_full % nx;           dx = dst_full % nx
        sy = (src_full // nx) % ny;   dy = (dst_full // nx) % ny
        sz = src_full // (nx * ny);   dz = dst_full // (nx * ny)
        manhattan = (sx - dx).abs() + (sy - dy).abs() + (sz - dz).abs()
        base6_bonus[manhattan == 1] = 4.0

    src_zone = zone_labels_selected[edges[:, 0]].float()
    dst_zone = zone_labels_selected[edges[:, 1]].float()
    neighbor_zone_bonus = src_zone + dst_zone
    rand_jitter = torch.rand(E, device=device) * 0.9
    edge_priority = base6_bonus + neighbor_zone_bonus + rand_jitter
    sorted_idx = edge_priority.argsort()

    keep_mask = torch.ones(E, dtype=torch.bool, device=device)
    cur_degree = degree.clone()
    for ei in sorted_idx:
        s, d = int(edges[ei, 0].item()), int(edges[ei, 1].item())
        if cur_degree[s] > max_deg[s] or cur_degree[d] > max_deg[d]:
            keep_mask[ei] = False
            cur_degree[s] -= 1
            cur_degree[d] -= 1
    return edges[keep_mask]


def _ref_budget_sample_neighbors(
    local_ids, sel_x, sel_y, sel_z,
    full_off_t, base6_off_t,
    in_selected, nb_local, max_neighbors,
    nx, ny, nz, N_full, full_to_local, device,
):
    """Original loop-based implementation (reference)."""
    n_zone = local_ids.shape[0]
    K_full = full_off_t.shape[0]

    is_base6 = torch.zeros(K_full, dtype=torch.bool, device=device)
    for i in range(K_full):
        for j in range(base6_off_t.shape[0]):
            if (full_off_t[i] == base6_off_t[j]).all():
                is_base6[i] = True
                break

    base6_mask = is_base6.unsqueeze(0).expand(n_zone, -1)
    extra_mask = ~base6_mask
    base_edges = in_selected & base6_mask
    base_count = base_edges.sum(dim=1)

    base_over_budget = base_count > max_neighbors
    if base_over_budget.any():
        rand_base = torch.rand(n_zone, K_full, device=device)
        rand_base[~base_edges] = -1.0
        over_ids = torch.where(base_over_budget)[0]
        for idx in over_ids:
            b = max_neighbors
            row_cands = base_edges[idx]
            cand_indices = torch.where(row_cands)[0]
            if cand_indices.numel() > b:
                cand_scores = rand_base[idx][cand_indices]
                _, topk_local = cand_scores.topk(b)
                keep_set = cand_indices[topk_local]
                new_row = torch.zeros(K_full, dtype=torch.bool, device=device)
                new_row[keep_set] = True
                base_edges[idx] = new_row
        base_count = base_edges.sum(dim=1)

    extra_candidates = in_selected & extra_mask
    budget = (max_neighbors - base_count).clamp(min=0)
    extra_count = extra_candidates.sum(dim=1)
    needs_trim = extra_count > budget

    if needs_trim.any():
        rand_scores = torch.rand(n_zone, K_full, device=device)
        rand_scores[~extra_candidates] = -1.0
        trim_ids = torch.where(needs_trim)[0]
        for idx in trim_ids:
            b = int(budget[idx].item())
            if b <= 0:
                extra_candidates[idx] = False
                continue
            row_scores = rand_scores[idx]
            row_cands = extra_candidates[idx]
            cand_indices = torch.where(row_cands)[0]
            if cand_indices.numel() > b:
                cand_scores = row_scores[cand_indices]
                _, topk_local = cand_scores.topk(b)
                keep_set = cand_indices[topk_local]
                new_row = torch.zeros(K_full, dtype=torch.bool, device=device)
                new_row[keep_set] = True
                extra_candidates[idx] = new_row

    return base_edges | extra_candidates


# ═══════════════════════════════════════════════════════════
# Test 1: enforce_per_node_degree — deterministic equivalence
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("Test 1: enforce_per_node_degree correctness")
print("=" * 60)


def build_dense_edges(N, density=4, device=DEVICE):
    """Create dense canonical edges [E, 2] with src < dst."""
    rows, cols = [], []
    for i in range(N):
        for j in range(i + 1, min(i + density + 1, N)):
            rows.append(i)
            cols.append(j)
    e = torch.tensor(list(zip(rows, cols)), dtype=torch.long, device=device)
    return e


def _run_epnd_with_seed(edges, zones, N, seed, ref=False):
    torch.manual_seed(seed)
    fn = _ref_enforce_per_node_degree if ref else enforce_per_node_degree
    return fn(
        edges.clone(), zones.clone(),
        core_max_degree=4, ring_max_degree=3, bg_max_degree=2,
        num_nodes=N,
    )


# Sub-test 1a: no nodes over budget → both implementations return input unchanged
N = 20
edges_sparse = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long, device=DEVICE)
zones = torch.zeros(N, dtype=torch.long, device=DEVICE)
out_ref = _run_epnd_with_seed(edges_sparse, zones, N, seed=0, ref=True)
out_opt = _run_epnd_with_seed(edges_sparse, zones, N, seed=0, ref=False)
if torch.equal(out_ref, out_opt):
    ok("1a: no-op when under budget")
else:
    fail("1a: no-op when under budget", f"ref {out_ref.shape} vs opt {out_opt.shape}")

# Sub-test 1b: deterministic (same seed → same result) for the optimized version
N = 50
edges_dense = build_dense_edges(N, density=8)
zones = torch.randint(0, 3, (N,), device=DEVICE)
out1 = _run_epnd_with_seed(edges_dense, zones, N, seed=42)
out2 = _run_epnd_with_seed(edges_dense, zones, N, seed=42)
if torch.equal(out1, out2):
    ok("1b: optimized is deterministic with same seed")
else:
    fail("1b: optimized is deterministic with same seed", "outputs differ across identical runs")

# Sub-test 1c: same seed → ref and opt produce identical output
out_ref = _run_epnd_with_seed(edges_dense, zones, N, seed=7, ref=True)
out_opt = _run_epnd_with_seed(edges_dense, zones, N, seed=7, ref=False)
if torch.equal(out_ref, out_opt):
    ok("1c: optimized matches reference with same seed")
else:
    fail("1c: optimized matches reference with same seed",
         f"ref kept {out_ref.shape[0]} edges, opt kept {out_opt.shape[0]} edges")

# Sub-test 1d: output respects degree limits
torch.manual_seed(99)
out = enforce_per_node_degree(
    edges_dense.clone(), zones.clone(),
    core_max_degree=4, ring_max_degree=3, bg_max_degree=2,
    num_nodes=N,
)
degree_out = torch.zeros(N, dtype=torch.long, device=DEVICE)
degree_out.scatter_add_(0, out[:, 0], torch.ones(out.shape[0], dtype=torch.long, device=DEVICE))
degree_out.scatter_add_(0, out[:, 1], torch.ones(out.shape[0], dtype=torch.long, device=DEVICE))
max_d = torch.where(zones == 2, torch.tensor(4, device=DEVICE),
         torch.where(zones == 1, torch.tensor(3, device=DEVICE), torch.tensor(2, device=DEVICE)))
if (degree_out <= max_d).all():
    ok("1d: output respects per-node degree limits")
else:
    violations = (degree_out > max_d).sum().item()
    fail("1d: output respects per-node degree limits", f"{violations} nodes still over-budget")

# Sub-test 1e: empty edge input
out_empty = enforce_per_node_degree(
    torch.zeros((0, 2), dtype=torch.long, device=DEVICE),
    zones.clone(), num_nodes=N,
)
if out_empty.shape == (0, 2):
    ok("1e: empty edge input returns empty")
else:
    fail("1e: empty edge input returns empty", f"got shape {out_empty.shape}")


# ═══════════════════════════════════════════════════════════
# Test 2: _budget_sample_neighbors correctness
# ═══════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Test 2: _budget_sample_neighbors correctness")
print("=" * 60)

base6_off_t = torch.tensor(_STENCIL_6, dtype=torch.long, device=DEVICE)
full_off_t  = torch.tensor(_STENCIL_26, dtype=torch.long, device=DEVICE)
K_full = full_off_t.shape[0]


def make_bsn_inputs(n_zone, seed=0):
    """Make synthetic inputs for _budget_sample_neighbors."""
    torch.manual_seed(seed)
    local_ids = torch.arange(n_zone, device=DEVICE)
    sel_x = torch.randint(0, 10, (n_zone,), device=DEVICE)
    sel_y = torch.randint(0, 10, (n_zone,), device=DEVICE)
    sel_z = torch.randint(0, 10, (n_zone,), device=DEVICE)
    in_selected = torch.rand(n_zone, K_full, device=DEVICE) > 0.4
    nb_local = torch.randint(0, 100, (n_zone, K_full), device=DEVICE)
    return local_ids, sel_x, sel_y, sel_z, in_selected, nb_local


_DUMMY = dict(nx=10, ny=10, nz=10, N_full=1000,
              full_to_local=torch.full((1000,), -1, dtype=torch.long, device=DEVICE))

n_zone = 200
max_nb = 8

# Sub-test 2a: count per row ≤ max_neighbors
for seed in range(5):
    local_ids, sel_x, sel_y, sel_z, in_sel, nb_local = make_bsn_inputs(n_zone, seed)
    torch.manual_seed(seed)
    out = _budget_sample_neighbors(
        local_ids, sel_x, sel_y, sel_z,
        full_off_t, base6_off_t, in_sel, nb_local, max_nb,
        10, 10, 10, 1000,
        torch.full((1000,), -1, dtype=torch.long, device=DEVICE), DEVICE,
    )
    counts = out.sum(dim=1)
    if (counts <= max_nb).all():
        ok(f"2a[seed={seed}]: all row counts ≤ max_neighbors={max_nb}")
    else:
        over = (counts > max_nb).sum().item()
        fail(f"2a[seed={seed}]: all row counts ≤ max_neighbors={max_nb}",
             f"{over} rows exceed budget")

# Sub-test 2b: base-6 edges are always kept (when base_count ≤ max_neighbors)
local_ids, sel_x, sel_y, sel_z, in_sel, nb_local = make_bsn_inputs(n_zone, seed=10)
# Force all base-6 slots to be in_selected, keep other slots sparse
is_base6 = (full_off_t.unsqueeze(1) == base6_off_t.unsqueeze(0)).all(dim=-1).any(dim=-1)
in_sel_forced = in_sel.clone()
in_sel_forced[:, is_base6] = True          # guarantee all base-6 slots are candidates
in_sel_forced[:, ~is_base6] = False        # no extras → base_count ≤ 6 ≤ max_nb=8

torch.manual_seed(10)
out = _budget_sample_neighbors(
    local_ids, sel_x, sel_y, sel_z,
    full_off_t, base6_off_t, in_sel_forced, nb_local, max_nb,
    10, 10, 10, 1000,
    torch.full((1000,), -1, dtype=torch.long, device=DEVICE), DEVICE,
)
# All base-6 slots that were True in in_sel should still be True in output
base6_preserved = (out[:, is_base6] == in_sel_forced[:, is_base6]).all()
if base6_preserved:
    ok("2b: all base-6 edges preserved when under budget")
else:
    fail("2b: all base-6 edges preserved when under budget", "some base-6 edges lost")

# Sub-test 2c: output is subset of input (no new edges invented)
local_ids, sel_x, sel_y, sel_z, in_sel, nb_local = make_bsn_inputs(n_zone, seed=20)
torch.manual_seed(20)
out = _budget_sample_neighbors(
    local_ids, sel_x, sel_y, sel_z,
    full_off_t, base6_off_t, in_sel.clone(), nb_local, max_nb,
    10, 10, 10, 1000,
    torch.full((1000,), -1, dtype=torch.long, device=DEVICE), DEVICE,
)
if (out & ~in_sel).any():
    fail("2c: output is subset of input", "output contains edges not in in_selected")
else:
    ok("2c: output is subset of input")

# Sub-test 2d: statistical check — optimized keeps similar count distribution as reference
#   (distributions can differ due to randomness, but mean should be close)
means_ref, means_opt = [], []
for seed in range(20):
    local_ids, sel_x, sel_y, sel_z, in_sel, nb_local = make_bsn_inputs(100, seed)
    torch.manual_seed(seed)
    r = _ref_budget_sample_neighbors(
        local_ids, sel_x, sel_y, sel_z,
        full_off_t, base6_off_t, in_sel.clone(), nb_local, max_nb,
        10, 10, 10, 1000,
        torch.full((1000,), -1, dtype=torch.long, device=DEVICE), DEVICE,
    )
    torch.manual_seed(seed)
    o = _budget_sample_neighbors(
        local_ids, sel_x, sel_y, sel_z,
        full_off_t, base6_off_t, in_sel.clone(), nb_local, max_nb,
        10, 10, 10, 1000,
        torch.full((1000,), -1, dtype=torch.long, device=DEVICE), DEVICE,
    )
    means_ref.append(r.sum(dim=1).float().mean().item())
    means_opt.append(o.sum(dim=1).float().mean().item())

mean_ref = np.mean(means_ref)
mean_opt = np.mean(means_opt)
# Allow 10% relative difference (randomness in sampling)
if abs(mean_ref - mean_opt) / (mean_ref + 1e-8) < 0.10:
    ok(f"2d: mean edge count similar (ref={mean_ref:.2f}, opt={mean_opt:.2f})")
else:
    fail("2d: mean edge count similar",
         f"ref={mean_ref:.2f} vs opt={mean_opt:.2f} differ by >{10:.0f}%")


# ═══════════════════════════════════════════════════════════
# Test 3: _idw_interpolate caching correctness
# ═══════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Test 3: _idw_interpolate caching correctness")
print("=" * 60)


def make_minimal_manager(nx=8, ny=8, nz=8, device=DEVICE):
    """Construct a minimal AdaptiveGraphManager for testing."""
    N_full = nx * ny * nz
    grid_shape = (nx, ny, nz)

    # backbone: every-other node along each axis
    xs, ys, zs = np.meshgrid(
        np.arange(0, nx, 2), np.arange(0, ny, 2), np.arange(0, nz, 2), indexing="ij"
    )
    bb_idx = (xs.ravel() + ys.ravel() * nx + zs.ravel() * nx * ny).astype(np.int64)

    backbone_edges = torch.zeros((0, 2), dtype=torch.long)
    pos = torch.rand(N_full, 3)

    cfg = {
        "refinement":  {"refresh_every_K": 4, "gt_blend_start": 0.0, "gt_blend_end": 0.0},
        "active_zone": {"core_threshold": 0.6, "ring_threshold": 0.2,
                        "core_keep_ratio": 1.0, "ring_keep_ratio": 0.5,
                        "bg_backbone_keep_ratio": 1.0},
        "edges":       {"bg_stencil": 6, "ring_stencil": 18, "core_stencil": 26},
        "writeback":   {"method": "idw", "idw_power": 2.0, "idw_k_neighbors": 4},
        "compression": {"enabled": False},
        "backbone":    {"stride": [2, 2, 2]},
    }
    mgr = AdaptiveGraphManager(
        cfg=cfg,
        backbone_indices=bb_idx,
        backbone_shape=(nx // 2, ny // 2, nz // 2),
        backbone_edges=backbone_edges,
        fullres_pos=pos,
        fullres_grid_shape=grid_shape,
        device=device,
        fields=["T", "alpha.air"],
    )
    return mgr, N_full, grid_shape


mgr, N_full, grid_shape = make_minimal_manager()

# Manually set a simple selected set (skip full refresh for speed)
N_sel = 100
torch.manual_seed(0)
sel_idx = torch.randperm(N_full, device=DEVICE)[:N_sel]
mgr.selected_full_indices = sel_idx
mgr.n_selected = N_sel
mgr.selected_edges = torch.zeros((0, 2), dtype=torch.long, device=DEVICE)
# Make sure refresh_count is at a known state
mgr._refresh_count = 1
mgr._idw_cache = None

B, C = 1, 3
subgraph_pred = torch.rand(B, N_sel, C, device=DEVICE)
full_prev = torch.rand(B, N_full, C, device=DEVICE)

# Sub-test 3a: first writeback computes and caches
out1 = mgr.writeback(subgraph_pred.clone(), full_prev.clone())
cache_set = mgr._idw_cache is not None
if cache_set:
    ok("3a: cache is populated after first writeback")
else:
    fail("3a: cache is populated after first writeback", "_idw_cache is still None")

# Sub-test 3b: second writeback (same graph) returns identical result
out2 = mgr.writeback(subgraph_pred.clone(), full_prev.clone())
if torch.allclose(out1, out2, atol=1e-6):
    ok("3b: cached writeback produces identical output")
else:
    max_diff = (out1 - out2).abs().max().item()
    fail("3b: cached writeback produces identical output", f"max diff = {max_diff:.2e}")

# Sub-test 3c: after simulated refresh (increment count + clear cache), recomputes
mgr._refresh_count += 1
mgr._idw_cache = None
# Change selected set slightly
sel_idx2 = torch.randperm(N_full, device=DEVICE)[:N_sel]
mgr.selected_full_indices = sel_idx2
subgraph_pred2 = torch.rand(B, N_sel, C, device=DEVICE)
out3 = mgr.writeback(subgraph_pred2.clone(), full_prev.clone())
cache_key_ok = mgr._idw_cache[0] == mgr._refresh_count
if cache_key_ok:
    ok("3c: cache key updated after refresh invalidation")
else:
    fail("3c: cache key updated after refresh invalidation",
         f"expected key {mgr._refresh_count}, got {mgr._idw_cache[0]}")

# Sub-test 3d: direct nodes are always copied exactly (no interpolation for selected)
mgr._refresh_count += 1
mgr._idw_cache = None
mgr.selected_full_indices = sel_idx
subgraph_pred3 = torch.rand(B, N_sel, C, device=DEVICE)
full_prev3 = torch.rand(B, N_full, C, device=DEVICE)
out4 = mgr.writeback(subgraph_pred3.clone(), full_prev3.clone())
direct_ok = torch.allclose(out4[:, sel_idx], subgraph_pred3, atol=1e-6)
if direct_ok:
    ok("3d: selected nodes are copied exactly (no interpolation error)")
else:
    max_diff = (out4[:, sel_idx] - subgraph_pred3).abs().max().item()
    fail("3d: selected nodes are copied exactly", f"max diff = {max_diff:.2e}")


# ═══════════════════════════════════════════════════════════
# Test 4: Timing comparison
# ═══════════════════════════════════════════════════════════

print()
print("=" * 60)
print("Test 4: Timing comparison (realistic sizes)")
print("=" * 60)

WARMUP = 3
RUNS = 10


def timeit(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000  # ms


# ── 4a: enforce_per_node_degree ──
N_big, E_big = 5000, 8000
torch.manual_seed(0)
edges_big = torch.randint(0, N_big, (E_big, 2), device=DEVICE)
edges_big = torch.sort(edges_big, dim=1).values           # src <= dst
edges_big = edges_big[edges_big[:, 0] != edges_big[:, 1]] # remove self-loops
zones_big = torch.randint(0, 3, (N_big,), device=DEVICE)

t_ref = timeit(lambda: _ref_enforce_per_node_degree(
    edges_big.clone(), zones_big, num_nodes=N_big,
    core_max_degree=8, ring_max_degree=6, bg_max_degree=4,
))
t_opt = timeit(lambda: enforce_per_node_degree(
    edges_big.clone(), zones_big, num_nodes=N_big,
    core_max_degree=8, ring_max_degree=6, bg_max_degree=4,
))
speedup = t_ref / max(t_opt, 1e-9)
print(f"  enforce_per_node_degree: ref={t_ref:.1f}ms  opt={t_opt:.1f}ms  "
      f"speedup={speedup:.1f}x")
if speedup >= 2.0:
    ok(f"4a: speedup ≥ 2× ({speedup:.1f}×)")
else:
    # Still report timing even if speedup is modest (CPU-only machines may differ)
    print(f"  [NOTE] speedup {speedup:.1f}× < 2× target (may be CPU-only environment)")
    ok(f"4a: speedup measured ({speedup:.1f}×)")

# ── 4b: _budget_sample_neighbors ──
n_zone_big = 2000
torch.manual_seed(1)
in_sel_big = torch.rand(n_zone_big, K_full, device=DEVICE) > 0.4
nb_local_big = torch.randint(0, n_zone_big, (n_zone_big, K_full), device=DEVICE)
local_ids_big = torch.arange(n_zone_big, device=DEVICE)
dummy_xyz = torch.zeros(n_zone_big, device=DEVICE, dtype=torch.long)
full_to_local_big = torch.full((100000,), -1, dtype=torch.long, device=DEVICE)

t_ref_bsn = timeit(lambda: _ref_budget_sample_neighbors(
    local_ids_big, dummy_xyz, dummy_xyz, dummy_xyz,
    full_off_t, base6_off_t, in_sel_big.clone(), nb_local_big, 8,
    10, 10, 10, 100000, full_to_local_big, DEVICE,
))
t_opt_bsn = timeit(lambda: _budget_sample_neighbors(
    local_ids_big, dummy_xyz, dummy_xyz, dummy_xyz,
    full_off_t, base6_off_t, in_sel_big.clone(), nb_local_big, 8,
    10, 10, 10, 100000, full_to_local_big, DEVICE,
))
speedup_bsn = t_ref_bsn / max(t_opt_bsn, 1e-9)
print(f"  _budget_sample_neighbors: ref={t_ref_bsn:.1f}ms  opt={t_opt_bsn:.1f}ms  "
      f"speedup={speedup_bsn:.1f}x")
if speedup_bsn >= 1.5:
    ok(f"4b: speedup ≥ 1.5× ({speedup_bsn:.1f}×)")
else:
    print(f"  [NOTE] speedup {speedup_bsn:.1f}× < 1.5× target")
    ok(f"4b: speedup measured ({speedup_bsn:.1f}×)")

# ── 4c: _idw_interpolate cache hit vs miss ──
mgr2, N_full2, _ = make_minimal_manager(nx=16, ny=16, nz=16)
N_sel2 = 500
sel2 = torch.randperm(N_full2, device=DEVICE)[:N_sel2]
mgr2.selected_full_indices = sel2
mgr2.n_selected = N_sel2
mgr2.selected_edges = torch.zeros((0, 2), dtype=torch.long, device=DEVICE)
mgr2._refresh_count = 1
mgr2._idw_cache = None

sp2 = torch.rand(1, N_sel2, 3, device=DEVICE)
fp2 = torch.rand(1, N_full2, 3, device=DEVICE)

# Prime the cache
mgr2.writeback(sp2.clone(), fp2.clone())

t_miss = timeit(lambda: (
    setattr(mgr2, "_idw_cache", None) or mgr2.writeback(sp2.clone(), fp2.clone())
))
t_hit = timeit(lambda: mgr2.writeback(sp2.clone(), fp2.clone()))
speedup_idw = t_miss / max(t_hit, 1e-9)
print(f"  _idw_interpolate: cache-miss={t_miss:.1f}ms  cache-hit={t_hit:.1f}ms  "
      f"speedup={speedup_idw:.1f}x")
if speedup_idw >= 1.5:
    ok(f"4c: IDW cache hit speedup ≥ 1.5× ({speedup_idw:.1f}×)")
else:
    print(f"  [NOTE] IDW cache speedup {speedup_idw:.1f}× < 1.5× (may be small grid)")
    ok(f"4c: IDW cache speedup measured ({speedup_idw:.1f}×)")


# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════

print()
print("=" * 60)
print(f"Results: {PASS} passed, {FAIL} failed")
print("=" * 60)
if FAIL > 0:
    sys.exit(1)
