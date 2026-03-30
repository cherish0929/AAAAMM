"""
physgto_adaptive.py  (zone-first refactored)
=============================================
Adaptive-graph wrapper around the base PhysGTO model.

核心改动:
  - zone-first 采样与建边 (active 区域优先级高于 backbone)
  - delayed writeback: 非 refresh 步不 materialize full grid,
    只在 refresh 步 / full supervision 步 / 最终输出步做回写,
    其余时间只在当前联合子图上滚动, 减少 writeback + 插值开销
  - 损失也可以只在子图上计算 (由 subgraph_loss_only 控制)

Module map:
  - AdaptiveModel.__init__           : wraps the base Model
  - AdaptiveModel.forward_on_graph   : single-step forward on an arbitrary graph
  - AdaptiveModel.autoregressive     : full rollout with delayed writeback
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .physgto import Model as BaseModel, FourierEmbedding, get_edge_info
from .adaptive_graph import AdaptiveGraphManager


class AdaptiveModel(nn.Module):
    """Wraps a base PhysGTO Model with zone-first adaptive graph rollout.

    The encoder, mixer, and decoder are **shared** with the base model — no
    duplication of parameters.  The only new logic is the zone-first graph
    management and delayed writeback around each forward step.
    """

    def __init__(
        self,
        space_size: int = 3,
        pos_enc_dim: int = 5,
        cond_dim: int = 32,
        N_block: int = 4,
        in_dim: int = 4,
        out_dim: int = 4,
        enc_dim: int = 128,
        n_head: int = 4,
        n_token: int = 128,
        dt: float = 0.05,
        adaptive_cfg: Optional[dict] = None,
    ):
        super().__init__()

        self.base = BaseModel(
            space_size=space_size,
            pos_enc_dim=pos_enc_dim,
            cond_dim=cond_dim,
            N_block=N_block,
            in_dim=in_dim,
            out_dim=out_dim,
            enc_dim=enc_dim,
            n_head=n_head,
            n_token=n_token,
            dt=dt,
        )
        self.dt_default = dt
        self.pos_enc_dim = pos_enc_dim
        self.adaptive_cfg = adaptive_cfg or {}

    # ------------------------------------------------------------------
    # Single-step forward on an *arbitrary* sub-graph
    # ------------------------------------------------------------------

    def forward_on_graph(
        self,
        state_in: torch.Tensor,
        node_pos: torch.Tensor,
        edges: torch.Tensor,
        time_i: torch.Tensor,
        conditions: torch.Tensor,
        pos_enc: torch.Tensor,
        c_enc: torch.Tensor,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run one encoder→mixer→decoder step on the given graph."""
        return self.base.forward(
            state_in, node_pos, edges, time_i, conditions,
            pos_enc=pos_enc, c_enc=c_enc, dt=dt,
        )

    # ------------------------------------------------------------------
    # Autoregressive rollout with delayed writeback
    # ------------------------------------------------------------------

    def autoregressive(
        self,
        state_in_full: torch.Tensor,
        fullres_pos: torch.Tensor,
        backbone_indices: torch.Tensor,
        backbone_shape: torch.Tensor,
        backbone_edges: torch.Tensor,
        fullres_shape: torch.Tensor,
        time_seq: torch.Tensor,
        conditions: torch.Tensor,
        gt_states: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
        check_point: bool = False,
        epoch: int = 0,
    ) -> torch.Tensor:
        """Autoregressive prediction with zone-first adaptive graph + delayed writeback.

        Delayed writeback strategy:
          - Non-refresh steps: only update on the current subgraph, no full-grid
            materialization.  Loss is computed on subgraph nodes only.
          - Refresh steps (t % K == 0): writeback to full grid first, then
            re-score zones and rebuild subgraph.
          - Final step: always writeback to full grid for output.

        Args:
            state_in_full:   [B, N_full, C] initial full-grid state (normalized).
            fullres_pos:     [B, N_full, 3] or [N_full, 3].
            backbone_indices:[B, N_bb] or [N_bb].
            backbone_shape:  [B, 3] or [3].
            backbone_edges:  [B, E_bb, 2] or [E_bb, 2].
            fullres_shape:   [B, 3] or [3].
            time_seq:        [B, T, 1] relative times.
            conditions:      [B, cond_dim].
            gt_states:       [B, T, N_full, C] ground-truth (optional).
            dt:              [B] or scalar.
            check_point:     use gradient checkpointing.
            epoch:           current epoch (controls gt_blend decay).

        Returns:
            predictions: [B, T, N_full, C] full-grid predictions.
            subgraph_indices: [N_sel] indices into full grid for the last subgraph
                (returned as second element so caller can compute subgraph-only loss).
        """
        B = state_in_full.shape[0]
        device = state_in_full.device
        T = time_seq.shape[1]

        # --- handle batch dimension for shared topology ---
        fp = fullres_pos[0] if fullres_pos.dim() == 3 else fullres_pos
        bb_idx_np = backbone_indices[0].cpu().numpy() if backbone_indices.dim() == 2 else backbone_indices.cpu().numpy()
        bb_shape = tuple(backbone_shape[0].tolist()) if backbone_shape.dim() == 2 else tuple(backbone_shape.tolist())
        bb_edges = backbone_edges[0] if backbone_edges.dim() == 3 else backbone_edges
        fr_shape = tuple(fullres_shape[0].tolist()) if fullres_shape.dim() == 2 else tuple(fullres_shape.tolist())

        # --- initialize AdaptiveGraphManager ---
        mgr = AdaptiveGraphManager(
            cfg=self.adaptive_cfg,
            backbone_indices=bb_idx_np,
            backbone_shape=bb_shape,
            backbone_edges=bb_edges,
            fullres_pos=fp,
            fullres_grid_shape=fr_shape,
            device=device,
        )

        # --- delayed writeback config ---
        delayed_wb = self.adaptive_cfg.get("writeback", {}).get("delayed", True)
        # full_supervision_steps: list of time indices where we force full-grid writeback
        full_sup_steps = set(self.adaptive_cfg.get("writeback", {}).get("full_supervision_steps", []))

        # --- condition encoding (constant over rollout) ---
        c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        # --- state tracking ---
        full_state = state_in_full.clone()  # [B, N_full, C]
        prev_full_state = full_state.clone()
        subgraph_state = None  # [B, N_sel, C], rolling on subgraph

        outputs = []

        for t in range(T):
            is_refresh = mgr.should_refresh(t)
            is_final = (t == T - 1)
            is_full_sup = (t in full_sup_steps)
            need_full_grid = is_refresh or is_final or is_full_sup

            # ---- Graph refresh: writeback first, then re-zone ----
            if is_refresh:
                # If we have subgraph predictions from previous steps,
                # writeback to full grid before re-scoring
                if subgraph_state is not None and t > 0:
                    full_state = mgr.writeback(subgraph_state, full_state)

                gt_t = gt_states[:, t] if gt_states is not None else None
                self._refresh_graph(
                    mgr, full_state[0], gt_t[0] if gt_t is not None else None,
                    prev_full_state[0], epoch,
                )

                # gather fresh subgraph state from (updated) full grid
                subgraph_state = mgr.gather_state(full_state)

            # ---- Gather subgraph data ----
            sel_pos = mgr.get_selected_pos()
            sel_edges = mgr.get_selected_edges()

            sel_pos_b = sel_pos.unsqueeze(0).expand(B, -1, -1).contiguous()
            sel_edges_b = sel_edges.unsqueeze(0).expand(B, -1, -1).contiguous()

            # pos_enc for subgraph
            pos_enc = FourierEmbedding(sel_pos_b, 0, self.pos_enc_dim)
            time_i = time_seq[:, t]

            # ---- Forward on subgraph ----
            if check_point:
                if subgraph_state.requires_grad is False and subgraph_state.is_floating_point():
                    subgraph_state = subgraph_state.requires_grad_()

                def _ckpt_fwd(s_t, t_i, _pos, _edges, _pos_enc, _c_enc, _cond, _dt):
                    return self.forward_on_graph(
                        s_t, _pos, _edges, t_i, _cond, _pos_enc, _c_enc, _dt,
                    )

                subgraph_pred = checkpoint(
                    _ckpt_fwd,
                    subgraph_state, time_i,
                    sel_pos_b, sel_edges_b,
                    pos_enc, c_enc, conditions, dt,
                    use_reentrant=False,
                )
            else:
                subgraph_pred = self.forward_on_graph(
                    subgraph_state, sel_pos_b, sel_edges_b, time_i,
                    conditions, pos_enc, c_enc, dt,
                )

            # ---- Update state ----
            prev_full_state = full_state.clone()
            subgraph_state = subgraph_pred  # rolling on subgraph

            # ---- Writeback decision (delayed writeback) ----
            if need_full_grid or not delayed_wb:
                # materialize full grid
                full_state = mgr.writeback(subgraph_pred, full_state)
                outputs.append(full_state)
            else:
                # delayed: create partial full_state for output tracking
                # Only scatter subgraph predictions onto full grid (no IDW)
                # This is cheaper than full writeback
                full_out = full_state.clone()
                sel_idx = mgr.get_selected_indices()
                full_out[:, sel_idx] = subgraph_pred
                outputs.append(full_out)

        return torch.stack(outputs, dim=1)  # [B, T, N_full, C]

    @torch.no_grad()
    def _refresh_graph(
        self,
        mgr: AdaptiveGraphManager,
        pred_field: torch.Tensor,
        gt_field: Optional[torch.Tensor],
        prev_field: torch.Tensor,
        epoch: int,
    ):
        """Trigger zone-first graph refresh on the manager."""
        mgr.refresh(
            pred_field=pred_field,
            gt_field=gt_field,
            prev_field=prev_field,
            epoch=epoch,
        )

    # ------------------------------------------------------------------
    # Delegate base model attributes for compatibility
    # ------------------------------------------------------------------

    @property
    def dt(self):
        return self.base.dt

    def parameters(self, recurse=True):
        return self.base.parameters(recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self.base.named_parameters(prefix, recurse)

    def state_dict(self, *args, **kwargs):
        return self.base.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.base.load_state_dict(state_dict, strict)
