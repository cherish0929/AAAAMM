"""
physgto_adaptive.py
===================
Adaptive-graph wrapper around the base PhysGTO model.

Architecture:
  - Reuses Encoder / Mixer / Decoder from physgto.py unchanged.
  - Overrides `autoregressive()` to implement:
      * Fixed backbone graph (always present)
      * Periodic refinement graph refresh every K steps
      * Activity-driven zone classification (core / ring / background)
      * Hybrid edge construction
      * IDW writeback from combined graph to full grid

Module map:
  - AdaptiveModel.__init__         : wraps the base Model
  - AdaptiveModel.forward_on_graph : single-step forward on an arbitrary graph
  - AdaptiveModel.autoregressive   : full rollout with dynamic refinement
  - _refresh_and_forward           : refresh logic + forward (for checkpointing)
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
    """Wraps a base PhysGTO Model with adaptive graph autoregressive rollout.

    The encoder, mixer, and decoder are **shared** with the base model — no
    duplication of parameters.  The only new logic is the graph management
    and writeback that happens *around* each forward step.
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

        # The base PhysGTO model (encoder-mixer-decoder, no change)
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
        """Run one encoder→mixer→decoder step on the given graph.

        This is essentially `BaseModel.forward` but accepts pre-computed
        pos_enc / c_enc and an arbitrary edge set.
        """
        return self.base.forward(
            state_in, node_pos, edges, time_i, conditions,
            pos_enc=pos_enc, c_enc=c_enc, dt=dt,
        )

    # ------------------------------------------------------------------
    # Autoregressive rollout with dynamic adaptive graph
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
        """Autoregressive prediction with backbone + dynamic refinement.

        Args:
            state_in_full:   [B, N_full, C] initial full-grid state (normalized).
            fullres_pos:     [B, N_full, 3] or [N_full, 3] (shared across batch).
            backbone_indices:[B, N_bb] or [N_bb] indices into fullres grid.
            backbone_shape:  [B, 3] or [3].
            backbone_edges:  [B, E_bb, 2] or [E_bb, 2].
            fullres_shape:   [B, 3] or [3].
            time_seq:        [B, T, 1] relative times.
            conditions:      [B, cond_dim].
            gt_states:       [B, T, N_full, C] ground-truth (for active scoring, optional).
            dt:              [B] or scalar.
            check_point:     use gradient checkpointing.
            epoch:           current epoch (controls gt_blend decay).

        Returns:
            predictions: [B, T, N_full, C] full-grid predictions.
        """
        B = state_in_full.shape[0]
        device = state_in_full.device
        T = time_seq.shape[1]

        # --- setup: handle batch dimension for shared topology ---
        # For batchsize=1, squeeze shared tensors for the graph manager
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

        # --- pre-compute condition encoding (constant over rollout) ---
        # conditions: [B, cond_dim]
        c_enc = FourierEmbedding(conditions, 0, self.pos_enc_dim)

        # --- rollout state on full grid ---
        full_state = state_in_full  # [B, N_full, C]
        prev_full_state = full_state.clone()
        outputs = []

        for t in range(T):
            # ---- Graph refresh every K steps ----
            if mgr.should_refresh(t):
                gt_t = gt_states[:, t] if gt_states is not None else None
                # use first batch element for scoring (batch=1 typical)
                self._refresh_graph(
                    mgr, full_state[0], gt_t[0] if gt_t is not None else None,
                    prev_full_state[0], epoch,
                )

            # ---- Gather combined-graph data ----
            combined_state = mgr.gather_state(full_state)       # [B, N_comb, C]
            combined_pos = mgr.get_combined_pos()                # [N_comb, 3]
            combined_edges = mgr.get_combined_edges()            # [E_comb, 2]

            # expand pos to batch dim; use contiguous copies so that each
            # time-step's tensors are independent (critical for checkpoint
            # which re-executes the closure during backward — if these were
            # views/refs the loop variable would be stale).
            combined_pos_b = combined_pos.unsqueeze(0).expand(B, -1, -1).contiguous()
            combined_edges_b = combined_edges.unsqueeze(0).expand(B, -1, -1).contiguous()

            # pos_enc for combined graph (recompute after graph topology changes)
            pos_enc = FourierEmbedding(combined_pos_b, 0, self.pos_enc_dim)

            time_i = time_seq[:, t]

            # ---- Forward on combined graph ----
            if check_point:
                if combined_state.requires_grad is False and combined_state.is_floating_point():
                    combined_state = combined_state.requires_grad_()

                # IMPORTANT: pass ALL graph-topology tensors as explicit
                # arguments so checkpoint saves/restores them correctly.
                # A closure would capture the *loop variable* by reference,
                # which points to a different graph after a refresh step.
                def _ckpt_fwd(s_t, t_i, _pos, _edges, _pos_enc, _c_enc, _cond, _dt):
                    return self.forward_on_graph(
                        s_t, _pos, _edges, t_i, _cond, _pos_enc, _c_enc, _dt,
                    )

                combined_pred = checkpoint(
                    _ckpt_fwd,
                    combined_state, time_i,
                    combined_pos_b, combined_edges_b,
                    pos_enc, c_enc, conditions, dt,
                    use_reentrant=False,
                )
            else:
                combined_pred = self.forward_on_graph(
                    combined_state, combined_pos_b, combined_edges_b, time_i,
                    conditions, pos_enc, c_enc, dt,
                )

            # ---- Writeback to full grid ----
            prev_full_state = full_state.clone()
            full_state = mgr.writeback(combined_pred, full_state)

            outputs.append(full_state)

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
        """Trigger refinement graph refresh on the manager."""
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
