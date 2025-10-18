#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gift_init.py
# Author            : Yiting Liu, Yibo Lin <yibolin@pku.edu.cn>
# Modified          : 2025-10 by Qzysme to include pin-risk aware adaptive spectral weighting

import logging
import time
from typing import List, Optional

import numpy as np
import scipy
import torch
from numpy.random import default_rng
from torch import nn

import dreamplace.ops.gift_init.gift_init_cpp as gift_init_cpp
import dreamplace.ops.gift_init.utils_gpu.mix_frequency_filter as mix_frequency_filter
import dreamplace.ops.gift_init.utils_gpu.util as util

logger = logging.getLogger(__name__)


class _GiFtBase(nn.Module):
    """Shared utilities for GiFt-based initialization."""

    def __init__(
        self,
        flat_netpin,
        netpin_start,
        pin2node_map,
        net_weights,
        net_mask,
        xl,
        yl,
        xh,
        yh,
        num_nodes,
        num_movable_nodes,
        scale=0.7,
        pin_risk_map=None,
        adapt_flag=0,
        adapt_samples=32,
        adapt_lambda_hpwl=1.0,
        adapt_lambda_density=0.2,
        adapt_lambda_risk=0.3,
        adapt_risk_step=5.0,
        adapt_seed=2025,
    ):
        super(_GiFtBase, self).__init__()

        self.num_nodes = num_nodes
        self.num_movable_nodes = num_movable_nodes
        self.num_fixed_nodes = max(num_nodes - num_movable_nodes, 0)
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.scale = scale

        self.adapt_flag = bool(adapt_flag)
        self.adapt_samples = max(int(adapt_samples), 0)
        self.adapt_lambda_hpwl = float(adapt_lambda_hpwl)
        self.adapt_lambda_density = float(adapt_lambda_density)
        self.adapt_lambda_risk = float(adapt_lambda_risk)
        self.adapt_risk_step = float(adapt_risk_step)
        self.adapt_rng = default_rng(adapt_seed)
        self.hpwl_sample_limit = 5000
        self.node_sample_limit = 20000

        logger.info("Construct adjacency matrix using clique model")
        ret = gift_init_cpp.adj_matrix_forward(
            flat_netpin.cpu(),
            netpin_start.cpu(),
            pin2node_map.cpu(),
            net_weights.cpu(),
            net_mask.cpu(),
            num_nodes,
        )
        data = ret[0]
        row = ret[1]
        col = ret[2]
        dtype = np.float32 if net_weights.dtype == torch.float32 else np.float64
        self.adj_mat = scipy.sparse.coo_matrix(
            (data.numpy(), (row.numpy(), col.numpy())),
            dtype=dtype,
            shape=(num_nodes, num_nodes),
        )
        logger.info("Done matrix construction")

        # cache netlist data in numpy for cost evaluation
        self.flat_netpin_np = flat_netpin.cpu().numpy()
        self.netpin_start_np = netpin_start.cpu().numpy()
        self.pin2node_np = pin2node_map.cpu().numpy()
        self.num_nets = len(self.netpin_start_np) - 1

        # risk map and its gradients
        if pin_risk_map is not None and pin_risk_map.size:
            self.risk_map = np.array(pin_risk_map, dtype=np.float32)
            self.risk_grad_y, self.risk_grad_x = np.gradient(self.risk_map)
            self.num_bins_y, self.num_bins_x = self.risk_map.shape
        else:
            self.risk_map = None
            self.risk_grad_x = None
            self.risk_grad_y = None
            approx = max(int(np.sqrt(max(self.num_movable_nodes, 1))), 1)
            self.num_bins_x = approx
            self.num_bins_y = approx

        self.density_bin_target = (
            self.num_movable_nodes / (self.num_bins_x * self.num_bins_y + 1e-9)
        )

    def _fixed_locations_np(self, pos):
        pos_t = pos.view([2, -1]).t().cpu().numpy()
        if self.num_fixed_nodes <= 0:
            return np.zeros((0, pos_t.shape[1]), dtype=pos_t.dtype)
        return pos_t[
            self.num_movable_nodes : self.num_movable_nodes + self.num_fixed_nodes
        ]

    def _random_initial(self, fixed_locations):
        risk_map = self.risk_map if self.risk_map is not None else None
        return util.generate_initial_locations(
            fixed_locations,
            self.num_movable_nodes,
            self.xl,
            self.yl,
            self.xh,
            self.yh,
            self.scale,
            risk_map=risk_map,
        )

    ## ----------------------- adaptive weighting helpers -----------------------

    def _risk_adjusted_location(
        self, base_location: torch.Tensor, device
    ) -> Optional[torch.Tensor]:
        if (
            not self.adapt_flag
            or self.risk_map is None
            or self.adapt_risk_step <= 0
            or self.num_movable_nodes <= 0
        ):
            return None

        movable = base_location[: self.num_movable_nodes].detach().cpu().numpy()
        xs = (movable[:, 0] - self.xl) / (self.xh - self.xl + 1e-9)
        ys = (movable[:, 1] - self.yl) / (self.yh - self.yl + 1e-9)
        xs = np.clip(xs, 0.0, 0.999999)
        ys = np.clip(ys, 0.0, 0.999999)
        ix = np.clip((xs * self.num_bins_x).astype(int), 0, self.num_bins_x - 1)
        iy = np.clip((ys * self.num_bins_y).astype(int), 0, self.num_bins_y - 1)

        grad_x = self.risk_grad_x[iy, ix]
        grad_y = self.risk_grad_y[iy, ix]
        grad = np.stack([grad_x, grad_y], axis=1)
        norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = grad / (norm + 1e-12)

        adjusted = movable - self.adapt_risk_step * grad
        adjusted[:, 0] = np.clip(adjusted[:, 0], self.xl, self.xh)
        adjusted[:, 1] = np.clip(adjusted[:, 1], self.yl, self.yh)

        loc = base_location.clone()
        loc[: self.num_movable_nodes] = torch.from_numpy(adjusted).to(
            device=device, dtype=base_location.dtype
        )
        return loc

    def _generate_weight_candidates(
        self, num_bases: int, default_weights: np.ndarray
    ) -> List[np.ndarray]:
        candidates: List[np.ndarray] = []

        # adaptively shrink samples for large designs
        effective_samples = int(self.adapt_samples)
        if self.num_nodes > 200000:
            effective_samples = max(4, min(self.adapt_samples, 12))
        elif self.num_nodes > 100000:
            effective_samples = max(6, min(self.adapt_samples, 24))

        weights = np.array(default_weights, dtype=np.float32)
        if weights.sum() <= 1e-8:
            weights = np.ones(num_bases, dtype=np.float32)
        weights = weights / weights.sum()
        candidates.append(weights)

        for idx in range(num_bases):
            unit = np.zeros(num_bases, dtype=np.float32)
            unit[idx] = 1.0
            candidates.append(unit)

        for _ in range(effective_samples):
            w = self.adapt_rng.dirichlet(np.ones(num_bases, dtype=np.float32))
            candidates.append(w.astype(np.float32))

        return candidates

    def _compute_hpwl(self, pos_np: np.ndarray) -> float:
        if self.num_nets <= 0:
            return 0.0

        num_nets = self.num_nets
        starts = self.netpin_start_np[:-1]
        ends = self.netpin_start_np[1:]

        if num_nets <= self.hpwl_sample_limit:
            net_indices = range(num_nets)
            scale = 1.0
        else:
            net_indices = self.adapt_rng.choice(
                num_nets, self.hpwl_sample_limit, replace=False
            )
            scale = float(num_nets) / float(self.hpwl_sample_limit)

        total = 0.0
        xs = pos_np[:, 0]
        ys = pos_np[:, 1]
        for net in net_indices:
            bgn = starts[net]
            end = ends[net]
            if end - bgn <= 1:
                continue
            pins = self.flat_netpin_np[bgn:end]
            nodes = self.pin2node_np[pins]
            x = xs[nodes]
            y = ys[nodes]
            total += (x.max() - x.min()) + (y.max() - y.min())
        return scale * total / (num_nets + 1e-9)

    def _compute_density(self, pos_np: np.ndarray) -> float:
        if self.num_movable_nodes <= 0:
            return 0.0

        xs_all = (pos_np[: self.num_movable_nodes, 0] - self.xl) / (
            self.xh - self.xl + 1e-9
        )
        ys_all = (pos_np[: self.num_movable_nodes, 1] - self.yl) / (
            self.yh - self.yl + 1e-9
        )

        if self.num_movable_nodes > self.node_sample_limit:
            sample_idx = self.adapt_rng.choice(
                self.num_movable_nodes, self.node_sample_limit, replace=False
            )
            xs = xs_all[sample_idx]
            ys = ys_all[sample_idx]
            scale = float(self.num_movable_nodes) / float(self.node_sample_limit)
        else:
            xs = xs_all
            ys = ys_all
            scale = 1.0

        xs = np.clip(xs, 0.0, 0.999999)
        ys = np.clip(ys, 0.0, 0.999999)
        ix = np.clip((xs * self.num_bins_x).astype(int), 0, self.num_bins_x - 1)
        iy = np.clip((ys * self.num_bins_y).astype(int), 0, self.num_bins_y - 1)

        bin_counts = np.zeros((self.num_bins_y, self.num_bins_x), dtype=np.float64)
        np.add.at(bin_counts, (iy, ix), 1.0)
        overflow = bin_counts - self.density_bin_target
        overflow[overflow < 0] = 0.0
        return scale * float(overflow.sum()) / (self.num_movable_nodes + 1e-9)

    def _compute_risk(self, pos_np: np.ndarray) -> float:
        if self.risk_map is None or self.num_movable_nodes <= 0:
            return 0.0
        xs_all = (pos_np[: self.num_movable_nodes, 0] - self.xl) / (
            self.xh - self.xl + 1e-9
        )
        ys_all = (pos_np[: self.num_movable_nodes, 1] - self.yl) / (
            self.yh - self.yl + 1e-9
        )

        if self.num_movable_nodes > self.node_sample_limit:
            sample_idx = self.adapt_rng.choice(
                self.num_movable_nodes, self.node_sample_limit, replace=False
            )
            xs = xs_all[sample_idx]
            ys = ys_all[sample_idx]
            scale = float(self.num_movable_nodes) / float(self.node_sample_limit)
        else:
            xs = xs_all
            ys = ys_all
            scale = 1.0

        xs = np.clip(xs, 0.0, 0.999999)
        ys = np.clip(ys, 0.0, 0.999999)
        ix = np.clip((xs * self.num_bins_x).astype(int), 0, self.num_bins_x - 1)
        iy = np.clip((ys * self.num_bins_y).astype(int), 0, self.num_bins_y - 1)
        penalty = self.risk_map[iy, ix].sum()
        return scale * penalty / (self.num_movable_nodes + 1e-9)

    def _evaluate_cost(self, location: torch.Tensor) -> float:
        pos_np = location.detach().cpu().numpy()
        hpwl = self._compute_hpwl(pos_np)
        density = self._compute_density(pos_np)
        risk = self._compute_risk(pos_np)
        return (
            self.adapt_lambda_hpwl * hpwl
            + self.adapt_lambda_density * density
            + self.adapt_lambda_risk * risk
        )

    def _enforce_fixed(self, location: torch.Tensor, fixed_tensor: torch.Tensor):
        if self.num_fixed_nodes > 0:
            location[
                self.num_movable_nodes : self.num_movable_nodes + self.num_fixed_nodes
            ] = fixed_tensor
        return location

    def _report_metrics(self, label: str, location: torch.Tensor) -> None:
        try:
            pos_np = location.detach().cpu().numpy()
            hpwl = self._compute_hpwl(pos_np)
            density = self._compute_density(pos_np)
            risk = self._compute_risk(pos_np)
            logger.info(
                "%s metrics: hpwl %.4e, density %.4e, pin-risk %.4e",
                label,
                hpwl,
                density,
                risk,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to report %s metrics: %s", label, exc)

    def _combine_locations(
        self,
        bases: List[torch.Tensor],
        default_weights: List[float],
        fixed_tensor: torch.Tensor,
        label: Optional[str] = None,
    ) -> torch.Tensor:
        if len(bases) == 1:
            combined = self._enforce_fixed(bases[0].clone(), fixed_tensor)
            if label:
                self._report_metrics(label, combined)
            return combined

        default_weights = np.array(default_weights, dtype=np.float32)
        if default_weights.sum() <= 1e-8:
            default_weights = np.ones(len(bases), dtype=np.float32)
        default_weights = default_weights / default_weights.sum()

        if not self.adapt_flag:
            combined = torch.zeros_like(bases[0])
            for w, loc in zip(default_weights, bases):
                combined += float(w) * loc
            combined = self._enforce_fixed(combined, fixed_tensor)
            if label:
                self._report_metrics(label, combined)
            return combined

        candidates = self._generate_weight_candidates(len(bases), default_weights)
        best_cost = float("inf")
        best_loc = None

        for weights in candidates:
            combined = torch.zeros_like(bases[0])
            for w, loc in zip(weights, bases):
                combined += float(w) * loc
            combined = self._enforce_fixed(combined, fixed_tensor)
            cost = self._evaluate_cost(combined)
            if cost < best_cost:
                best_cost = cost
                best_loc = combined

        if best_loc is None:
            best_loc = torch.zeros_like(bases[0])
            for w, loc in zip(default_weights, bases):
                best_loc += float(w) * loc
            best_loc = self._enforce_fixed(best_loc, fixed_tensor)

        if label:
            self._report_metrics(label, best_loc)

        return best_loc


class GiFtInit(_GiFtBase):
    """
    @brief Compute initial position using GiFt technique published at ICCAD 2024,
    augmented with adaptive spectral weighting and pin-risk aware basis.
    """

    def forward(self, pos):
        with torch.no_grad():
            fixed_np = self._fixed_locations_np(pos)
            random_initial = self._random_initial(fixed_np)
            random_initial = np.concatenate((random_initial, fixed_np), 0)
            random_initial = torch.from_numpy(random_initial).float().to(pos.device)
            fixed_tensor = torch.from_numpy(fixed_np).float().to(pos.device)

            start = time.time()
            gsp_filter = mix_frequency_filter.GiFt_GPU(self.adj_mat, pos.device)
            gsp_filter.train(4)
            location_low = gsp_filter.get_cell_position(4, random_initial)

            gsp_filter.train(4)
            location_m = gsp_filter.get_cell_position(2, random_initial)

            gsp_filter.train(2)
            location_h = gsp_filter.get_cell_position(2, random_initial)
            end = time.time()
            logger.info("GiFt passes finished in %g sec", end - start)

            bases: List[torch.Tensor] = [
                location_low,
                location_m,
                location_h,
                random_initial,
            ]
            default_weights = [0.2, 0.7, 0.1, 0.0]

            risk_basis = self._risk_adjusted_location(location_m, pos.device)
            if risk_basis is not None:
                bases.append(risk_basis)
                default_weights.append(0.0)

            location = self._combine_locations(
                bases, default_weights, fixed_tensor, label="GiFt init"
            )
            return location.t()


class GiFtPlusInit(_GiFtBase):
    """
    Enhanced GiFt initializer that incorporates graph sparsification and
    Dirichlet boundary handling inspired by GiFtPlus, with adaptive spectral weighting.
    """

    def __init__(
        self,
        flat_netpin,
        netpin_start,
        pin2node_map,
        net_weights,
        net_mask,
        xl,
        yl,
        xh,
        yh,
        num_nodes,
        num_movable_nodes,
        scale=0.7,
        pin_risk_map=None,
        sparsify_k=16,
        dirichlet_iters=3,
        refine_iters=2,
        **adapt_kwargs,
    ):
        super(GiFtPlusInit, self).__init__(
            flat_netpin,
            netpin_start,
            pin2node_map,
            net_weights,
            net_mask,
            xl,
            yl,
            xh,
            yh,
            num_nodes,
            num_movable_nodes,
            scale=scale,
            pin_risk_map=pin_risk_map,
            **adapt_kwargs,
        )
        self.sparsify_k = sparsify_k
        self.dirichlet_iters = max(dirichlet_iters, 0)
        self.refine_iters = max(refine_iters, 0)

        fixed_mask = np.zeros(self.num_nodes, dtype=np.bool_)
        if self.num_fixed_nodes > 0:
            fixed_mask[
                self.num_movable_nodes : self.num_movable_nodes + self.num_fixed_nodes
            ] = True
        self.fixed_mask = fixed_mask

    def forward(self, pos):
        with torch.no_grad():
            fixed_np = self._fixed_locations_np(pos)
            random_initial = self._random_initial(fixed_np)
            random_initial = np.concatenate((random_initial, fixed_np), 0)
            random_initial = torch.from_numpy(random_initial).float().to(pos.device)
            fixed_tensor = torch.from_numpy(fixed_np).float().to(pos.device)

            start = time.time()
            gsp_filter = mix_frequency_filter.GiFtPlus_GPU(
                self.adj_mat,
                pos.device,
                sparsify_k=self.sparsify_k,
                fixed_mask=self.fixed_mask,
            )

            gsp_filter.train(4)
            location_low = gsp_filter.get_cell_position_with_boundaries(
                4, random_initial, fixed_tensor, self.dirichlet_iters
            )

            gsp_filter.train(4)
            location_m = gsp_filter.get_cell_position_with_boundaries(
                2, random_initial, fixed_tensor, self.dirichlet_iters
            )

            gsp_filter.train(2)
            location_h = gsp_filter.get_cell_position_with_boundaries(
                2, random_initial, fixed_tensor, self.dirichlet_iters
            )

            bases: List[torch.Tensor] = [
                location_low,
                location_m,
                location_h,
                random_initial,
            ]
            default_weights = [0.2, 0.7, 0.1, 0.0]

            risk_basis = self._risk_adjusted_location(location_m, pos.device)
            if risk_basis is not None:
                bases.append(risk_basis)
                default_weights.append(0.0)

            location = self._combine_locations(
                bases, default_weights, fixed_tensor, label="GiFtPlus init"
            )

            if self.refine_iters > 0:
                location = gsp_filter.refine_with_boundaries(
                    location, fixed_tensor, self.refine_iters
                )
                self._report_metrics("GiFtPlus init refined", location)

            end = time.time()
            logger.info("GiFtPlus passes finished in %g sec", end - start)
            return location.t()
