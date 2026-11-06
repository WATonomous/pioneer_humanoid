import cupy as cp
import cupyx.scipy.ndimage as ndimage
from typing import List

from .plugin_manager import PluginBase


class PositiveSpikeFilter(PluginBase):
    """Remove isolated positive height spikes while keeping extended structures.

    The filter uses two cues:
    1. finite-only local background prominence for isolated tower components
    2. exposed-tip prominence over the highest finite neighbor near NaN frontiers

    This keeps trench drops and supported walls while allowing spikes to be
    rejected even when several neighbors are invalid.
    """

    def __init__(
        self,
        cell_n: int = 100,
        input_layer_name: str = "elevation",
        median_filter_size: int = 5,
        spike_height_diff_m: float = 0.35,
        support_height_tolerance_m: float = 0.12,
        max_support_neighbor_count: int = 1,
        max_component_cells: int = 2,
        edge_invalid_neighbor_count_min: int = 9,
        edge_peak_diff_m: float = 0.0,
        edge_similarity_tolerance_m: float = 0.05,
        edge_max_similar_neighbor_count: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.median_filter_size = max(3, int(median_filter_size))
        if self.median_filter_size % 2 == 0:
            self.median_filter_size += 1
        self.spike_height_diff_m = float(spike_height_diff_m)
        self.support_height_tolerance_m = float(support_height_tolerance_m)
        self.max_support_neighbor_count = int(max_support_neighbor_count)
        self.max_component_cells = int(max_component_cells)
        self.edge_invalid_neighbor_count_min = int(edge_invalid_neighbor_count_min)
        self.edge_peak_diff_m = float(edge_peak_diff_m)
        self.edge_similarity_tolerance_m = float(edge_similarity_tolerance_m)
        self.edge_max_similar_neighbor_count = int(edge_max_similar_neighbor_count)

    def _get_input_layer(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
    ) -> cp.ndarray:
        if self.input_layer_name in layer_names:
            layer = elevation_map[layer_names.index(self.input_layer_name)].copy()
            if self.input_layer_name == "elevation":
                valid_mask = elevation_map[2] > 0.5
                layer = cp.where(valid_mask & cp.isfinite(layer), layer, cp.nan)
            return layer
        if self.input_layer_name in plugin_layer_names:
            return plugin_layers[plugin_layer_names.index(self.input_layer_name)].copy()
        valid_mask = elevation_map[2] > 0.5
        layer = elevation_map[0].copy()
        return cp.where(valid_mask & cp.isfinite(layer), layer, cp.nan)

    def _count_supporting_neighbors(
        self,
        height_map: cp.ndarray,
        finite_mask: cp.ndarray,
        candidate_mask: cp.ndarray,
    ) -> cp.ndarray:
        support = cp.zeros(height_map.shape, dtype=cp.int32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = cp.roll(height_map, shift=(dy, dx), axis=(0, 1))
                shifted_finite = cp.roll(finite_mask, shift=(dy, dx), axis=(0, 1))
                shifted_candidate = cp.roll(candidate_mask, shift=(dy, dx), axis=(0, 1))
                if dy > 0:
                    shifted_finite[:dy, :] = False
                    shifted_candidate[:dy, :] = False
                elif dy < 0:
                    shifted_finite[dy:, :] = False
                    shifted_candidate[dy:, :] = False
                if dx > 0:
                    shifted_finite[:, :dx] = False
                    shifted_candidate[:, :dx] = False
                elif dx < 0:
                    shifted_finite[:, dx:] = False
                    shifted_candidate[:, dx:] = False
                support += (
                    shifted_finite
                    & (~shifted_candidate)
                    & (cp.abs(shifted - height_map) <= self.support_height_tolerance_m)
                )
        return support

    def _get_local_background(
        self,
        height_map: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        pad = self.median_filter_size // 2
        padded = cp.pad(height_map, pad_width=pad, mode="constant", constant_values=cp.nan)
        window_views = []
        for dy in range(self.median_filter_size):
            for dx in range(self.median_filter_size):
                if dy == pad and dx == pad:
                    continue
                window_views.append(
                    padded[dy : dy + height_map.shape[0], dx : dx + height_map.shape[1]]
                )
        window_stack = cp.stack(window_views, axis=0)
        finite_neighbor_count = cp.sum(cp.isfinite(window_stack), axis=0, dtype=cp.int32)
        local_background = cp.nanmedian(window_stack, axis=0)
        # If a cell has no finite neighbors in the full window, leave the background
        # equal to the cell itself so it won't become a false candidate.
        local_background = cp.where(cp.isfinite(local_background), local_background, height_map)
        return local_background, finite_neighbor_count

    def _get_edge_peak_reject_mask(
        self,
        height_map: cp.ndarray,
        finite_mask: cp.ndarray,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        reject = cp.zeros(height_map.shape, dtype=cp.bool_)
        replacement = cp.full(height_map.shape, cp.nan, dtype=height_map.dtype)
        if self.edge_invalid_neighbor_count_min > 8 or self.edge_peak_diff_m <= 0.0:
            return reject, replacement

        finite_neighbor_count = cp.zeros(height_map.shape, dtype=cp.int32)
        similar_neighbor_count = cp.zeros(height_map.shape, dtype=cp.int32)
        max_finite_neighbor_height = cp.full(height_map.shape, -cp.inf, dtype=height_map.dtype)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = cp.roll(height_map, shift=(dy, dx), axis=(0, 1))
                shifted_finite = cp.roll(finite_mask, shift=(dy, dx), axis=(0, 1))
                if dy > 0:
                    shifted_finite[:dy, :] = False
                elif dy < 0:
                    shifted_finite[dy:, :] = False
                if dx > 0:
                    shifted_finite[:, :dx] = False
                elif dx < 0:
                    shifted_finite[:, dx:] = False
                finite_neighbor_count += shifted_finite
                similar_neighbor_count += shifted_finite & (
                    cp.abs(shifted - height_map) <= self.edge_similarity_tolerance_m
                )
                max_finite_neighbor_height = cp.where(
                    shifted_finite,
                    cp.maximum(max_finite_neighbor_height, shifted),
                    max_finite_neighbor_height,
                )

        invalid_neighbor_count = 8 - finite_neighbor_count
        has_finite_neighbor = finite_neighbor_count > 0
        reject = (
            finite_mask
            & has_finite_neighbor
            & (invalid_neighbor_count >= self.edge_invalid_neighbor_count_min)
            & (similar_neighbor_count <= self.edge_max_similar_neighbor_count)
            & ((height_map - max_finite_neighbor_height) > self.edge_peak_diff_m)
        )
        replacement = cp.where(has_finite_neighbor, max_finite_neighbor_height, replacement)
        return reject, replacement

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        height_map = self._get_input_layer(elevation_map, layer_names, plugin_layers, plugin_layer_names)
        finite_mask = cp.isfinite(height_map)
        if not cp.any(finite_mask):
            return height_map

        local_background, _ = self._get_local_background(height_map)
        candidate = finite_mask & ((height_map - local_background) > self.spike_height_diff_m)
        reject = cp.zeros(height_map.shape, dtype=cp.bool_)
        if cp.any(candidate):
            labels, _ = ndimage.label(candidate, structure=ndimage.generate_binary_structure(2, 2))
            component_sizes = cp.bincount(labels.ravel())
            small_component = labels > 0
            small_component &= component_sizes[labels] <= self.max_component_cells

            support = self._count_supporting_neighbors(height_map, finite_mask, candidate)
            reject = small_component & (support <= self.max_support_neighbor_count)

        edge_peak_reject, edge_peak_replacement = self._get_edge_peak_reject_mask(
            height_map,
            finite_mask,
        )
        if not cp.any(reject) and not cp.any(edge_peak_reject):
            return height_map
        filtered = cp.where(reject, local_background, height_map)
        filtered = cp.where(edge_peak_reject, edge_peak_replacement, filtered)
        return cp.where(finite_mask, filtered, height_map)
