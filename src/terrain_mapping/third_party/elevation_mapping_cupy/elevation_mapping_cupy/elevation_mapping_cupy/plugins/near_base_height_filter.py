import cupy as cp
from typing import List

from .plugin_manager import PluginBase


class NearBaseHeightFilter(PluginBase):
    """Invalidate close-range cells that sit above a fixed base-frame height threshold.

    This is intended for BASE-centered digging maps, where spurious close lidar returns
    from dust, rain, or snow can float above the real ground and then persist because
    later rays no longer see through them. Cells that violate the near-base gate are
    marked invalid (NaN) so downstream despiking/inpainting can treat them as holes.
    """

    def __init__(
        self,
        cell_n: int = 100,
        input_layer_name: str = "elevation",
        resolution: float = 0.1,
        radius_m: float = 3.5,
        max_allowed_height_m: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.input_layer_name = input_layer_name
        self.radius_m = float(radius_m)
        self.max_allowed_height_m = float(max_allowed_height_m)

        center = (float(cell_n) - 1.0) / 2.0
        coords = (cp.arange(cell_n, dtype=cp.float32) - center) * float(resolution)
        yy, xx = cp.meshgrid(coords, coords, indexing="ij")
        self.radial_mask = (xx * xx + yy * yy) < (self.radius_m * self.radius_m)

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
        return elevation_map[0].copy()

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        height_map = self._get_input_layer(elevation_map, layer_names, plugin_layers, plugin_layer_names)
        reject_mask = cp.isfinite(height_map) & self.radial_mask & (height_map > self.max_allowed_height_m)
        return cp.where(reject_mask, cp.nan, height_map)
