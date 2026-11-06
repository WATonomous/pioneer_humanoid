#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import cupy as cp
import string
from typing import List

from .plugin_manager import PluginBase


class MinFilter(PluginBase):
    """This is a filter to fill in invalid cells with minimum values around.

    Args:
        cell_n (int): width of the elevation map.
        dilation_size (int): The size of the patch to search for minimum value for each iteration.
        iteration_n (int): The number of iteration to repeat the same filter.
        **kwargs ():
    """

    def __init__(self, cell_n: int = 100, dilation_size: int = 5, iteration_n: int = 5, **kwargs):
        super().__init__()
        self.iteration_n = iteration_n
        self.width = cell_n
        self.height = cell_n
        self.min_filtered = cp.zeros((self.width, self.height), dtype=cp.float32)
        self.min_filtered_mask = cp.zeros((self.width, self.height), dtype=cp.float32)
        self.next_min_filtered = cp.zeros((self.width, self.height), dtype=cp.float32)
        self.next_min_filtered_mask = cp.zeros((self.width, self.height), dtype=cp.float32)
        self.min_filter_kernel = cp.ElementwiseKernel(
            in_params="raw U map, raw U mask",
            out_params="raw U newmap, raw U newmask",
            preamble=string.Template(
                """
                __device__ int get_map_idx(int idx, int layer_n) {
                    const int layer = ${width} * ${height};
                    return layer * layer_n + idx;
                }

                __device__ bool is_inside(int row, int col) {
                    return row > 0 && row < ${height} - 1 && col > 0 && col < ${width} - 1;
                }
                """
            ).substitute(width=self.width, height=self.height),
            operation=string.Template(
                """
                U h = map[get_map_idx(i, 0)];
                U valid = mask[get_map_idx(i, 0)];
                newmap[get_map_idx(i, 0)] = h;
                newmask[get_map_idx(i, 0)] = valid;
                if (valid < 0.5) {
                    int row = i / ${width};
                    int col = i % ${width};
                    U min_value = 1000000.0;
                    for (int dy = -${dilation_size}; dy <= ${dilation_size}; dy++) {
                        for (int dx = -${dilation_size}; dx <= ${dilation_size}; dx++) {
                            int candidate_row = row + dy;
                            int candidate_col = col + dx;
                            if (!is_inside(candidate_row, candidate_col)) {continue;}
                            int idx = candidate_row * ${width} + candidate_col;
                            U candidate_valid = mask[idx];
                            U value = map[idx];
                            if(candidate_valid > 0.5 && value < min_value) {
                                min_value = value;
                            }
                        }
                    }
                    if (min_value < 1000000 - 1) {
                        newmap[get_map_idx(i, 0)] = min_value;
                        newmask[get_map_idx(i, 0)] = 0.6;
                    }
                }
                """
            ).substitute(dilation_size=dilation_size, width=self.width),
            name=f"min_filter_{self.width}_{self.height}_{dilation_size}",
        )

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        source_map = elevation_map[0]
        source_mask = elevation_map[2]
        output_map = self.min_filtered
        output_mask = self.min_filtered_mask
        for _ in range(self.iteration_n):
            self.min_filter_kernel(
                source_map,
                source_mask,
                output_map,
                output_mask,
                size=(self.width * self.height),
            )
            source_map, output_map = output_map, self.next_min_filtered if output_map is self.min_filtered else self.min_filtered
            source_mask, output_mask = (
                output_mask,
                self.next_min_filtered_mask if output_mask is self.min_filtered_mask else self.min_filtered_mask,
            )
        min_filtered = cp.where(source_mask > 0.5, source_map, cp.nan)
        return min_filtered
