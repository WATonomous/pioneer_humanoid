import cupy as cp

from elevation_mapping_cupy.plugins.plugin_manager import PluginManager, PluginParams


class CountingPlugin:
    def __init__(self):
        self.calls = 0

    def __call__(self, elevation_map, layer_names, plugin_layers, plugin_layer_names, *args):
        self.calls += 1
        return cp.full(elevation_map.shape[1:], self.calls, dtype=cp.float32)


def test_plugin_layer_is_computed_once_per_map_generation():
    manager = PluginManager(cell_n=4)
    plugin = CountingPlugin()
    manager.plugin_params = [PluginParams(name="counting", layer_name="counted")]
    manager.plugins = [plugin]
    manager.layers = cp.full((1, 4, 4), cp.nan, dtype=cp.float32)
    manager.layer_names = ["counted"]
    manager.plugin_names = ["counting"]
    manager._generation = 0
    manager._layer_generations = [-1]
    manager._empty_semantic_map = cp.zeros((0, 4, 4), dtype=cp.float32)
    elevation_map = cp.zeros((7, 4, 4), dtype=cp.float32)

    manager.update_with_name("counted", elevation_map, ["elevation"])
    manager.update_with_name("counted", elevation_map, ["elevation"])
    assert plugin.calls == 1

    manager.reset_layers()
    manager.update_with_name("counted", elevation_map, ["elevation"])
    assert plugin.calls == 2
