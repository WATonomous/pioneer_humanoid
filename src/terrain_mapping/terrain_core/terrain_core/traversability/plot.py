"""Draw terrain layers using an off-screen Matplotlib figure."""
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure


def plot_layers(elevation, layers, caption=None):
    figure = Figure(figsize=(13, 8), constrained_layout=True)
    if caption:
        figure.suptitle(caption)
    axes = figure.subplots(2, 3).ravel()
    for ax, (title, values) in zip(axes, [
        ("Elevation (m)", elevation),
        ("Slope (degrees)", np.rad2deg(layers["slope"])),
        ("Step height (m)", layers["step"]),
        ("Roughness (m)", layers["roughness"]),
        ("Traversability score", layers["traversability"]),
    ]):
        image = ax.imshow(values, origin="lower", interpolation="nearest")
        ax.set_title(title)
        figure.colorbar(image, ax=ax, shrink=0.8)

    known = np.isfinite(layers["traversability"])
    decisions = np.where(known, layers["traversable"], np.nan)
    cmap = ListedColormap(["#c44e52", "#36875c"]).with_extremes(bad="#bdbdbd")
    image = axes[-1].imshow(decisions, origin="lower", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=1.5)
    bar = figure.colorbar(image, ax=axes[-1], shrink=0.8, ticks=[0, 1])
    bar.set_ticklabels(["Blocked", "Traversable"])
    axes[-1].set_title("TR-8 decision (gray = unknown)")
    return figure
