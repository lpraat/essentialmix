import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid


def image_grid(images: Sequence[np.ndarray], n_cols=5, figsize=(10, 10), axes_pad=0.1) -> plt.Figure:
    n_cols = min(len(images), n_cols)
    n = len(images)
    n_rows = math.ceil(n / 5)
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(n_rows, n_cols),
        axes_pad=axes_pad,
    )

    for ax, im in zip(grid, images):
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
    return fig