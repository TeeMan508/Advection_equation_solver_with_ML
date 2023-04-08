import solver
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray
from datadrivenpdes.core import grids
from datadrivenpdes.core import integrate
from datadrivenpdes.core import models
from datadrivenpdes.core import tensor_ops
from datadrivenpdes.advection import equations as advection_equations
from datadrivenpdes.pipelines import model_utils


def generate_grid(grid_resolution: int, grid_length: int) -> grids.Grid:
    result_grid = grids.Grid(
        size_x=grid_resolution, size_y=1,
        step=grid_length / grid_resolution
    )
    return result_grid


def generate_initial_conditions(grid: grids.Grid, height_list: np.array, width_list: np.array) -> dict:
    def make_square(x, height=1.0, center=0.25, width=0.1):
        nx = x.shape[0]
        c = np.zeros_like(x)
        c[int((center - width) * nx):int((center + width) * nx)] = height
        return c

    def make_multi_square(x, height_list, width_list):
        c_list = []
        for height in height_list:
            for width in width_list:
                c_temp = make_square(x, height=height, width=width)
                c_list.append(c_temp)

        return np.array(c_list)

    x, _ = grid.get_mesh()
    c_init = make_multi_square(x, height_list=height_list, width_list=width_list)
    initial_state = {
        'concentration': c_init.astype(np.float32),  # tensorflow code expects float32
        'x_velocity': np.ones(c_init.shape, np.float32) * 1.0,
        'y_velocity': np.zeros(c_init.shape, np.float32)
    }

    return initial_state
