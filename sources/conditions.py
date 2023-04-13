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



def generate_initial_conditions(grid: grids.Grid, height_list: np.array, width_list: np.array, geometry_type="gaussian") -> dict:

    """
    choose geometry_type square or gaussian - quasi-random square
    """

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

    def make_triangle(x, height=1.0, center=0.25, width=0.1):
        nx = x.shape[0]
        c = np.zeros_like(x)
        c[int((center - width) * nx):int((center + width) * nx)] = height
        return c

    def make_multi_triangle(x, height_list, width_list):
        c_list = []
        for height in height_list:
            for width in width_list:
                c_temp = make_square(x, height=height, width=width)
                c_list.append(c_temp)

        return np.array(c_list)

    def make_gaussian(x, height=1.0, center=0.25, width=0.1):
        """
        Args:
        x: Numpy array. Shape should be (nx, 1) or (nx,)
        height: float, peak concentration
        center: float, relative center position in 0~1
        width: float, relative width in 0~0.5

        Returns:
        Numpy array, same shape as `x`
        """
        nx = x.shape[0]
        x_max = x.max()
        center *= x_max
        width *= x_max
        c = height * np.exp(-(x - center) ** 2 / width ** 2)
        return c

    def make_multi_gaussian(x, height_list, width_list):

        c_list = []
        for height in height_list:
            for width in width_list:
                c_temp = make_gaussian(x, height=height, width=width)
                c_list.append(c_temp)

        return np.array(c_list)

    if geometry_type == "square":
        x, _ = grid.get_mesh()
        c_init = make_multi_square(x, height_list=height_list, width_list=width_list)
        initial_state = {
            'concentration': c_init.astype(np.float32),  # tensorflow code expects float32
            'x_velocity': np.ones(c_init.shape, np.float32) * 1.0,
            'y_velocity': np.zeros(c_init.shape, np.float32)
        }

    elif geometry_type == "gaussian":
        np.random.seed(41)
        height_list_guass = np.random.uniform(0.1, 0.5, size=10)
        width_list_guass = np.random.uniform(1 / 16, 1 / 4, size=3)

        c_init_guass = make_multi_gaussian(
            x_coarse,
            height_list=height_list_guass,
            width_list=width_list_guass
        )

        c_init_guass.shape  # (sample, x, y)
        initial_state = {
            'concentration': c_init_guass.astype(np.float32),  # tensorflow code expects float32
            'x_velocity': np.ones(c_init_guass.shape, np.float32) * 1.0,
            'y_velocity': np.zeros(c_init_guass.shape, np.float32)
        }
    elif geometry_type == "triangle":
        x, _ = grid.get_mesh()
        c_init = make_multi_triangle(x, height_list=height_list, width_list=width_list)
        initial_state = {
            'concentration': c_init.astype(np.float32),  # tensorflow code expects float32
            'x_velocity': np.ones(c_init.shape, np.float32) * 1.0,
            'y_velocity': np.zeros(c_init.shape, np.float32)
        }

    return initial_state


def wrap_as_xarray(integrated, time_steps, x):
    dr = xarray.DataArray(
        integrated['concentration'].numpy().squeeze(),
        dims=('time', 'sample', 'x'),
        coords={'time': time_steps, 'x': x.squeeze()}
    )
    return dr
