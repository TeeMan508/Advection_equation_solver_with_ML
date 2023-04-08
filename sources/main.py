# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import xarray
# from datadrivenpdes.core import grids
# from datadrivenpdes.core import integrate
# from datadrivenpdes.core import models
# from datadrivenpdes.core import tensor_ops
# from datadrivenpdes.advection import equations as advection_equations
# from datadrivenpdes.pipelines import model_utils
from solver import Solver
from conditions import *

tf.enable_eager_execution()
plt.rcParams['font.size'] = 14

grid_length = 32
fine_grid_resolution = 256
coarse_grid_resolution = 32
assert fine_grid_resolution % coarse_grid_resolution == 0

height_list = np.arange(0.1, 1.1, 0.1)
width_list = np.arange(1/16, 1/4, 1/16)

if __name__ == "__main__":
    test_grid = generate_grid(coarse_grid_resolution, grid_length)
    test_cond = generate_initial_conditions(test_grid, height_list, width_list)
    my_solv = Solver(test_grid, test_cond, advection_equations)
    my_model = my_solv.generate_model(2)
    res = my_solv.integrate_until(my_model, 256)
    dr = xarray.DataArray(
        res['concentration'].numpy().squeeze(),
        dims=('time', 'sample', 'x'),
    )
    dr.isel(time=[0, 10, 128], sample=[4, 10, 16]).plot(col='sample', hue='time')
    plt.show()