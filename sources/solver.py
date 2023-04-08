import os
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


class Solver:
    def __init__(self, grid: grids.Grid, initial_state: dict, equation_system: advection_equations):
        self.grid = grid
        self.initial_state = initial_state
        self.equation_system = equation_system

    def generate_model(self, order: int):
        assert 3 > order > 0
        if order == 1:
            model = models.FiniteDifferenceModel(
                self.equation_system.UpwindAdvection(cfl_safety_factor=0.5),
                self.grid
            )
        else:
            model = models.FiniteDifferenceModel(
                self.equation_system.VanLeerAdvection(cfl_safety_factor=0.5),
                self.grid
            )
        return model

    def integrate_until(self, model, final_time_step: int):
        time_steps = np.arange(0, final_time_step + 1)
        solution = integrate.integrate_steps(model, self.initial_state, time_steps)
        return solution