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

    # def integrate_until(self, model, initial_cond: dict, final_time_step: int):
    #     time_steps = np.arange(0, final_time_step + 1)
    #     solution = integrate.integrate_steps(model, initial_cond, time_steps)
    #     # key_defs = solution.key_definitions
    #     return solution

    def integrate_until(self, model, initial_cond: dict, time_steps: int):
        # time_steps = np.arange(0, final_time_step + 1)
        solution = integrate.integrate_steps(model, initial_cond, time_steps)
        # key_defs = solution.key_definitions
        return solution

    # def integrate_until_we_all_die(self, model, initial_cond: dict, coarse_ratio: int, fine_time_steps: int):
    #     time_steps = np.arange(0, fine_time_steps * coarse_ratio + 1, coarse_ratio)
    #     solution = integrate.integrate_steps(model, initial_cond, time_steps)
    #     # key_defs = solution.key_definitions
    #     return solution

    def integrate_gap(self, model, begin_cond: dict, begin_time_step:int, end_time_step: int):
        new_cond = {'concentration': begin_cond['concentration'][begin_time_step],
                    'x_velocity': begin_cond['x_velocity'][begin_time_step],
                    'y_velocity': begin_cond['y_velocity'][begin_time_step]}
        time_steps = np.arange(begin_time_step, end_time_step + 1)
        solution = integrate.integrate_steps(model, new_cond, time_steps)
        return solution

    # def solve_gap(self, model, initial_cond: dict, final_time_step: int):
    #     sub_result = self.integrate_until(model, initial_cond, final_time_step)
    #     result = self.integrate_gap(model, initial_cond, final_time_step)
    #