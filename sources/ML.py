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


def integrate_until(model, initial_cond: dict, time_steps):
    # time_steps = np.arange(0, final_time_step + 1)
    solution = integrate.integrate_steps(model, initial_cond, time_steps)
    # key_defs = solution.key_definitions
    return solution

def reference_solution(fine_grid, coarse_grid,
                       coarse_time_steps, initial_cond):
    equation = advection_equations.VanLeerAdvection(cfl_safety_factor=0.5)
    key_defs = equation.key_definitions  # записали наши концентрации и скорости
    model = models.FiniteDifferenceModel(equation, fine_grid)  # МКО
    coarse_ratio = fine_grid.size_x // coarse_grid.size_x  # целая часть = 256/32
    steps = np.arange(0, coarse_time_steps * coarse_ratio + 1, coarse_ratio)
    integrated_fine = integrate_until(model, initial_cond, steps)
    # integrated_fine = integrate.integrate_steps(model, self.initial_state, steps)

    integrated_coarse = tensor_ops.regrid(integrated_fine, key_defs, fine_grid,
                                          coarse_grid)  # из мелкой в грубую подставили вычисленные значения
    return integrated_coarse

class ML:

    # def __init__(self):

    def generate_model(self, grid: grids.Grid):
        model = models.PseudoLinearModel(
            advection_equations.FiniteVolumeAdvection(0.5),
            grid,
            num_time_steps=4,  # multi-step loss function
            stencil_size=3, kernel_size=(3, 1), num_layers=4, filters=32,
            constrained_accuracy_order=1,
            learned_keys={'concentration_edge_x', 'concentration_edge_y'},  # finite volume view, use edge concentration
            activation='relu',
        )
        return model


def make_train_data(integrated_coarse, fine_time_steps, example_time_steps=4):
    # equation = advection_equations.VanLeerAdvection(cfl_safety_factor=0.5)
    # key_defs = equation.key_definitions
    #
    # integrated_coarse = tensor_ops.regrid(integrated_fine, key_defs, fine_grid, coarse_grid)

    train_input = {k: v[:-example_time_steps] for k, v in integrated_coarse.items()}

    n_time, n_sample, n_x, n_y = train_input['concentration'].shape
    for k in train_input:
        train_input[k] = tf.reshape(train_input[k], [n_sample * n_time, n_x, n_y])

    # print('\n train_input shape:')
    # for k, v in train_input.items():
    #     print(k, v.shape)  # (merged_sample, x, y)

    output_list = []
    for shift in range(1, example_time_steps + 1):
        output_slice = integrated_coarse['concentration'][shift:fine_time_steps - example_time_steps + shift + 1]
        n_time, n_sample, n_x, n_y = output_slice.shape
        output_slice = tf.reshape(output_slice, [n_sample * n_time, n_x, n_y])
        output_list.append(output_slice)

    train_output = tf.stack(output_list, axis=1)

    assert train_output.shape[0] == train_input['concentration'].shape[0]  # merged_sample
    assert train_output.shape[2] == train_input['concentration'].shape[1]  # x
    assert train_output.shape[3] == train_input['concentration'].shape[2]  # y
    assert train_output.shape[1] == example_time_steps

    return train_input, train_output

def fit(model, train_input, train_output):

    model.compile(
        optimizer='adam', loss='mae'
    )

    tf.random.set_random_seed(42)
    np.random.seed(42)

    history = model.fit(
        train_input, train_output, epochs=120, batch_size=32,
        verbose=1, shuffle=True
    )

    model_utils.save_weights(model, f'weights_1d_120epochs.h5')

    # df_history = pd.DataFrame(history.history)
    # df_history.plot(marker='.')
    # plt.suptitle("loss",fontsize=8)
    # plt.show()
    #
    # df_history['loss'][3:].plot(marker='.')
    # plt.show()

    return 1





