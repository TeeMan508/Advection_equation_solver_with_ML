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
from ML import *

tf.enable_eager_execution()
plt.rcParams['font.size'] = 14  # размер шрифта

grid_length = 32
fine_grid_resolution = 256
coarse_grid_resolution = 32
assert fine_grid_resolution % coarse_grid_resolution == 0

height_list = np.arange(0.1, 1.1, 0.1)
width_list = np.arange(1/16, 1/4, 1/16)

begin_time_step = 0
end_time_step = 256

if __name__ == "__main__":
    # test = reference on fine grid
    # test_2d = 2 orders of accuracy on coarse grid
    fine_grid = generate_grid(fine_grid_resolution, grid_length)
    coarse_grid = generate_grid(coarse_grid_resolution, grid_length)
    test_cond = generate_initial_conditions(fine_grid, height_list, width_list) # c_init_fine
    # for k, v in test_cond.items():
    #     print(k, v.shape)
    test_cond_2d = generate_initial_conditions(coarse_grid, height_list, width_list) # c_init

    my_solv_nn= Solver(fine_grid, test_cond, advection_equations)
    my_model_nn = my_solv_nn.generate_model(2)

    my_solv = Solver(coarse_grid, test_cond_2d, advection_equations)
    my_model = my_solv.generate_model(2)

    time_steps = np.arange(begin_time_step, end_time_step + 1)

    res = my_solv.integrate_until(my_model, test_cond_2d, time_steps)
    # res2 = my_solv.integrate_gap(my_model, res, begin_time_step, end_time_step)
    # for k, v in res2.items():
    #     print(k, v.shape)

    x_f, _ = fine_grid.get_mesh()
    x_c, _ = coarse_grid.get_mesh()

    dr = wrap_as_xarray(res, time_steps, x_c)

    # plot initial
    # dr.isel(time=[0, 10, 128], sample=[4, 10, 16]).plot(col='sample', hue='time')
    # plt.show()

    nn = ML(test_cond)
    model_nn = nn.generate_model(coarse_grid)
    integrated_ref = nn.reference_solution(fine_grid, coarse_grid, end_time_step)
    # for k, v in integrated_ref.items():
    #     print(k, v.shape)
    train_input, train_output = nn.make_train_data(integrated_ref, end_time_step)

    # # plot train data
    # i_sample = 48  # any number between 0 and train_output.shape[0]  любое число от 0 до train_output.shape[0]
    # plt.plot(train_input['concentration'][i_sample].numpy(), label='init')
    # for shift in range(train_output.shape[1])[:3]:
    #     plt.plot(train_output[i_sample, shift].numpy(), label=f'shift={shift+1}')
    #
    # plt.title(f'no. {i_sample} sample')
    # plt.legend()
    # plt.show()

    # nn.fit(model_nn, train_input, train_output)

    model_utils.load_weights(model_nn, "weights_1d_10epochs.h5")

    integrated_nn = nn.integrate_until(model_nn, test_cond_2d, time_steps)

    dr_nn = wrap_as_xarray(integrated_nn, time_steps, x_c)
    dr_nn.isel(time=[0, 10, 100, 120, 128], sample=[4, 10, 16]).plot(col='sample', hue='time')
    plt.show()

    # Evaluate accuracy on training set

    dr_ref = wrap_as_xarray(integrated_ref, time_steps, x_c)  # reference "truth"
    dr_all_train = xarray.concat([dr_nn, dr, dr_ref], dim='model')
    dr_all_train.coords['model'] = ['nn', '2nd', 'ref']

    (dr_all_train.isel(time=[0, 16, 64, 128, 256], sample=[4, 10, 16])
     .plot(hue='model', col='time', row='sample', alpha=0.6, linewidth=2))
    plt.show()

    (
        (dr_all_train.sel(model=['nn',  '2nd']) - dr_all_train.sel(model='ref'))
            .pipe(abs).mean(dim=['x', 'sample'])
            .isel(time=slice(0, 129, 2)).plot(hue='model')
    )
    plt.title('Error on training set')
    plt.grid()
    plt.show()




    # Prediction on new test data

    np.random.seed(41)
    height_list_test = np.random.uniform(0.1, 0.9, size=10)
    c_init_test = generate_initial_conditions(coarse_grid, height_list=height_list_test,
                                              width_list=width_list)
    print("c_init_test")
    for k, v in c_init_test.items():
        print(k, v.shape)

    print("c_init_test.shape = ", c_init_test.shape)  # (sample, x, y)

    print("height_list_test = ", height_list_test)

    plt.plot(x_c, c_init_test[5])

    # initial_state_test = {
    #     'concentration': c_init_test.astype(np.float32),  # tensorflow code expects float32
    #     'x_velocity': np.ones(c_init_test.shape, np.float32) * 1.0,
    #     'y_velocity': np.zeros(c_init_test.shape, np.float32)
    # }

    dr_nn_test = wrap_as_xarray(nn.integrate_until(model_nn, c_init_test, time_steps))

    dr_2nd_test = wrap_as_xarray(nn.integrate_until(my_model, c_init_test, time_steps))

    dr_sol_test = xarray.concat([dr_nn_test, dr_2nd_test], dim='model')
    dr_sol_test.coords['model'] = ['Neural net', 'Second order']

    (dr_sol_test.isel(time=[0, 16, 64, 128, 256], sample=[4, 10, 16])
     .plot(hue='model', col='time', row='sample', alpha=0.6, linewidth=2)
     )
#
    plt.ylim(0, 1)

    (dr_sol_test.isel(time=[0, 16, 64, 256], sample=16).rename({'time': 'Time step'})
     .plot(hue='model', col='Time step', alpha=0.6, col_wrap=2, linewidth=2, figsize=[6, 4.5], ylim=[None, 0.8])
     )

    plt.suptitle('Advection under 1-D constant velocity', y=1.05)
    plt.savefig('1d-test-sample.png', dpi=288, bbox_inches='tight')

    print("Вы молодцы!")
