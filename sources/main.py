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
    # # test = reference on fine grid
    # # test_2d = 2 orders of accuracy on coarse grid

    fine_grid = generate_grid(fine_grid_resolution, grid_length)
    coarse_grid = generate_grid(coarse_grid_resolution, grid_length)
    test_cond = generate_initial_conditions(fine_grid, height_list, width_list) # c_init_fine
    test_cond_2d = generate_initial_conditions(coarse_grid, height_list, width_list) # c_init

    my_solv_nn= Solver(fine_grid, test_cond, advection_equations)
    my_model_nn = my_solv_nn.generate_model(2)

    my_solv = Solver(coarse_grid, test_cond_2d, advection_equations)
    my_model = my_solv.generate_model(2)

    time_steps = np.arange(begin_time_step, end_time_step + 1)

    res = my_solv.integrate_until(my_model, test_cond_2d, time_steps)

    x_f, _ = fine_grid.get_mesh()
    x_c, _ = coarse_grid.get_mesh()

    dr = wrap_as_xarray(res, time_steps, x_c)

    # # plot initial

    # dr.isel(time=[0, 10, 128], sample=[4, 10, 16]).plot(col='sample', hue='time')
    # plt.show()

    nn = ML()
    model_nn = nn.generate_model(coarse_grid)
    integrated_ref = nn.reference_solution(fine_grid, coarse_grid, end_time_step, test_cond)
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

    # # Включить при первом запуске, далее закомментировать
    nn.fit(model_nn, train_input, train_output)

    model_utils.load_weights(model_nn, "weights_1d_120epochs.h5")

    integrated_nn = nn.integrate_until(model_nn, test_cond_2d, time_steps)

    # # Построение графиков работы модели на train set
    # dr_nn = wrap_as_xarray(integrated_nn, time_steps, x_c)
    # dr_nn.isel(time=[0, 10, 100, 120, 128], sample=[4, 10, 16]).plot(col='sample', hue='time')
    # plt.show()

    # # Evaluate accuracy on training set, построение графика ошибки моделей на train set

    # (
    #     (dr_all_train.sel(model=['nn',  '2nd']) - dr_all_train.sel(model='ref'))
    #         .pipe(abs).mean(dim=['x', 'sample'])
    #         .isel(time=slice(0, 129, 2)).plot(hue='model')
    # )
    # plt.title('Error on training set')
    # plt.grid()
    # plt.show()

    # Prediction on new test data

    np.random.seed(41)
    height_list_test = np.random.uniform(0.1, 0.9, size=10)
    c_init_test = generate_initial_conditions(coarse_grid, height_list=height_list_test,
                                              width_list=width_list)
    dr_nn_test = wrap_as_xarray(nn.integrate_until(model_nn, c_init_test, time_steps), time_steps, x_c)
    dr_2nd_test = wrap_as_xarray(nn.integrate_until(my_model, c_init_test, time_steps), time_steps, x_c)

    # Reference solution for test set

    c_init_fine_test = generate_initial_conditions(fine_grid, height_list=height_list_test, width_list=width_list)

    integrated_ref_test = nn.reference_solution(fine_grid, coarse_grid, end_time_step, c_init_fine_test)

    dr_ref_test = wrap_as_xarray(integrated_ref_test, time_steps, x_c)  # reference "truth"
    dr_all_test = xarray.concat([dr_nn_test, dr_2nd_test, dr_ref_test], dim='model')
    dr_all_test.coords['model'] = ['Neural net', 'Second order',  'Reference']

    (dr_all_test.isel(time=[0, 16, 64, 128, 256], sample=[4, 10, 16])
     .plot(hue='model', col='time', row='sample', alpha=0.6, linewidth=2)
     )
    plt.show()

    (dr_all_test.isel(time=[0, 16, 64, 256], sample=16).rename({'time': 'Time step'})
     .plot(hue='model', col='Time step', alpha=0.6, col_wrap=2, linewidth=2, figsize=[6, 4.5], ylim=[None, 0.8])
     )
    plt.suptitle('Advection under 1-D constant velocity',fontsize=8)
    plt.show()
    plt.savefig('1d-test-sample.png', dpi=288, bbox_inches='tight')

    # Plot test accuracy

    (dr_all_test.sel(model=['Neural net', 'Second order']) - dr_all_test.sel(model='Reference')).pipe(abs).mean(dim=['x', 'sample']).isel(time=slice(0, 257, 2)).plot(hue='model', figsize=[4.5, 3.5], linewidth=2.0)
    plt.title('Error for 1-D advection')
    plt.xlabel('Time step')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.grid()
    plt.xticks(range(0, 257, 50))
    plt.show()
    plt.savefig('1d-test-mae.png', dpi=288, bbox_inches='tight')

    print("Вы молодцы!")