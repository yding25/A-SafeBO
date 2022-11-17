import os
import argparse

import pandas as pd

def options():
    parser = argparse.ArgumentParser(description='Run experiments for performance comparison of the three algorithms\n'
                                                 '(A-SafeBO, SafeOPT, StageOPT)')
    parser.add_argument("--max_iter", default=100, type=int, help='Maximum number of iterations for each algorithm')
    parser.add_argument("--max_test", default=1, type=int, help="A number indicating how many times each experiment will be repeated.")
    args = parser.parse_args()
    print(args)
    return args

def exp(env_dict):

    kwargs = ['--max_iter', '--num_exp', '--env_name', '--threshold', '--start_point']

    env_setting = env_dict['env_setting']
    hp_value = env_dict['hp_value']
    threshold = env_dict['threshold']
    initial_sets = env_dict['initial_sets']

    env_name = env_setting[0]
    max_iter = env_setting[1]
    max_test = env_setting[2]
    ############
    # A-SafeBO #
    ############

    for start_point in initial_sets:
        for i in range(max_test):
            kwargs_value = [max_iter, i, env_name, threshold, start_point]
            arg_options = ''
            for k, v in zip(kwargs, kwargs_value):
                arg_options += f' {k} {v}'
            os.system(f'python3 environment_a_safebo.py {arg_options}')

    ###########
    # SafeOPT #
    ###########
    # hyper parameters
    hyper_parameters = ['--length_scale', '--beta']

    for start_point in initial_sets:
        for i in range(max_test):
            kwargs_value = [max_iter, i, env_name, threshold, start_point]
            arg_options = ''
            for k, v in zip(kwargs + hyper_parameters, kwargs_value + hp_value):
                arg_options += f' {k} {v}'
            os.system(f'python3 environment_safeopt.py {arg_options}')

    ############
    # StageOPT #
    ############
    # hyper parameters
    hyper_parameters = ['--length_scale', '--beta']

    # rate of samples for stage two from total samples
    rate_stagetwo = ['--rate_stagetwo']

    for start_point in initial_sets:
        for i in range(max_test):
            kwargs_value = [max_iter, i, env_name, threshold, start_point, 0.1]
            arg_options = ''
            for k, v in zip(kwargs + rate_stagetwo + hyper_parameters, kwargs_value + hp_value):
                arg_options += f' {k} {v}'
            os.system(f'python3 environment_stageopt.py {arg_options}')

if __name__=='__main__' :
    ##########
    # Common #
    ##########
    print("Load Options")
    args = options()

    "--max_iter, --max_test"
    max_iter = args.max_iter
    max_test = args.max_test

    "The name of the environment in which the experiment is run\n"
    "possible env : GRIEWANK, ADJIMAN, HARTMANN_6D, PERIODIC_10D, POWERPLANT"

    #############################
    # exp for Griewank function #
    #############################
    env_dict = {'env_setting': ['GRIEWANK', max_iter, max_test],
                'hp_value': [0.3, 2.0],
                'threshold': -0.9,
                'initial_sets': ['[3.0,4.5]', '[-3.0,4.5]', '[3.0,-4.5]', '[-3.0,-4.5]',
                                  '[4.0,5.0]', '[-4.0,-5.0]',
                                  '[2.5,5.0]', '[-2.5,5.0]', '[2.5,-5.0]', '[-2.5,-5.0]']}
    exp(env_dict)

    ############################
    # exp for Adjiman function #
    ############################
    env_dict = {'env_setting': ['ADJIMAN', max_iter, max_test],
                'hp_value': [0.3, 2.5],
                'threshold': -0.1,
                'initial_sets': ['[-3.0,2.0]', '[-3.25,1.8]', '[-2.75,1.8]', '[-3.25,2.2]',
                                 '[-2.75,2.2]', '[-3.0,-4.0]',
                                 '[-3.25,-4.25]', '[-2.75,-4.5]', '[-3.25,-4.5]', '[-2.75,-4.25]']}
    exp(env_dict)

    ################################
    # exp for HARTMANN 6D function #
    ################################
    env_dict = {'env_setting': ['HARTMANN_6D', max_iter, max_test],
                'hp_value': [0.7, 3.0],
                'threshold': 1.2,
                'initial_sets': ['[0.493,0.921,0.246,0.667,0.020,0.560]',
                                 '[0.354,0.876,0.337,0.814,0.699,0.442]',
                                 '[0.483,0.731,0.331,0.409,0.944,0.370]',
                                 '[0.265,0.819,0.579,0.590,0.909,0.240]',
                                 '[0.263,0.822,0.049,0.659,0.602,0.305]',
                                 '[0.314,0.948,0.789,0.366,0.269,0.333]',
                                 '[0.581,0.867,0.513,0.535,0.815,0.425]',
                                 '[0.358,0.949,0.734,0.380,0.355,0.526]',
                                 '[0.535,0.956,0.847,0.482,0.074,0.498]',
                                 '[0.530,0.944,0.889,0.462,0.533,0.517]']}
    exp(env_dict)

    # ################################
    # exp for Periodic 10D function #
    # ################################
    env_dict = {'env_setting': ['PERIODIC_10D', max_iter, max_test],
                'hp_value': [0.7, 3.0],
                'threshold': -5.,
                'initial_sets': ['[0.3,-0.5,-0.3,0.5,0.3,0.5,-0.3,-0.5,0.3,-0.5]',
                                 '[0.286,-0.294,0.899,-0.268,-0.192,0.685,0.560,-0.216,-0.224,-0.257]',
                                 '[-0.086,-0.254,-0.799,0.227,0.291,-0.281,0.668,0.827,-0.177,-0.261]',
                                 '[-0.209,-0.218,0.218,-0.138,-0.651,-0.956,0.453,-0.339,0.493,0.432]',
                                 '[0.973,0.904,0.280,-0.367,0.004,0.162,-0.442,0.209,0.373,0.364]',
                                 '[-0.873,0.457,-0.364,-0.544,-0.061,0.659,0.199,-0.038,0.120,-0.440]',
                                 '[-0.535,-0.075,-0.429,0.459,0.661,0.363,0.091,-0.521,-0.388,-0.673]',
                                 '[0.048,-0.517,0.663,0.207,0.487,0.389,-0.395,0.379,0.562,-0.254]',
                                 '[0.767,0.558,0.008,-0.307,-0.644,0.172,-0.474,-0.027,-0.297,-0.067]',
                                 '[0.659,0.600,0.355,0.018,0.203,-0.625,-0.404,0.419,0.298,-0.369]']}
    exp(env_dict)

    ###################################
    # exp for Power Plant environment #
    ###################################
    env_dict = {'env_setting': ['POWERPLANT', max_iter],
                'hp_value': [0.3, 3.0],
                'threshold': 430.,
                'initial_sets': ['[30.663,70.5748,1010.1157,75.2918]',
                                 '[33.08995296,80.40255314,994.20235075,25.67331007]',
                                 '[34.19279221,30.29279353,1014.14653527,68.90239283]',
                                 '[32.15947294,71.12624406,1010.52548583,45.09570016]',
                                 '[36.0114917,78.27162509,1013.41056311,97.4628945]',
                                 '[30.16595202,69.28065608,1008.1440472,61.48579501]',
                                 '[27.75832493,68.82797808,1014.84171746,43.66269532]',
                                 '[29.62999749,25.45803955,1008.64206441,97.61239267]',
                                 '[36.09885622,31.91987892,1030.16868218,79.76351214]',
                                 '[36.67673215,31.35579247,1031.98630342,78.2728365]']}
    exp(env_dict)