# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import sys


def options():
    parser = argparse.ArgumentParser(
        description="Run experiments for performance comparison of the algorithms: A-SafeBO"
    )
    parser.add_argument(
        "--max_iter",
        default=100,
        type=int,
        help="Maximum number of iterations for each algorithm",
    )
    parser.add_argument(
        "--max_test",
        default=1,
        type=int,
        help="A number indicating how many times each experiment will be repeated.",
    )
    args = parser.parse_args()
    print(args)
    return args


def exp(env_dict):

    kwargs = ["--max_iter", "--num_exp", "--env_name", "--threshold", "--start_point"]

    env_setting = env_dict["env_setting"]
    hp_value = env_dict["hp_value"]
    threshold = env_dict["threshold"]
    initial_sets = env_dict["initial_sets"]

    env_name = env_setting[0]
    max_iter = env_setting[1]
    max_test = env_setting[2]

    ############
    # A-SafeBO #
    ############
    global n_sample_array

    n_sample_array = [10]

    for n in n_sample_array:
        # Yan: 在代码中，虽然定义了10个不同的起始点，但在实际运行中，通过嵌套的循环结构，每次只选择一个起始点来运行实验。这是因为在循环的第二层，针对每个起始点，还有一个嵌套的循环，用于执行多次实验。
        for start_point in initial_sets:
            for i in range(max_test):
                n_sample = ["--num_smaple"]
                kwargs_value = [max_iter, i, env_name, threshold, start_point, n]
                arg_options = ""
                for k, v in zip(kwargs + n_sample, kwargs_value):
                    arg_options += f" {k} {v}"
                os.system(f"python3 environment_a_safebo.py {arg_options}")


if __name__ == "__main__":
    ##########
    # Common #
    ##########
    print("Load Options")
    args = options()

    "--max_iter, --max_test"
    max_iter = args.max_iter
    max_test = args.max_test

    "The name of the environment in which the experiment is run\n"
    "possible env : POWERPLANT"

    #############################
    # exp for POWERPLANT function #
    #############################
    # Yan: 真实的工厂数据。

    env_dict = {
        "env_setting": ["POWERPLANT", max_iter, max_test],
        "hp_value": [0.2, 3.0],
        "threshold": 453.0,
        # "initial_sets": [
        #     "[22.7710,29.2801,1032.5030,91.3730]",
        #     "[21.6642,40.3449,1011.8862,85.6849]",
        #     "[20.9155,46.5602,1032.5521,53.6079]",
        #     "[20.2352,34.0057,1015.6979,90.9239]",
        #     "[18.3926,64.8709,1028.1254,31.8348]",
        #     "[15.0425,74.8759,999.9600,69.0304]",
        #     "[20.4701,41.1833,1008.3691,86.6184]",
        #     "[15.6398,72.1674,1021.6926,27.2426]",
        #     "[18.0123,36.9396,1015.5951,88.7465]",
        #     "[14.9420,74.8896,1002.0407,67.8375]",
        # ],
        "initial_sets": ["[22.7710,29.2801,1032.5030,91.3730]"],
    }

    exp(env_dict)

    print(
        f"results are saved in /benchmark_map/POWERPLANT/A-SafeBO_{max_iter}_N_{n_sample_array[0]}!"
    )
