import os
import argparse
import random
import sys

import GPy
import numpy as np
import pandas as pd
from matplotlib import pylab as plt

import time
import copy

import json
import os.path

from a_safebo import asafebo


def options():
    parser = argparse.ArgumentParser(description="A-SafeBO experiments")
    parser.add_argument(
        "--max_iter", default=1000, type=int, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--num_exp",
        default=0,
        type=int,
        help="A number indicating the number of experiments",
    )
    parser.add_argument(
        "--env_name",
        default="GRIEWANK",
        type=str,
        help="The name of the environment in which the experiment is run\n"
        "possible env : GRIEWANK, ADJIMAN, ALPINE, HARTMANN_6D, PERIODIC_10D, POWERPLANT",
    )
    parser.add_argument(
        "--threshold",
        default=-0.9,
        type=float,
        help="Safety threshold for safety constraint",
    )
    parser.add_argument(
        "--start_point",
        default="[3.0,4.5]",
        type=json.loads,
        help="Starting point that included in the safe set for optimization",
    )
    parser.add_argument(
        "--num_smaple",
        default=300,
        type=int,
        help="Number of samples for building a gp in ensemble GP",
    )
    args = parser.parse_args()
    print(args)
    return args


def make_dir(env_name, start_point, num_exp, max_iter, n_sample):

    path = (
        "./benchmark_map/"
        + env_name
        + "/A-SafeBO_"
        + str(max_iter)
        + "_N_"
        + str(n_sample)
    )

    if not (os.path.isdir(os.path.expanduser("./benchmark_map/" + env_name))):
        os.makedirs(os.path.join(os.path.expanduser("./benchmark_map/" + env_name)))
    if not (os.path.isdir(os.path.expanduser(path))):
        os.makedirs(os.path.join(os.path.expanduser(path)))
    if not (os.path.isdir(os.path.expanduser(path + "/" + str(start_point)))):
        os.makedirs(os.path.join(os.path.expanduser(path + "/" + str(start_point))))
    if not (
        os.path.isdir(
            os.path.expanduser(path + "/" + str(start_point) + "/" + str(num_exp))
        )
    ):
        os.makedirs(
            os.path.join(
                os.path.expanduser(path + "/" + str(start_point) + "/" + str(num_exp))
            )
        )

    if not (os.path.exists(path + "/final_result_A-SafeBO.xlsx")):
        final_result = [
            ["start pint", "num exp", "time", "esti best", "safe rate", "min regret"]
        ]
        df1 = pd.DataFrame.from_records(final_result)
        df1.to_excel(path + "/final_result_A-SafeBO.xlsx", index=False)
    if not (os.path.exists(path + "/final_regret_A-SafeBO.xlsx")):
        df2 = pd.DataFrame.from_records([])
        df2.to_excel(path + "/final_regret_A-SafeBO.xlsx", index=False)
    if not (os.path.exists(path + "/final_cumregret_A-SafeBO.xlsx")):
        df3 = pd.DataFrame.from_records([])
        df3.to_excel(path + "/final_cumregret_A-SafeBO.xlsx", index=False)

    sys.stdout = open(
        path + "/" + str(start_point) + "/" + str(num_exp) + "/A-SafeBO_log.txt", "w"
    )


def save_result(path, final_result, regret_list):

    df1 = pd.read_excel(path + "/final_result_A-SafeBO.xlsx")
    df1 = df1.append(final_result)
    writer1 = pd.ExcelWriter(path + "/final_result_A-SafeBO.xlsx", engine="openpyxl")
    df1.to_excel(writer1, index=False)
    writer1.save()

    df2 = pd.read_excel(path + "/final_regret_A-SafeBO.xlsx")
    df2 = df2.append([np.array(regret_list)[:, 0].T.tolist()])
    writer2 = pd.ExcelWriter(path + "/final_regret_A-SafeBO.xlsx", engine="openpyxl")
    df2.to_excel(writer2, index=False)
    writer2.save()

    df3 = pd.read_excel(path + "/final_cumregret_A-SafeBO.xlsx")
    df3 = df3.append([np.array(regret_list)[:, 1].T.tolist()])
    writer3 = pd.ExcelWriter(path + "/final_cumregret_A-SafeBO.xlsx", engine="openpyxl")
    df3.to_excel(writer3, index=False)
    writer3.save()

    sys.stdout.close()


# make environment
class benchmark_env(object):
    def __init__(self, name):

        self.name = name

        self.noise_var = 0.05**2
        self.bound = []
        self.domain = []
        self.grid = []
        self.target = None

        self.function = None
        self.CCPPmodel = None  # for POWERPLANT

        if self.name == "GRIEWANK":

            # dimensions of the environment
            self.dim = 2

            # original
            # self.function = lambda x: (1 + (np.power(x[:, 0], 2.) / 4000. + np.power(x[:, 1], 2.) / 4000.) -
            #                            (np.cos(x[:, 0]) * np.cos(x[:, 1]) / 2.))

            # version for maximization test
            self.function = lambda x: -(
                1
                + (np.power(x[:, 0], 2.0) / 4000.0 + np.power(x[:, 1], 2.0) / 4000.0)
                - (np.cos(x[:, 0]) * np.cos(x[:, 1] / 2.0))
            )

            # boundary of the environment
            self.bound = [(-5.0, 5.0), (-5.0, 5.0)]
            self.domain = [[-5.0, 5.0], [-5.0, 5.0]]

            # global optimum
            self.max_coordi = np.array([[0, 0]])
            self.max_value = np.array([[0.0]])

            for i in range(int(self.dim)):
                self.grid.append(
                    np.linspace(int(self.bound[i][0]), int(self.bound[i][1]), num=100)
                )

            self.grid = np.array([x.ravel() for x in np.meshgrid(*self.grid)]).T

            file = "./benchmark_map/" + self.name + "/" + self.name + ".dat"
            if not (os.path.isfile(file)):

                self.target = self.function(self.grid)
                self.target.tofile(
                    "./benchmark_map/" + self.name + "/" + self.name + ".dat"
                )
            else:
                self.target = np.fromfile(
                    "./benchmark_map/" + self.name + "/" + self.name + ".dat",
                    dtype=float,
                )

        elif self.name == "ADJIMAN":

            # dimensions of the environment
            self.dim = 2
            # original
            # self.function = lambda x: (np.cos(x[:, 0]) * np.sin(x[:, 1]) - x[:, 0] / (np.power(x[:, 1], 2) + 1.))

            # version for maximization test
            self.function = lambda x: -(
                np.cos(x[:, 0]) * np.sin(x[:, 1])
                - x[:, 0] / (np.power(x[:, 1], 2) + 1.0)
            )

            # boundary of the environment
            self.bound = [(-5.0, 5.0), (-5.0, 5.0)]
            self.domain = [[-5.0, 5.0], [-5.0, 5.0]]

            # global optimum
            self.max_coordi = np.array([[5, 0.0]])
            self.max_value = np.array([5.0])

            for i in range(int(self.dim)):
                self.grid.append(
                    np.linspace(int(self.bound[i][0]), int(self.bound[i][1]), num=100)
                )

            self.grid = np.array([x.ravel() for x in np.meshgrid(*self.grid)]).T

            file = "./benchmark_map/" + self.name + "/" + self.name + ".dat"
            if not (os.path.isfile(file)):

                self.target = self.function(self.grid)
                self.target.tofile(
                    "./benchmark_map/" + self.name + "/" + self.name + ".dat"
                )
            else:
                self.target = np.fromfile(
                    "./benchmark_map/" + self.name + "/" + self.name + ".dat",
                    dtype=float,
                )

        elif self.name == "HARTMANN_6D":

            # dimensions of the environment
            self.dim = 6

            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A = np.array(
                [
                    [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                    [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                    [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                    [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
                ]
            )
            P = np.array(
                [
                    [0.1312, 0.1696, 0.5569, 0.124, 0.8283, 0.5886],
                    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.381],
                ]
            )
            # original
            # self.function = lambda x: -np.dot(alpha, np.exp(-np.sum(np.multiply(A, np.power((x - P), 2.)), axis=1)))

            # version for maximization test
            self.function = lambda x: np.dot(
                alpha, np.exp(-np.sum(np.multiply(A, np.power((x - P), 2.0)), axis=1))
            )

            # boundary of the environment
            self.bound = [
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
                (0.0, 1.0),
            ]
            self.domain = [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]

            # global optimum
            self.max_coordi = np.array(
                [[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]
            )
            self.max_value = np.array([[3.32237]])

            self.grid = None
            self.target = None

        elif self.name == "PERIODIC_10D":

            # dimensions of the environment
            self.dim = 10
            # original
            # self.function = lambda x: 1. + (
            #         np.power(np.sin(x[:, 0]), 2.) + np.power(np.sin(x[:, 1]), 2.) + np.power(np.sin(x[:, 2]), 2.) +
            #         np.power(np.sin(x[:, 3]), 2.) + np.power(np.sin(x[:, 4]), 2.) + np.power(np.sin(x[:, 5]), 2.) +
            #         np.power(np.sin(x[:, 6]), 2.) + np.power(np.sin(x[:, 7]), 2.) + np.power(np.sin(x[:, 8]), 2.) +
            #         np.power(np.sin(x[:, 9]), 2.)) - 0.1 * (np.exp(-(np.power(x[:, 0], 2.) + np.power(x[:, 1], 2.) +
            #                                                          np.power(x[:, 2], 2.) + np.power(x[:, 3], 2.) +
            #                                                          np.power(x[:, 4], 2.) + np.power(x[:, 5], 2.) +
            #                                                          np.power(x[:, 6], 2.) + np.power(x[:, 7], 2.) +
            #                                                          (x[:, 8], 2.) + np.power(x[:, 9], 2.))))

            # version for maximization test
            self.function = lambda x: -(
                1.0
                + (
                    np.power(np.sin(x[:, 0]), 2.0)
                    + np.power(np.sin(x[:, 1]), 2.0)
                    + np.power(np.sin(x[:, 2]), 2.0)
                    + np.power(np.sin(x[:, 3]), 2.0)
                    + np.power(np.sin(x[:, 4]), 2.0)
                    + np.power(np.sin(x[:, 5]), 2.0)
                    + np.power(np.sin(x[:, 6]), 2.0)
                    + np.power(np.sin(x[:, 7]), 2.0)
                    + np.power(np.sin(x[:, 8]), 2.0)
                    + np.power(np.sin(x[:, 9]), 2.0)
                )
                - 0.1
                * (
                    np.exp(
                        -(
                            np.power(x[:, 0], 2.0)
                            + np.power(x[:, 1], 2.0)
                            + np.power(x[:, 2], 2.0)
                            + np.power(x[:, 3], 2.0)
                            + np.power(x[:, 4], 2.0)
                            + np.power(x[:, 5], 2.0)
                            + np.power(x[:, 6], 2.0)
                            + np.power(x[:, 7], 2.0)
                            + np.power(x[:, 8], 2.0)
                            + np.power(x[:, 9], 2.0)
                        )
                    )
                )
            )

            # boundary of the environment
            self.bound = [
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
                (-2.0, 2.0),
            ]
            self.domain = [
                [-2.0, 2.0],
                [-2.0, 2.0],
                [-2.0, 2.0],
                [-2.0, 2.0],
                [-2.0, 2.0],
                [-2.0, 2.0],
                [-2.0, 2.0],
                [-2.0, 2.0],
                [-2.0, 2.0],
                [-2.0, 2.0],
            ]

            # global optimum
            self.max_coordi = np.array(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            )
            self.max_value = np.array([[-0.9]])

            self.grid = None
            self.target = None

        elif self.name == "POWERPLANT":

            from datasets import powerplant

            X, y = powerplant.load_data()
            self.CCPPmodel = powerplant.train(X, y)

            # dimensions of the environment
            self.dim = 4
            self.function = lambda x: powerplant.predict(self.CCPPmodel, x)

            # boundary of the environment
            self.bound = [
                (1.81, 37.11),
                (25.36, 81.56),
                (992.89, 1033.30),
                (25.56, 100.16),
            ]
            self.domain = [
                [1.81, 37.11],
                [25.36, 81.56],
                [992.89, 1033.30],
                [25.56, 100.16],
            ]

            # global optimum
            self.max_coordi = np.array([[5.48, 40.07, 1019.63, 65.62]])
            self.max_value = np.array([[495.76]])

            self.grid = None
            self.target = None

    def sample(self, x):

        if self.name == "HARTMANN_6D":
            x = np.atleast_2d(x)
            y = []
            for i in range(len(x)):
                x_i = [x[i], x[i], x[i], x[i]]
                np.random.seed(int(time.time()))
                noise = np.random.normal(0, np.sqrt(self.noise_var), 1)
                y.append(float(self.function(x_i)) + float(noise))

        else:
            x = np.atleast_2d(x)
            np.random.seed(int(time.time()))
            noise = np.random.normal(0, np.sqrt(self.noise_var), len(x))
            y = self.function(x) + noise

        return np.atleast_2d(y)


class Testbenchmark(object):
    def __init__(self, name, max_iter, num_exp, threshold, start_point, n_sample):

        self.name = name
        self.num_exp = num_exp
        self.env = benchmark_env(name)
        self.n_sample = n_sample

        self.path = (
            "./benchmark_map/"
            + self.name
            + "/A-SafeBO_"
            + str(max_iter)
            + "_N_"
            + str(self.n_sample)
        )

        self.bound = copy.deepcopy(self.env.bound)
        self.domain = copy.deepcopy(self.env.domain)
        self.dim = self.env.dim
        self.length = [float(b[1] - b[0]) for b in (self.bound)]

        self.threshold = np.atleast_2d([threshold])

        self.max_iters = max_iter
        self.point_color = []
        self.point_x = []
        self.point_y = []

        self.cum_regret = 0
        self.regret_list = []

        self.safe_rate = None
        self.esti_best = None

        if self.env.dim == 2:
            x1 = np.linspace(int(self.domain[0][0]), int(self.domain[0][1]), num=100)
            x2 = np.linspace(int(self.domain[1][0]), int(self.domain[1][1]), num=100)

            self.grid = copy.deepcopy(self.env.grid)
            self.mesh = np.array(np.meshgrid(x1, x2))
        else:
            self.grid = None
            self.mesh = None

        self.init_x = np.atleast_2d(start_point)
        self.init_y = np.atleast_2d(self.env.sample(self.init_x))

        self.target = copy.deepcopy(self.env.target)
        self.max_coordi = self.env.max_coordi
        self.max_value = self.env.max_value

        # set optimizer
        length = []
        for i in range(self.dim):
            length.append(1.0)
        kernel = GPy.kern.RBF(
            input_dim=self.dim, variance=2.0, lengthscale=length, ARD=True
        )

        gp = GPy.models.GPRegression(self.init_x, self.init_y, kernel, noise_var=0.01)

        self.opt = asafebo(
            gp=gp,
            bounds=self.bound,
            threshold=self.threshold,
            max_iters=self.max_iters,
            name=self.name,
            n_sample=self.n_sample,
        )

    def plot_initial_setting(self):

        if self.env.dim == 2:

            fig = plt.figure()
            ax = fig.gca()

            c = ax.contour(
                self.mesh[0], self.mesh[1], self.target.reshape(self.mesh[0].shape)
            )

            plt.colorbar(c)

            fig.suptitle("Threshold : %f" % self.opt.threshold)

            ax.plot(self.init_x[0][0], self.init_x[0][1], "om", markersize=10)
            ax.plot(
                self.max_coordi.reshape(-1, 2)[:, 0],
                self.max_coordi.reshape(-1, 2)[:, 1],
                "ob",
                markersize=10,
            )

            plt.savefig(
                self.path
                + "/"
                + str(start_point)
                + "/"
                + str(num_exp)
                + "/contour_safe_fig_00.png"
            )
            plt.close()
        else:
            print("Plot only in 2D environment")

    def plot(self, iter):

        if self.env.dim == 2:

            fig = plt.figure()
            ax = fig.gca()

            c = ax.contour(
                self.mesh[0], self.mesh[1], self.target.reshape(self.mesh[0].shape)
            )

            plt.colorbar(c)

            fig.suptitle("Threshold : %f" % self.opt.threshold)

            x = np.array(self.point_x)
            x = x.reshape(-1, 2)

            ax.scatter(x[:, 0], x[:, 1], c=self.point_color, marker="o", alpha=1.0)

            max_point, _ = self.opt.get_maximum()
            ax.plot(max_point[0][0], max_point[0][1], "ok", markersize=10)
            ax.plot(
                self.max_coordi.reshape(-1, 2)[:, 0],
                self.max_coordi.reshape(-1, 2)[:, 1],
                "ob",
                markersize=10,
            )

            ax.plot(self.init_x[0][0], self.init_x[0][1], "om", markersize=10)

            plt.savefig(
                self.path
                + "/"
                + str(start_point)
                + "/"
                + str(num_exp)
                + "/contour_safe_fig_%02d.png" % iter
            )
            plt.close()

        else:
            print("Plot only in 2D environment")

    def plot_for_2D_dim(self, regret_list):

        self.target_plot()
        self.plot(self.max_iters)
        self.best_plot()
        self.regret_plot(regret_list)

    def plot_for_high_dim(self, regret_list):

        a_regret_list = np.array(regret_list)
        x = np.arange(1, len(regret_list) + 1)
        plt.plot(x, a_regret_list[:, 0])
        plt.title("regret")
        plt.savefig(
            self.path + "/" + str(start_point) + "/" + str(num_exp) + "/regret.png"
        )
        plt.close()

        plt.plot(x, a_regret_list[:, 1])
        plt.title("cum_regret")
        plt.savefig(
            self.path + "/" + str(start_point) + "/" + str(num_exp) + "/cum_regret.png"
        )
        plt.close()

        self.safe_rate = float(self.point_color.count("g") - 1.0) / float(
            self.max_iters
        )

        print(
            "Total exploration steps : %d\nSafe exploration steps : %d\nUnsafe exploration steps : %d\nSafe exploration rates : %f"
            % (
                self.max_iters,
                self.point_color.count("g") - 1.0,
                self.point_color.count("r"),
                self.safe_rate,
            )
        )

        max_point, _ = self.opt.get_maximum()
        self.esti_best = self.env.sample(max_point)

        print(
            "Target best position : {}, result : {}".format(
                self.max_coordi, self.max_value
            )
        )
        print(
            "Estimated target best points : {}, Estimated target best result : {}".format(
                max_point, self.esti_best
            )
        )

    def target_plot(self):

        if self.env.dim == 2:

            fig = plt.figure()
            ax = fig.gca()

            fig.suptitle("env name : {}".format(self.name))

            c = ax.contour(
                self.mesh[0], self.mesh[1], self.target.reshape(self.mesh[0].shape)
            )
            plt.colorbar(c)

            ax.plot(
                self.max_coordi.reshape(-1, 2)[:, 0],
                self.max_coordi.reshape(-1, 2)[:, 1],
                "ob",
                markersize=10,
            )

            print("2D map min : %f, max : %f" % (min(self.target), max(self.target)))

            plt.savefig("./benchmark_map/" + self.name + "/contour_target.png")
            plt.close()

        else:
            print("Plot only in 2D environment")

    def best_plot(self):

        if self.env.dim == 2:

            fig = plt.figure()
            ax = fig.gca()

            c = ax.contour(
                self.mesh[0], self.mesh[1], self.target.reshape(self.mesh[0].shape)
            )
            plt.colorbar(c)

            fig.suptitle("Threshold : %f" % self.opt.threshold)

            x = np.array(self.point_x)
            x = x.reshape(-1, 2)

            ax.scatter(x[:, 0], x[:, 1], c=self.point_color, marker="o", alpha=1.0)

            self.safe_rate = float(self.point_color.count("g") - 1.0) / float(
                self.max_iters
            )

            print(
                "Total exploration steps : %d\nSafe exploration steps : %d\nUnsafe exploration steps : %d\nSafe exploration rates : %f"
                % (
                    self.max_iters,
                    self.point_color.count("g") - 1.0,
                    self.point_color.count("r"),
                    self.safe_rate,
                )
            )

            max_point, _ = self.opt.get_maximum()
            self.esti_best = self.env.sample(max_point)
            ax.plot(max_point[0][0], max_point[0][1], "ok", markersize=10)
            ax.plot(self.max_coordi[:, 0], self.max_coordi[:, 1], "ob", markersize=10)
            ax.plot(self.init_x[:, 0], self.init_x[:, 1], "om", markersize=10)

            print(
                "Target best position : {}, result : {}".format(
                    self.max_coordi, self.max_value
                )
            )
            print(
                "Estimated target best points : {}, Estimated target best result : {}".format(
                    max_point, self.esti_best
                )
            )

            plt.savefig(
                self.path
                + "/"
                + str(start_point)
                + "/"
                + str(num_exp)
                + "/contour_max_fig.png"
            )
            plt.close()

        else:
            print("Plot only in 2D environment")

    def regret_plot(self, regret_list):

        regret_list = np.array(regret_list)
        x = np.arange(1, len(regret_list) + 1)

        plt.plot(x, regret_list[:, 0])
        plt.ylim(min(regret_list[:, 0]), max(regret_list[:, 0]))
        plt.title("regret")
        plt.savefig(
            self.path + "/" + str(start_point) + "/" + str(num_exp) + "/regret.png"
        )
        plt.close()

        plt.plot(x, regret_list[:, 1])
        plt.ylim(min(regret_list[:, 1]), max(regret_list[:, 1]))
        plt.title("cum_regret")
        plt.savefig(
            self.path + "/" + str(start_point) + "/" + str(num_exp) + "/cum_regret.png"
        )
        plt.close()


if __name__ == "__main__":
    print("A-SafeBO Load Options")
    args = options()

    "--max_iter, --num_exp, --env_name, --threshold, --start_point"
    max_iter = args.max_iter
    env_name = args.env_name
    num_exp = args.num_exp
    threshold = args.threshold
    start_point = args.start_point
    n_sample = args.num_smaple

    make_dir(env_name, start_point, num_exp, max_iter, n_sample)

    env_setting = Testbenchmark(
        env_name, max_iter, num_exp, threshold, start_point, n_sample
    )

    print("Option : {}".format(args))
    x = np.copy(env_setting.opt.gp[0].X[-1])
    for i in range(len(x)):
        x[i] = x[i] * env_setting.length[i]
    y = env_setting.env.sample(x)
    init_param = np.copy(x)
    init_value = np.copy(y)

    regret = env_setting.max_value - y
    env_setting.cum_regret += regret
    env_setting.regret_list.append([float(regret), float(env_setting.cum_regret)])

    env_setting.point_x.append(np.squeeze(x))
    env_setting.point_y.append(np.squeeze(y))

    if y > env_setting.opt.threshold:
        env_setting.point_color.append("g")
    else:
        env_setting.point_color.append("r")

    # plot initial point
    env_setting.plot_initial_setting()

    t = time.time()

    for i in range(env_setting.max_iters):

        t_loop_1 = time.time()
        next_point = env_setting.opt.optimize()

        next_cpk = np.atleast_2d(env_setting.env.sample(next_point))

        regret = env_setting.max_value - next_cpk

        env_setting.cum_regret += regret

        env_setting.regret_list.append([float(regret), float(env_setting.cum_regret)])

        env_setting.point_x.append(np.squeeze(next_point))
        env_setting.point_y.append(np.squeeze(next_cpk))

        env_setting.opt.add_new_data(next_point, next_cpk)
        t_loop_2 = time.time()

        if next_cpk > env_setting.opt.threshold:
            env_setting.point_color.append("g")
        else:
            env_setting.point_color.append("r")

    t1 = time.time()

    print(
        "init param :{}, init value ;{}, threshold :{}".format(
            init_param, init_value, env_setting.opt.threshold
        )
    )
    print("num of samples : {}".format(len(env_setting.opt.gp[0].X)))
    print("\n\ntotal time : %f s" % (t1 - t))
    print(
        "Cnt FNS : {}, EXP : {}, MAX : {}".format(
            env_setting.opt.cnt_FNS, env_setting.opt.cnt_EXP, env_setting.opt.cnt_MAX
        )
    )

    if env_setting.dim == 2:
        env_setting.plot_for_2D_dim(env_setting.regret_list)
    else:
        env_setting.plot_for_high_dim(env_setting.regret_list)

    final_result = [
        [
            start_point,
            num_exp,
            float((t1 - t)),
            float(env_setting.esti_best),
            float(env_setting.safe_rate),
            float(min(np.array(env_setting.regret_list)[:, 0])),
        ]
    ]
    save_result(env_setting.path, final_result, env_setting.regret_list)
