"""
A-SafeBO(Adaptive Safe Bayesian Optimization)
Authors: - Guk Han (khan@rit.kaist.ac.kr)
"""
import numpy as np

from a_safebo_swarm import SwarmOptimization
from scipy.spatial.distance import cdist
import copy

class asafebo(object):

    def __init__(self, gp, bounds, threshold=None, max_iters=None, name=None, n_sample=None):

        self.dim = gp.input_dim
        self.name = name
        self.n_Sample = n_sample

        self.gp = [copy.deepcopy(gp)]

        self.beta = 1.0
        self.init_length_scale = (np.ones(self.dim, dtype=np.float)).tolist()
        self.length_scale = (np.ones(self.dim, dtype=np.float)).tolist()
        self.init_norm_bound = 1.0
        self.norm_bound = 1.0

        self.max_iters = max_iters
        self.bounds = bounds
        self.density_score = 0.
        self.window_threshold = 0.05
        self.density_threshold = 0.05
        self.lipschitz = 0.
        self.swarm_type = 'expand'

        self.t = 0
        self.scale_h = 1.

        self.esti_y = None
        self.std = 0.0

        self.noise = gp.Gaussian_noise.variance
        self.delta = 0.05  # with probability at least 1 - delta, the function exists in the confidence bound

        self.length = [float(b[1] - b[0]) for b in (self.bounds)]
        self.info_metric = None

        self.step_size = 10
        self.swarm_size = self.step_size * self.dim
        self.max_swarm_iters = 100

        self.optimal_velocities = np.ones(self.dim) * 0.5
        self.hyper_list = []
        self.result_list = []

        for i in range(self.dim):
            self.gp[0].X[:, i] = self.gp[0].X[:, i] / self.length[i]

        # Safe set
        self.S = np.asarray(self.gp[0].X)
        self.S_density = np.array([[0]])

        self.memory_bank = {'X': np.copy(self.gp[0].X), 'Y': np.copy(self.gp[0].Y)}
        self.W_MB = np.zeros_like(self.gp[0].Y)
        self.P_MB = np.zeros_like(self.gp[0].Y)

        self.threshold = np.copy(threshold)
        self.unsafe_rate = 0.
        self.gamma = 0.9

        self.cnt_FNS = 0
        self.cnt_EXP = 0
        self.cnt_MAX = 0

        self.swarm = SwarmOptimization(self.beta, self.threshold, self.length, self.gp, self.swarm_size,
                                       self.optimal_velocities, self.window_threshold, self.density_threshold, self.delta,
                                       bounds=self.bounds)

    def cal_next_point(self, weight):

        self.select_acquisition()
        self.cal_lipschitz(self.S)

        if len(self.S) > 0:
            # make safe candidate
            if self.swarm_type == 'maximize':
                safe_size, input_dim = self.S.shape

                if safe_size < self.swarm_size:
                    random_id = np.random.randint(safe_size, size=self.swarm_size)

                    candidate = self.S[random_id, :]
                else:
                    s = np.squeeze(self.S_density).argsort()

                    candidate = self.S[s][:self.swarm_size]

            else:
                candidate = \
                    (np.random.uniform(self.swarm.bounds[0][0], self.swarm.bounds[0][1], self.swarm_size)).reshape(
                        -1, 1)
                for i in range(self.dim - 1):
                    added_axis = (np.random.uniform(self.swarm.bounds[i + 1][0], self.swarm.bounds[i + 1][1],
                                                    self.swarm_size)).reshape(-1, 1)
                    candidate = np.concatenate((candidate, added_axis), axis=1)

        else:
            candidate = \
                (np.random.uniform(self.swarm.bounds[0][0], self.swarm.bounds[0][1], self.swarm_size)).reshape(
                    -1, 1)
            for i in range(self.dim - 1):
                added_axis = (np.random.uniform(self.swarm.bounds[i + 1][0], self.swarm.bounds[i + 1][1],
                                                self.swarm_size)).reshape(-1, 1)
                candidate = np.concatenate((candidate, added_axis), axis=1)

        safe_swarm = self.swarm

        safe_swarm.init_swarm(self.gp, candidate, self.beta, self.swarm_size, self.max_swarm_iters,
                              self.S, self.lipschitz, self.swarm_type, self.unsafe_rate, self.density_score, weight, self.max_iters)

        next_point = safe_swarm.run_swarm()

        if next_point.any() == None:
            id = np.argmin(self.S_density)
            next_point = np.atleast_2d(self.S[id])

        result = np.array([np.squeeze(gp.predict_noiseless(next_point)) for gp in self.gp])

        mean = np.squeeze(np.dot(weight, result[:, 0]))
        m = np.atleast_2d(np.copy(mean))
        for i in range(self.num_model - 1):
            m = np.vstack([m, np.atleast_2d(mean)])
        var = np.squeeze(np.dot(weight, result[:, 1]) + np.dot(weight, result[:, 0] - m))

        self.std = np.sqrt(abs(var))

        return np.atleast_2d(next_point), mean

    def optimize(self):

        model_size = min(int(self.max_iters), self.n_Sample)
        self.num_model = (max(len(self.memory_bank['X']) - self.n_Sample, 0) // model_size) + 1

        if len(self.memory_bank['X']) > 1:

            if len(self.memory_bank['X']) < model_size:
                self.gp[0].set_XY(np.atleast_2d(self.memory_bank['X']),
                                  np.atleast_2d(self.memory_bank['Y']))
            else:
                for i in range(int(self.num_model)):
                    new_gp = copy.deepcopy(self.gp[0])
                    idx = np.random.choice(len(self.memory_bank['X']), model_size, replace=False, p=self.P_MB.squeeze())
                    new_gp.set_XY(np.atleast_2d(self.memory_bank['X'][idx]),
                                  np.atleast_2d(self.memory_bank['Y'][idx]))
                    if len(self.gp) - 1 < i:
                        self.gp.append(new_gp)
                    else:
                        self.gp[i] = new_gp

        else:
            self.density_score = 0.

        weight = []
        if self.num_model > 1:
            for i in range(self.num_model):
                if i < self.num_model - 1:
                    mean_MSE = np.mean(
                        np.power(self.gp[i + 1].Y - self.gp[i].predict_noiseless(self.gp[i + 1].X)[0], 2))
                else:
                    mean_MSE = np.mean(np.power(self.gp[0].Y - self.gp[i].predict_noiseless(self.gp[0].X)[0], 2))
                weight.append(mean_MSE)
            weight = (1. - (weight / np.sum(weight))) / (self.num_model - 1.)
            weight = weight.reshape(1, self.num_model)
        else:
            weight.append(1)

        next_point, self.esti_y = self.cal_next_point(weight)

        ori_x = np.copy(next_point)
        for i in range(self.dim):
            ori_x[0][i] = next_point[0][i] * self.length[i]

        return np.atleast_2d(ori_x)

    def get_maximum(self):
        """
        Return the current estimate for the maximum.

        Returns
        -------
        x : ndarray
            Location of the maximum
        y : 0darray
            Maximum value

        """
        max_idx = np.argmax(self.memory_bank['Y'])
        max_point = np.atleast_2d(np.copy(self.memory_bank['X'][max_idx, :]))

        for i in range(self.dim):
            max_point[:, i] = max_point[:, i] * self.length[i]

        return np.atleast_2d(max_point), np.atleast_2d(self.memory_bank['Y'][max_idx])

    def scale(self):

        self.info_metric = 0
        for i in range(len(self.gp)):
            K = self.gp[i].kern.Kdiag(self.gp[i].X)
            self.info_metric += 0.5 * np.log(np.linalg.det(np.identity(len(K)) + np.power(self.noise, -2.) * K))

        self.scale_h = np.exp((5 * np.log(2) / (float(self.max_iters) * 0.9)) * self.t)
        self.length_scale = self.init_length_scale / np.float64(np.power(self.scale_h, 1 / self.dim))

        self.cal_beta()

        for i in range(len(self.gp)):
            self.gp[i].kern.lengthscale = self.length_scale
            self.gp[i].kern.parameters_changed()

    def cal_beta(self):

        self.norm_bound = np.float64(self.scale_h) * self.init_norm_bound

        beta_star = self.norm_bound + 4. * self.noise * np.sqrt(max(self.info_metric + 1. + np.log(1. / self.delta), 0.))
        self.beta = max(self.beta, beta_star)

    def add_new_data(self, inputs, outputs):
        scaled_inputs = np.copy(inputs)
        scaled_outputs = np.copy(outputs)

        self.result_list.append([float(self.esti_y.squeeze()), outputs, float(self.threshold.squeeze()),
                                 float(float(outputs.squeeze()) > float(self.threshold.squeeze())),
                                 self.density_score, self.unsafe_rate])

        self.hyper_list.append([float(self.length_scale[0]), float(self.beta), float(self.norm_bound)])

        for i in range(self.dim):
            scaled_inputs[:, i] = scaled_inputs[:, i] / self.length[i]

        self.memory_bank['X'] = np.vstack([self.memory_bank['X'], np.atleast_2d(scaled_inputs)])
        self.memory_bank['Y'] = np.vstack([self.memory_bank['Y'], np.atleast_2d(scaled_outputs)])
        self.W_MB = np.vstack([self.W_MB, np.atleast_2d(np.zeros_like(scaled_outputs))])

        self.S = self.memory_bank['X'][np.squeeze(self.memory_bank['Y'] > float(self.threshold))].reshape(-1, self.dim)
        self.S_density = self.W_MB[np.squeeze(self.memory_bank['Y'] > float(self.threshold))].reshape(-1, 1)

        window_upper_bound = scaled_inputs + self.window_threshold
        window_lower_bound = scaled_inputs - self.window_threshold
        in_window = np.all(np.logical_and((np.atleast_2d(self.memory_bank['X'])[:-1] > window_lower_bound),
                                          (np.atleast_2d(self.memory_bank['X'])[:-1] < window_upper_bound)), axis=1)

        self.W_MB[:-1][in_window] += 1.
        self.W_MB[-1] = sum(in_window)

        self.density_score = sum(in_window) / self.max_iters

        if sum(self.W_MB) != 0 and len(self.W_MB) != 1:
            self.P_MB = (1. - self.W_MB / sum(self.W_MB)) / (len(self.W_MB) - 1.)
        else:
            if sum(self.W_MB) == 0 and len(self.W_MB) != 1:
                self.P_MB = np.ones_like(self.W_MB) / (len(self.W_MB))
            else:
                self.P_MB = np.atleast_2d(1.)

        unsafe_point = float(sum(self.memory_bank['Y'] < self.threshold))
        self.unsafe_rate = unsafe_point / self.max_iters

        if self.t < float(self.max_iters) * self.gamma:
            self.scale()

        self.t += 1

    def cal_lipschitz(self, safe_set):

        if len(safe_set) > 1:
            safe_result = np.array([np.squeeze(gp.predict_noiseless(safe_set)) for gp in self.gp])

            safe_y = np.mean(safe_result[:, 0], axis=0).reshape(-1, 1)

            dist_x = cdist(safe_set, safe_set)
            dist_y = cdist(safe_y, safe_y)


            dist_x[dist_x == 0] = np.inf
            slope = dist_y / dist_x
            self.lipschitz = np.max(slope)
        else:
            self.lipschitz = 0.

    def select_acquisition(self):

        if float(self.max_iters) * self.gamma > self.t:

            if (self.density_score < self.density_threshold) or (self.unsafe_rate > self.delta):
                if np.random.binomial(1, self.density_score) == 1:
                    self.swarm_type = 'find_new'
                    self.cnt_FNS += 1
                else:
                    self.swarm_type = 'expand'
                    self.cnt_EXP += 1
            else:
                self.swarm_type = 'find_new'
                self.cnt_FNS += 1
        else:
            self.swarm_type = 'maximize'
            self.cnt_MAX += 1
