"""
A-SafeBO(Adaptive Safe Bayesian Optimization)
Authors: - Guk Han (khan@rit.kaist.ac.kr)
"""
import numpy as np
from scipy.spatial.distance import cdist

from scipy.special import expit

__all__ = ['SwarmOptimization']

class SwarmOptimization(object):
    def __init__(self, beta, threshold, length, gp, swarm_size, velocity, bounds=None):

        self.c1 = 1.
        self.c2 = 1.

        self.gp = gp
        self.input_dim = self.gp[0].input_dim
        self.threshold = np.copy(threshold)
        self.noise = self.gp[0].Gaussian_noise.variance

        self.beta = beta
        self.length = length

        self.bounds = bounds
        if self.bounds is not None:
            self.bounds = np.asarray(self.bounds)
            for i in range(self.input_dim):
                self.bounds[i, :] = self.bounds[i, :] / self.length[i]

        self.random_inertia = True
        self.initial_inertia = 1.0
        self.final_inertia = 0.1
        self.velocity_scale = velocity

        self.swarm_size = swarm_size

        self.positions = np.empty((swarm_size, self.input_dim), dtype=np.float)
        self.velocities = np.empty_like(self.positions)

        self.best_positions = np.empty_like(self.positions)
        self.best_values = np.empty(len(self.best_positions), dtype=np.float)
        self.global_best_position = None
        self.global_best_value = None

    @property
    def max_velocity(self):
        """Return the maximum allowed velocity of particles."""
        return self.velocity_scale

    def init_swarm(self, gp, positions, beta, swarm_size, max_iter, weight):
        """Initialize the swarm.

        Parameters
        ----------
        positions: ndarray
            The initial positions of the particles.
        """
        self.gp = gp
        self.weight = weight

        self.swarm_size = swarm_size
        self.max_iter = max_iter

        self.positions = np.copy(positions)

        self.beta = beta

        self.best_positions = np.empty_like(self.positions)
        self.best_values = np.empty(len(self.best_positions), dtype=np.float)
        self.global_best_position = None
        self.global_best_value = None

        self.velocities = (np.random.uniform(-1, 1, size=self.positions.shape) *
                           self.max_velocity)

        values = self.fitness(self.positions)

        # Initialize best estimates
        self.best_positions[:] = self.positions
        self.best_values = values

        best_value_id = np.argmax(self.best_values)

        self.global_best_position = np.atleast_2d(self.best_positions[best_value_id, :])
        self.global_best_value = self.best_values[best_value_id]

    def run_swarm(self):
        """Let the swarm explore the parameter space.

        Parameters
        ----------
        max_iter : int
            The number of iterations for which to run the swarm.
        """
        # run the core swarm optimization

        inertia = 0.
        inertia_step = 0.

        for i in range(self.max_iter):
            # update velocities
            delta_global_best = self.global_best_position - self.positions
            delta_self_best = self.best_positions - self.positions

            r = np.random.uniform(-1., 1., size=(2 * self.swarm_size, self.input_dim))
            r1 = r[:self.swarm_size]
            r2 = r[self.swarm_size:]

            if not (self.random_inertia):
                if i == 0:
                    inertia = self.initial_inertia
                    inertia_step = (self.final_inertia - self.initial_inertia) / self.max_iter
                else:
                    inertia += inertia_step
            else:
                inertia = 0.5 + np.random.uniform(-1, 1) / 2.

            self.velocities = self.velocities * inertia \
                              + ((r1 * self.c1 * delta_self_best + r2 * self.c2 * delta_global_best))

            # clip
            np.clip(self.velocities,
                    -self.max_velocity,
                    self.max_velocity,
                    out=self.velocities)

            self.positions += self.velocities

            # Clip particles to domain
            if self.bounds is not None:
                np.clip(self.positions,
                        self.bounds[:, 0],
                        self.bounds[:, 1],
                        out=self.positions)
            values = self.fitness(self.positions)

            # find out which particles are improving

            update_set = values > self.best_values

            if update_set.any() != False:
                self.best_values[update_set] = values[update_set]
                self.best_positions[update_set] = self.positions[update_set]

                best_value_id = np.argmax(self.best_values)

                self.global_best_position = np.atleast_2d(self.best_positions[best_value_id, :])
                self.global_best_value = self.best_values[best_value_id]

            else:
                pass

        return self.global_best_position

    def fitness(self, position):

        result = np.array([np.squeeze(gp.predict_noiseless(position)) for gp in self.gp])

        mean = np.squeeze(np.dot(self.weight, result[:, 0]))
        m = np.atleast_2d(np.copy(mean))
        for i in range(len(self.gp) - 1):
            m = np.vstack([m, np.atleast_2d(mean)])
        var = np.squeeze(np.dot(self.weight, result[:, 1]) + np.dot(self.weight, result[:, 0] - m))
        std_dev = np.sqrt(abs(var))

        upper_bound = np.atleast_1d((mean - np.mean(mean)) + self.beta * std_dev)

        value = np.squeeze(upper_bound)

        return value

