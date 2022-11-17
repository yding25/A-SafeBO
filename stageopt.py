"""
Classes that implement SafeOpt.

Authors: - Felix Berkenkamp (befelix at inf dot ethz dot ch)
         - Nicolas Carion (carion dot nicolas at gmail dot com)
"""

from collections import Sequence
from functools import partial

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from scipy.stats import norm
from builtins import range

from stageopt_swarm import SwarmOptimization


import logging

__all__ = ['StageOptSwarm']

class GaussianProcessOptimization(object):
    """
    Base class for GP optimization.

    Handles common functionality.

    Parameters
    ----------
    gp: GPy Gaussian process
    fmin : float or list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted).
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float or list of floats
        The algorithm will not try to expand any points that are below this
        threshold. This makes the algorithm stop expanding points eventually.
        If a list, this represents the stopping criterion for all the gps.
        This ignores the scaling factor.
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of
        each kernel. You should probably leave this to "auto" unless your
        kernel is non-stationary.
    """

    def __init__(self, gp, fmin, beta=2, num_contexts=0, threshold=0,
                 scaling='auto'):
        """Initialization, see `GaussianProcessOptimization`."""
        super(GaussianProcessOptimization, self).__init__()

        if isinstance(gp, list):
            self.gps = gp
        else:
            self.gps = [gp]
        self.gp = self.gps[0]

        self.fmin = fmin
        if not isinstance(self.fmin, list):
            self.fmin = [self.fmin] * len(self.gps)
        self.fmin = np.atleast_1d(np.asarray(self.fmin).squeeze())

        if hasattr(beta, '__call__'):
            # Beta is a function of t
            self.beta = beta
        else:
            # Assume that beta is a constant
            self.beta = lambda t: beta

        if scaling == 'auto':
            dummy_point = np.zeros((1, self.gps[0].input_dim))
            self.scaling = [gpm.kern.Kdiag(dummy_point)[0] for gpm in self.gps]
            self.scaling = np.sqrt(np.asarray(self.scaling))
        else:
            self.scaling = np.asarray(scaling)
            if self.scaling.shape[0] != len(self.gps):
                raise ValueError("The number of scaling values should be "
                                 "equal to the number of GPs")

        self.threshold = threshold
        self._parameter_set = None
        self.bounds = None
        self.num_samples = 0
        self.num_contexts = num_contexts

        self._x = None
        self._y = None
        self._get_initial_xy()

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def data(self):
        """Return the data within the GP models."""
        return self._x, self._y

    @property
    def t(self):
        """Return the time step (number of measurements)."""
        return self._x.shape[0]

    def _get_initial_xy(self):
        """Get the initial x/y data from the GPs."""
        self._x = self.gp.X
        y = [self.gp.Y]

        for gp in self.gps[1:]:
            if np.allclose(self._x, gp.X):
                y.append(gp.Y)
            else:
                raise NotImplemented('The GPs have different measurements.')

        self._y = np.concatenate(y, axis=1)

    def _add_context(self, x, context):
        """Add the context to a vector.

        Parameters
        ----------
        x : ndarray
        context : ndarray

        Returns
        -------
        x_extended : ndarray
        """
        context = np.atleast_2d(context)
        num_contexts = context.shape[1]

        x2 = np.empty((x.shape[0], x.shape[1] + num_contexts), dtype=float)
        x2[:, :x.shape[1]] = x
        x2[:, x.shape[1]:] = context
        return x2

    def _add_data_point(self, gp, x, y, context=None):
        """Add a data point to a particular GP.

        This should only be called on its own if you know what you're doing.
        This does not update the global data stores self.x and self.y.

        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        context: array_like
            The context(s) used for the data points
        gp: instance of GPy.model.GPRegression
            If specified, determines the GP to which we add the data point
            to. Note that this should only be used if that data point is going
            to be removed again.
        """
        if context is not None:
            x = self._add_context(x, context)

        gp.set_XY(np.vstack([gp.X, x]),
                  np.vstack([gp.Y, y]))

    def add_new_data_point(self, x, y, context=None):
        """
        Add a new function observation to the GPs.

        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        context: array_like
            The context(s) used for the data points.
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        if self.num_contexts:
            x = self._add_context(x, context)

        for i, gp in enumerate(self.gps):
            not_nan = ~np.isnan(y[:, i])
            if np.any(not_nan):
                # Add data to GP (context already included in x)
                self._add_data_point(gp, x[not_nan, :], y[not_nan, [i]])

        # Update global data stores
        self._x = np.concatenate((self._x, x), axis=0)
        self._y = np.concatenate((self._y, y), axis=0)

    def _remove_last_data_point(self, gp):
        """Remove the last data point of a specific GP.

        This does not update global data stores, self.x and self.y.

        Parameters
        ----------
            gp: Instance of GPy.models.GPRegression
                The gp that the last data point should be removed from
        """
        gp.set_XY(gp.X[:-1, :], gp.Y[:-1, :])

    def remove_last_data_point(self):
        """Remove the data point that was last added to the GP."""
        last_y = self._y[-1]

        for gp, yi in zip(self.gps, last_y):
            if not np.isnan(yi):
                gp.set_XY(gp.X[:-1, :], gp.Y[:-1, :])

        self._x = self._x[:-1, :]
        self._y = self._y[:-1, :]

class StageOptSwarm(GaussianProcessOptimization):

    def __init__(self, gp, fmin, bounds, beta=2, scaling='auto', threshold=0,
                 swarm_size=20):
        """Initialization, see `SafeOptSwarm`."""
        super(StageOptSwarm, self).__init__(gp,
                                            fmin=fmin,
                                            beta=beta,
                                            num_contexts=0,
                                            threshold=threshold,
                                            scaling=scaling)

        # Safe set
        self.S = np.asarray(self.gps[0].X)

        self.swarm_size = swarm_size
        self.max_iters = 100  # number of swarm iterations

        if not isinstance(bounds, list):
            self.bounds = [bounds] * self.S.shape[1]
        else:
            self.bounds = bounds

        # These are estimates of the best lower bound, and its location
        self.best_lower_bound = -np.inf
        self.greedy_point = self.S[0, :]

        self.optimal_velocities = self.optimize_particle_velocity()

        swarm_types = ['greedy', 'maximizers', 'expanders']

        self.swarms = {swarm_type:
                       SwarmOptimization(
                           swarm_size,
                           self.optimal_velocities,
                           partial(self._compute_particle_fitness,
                                   swarm_type),
                           bounds=self.bounds)
                       for swarm_type in swarm_types}

    def optimize_particle_velocity(self):
        """Optimize the velocities of the particles.

        Note that this only works well for stationary kernels and constant mean
        functions. Otherwise the velocity depends on the position!

        Returns
        -------
        velocities: ndarray
            The estimated optimal velocities in each direction.
        """
        parameters = np.zeros((1, self.gp.input_dim), dtype=np.float)
        velocities = np.empty((len(self.gps), self.gp.input_dim),
                              dtype=np.float)

        for i, gp in enumerate(self.gps):
            for j in range(self.gp.input_dim):
                tmp_velocities = np.zeros((1, self.gp.input_dim),
                                          dtype=np.float)

                # lower and upper bounds on velocities
                upper_velocity = 1000.
                lower_velocity = 0.

                # Binary search over optimal velocities
                while True:
                    mid = (upper_velocity + lower_velocity) / 2
                    tmp_velocities[0, j] = mid

                    kernel_matrix = gp.kern.K(parameters, tmp_velocities)
                    covariance = kernel_matrix.squeeze() / self.scaling[i] ** 2

                    # Make sure the correlation is in the sweet spot
                    velocity_enough = covariance > 0.94
                    not_too_fast = covariance < 0.95

                    if not_too_fast:
                        upper_velocity = mid
                    elif velocity_enough:
                        lower_velocity = mid

                    if ((not_too_fast and velocity_enough) or
                            upper_velocity - lower_velocity < 1e-5):
                        break

                # Store optimal velocity
                velocities[i, j] = mid

        # Select the minimal velocity (for the toughest safety constraint)
        velocities = np.min(velocities, axis=0)

        # Scale for number of parameters (this might not be so clever if they
        # are all independent, additive kernels).
        velocities /= np.sqrt(self.gp.input_dim)
        return velocities

    def _compute_penalty(self, slack):
        """Return the penalty associated to a constraint violation.

        The penalty is a piecewise linear function that is nonzero only if the
        safety constraints are violated. This penalty encourages particles to
        stay within the safe set.

        Parameters
        ----------
        slack: ndarray
            A vector corresponding to how much the constraint was violated.

        Returns
        -------
        penalties - ndarray
            The value of the penalties
        """
        penalties = np.atleast_1d(np.clip(slack, None, 0))

        penalties[(slack < 0) & (slack > -0.001)] *= 2
        penalties[(slack <= -0.001) & (slack > -0.1)] *= 5
        penalties[(slack <= -0.1) & (slack > -1)] *= 10

        slack_id = slack < -1
        penalties[slack_id] = -300 * penalties[slack_id] ** 2
        return penalties

    def _compute_particle_fitness(self, swarm_type, particles):
        """
        Return the value of the particles and the safety information.

        Parameters
        ----------
        particles : ndarray
            A vector containing the coordinates of the particles
        swarm_type : string
            A string corresponding to the swarm type. It can be any of the
            following strings:

                * 'greedy' : Optimal value(best lower bound).
                * 'expander' : Expanders (lower bound close to constraint)
                * 'maximizer' : Maximizers (Upper bound better than best l)
                * 'safe_set' : Only check the safety of the particles
        Returns
        -------
        values : ndarray
            The values of the particles
        global_safe : ndarray
            A boolean mask indicating safety status of all particles
            (note that in the case of a greedy swarm, this is not computed and
            we return a True mask)
        """
        beta = self.beta(self.t)

        # classify the particle's function values
        mean, var = self.gps[0].predict_noiseless(particles)
        mean = mean.squeeze()
        std_dev = np.sqrt(var.squeeze())

        # compute the confidence interval
        lower_bound = np.atleast_1d(mean - beta * std_dev)
        upper_bound = np.atleast_1d(mean + beta * std_dev)

        # the greedy swarm optimizes for the lower bound
        if swarm_type == 'greedy':
            return lower_bound, np.broadcast_to(True, len(lower_bound))

        # value we are optimizing for. Expanders and maximizers seek high
        # variance points
        values = std_dev / self.scaling[0]

        #
        is_safe = swarm_type == 'safe_set'
        is_expander = swarm_type == 'expanders'
        is_maximizer = swarm_type == 'maximizers'

        if is_safe:
            interest_function = None
        else:
            if is_expander:
                # For expanders, the interest function is updated depending on
                # the lower bounds
                interest_function = (len(self.gps) *
                                     np.ones(np.shape(values), dtype=np.float))
            elif is_maximizer:
                improvement = upper_bound - self.best_lower_bound
                interest_function = expit(10 * improvement / self.scaling[0])
            else:
                # unknown particle type (shouldn't happen)
                raise AssertionError("Invalid swarm type")

        # boolean mask that tell if the particles are safe according to all gps
        global_safe = np.ones(particles.shape[0], dtype=np.bool)
        total_penalty = np.zeros(particles.shape[0], dtype=np.float)

        for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
            # Only recompute confidence intervals for constraints
            if i > 0:
                # classify using the current GP
                mean, var = gp.predict_noiseless(particles)
                mean = mean.squeeze()
                std_dev = np.sqrt(var.squeeze())
                lower_bound = mean - beta * std_dev

                values = np.maximum(values, std_dev / scaling)

            # if the current GP has no safety constrain, we skip it
            if self.fmin[i] == -np.inf:
                continue

            slack = np.atleast_1d(lower_bound - self.fmin[i])

            # computing penalties
            global_safe &= slack >= 0

            # Skip cost update for safety evaluation
            if is_safe:
                continue

            # Normalize the slack somewhat
            slack /= scaling

            total_penalty += self._compute_penalty(slack)

            if is_expander:
                # check if the particles are expanders for the current gp
                interest_function *= norm.pdf(slack, scale=0.2)

        # this swarm type is only interested in knowing whether the particles
        # are safe.
        if is_safe:
            return lower_bound, global_safe

        # add penalty
        values += total_penalty

        # apply the mask for current interest function
        values *= interest_function

        return values, global_safe

    def get_new_query_point(self, swarm_type):
        """
        Compute a new point at which to evaluate the function.

        This function relies on a Particle Swarm Optimization (PSO) to find the
        optimum of the objective function (which depends on the swarm type).

        Parameters
        ----------
        swarm_type: string
            This parameter controls the type of point that should be found. It
            can take one of the following values:

                * 'expanders' : find a point that increases the safe set
                * 'maximizers' : find a point that maximizes the objective
                                 function within the safe set.
                * 'greedy' : retrieve an estimate of the best currently known
                             parameters (best lower bound).

        Returns
        -------
        global_best: np.array
            The next parameters that should be evaluated.
        max_std_dev: float
            The current standard deviation in the point to be evaluated.
        """
        beta = self.beta(self.t)
        safe_size, input_dim = self.S.shape

        # Make sure the safe set is still safe
        _, safe = self._compute_particle_fitness('safe_set', self.S)

        num_safe = safe.sum()
        if num_safe == 0:
            raise RuntimeError('The safe set is empty.')

        # Prune safe set if points in the discrete approximation of the safe
        # ended up being unsafe, but never prune below swarm size to avoid
        # empty safe set.
        if num_safe >= self.swarm_size and num_safe != len(safe):
            # Warn that the safe set has decreased
            logging.warning("Warning: {} unsafe points removed. "
                            "Model might be violated"
                            .format(np.count_nonzero(~safe)))

            # Remove unsafe points
            self.S = self.S[safe]
            safe_size = self.S.shape[0]

        # initialize particles
        if swarm_type == 'greedy':
            # we pick particles u.a.r in the safe set
            random_id = np.random.randint(safe_size, size=self.swarm_size - 3)
            best_sampled_point = np.argmax(self.gp.Y)

            # Particles are drawn at random from the safe set, but include the
            # - Previous greedy estimate
            # - last point
            # - best sampled point
            particles = np.vstack((self.S[random_id, :],
                                   self.greedy_point,
                                   self.gp.X[-1, :],
                                   self.gp.X[best_sampled_point]))
        else:
            # we pick particles u.a.r in the safe set
            random_id = np.random.randint(safe_size, size=self.swarm_size)
            particles = self.S[random_id, :]

        # Run the swarm optimization
        swarm = self.swarms[swarm_type]
        swarm.init_swarm(particles)
        swarm.run_swarm(self.max_iters)

        # expand safe set
        if swarm_type != 'greedy':
            num_added = 0

            # compute correlation between new candidates and current safe set
            covariance = self.gp.kern.K(swarm.best_positions,
                                        np.vstack((self.S,
                                                   swarm.best_positions)))
            covariance /= self.scaling[0] ** 2

            initial_safe = len(self.S)
            n, m = np.shape(covariance)

            # this mask keeps track of the points that we have added in the
            # safe set to account for them when adding a new point
            mask = np.zeros(m, dtype=np.bool)
            mask[:initial_safe] = True

            for j in range(n):
                # make sure correlation with old points is relatively low
                if np.all(covariance[j, mask] <= 0.95):
                    self.S = np.vstack((self.S, swarm.best_positions[[j], :]))
                    num_added += 1
                    mask[initial_safe + j] = True

            logging.debug("At the end of swarm {}, {} points were appended to"
                          " the safeset".format(swarm_type, num_added))
        else:
            # check whether we found a better estimate of the lower bound
            mean, var = self.gp.predict_noiseless(self.greedy_point[None, :])
            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())

            lower_bound = mean - beta * std_dev
            if lower_bound < np.max(swarm.best_values):
                self.greedy_point = swarm.global_best.copy()

        if swarm_type == 'greedy':
            return swarm.global_best.copy(), np.max(swarm.best_values)

        # compute the variance of the point picked
        var = np.empty(len(self.gps), dtype=np.float)
        # max_std_dev = 0.
        for i, (gp, scaling) in enumerate(zip(self.gps, self.scaling)):
            var[i] = gp.predict_noiseless(swarm.global_best[None, :])[1]

        return swarm.global_best, np.sqrt(var)

    def optimize(self, expand=True):
        """Run Safe Bayesian optimization and get the next parameters.

        Parameters
        ----------
        ucb: bool
            Whether to only compute maximizers (best upper bound).

        Returns
        -------
        x: np.array
            The next parameters that should be evaluated.
        """
        # compute estimate of the lower bound
        self.greedy, self.best_lower_bound = self.get_new_query_point('greedy')
        if not(expand):
            x_maxi, std_maxi = self.get_new_query_point('maximizers')
            return x_maxi
        else:
            x_exp, std_exp = self.get_new_query_point('expanders')
            return x_exp

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
        maxi = np.argmax(self.gp.Y)
        return self.gp.X[maxi, :], self.gp.Y[maxi]
