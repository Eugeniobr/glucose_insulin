from dataclasses import asdict, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares

from model import UltradianParams, _rk4_step, build_insulin_depot_input, build_meal_input, f1, f2, f3, f4, glucose_mgdl_to_total_mg, total_mg_to_glucose_mgdl


class LatentStateBootstrap:
    def __init__(self, t_data, glucose_data_mg_dl, params: UltradianParams, smoothing_factor: float):
        self.t_data = np.asarray(t_data, dtype=float)
        self.glucose_data_mg_dl = np.asarray(glucose_data_mg_dl, dtype=float)
        self.params = params
        smoothing = max(len(self.t_data) * smoothing_factor, 1.0)
        self.glucose_spline = UnivariateSpline(self.t_data, self.glucose_data_mg_dl, s=smoothing, k=3)

    def glucose_total(self, t):
        return glucose_mgdl_to_total_mg(self.glucose_spline(t), self.params)

    def rhs(self, t, state):
        ip, ii, h1, h2, h3 = state
        glucose_total = self.glucose_total(t)
        dip = f1(glucose_total, self.params) - ((ip / self.params.Vp) - (ii / self.params.Vi)) * self.params.E - ip / self.params.tp
        dii = ((ip / self.params.Vp) - (ii / self.params.Vi)) * self.params.E - ii / self.params.ti
        dh1 = (ip - h1) / self.params.td
        dh2 = (h1 - h2) / self.params.td
        dh3 = (h2 - h3) / self.params.td
        return [dip, dii, dh1, dh2, dh3]

    def simulate(self, initial_state):
        solution = solve_ivp(
            fun=lambda t, state: self.rhs(t, state),
            t_span=(self.t_data[0], self.t_data[-1]),
            y0=np.asarray(initial_state, dtype=float),
            t_eval=self.t_data,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution

    def residuals(self, initial_state):
        solution = self.simulate(initial_state)
        glucose_total = self.glucose_total(solution.t)
        derivative_total = 10.0 * self.params.Vg * self.glucose_spline.derivative()(solution.t)
        model_derivative = f4(solution.y[4], self.params) - f2(glucose_total, self.params) - f3(solution.y[1], self.params) * glucose_total
        penalty = 5e-5 * np.asarray(initial_state, dtype=float)
        return np.concatenate([derivative_total - model_derivative, penalty])

    def fit(self) -> np.ndarray:
        initial_guess = np.array([120.0, 700.0, 120.0, 120.0, 120.0], dtype=float)
        bounds = (np.zeros(5), np.full(5, 1e5))
        result = least_squares(
            fun=self.residuals,
            x0=initial_guess,
            bounds=bounds,
            method="trf",
            max_nfev=200,
        )
        return np.asarray(result.x, dtype=float)


class GlucoseDrivenInverseModel:
    def __init__(self, t_data, glucose_data_mg_dl, params: UltradianParams, smoothing_factor: float):
        self.t_data = np.asarray(t_data, dtype=float)
        self.glucose_data_mg_dl = np.asarray(glucose_data_mg_dl, dtype=float)
        self.params = params

        smoothing = max(len(self.t_data) * smoothing_factor, 1.0)
        self.glucose_spline = UnivariateSpline(self.t_data, self.glucose_data_mg_dl, s=smoothing, k=3)

    def glucose_total(self, t):
        return glucose_mgdl_to_total_mg(self.glucose_spline(t), self.params)

    def dglucose_total_dt(self, t):
        return 10.0 * self.params.Vg * self.glucose_spline.derivative()(t)

    def rhs(self, t, state):
        ip, ii, h1, h2, h3 = state
        glucose_total = self.glucose_total(t)
        dip = f1(glucose_total, self.params) - ((ip / self.params.Vp) - (ii / self.params.Vi)) * self.params.E - ip / self.params.tp
        dii = ((ip / self.params.Vp) - (ii / self.params.Vi)) * self.params.E - ii / self.params.ti
        dh1 = (ip - h1) / self.params.td
        dh2 = (h1 - h2) / self.params.td
        dh3 = (h2 - h3) / self.params.td
        return [dip, dii, dh1, dh2, dh3]

    def simulate(self, initial_state):
        solution = solve_ivp(
            fun=lambda t, state: self.rhs(t, state),
            t_span=(self.t_data[0], self.t_data[-1]),
            y0=np.asarray(initial_state, dtype=float),
            t_eval=self.t_data,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution

    def residuals(self, initial_state):
        solution = self.simulate(initial_state)
        ii = solution.y[1]
        h3 = solution.y[4]
        glucose_total = self.glucose_total(solution.t)
        observed_derivative = self.dglucose_total_dt(solution.t)
        model_derivative = f4(h3, self.params) - f2(glucose_total, self.params) - f3(ii, self.params) * glucose_total
        penalty = 1e-4 * np.asarray(initial_state, dtype=float)
        return np.concatenate([observed_derivative - model_derivative, penalty])

    def fit(self):
        initial_guess = np.array([100.0, 800.0, 100.0, 100.0, 100.0], dtype=float)
        bounds = (np.zeros(5), np.full(5, 1e5))
        result = least_squares(
            fun=self.residuals,
            x0=initial_guess,
            bounds=bounds,
            method="trf",
            max_nfev=500,
        )
        solution = self.simulate(result.x)
        glucose_total = self.glucose_total(solution.t)
        inferred_input = self.dglucose_total_dt(solution.t) - (
            f4(solution.y[4], self.params) - f2(glucose_total, self.params) - f3(solution.y[1], self.params) * glucose_total
        )
        return {
            "method": "legacy",
            "x0_est": result.x,
            "t": solution.t,
            "Ip": solution.y[0],
            "Ii": solution.y[1],
            "h1": solution.y[2],
            "h2": solution.y[3],
            "h3": solution.y[4],
            "states": np.column_stack(
                [
                    np.zeros_like(solution.t, dtype=float),
                    solution.y[0],
                    solution.y[1],
                    glucose_total,
                    solution.y[2],
                    solution.y[3],
                    solution.y[4],
                ]
            ),
            "glucose_smooth_mg_dl": self.glucose_spline(solution.t),
            "predicted_glucose_mg_dl": self.glucose_spline(solution.t),
            "innovations_mg_dl": np.zeros_like(solution.t, dtype=float),
            "innovation_variances": np.zeros_like(solution.t, dtype=float),
            "inferred_glucose_input": inferred_input,
            "optimization_cost": float(result.cost),
            "optimization_success": bool(result.success),
        }


class GlucoseStateUKF:
    def __init__(
        self,
        t_data,
        glucose_data_mg_dl,
        meal_events,
        insulin_events,
        params: UltradianParams,
        smoothing_factor: float,
        ukf_settings: dict | None = None,
    ):
        self.t_data = np.asarray(t_data, dtype=float)
        self.glucose_data_mg_dl = np.asarray(glucose_data_mg_dl, dtype=float)
        self.meal_events = meal_events
        self.insulin_events = insulin_events
        self.params = params

        smoothing = max(len(self.t_data) * smoothing_factor, 1.0)
        self.glucose_spline = UnivariateSpline(self.t_data, self.glucose_data_mg_dl, s=smoothing, k=3)

        ukf_settings = ukf_settings or {}
        self.state_dim = 7
        self.measurement_dim = 1
        self.alpha = float(ukf_settings.get("alpha", 0.15))
        self.beta = float(ukf_settings.get("beta", 2.0))
        self.kappa = float(ukf_settings.get("kappa", 0.0))
        self.lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        self.gamma = np.sqrt(self.state_dim + self.lambda_)

        self.weight_mean = np.full(2 * self.state_dim + 1, 1.0 / (2.0 * (self.state_dim + self.lambda_)), dtype=float)
        self.weight_cov = self.weight_mean.copy()
        self.weight_mean[0] = self.lambda_ / (self.state_dim + self.lambda_)
        self.weight_cov[0] = self.weight_mean[0] + (1.0 - self.alpha**2 + self.beta)

        q_s = float(ukf_settings.get("q_s", 8_000.0))
        q_ip = float(ukf_settings.get("q_ip", 2_500.0))
        q_ii = float(ukf_settings.get("q_ii", 2_500.0))
        q_g = float(ukf_settings.get("q_g", 18_000.0))
        q_h = float(ukf_settings.get("q_h", 3_500.0))
        measurement_noise_mgdl = float(ukf_settings.get("measurement_noise_mgdl", 6.0))
        self.process_noise = np.diag([q_s, q_ip, q_ii, q_g, q_h, q_h, q_h])
        self.measurement_noise = np.array([[measurement_noise_mgdl**2]], dtype=float)
        self.settings = {
            "alpha": self.alpha,
            "beta": self.beta,
            "kappa": self.kappa,
            "measurement_noise_mgdl": measurement_noise_mgdl,
            "q_s": q_s,
            "q_ip": q_ip,
            "q_ii": q_ii,
            "q_g": q_g,
            "q_h": q_h,
        }

    def glucose_total(self, t):
        return glucose_mgdl_to_total_mg(self.glucose_spline(t), self.params)

    def dglucose_total_dt(self, t):
        return 10.0 * self.params.Vg * self.glucose_spline.derivative()(t)

    def _initial_state(self, bootstrap_state: np.ndarray | None) -> np.ndarray:
        glucose_total_initial = glucose_mgdl_to_total_mg(self.glucose_data_mg_dl[0], self.params)
        if bootstrap_state is None:
            basal_remote = self.params.C4 * self.params.Vi
            basal_delayed = self.params.C5 * self.params.Vp
            ip0, ii0, h10, h20, h30 = 90.0, basal_remote, 90.0, 90.0, basal_delayed
        else:
            ip0, ii0, h10, h20, h30 = [float(value) for value in bootstrap_state]
        return np.array(
            [
                0.0,
                ip0,
                ii0,
                glucose_total_initial,
                h10,
                h20,
                h30,
            ],
            dtype=float,
        )

    def _initial_covariance(self) -> np.ndarray:
        glucose_variance = (12.0 * self.params.Vg * 10.0) ** 2
        return np.diag([50_000.0, 12_000.0, 12_000.0, glucose_variance, 15_000.0, 15_000.0, 15_000.0]).astype(float)

    def _sigma_points(self, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        jitter = 1e-6
        adjusted = covariance.copy()
        while True:
            try:
                chol = np.linalg.cholesky(adjusted)
                break
            except np.linalg.LinAlgError:
                adjusted = adjusted + np.eye(self.state_dim) * jitter
                jitter *= 10.0

        sigma = np.zeros((2 * self.state_dim + 1, self.state_dim), dtype=float)
        sigma[0] = mean
        for index in range(self.state_dim):
            delta = self.gamma * chol[:, index]
            sigma[index + 1] = mean + delta
            sigma[self.state_dim + index + 1] = mean - delta
        return sigma

    def _measurement_model(self, state: np.ndarray) -> np.ndarray:
        return np.array([total_mg_to_glucose_mgdl(state[3], self.params)], dtype=float)

    def _interval_grid(self, start_time: float, end_time: float) -> np.ndarray:
        if end_time <= start_time:
            return np.array([start_time, end_time], dtype=float)

        dt = min(1.0, max(end_time - start_time, 1e-6))
        base_grid = np.arange(start_time, end_time + dt, dt)
        insulin_times = np.array(
            [event.minutes for event in self.insulin_events if start_time <= event.minutes <= end_time],
            dtype=float,
        )
        grid = np.unique(np.concatenate([base_grid, insulin_times, np.array([start_time, end_time])]))
        return grid

    def _propagate_state(self, state: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
        grid = self._interval_grid(start_time, end_time)

        propagated = np.asarray(state, dtype=float).copy()
        meal_input = build_meal_input(self.meal_events, self.params, disturbance=None)
        insulin_depot_input = build_insulin_depot_input(self.insulin_events, self.params)
        for index, current_time in enumerate(grid):
            if index == len(grid) - 1:
                continue
            next_time = float(grid[index + 1])
            propagated = _rk4_step(
                float(current_time),
                propagated,
                next_time - float(current_time),
                self.params,
                meal_input,
                insulin_depot_input,
            )

        return np.maximum(propagated, 0.0)

    def _predict(self, mean: np.ndarray, covariance: np.ndarray, start_time: float, end_time: float):
        sigma_points = self._sigma_points(mean, covariance)
        propagated_sigma = np.array(
            [self._propagate_state(point, start_time, end_time) for point in sigma_points],
            dtype=float,
        )
        predicted_mean = np.sum(self.weight_mean[:, None] * propagated_sigma, axis=0)
        centered = propagated_sigma - predicted_mean
        predicted_covariance = self.process_noise.copy()
        for index, point in enumerate(centered):
            predicted_covariance += self.weight_cov[index] * np.outer(point, point)
        predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
        return propagated_sigma, predicted_mean, predicted_covariance

    def _update(self, propagated_sigma: np.ndarray, predicted_mean: np.ndarray, predicted_covariance: np.ndarray, measurement_mg_dl: float):
        measurement_sigma = np.array([self._measurement_model(point) for point in propagated_sigma], dtype=float)
        predicted_measurement = np.sum(self.weight_mean[:, None] * measurement_sigma, axis=0)

        innovation_covariance = self.measurement_noise.copy()
        cross_covariance = np.zeros((self.state_dim, self.measurement_dim), dtype=float)
        for index in range(len(propagated_sigma)):
            dz = measurement_sigma[index] - predicted_measurement
            dx = propagated_sigma[index] - predicted_mean
            innovation_covariance += self.weight_cov[index] * np.outer(dz, dz)
            cross_covariance += self.weight_cov[index] * np.outer(dx, dz)

        kalman_gain = cross_covariance @ np.linalg.inv(innovation_covariance)
        innovation = np.array([measurement_mg_dl], dtype=float) - predicted_measurement
        updated_mean = predicted_mean + kalman_gain @ innovation
        updated_covariance = predicted_covariance - kalman_gain @ innovation_covariance @ kalman_gain.T
        updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
        updated_mean = np.maximum(updated_mean, 0.0)
        updated_mean[3] = glucose_mgdl_to_total_mg(measurement_mg_dl, self.params)
        updated_covariance += np.eye(self.state_dim) * 1e-6
        return updated_mean, updated_covariance, predicted_measurement, innovation_covariance

    def run(self, bootstrap_state: np.ndarray | None) -> dict:
        state_mean = self._initial_state(bootstrap_state=bootstrap_state)
        state_covariance = self._initial_covariance()

        filtered_states = np.zeros((len(self.t_data), self.state_dim), dtype=float)
        predicted_glucose = np.zeros(len(self.t_data), dtype=float)
        innovations = np.zeros(len(self.t_data), dtype=float)
        innovation_variances = np.zeros(len(self.t_data), dtype=float)

        filtered_states[0] = state_mean
        predicted_glucose[0] = self.glucose_data_mg_dl[0]
        innovations[0] = 0.0
        innovation_variances[0] = float(self.measurement_noise[0, 0])

        for index in range(1, len(self.t_data)):
            propagated_sigma, predicted_mean, predicted_covariance = self._predict(
                mean=state_mean,
                covariance=state_covariance,
                start_time=float(self.t_data[index - 1]),
                end_time=float(self.t_data[index]),
            )
            updated_mean, updated_covariance, predicted_measurement, innovation_covariance = self._update(
                propagated_sigma=propagated_sigma,
                predicted_mean=predicted_mean,
                predicted_covariance=predicted_covariance,
                measurement_mg_dl=float(self.glucose_data_mg_dl[index]),
            )
            state_mean = updated_mean
            state_covariance = updated_covariance
            filtered_states[index] = state_mean
            predicted_glucose[index] = float(predicted_measurement[0])
            innovations[index] = float(self.glucose_data_mg_dl[index] - predicted_measurement[0])
            innovation_variances[index] = float(innovation_covariance[0, 0])

        glucose_total_smooth = self.glucose_total(self.t_data)
        inferred_input = self.dglucose_total_dt(self.t_data) - (
            f4(filtered_states[:, 6], self.params)
            - f2(glucose_total_smooth, self.params)
            - f3(filtered_states[:, 2], self.params) * glucose_total_smooth
        )
        return {
            "method": "ukf",
            "t": self.t_data,
            "states": filtered_states,
            "predicted_glucose_mg_dl": predicted_glucose,
            "innovations_mg_dl": innovations,
            "innovation_variances": innovation_variances,
            "glucose_smooth_mg_dl": self.glucose_spline(self.t_data),
            "inferred_glucose_input": inferred_input,
        }


class GlucoseStateParticleFilter:
    PARAMETER_NAMES = (
        "insulin_sensitivity_scale",
        "meal_tau",
        "insulin_rapid_bio_scale",
        "insulin_basal_bio_scale",
    )

    PARAMETER_BOUNDS = {
        "insulin_sensitivity_scale": (0.5, 1.8),
        "meal_tau": (20.0, 90.0),
        "insulin_rapid_bio_scale": (0.7, 1.3),
        "insulin_basal_bio_scale": (0.7, 1.3),
    }

    def __init__(self, t_data, glucose_data_mg_dl, meal_events, insulin_events, params: UltradianParams, smoothing_factor: float, settings: dict | None = None):
        self.t_data = np.asarray(t_data, dtype=float)
        self.glucose_data_mg_dl = np.asarray(glucose_data_mg_dl, dtype=float)
        self.meal_events = meal_events
        self.insulin_events = insulin_events
        self.base_params = params
        smoothing = max(len(self.t_data) * smoothing_factor, 1.0)
        self.glucose_spline = UnivariateSpline(self.t_data, self.glucose_data_mg_dl, s=smoothing, k=3)
        settings = settings or {}
        self.particle_count = int(settings.get("particle_count", 80))
        self.measurement_noise_mgdl = float(settings.get("measurement_noise_mgdl", 12.0))
        self.state_noise_scale = float(settings.get("state_noise_scale", 0.03))
        self.param_noise_scale = float(settings.get("param_noise_scale", 0.015))
        self.settings = {
            "particle_count": self.particle_count,
            "measurement_noise_mgdl": self.measurement_noise_mgdl,
            "state_noise_scale": self.state_noise_scale,
            "param_noise_scale": self.param_noise_scale,
        }

    def dglucose_total_dt(self, t):
        return 10.0 * self.base_params.Vg * self.glucose_spline.derivative()(t)

    def _initial_particles(self, bootstrap_state: np.ndarray):
        state_center = np.array(
            [0.0, bootstrap_state[0], bootstrap_state[1], glucose_mgdl_to_total_mg(self.glucose_data_mg_dl[0], self.base_params), bootstrap_state[2], bootstrap_state[3], bootstrap_state[4]],
            dtype=float,
        )
        state_scales = np.array([80.0, 120.0, 180.0, 80.0 * self.base_params.Vg * 10.0, 120.0, 120.0, 120.0], dtype=float)
        particles = state_center + np.random.normal(scale=state_scales, size=(self.particle_count, 7))
        particles = np.maximum(particles, 0.0)
        parameter_center = np.array([float(getattr(self.base_params, name)) for name in self.PARAMETER_NAMES], dtype=float)
        parameter_scales = np.array([0.08, 8.0, 0.06, 0.06], dtype=float)
        parameter_particles = parameter_center + np.random.normal(scale=parameter_scales, size=(self.particle_count, len(self.PARAMETER_NAMES)))
        for index, name in enumerate(self.PARAMETER_NAMES):
            lower, upper = self.PARAMETER_BOUNDS[name]
            parameter_particles[:, index] = np.clip(parameter_particles[:, index], lower, upper)
        weights = np.full(self.particle_count, 1.0 / self.particle_count, dtype=float)
        return particles, parameter_particles, weights

    def _params_from_vector(self, vector: np.ndarray) -> UltradianParams:
        updates = {name: float(vector[index]) for index, name in enumerate(self.PARAMETER_NAMES)}
        return replace(self.base_params, **updates)

    def _interval_grid(self, start_time: float, end_time: float) -> np.ndarray:
        if end_time <= start_time:
            return np.array([start_time, end_time], dtype=float)
        dt = min(1.0, max(end_time - start_time, 1e-6))
        base_grid = np.arange(start_time, end_time + dt, dt)
        insulin_times = np.array([event.minutes for event in self.insulin_events if start_time <= event.minutes <= end_time], dtype=float)
        return np.unique(np.concatenate([base_grid, insulin_times, np.array([start_time, end_time])]))

    def _propagate_single(self, state: np.ndarray, params: UltradianParams, start_time: float, end_time: float) -> np.ndarray:
        grid = self._interval_grid(start_time, end_time)
        propagated = np.asarray(state, dtype=float).copy()
        meal_input = build_meal_input(self.meal_events, params, disturbance=None)
        insulin_depot_input = build_insulin_depot_input(self.insulin_events, params)
        for index, current_time in enumerate(grid[:-1]):
            next_time = float(grid[index + 1])
            propagated = _rk4_step(float(current_time), propagated, next_time - float(current_time), params, meal_input, insulin_depot_input)
        return np.maximum(propagated, 0.0)

    def _resample(self, particles: np.ndarray, parameter_particles: np.ndarray, weights: np.ndarray):
        cumulative = np.cumsum(weights)
        step = 1.0 / len(weights)
        start = np.random.uniform(0.0, step)
        positions = start + step * np.arange(len(weights))
        indexes = np.searchsorted(cumulative, positions)
        return particles[indexes], parameter_particles[indexes], np.full(len(weights), 1.0 / len(weights), dtype=float)

    def run(self, bootstrap_state: np.ndarray):
        particles, parameter_particles, weights = self._initial_particles(bootstrap_state)
        filtered_states = np.zeros((len(self.t_data), 7), dtype=float)
        predicted_glucose = np.zeros(len(self.t_data), dtype=float)
        innovations = np.zeros(len(self.t_data), dtype=float)
        innovation_variances = np.zeros(len(self.t_data), dtype=float)
        posterior_params = np.zeros((len(self.t_data), len(self.PARAMETER_NAMES)), dtype=float)

        filtered_states[0] = np.average(particles, axis=0, weights=weights)
        filtered_states[0, 3] = glucose_mgdl_to_total_mg(self.glucose_data_mg_dl[0], self.base_params)
        predicted_glucose[0] = self.glucose_data_mg_dl[0]
        innovation_variances[0] = self.measurement_noise_mgdl**2
        posterior_params[0] = np.average(parameter_particles, axis=0, weights=weights)

        for index in range(1, len(self.t_data)):
            start_time = float(self.t_data[index - 1])
            end_time = float(self.t_data[index])
            for particle_index in range(self.particle_count):
                parameter_particles[particle_index] += np.random.normal(
                    scale=np.array([0.02, 1.0, 0.01, 0.01], dtype=float) * self.param_noise_scale,
                    size=len(self.PARAMETER_NAMES),
                )
                for param_index, name in enumerate(self.PARAMETER_NAMES):
                    lower, upper = self.PARAMETER_BOUNDS[name]
                    parameter_particles[particle_index, param_index] = np.clip(parameter_particles[particle_index, param_index], lower, upper)
                particle_params = self._params_from_vector(parameter_particles[particle_index])
                particles[particle_index] = self._propagate_single(particles[particle_index], particle_params, start_time, end_time)
                particles[particle_index] += np.random.normal(
                    scale=np.maximum(np.abs(particles[particle_index]) * self.state_noise_scale, 1.0),
                    size=particles.shape[1],
                )
                particles[particle_index] = np.maximum(particles[particle_index], 0.0)

            predicted_measurements = total_mg_to_glucose_mgdl(particles[:, 3], self.base_params)
            residuals = self.glucose_data_mg_dl[index] - predicted_measurements
            log_weights = -0.5 * (residuals / self.measurement_noise_mgdl) ** 2
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            weights /= np.sum(weights)

            predicted_glucose[index] = float(np.average(predicted_measurements, weights=weights))
            innovations[index] = float(self.glucose_data_mg_dl[index] - predicted_glucose[index])
            innovation_variances[index] = float(np.average((predicted_measurements - predicted_glucose[index]) ** 2, weights=weights) + self.measurement_noise_mgdl**2)
            filtered_states[index] = np.average(particles, axis=0, weights=weights)
            filtered_states[index, 3] = glucose_mgdl_to_total_mg(self.glucose_data_mg_dl[index], self.base_params)
            posterior_params[index] = np.average(parameter_particles, axis=0, weights=weights)

            effective_sample_size = 1.0 / np.sum(weights**2)
            if effective_sample_size < 0.5 * self.particle_count:
                particles, parameter_particles, weights = self._resample(particles, parameter_particles, weights)

        final_param_vector = posterior_params[-1]
        final_params = self._params_from_vector(final_param_vector)
        inferred_input = self.dglucose_total_dt(self.t_data) - (
            f4(filtered_states[:, 6], final_params)
            - f2(filtered_states[:, 3], final_params)
            - f3(filtered_states[:, 2], final_params) * filtered_states[:, 3]
        )
        adapted_params = {name: float(final_param_vector[index]) for index, name in enumerate(self.PARAMETER_NAMES)}
        return {
            "method": "bayes",
            "t": self.t_data,
            "states": filtered_states,
            "predicted_glucose_mg_dl": predicted_glucose,
            "innovations_mg_dl": innovations,
            "innovation_variances": innovation_variances,
            "glucose_smooth_mg_dl": self.glucose_spline(self.t_data),
            "inferred_glucose_input": inferred_input,
            "adapted_params": adapted_params,
            "posterior_params": posterior_params,
        }


def estimate_initial_state(
    t_data,
    glucose_data_mg_dl,
    params: UltradianParams,
    smoothing_factor: float,
    meal_events,
    insulin_events,
    estimator_mode: str = "legacy",
    ukf_settings: dict | None = None,
    bayes_settings: dict | None = None,
):
    estimator_mode = (estimator_mode or "legacy").strip().lower()
    if estimator_mode == "legacy":
        model = GlucoseDrivenInverseModel(
            t_data=t_data,
            glucose_data_mg_dl=glucose_data_mg_dl,
            params=params,
            smoothing_factor=smoothing_factor,
        )
        result = model.fit()
        glucose_total_initial = glucose_mgdl_to_total_mg(glucose_data_mg_dl[0], params)
        initial_state = np.array(
            [0.0, result["x0_est"][0], result["x0_est"][1], glucose_total_initial, result["x0_est"][2], result["x0_est"][3], result["x0_est"][4]],
            dtype=float,
        )
        state_payload = {
            "inverse": {
                "method": "legacy",
                "optimization_cost": result["optimization_cost"],
                "optimization_success": result["optimization_success"],
                "measurement_rmse_mg_dl": None,
                "mean_inferred_glucose_input_mg_min": float(np.mean(result["inferred_glucose_input"])),
                "estimated_initial_state": {
                    "S0": float(initial_state[0]),
                    "Ip0": float(initial_state[1]),
                    "Ii0": float(initial_state[2]),
                    "G0": float(initial_state[3]),
                    "h10": float(initial_state[4]),
                    "h20": float(initial_state[5]),
                    "h30": float(initial_state[6]),
                },
            },
            "params": asdict(params),
        }
        return initial_state, result, state_payload

    if estimator_mode == "bayes":
        bootstrap = LatentStateBootstrap(
            t_data=t_data,
            glucose_data_mg_dl=glucose_data_mg_dl,
            params=params,
            smoothing_factor=smoothing_factor,
        )
        bootstrap_state = bootstrap.fit()
        observer = GlucoseStateParticleFilter(
            t_data=t_data,
            glucose_data_mg_dl=glucose_data_mg_dl,
            meal_events=meal_events,
            insulin_events=insulin_events,
            params=params,
            smoothing_factor=smoothing_factor,
            settings=bayes_settings,
        )
        result = observer.run(bootstrap_state=bootstrap_state)
        filtered_states = result["states"]
        initial_state = filtered_states[0].copy()
        residuals = result["predicted_glucose_mg_dl"] - np.asarray(glucose_data_mg_dl, dtype=float)
        rmse = float(np.sqrt(np.mean(residuals**2)))
        state_payload = {
            "inverse": {
                "method": "bayes",
                "bayes_settings": observer.settings,
                "optimization_cost": None,
                "optimization_success": True,
                "measurement_rmse_mg_dl": rmse,
                "mean_inferred_glucose_input_mg_min": float(np.mean(result["inferred_glucose_input"])),
                "estimated_initial_state": {
                    "S0": float(initial_state[0]),
                    "Ip0": float(initial_state[1]),
                    "Ii0": float(initial_state[2]),
                    "G0": float(initial_state[3]),
                    "h10": float(initial_state[4]),
                    "h20": float(initial_state[5]),
                    "h30": float(initial_state[6]),
                },
                "bootstrap_latent_state": {
                    "Ip0": float(bootstrap_state[0]),
                    "Ii0": float(bootstrap_state[1]),
                    "h10": float(bootstrap_state[2]),
                    "h20": float(bootstrap_state[3]),
                    "h30": float(bootstrap_state[4]),
                },
                "latest_filtered_state": {
                    "S": float(filtered_states[-1, 0]),
                    "Ip": float(filtered_states[-1, 1]),
                    "Ii": float(filtered_states[-1, 2]),
                    "G": float(filtered_states[-1, 3]),
                    "h1": float(filtered_states[-1, 4]),
                    "h2": float(filtered_states[-1, 5]),
                    "h3": float(filtered_states[-1, 6]),
                },
                "adapted_params": result["adapted_params"],
            },
            "params": asdict(replace(params, **result["adapted_params"])),
        }
        return initial_state, result, state_payload

    bootstrap = LatentStateBootstrap(
        t_data=t_data,
        glucose_data_mg_dl=glucose_data_mg_dl,
        params=params,
        smoothing_factor=smoothing_factor,
    )
    bootstrap_state = bootstrap.fit()
    observer = GlucoseStateUKF(
        t_data=t_data,
        glucose_data_mg_dl=glucose_data_mg_dl,
        meal_events=meal_events,
        insulin_events=insulin_events,
        params=params,
        smoothing_factor=smoothing_factor,
        ukf_settings=ukf_settings,
    )
    result = observer.run(bootstrap_state=bootstrap_state)
    filtered_states = result["states"]
    initial_state = filtered_states[0].copy()
    residuals = result["predicted_glucose_mg_dl"] - np.asarray(glucose_data_mg_dl, dtype=float)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    state_payload = {
        "inverse": {
            "method": "ukf",
            "ukf_settings": observer.settings,
            "optimization_cost": None,
            "optimization_success": True,
            "measurement_rmse_mg_dl": rmse,
            "mean_inferred_glucose_input_mg_min": float(np.mean(result["inferred_glucose_input"])),
            "estimated_initial_state": {
                "S0": float(initial_state[0]),
                "Ip0": float(initial_state[1]),
                "Ii0": float(initial_state[2]),
                "G0": float(initial_state[3]),
                "h10": float(initial_state[4]),
                "h20": float(initial_state[5]),
                "h30": float(initial_state[6]),
            },
            "bootstrap_latent_state": {
                "Ip0": float(bootstrap_state[0]),
                "Ii0": float(bootstrap_state[1]),
                "h10": float(bootstrap_state[2]),
                "h20": float(bootstrap_state[3]),
                "h30": float(bootstrap_state[4]),
            },
            "latest_filtered_state": {
                "S": float(filtered_states[-1, 0]),
                "Ip": float(filtered_states[-1, 1]),
                "Ii": float(filtered_states[-1, 2]),
                "G": float(filtered_states[-1, 3]),
                "h1": float(filtered_states[-1, 4]),
                "h2": float(filtered_states[-1, 5]),
                "h3": float(filtered_states[-1, 6]),
            },
        },
        "params": asdict(params),
    }
    return initial_state, result, state_payload
