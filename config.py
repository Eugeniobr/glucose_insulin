from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class AppConfig:
    base_dir: Path = Path(__file__).resolve().parent
    glucose_json_path: Path = Path("dados_glicose_final.json")
    outputs_dir: Path = Path("outputs")
    plots_dir: Path = Path("outputs/plots")
    metrics_path: Path = Path("outputs/metrics.json")
    state_path: Path = Path("outputs/state_latest.json")
    series_path: Path = Path("outputs/forward_series.json")
    forecast_history_path: Path = Path("outputs/forecast_history.json")
    calibrated_params_path: Path = Path("outputs/calibrated_params.json")
    forecast_model_path: Path = Path("outputs/forecast_model.json")
    simple_forecast_model_path: Path = Path("outputs/simple_forecast_model.json")
    takagi_sugeno_dynamic_model_path: Path = Path("outputs/takagi_sugeno_dynamic_model.json")
    takagi_sugeno_ml_benchmark_path: Path = Path("outputs/takagi_sugeno_ml_benchmark.json")
    external_forecast_model_path: Path = Path("outputs/external_forecast_model.json")
    external_history_report_path: Path = Path("outputs/external_history_report.json")
    external_history_csv_path: str = os.getenv("EXTERNAL_HISTORY_CSV_PATH", "").strip()
    external_physiology_priors_path: Path = Path("outputs/external_physiology_priors.json")
    personal_model_prototype_path: Path = Path("outputs/personal_model_prototype.json")
    personal_model_residual_prototype_path: Path = Path("outputs/personal_model_residual_prototype.json")
    adaptive_params_path: Path = Path("outputs/adaptive_params.json")
    extract_script: Path = Path("extract.py")
    meals_source: str = os.getenv(
        "MEALS_CSV_URL",
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQRSsQr5W1fR96SXCfRxc_U6yK56qraQmrpnbrv8LBVtNPBcyp2cQ5QAKH4jgwLsmMF3X3obTFyg_Qq/pub?gid=303496725&single=true&output=csv",
    ).strip()
    insulin_source: str = os.getenv(
        "INSULIN_CSV_URL",
        "https://docs.google.com/spreadsheets/d/e/2PACX-1vQhuQhka421OrNDETfQHE5JN__PKTLzSS80hZgplXqBU460gIo3e8dZ_YBXC7mUKRM0tcOSQg3LMjxv/pub?gid=976380391&single=true&output=csv",
    ).strip()
    exercise_source: str = os.getenv("EXERCISE_CSV_URL", "").strip()
    refresh_minutes: int = int(os.getenv("REFRESH_MINUTES", "15"))
    forward_dt_minutes: float = float(os.getenv("FORWARD_DT_MINUTES", "1.0"))
    smoothing_factor: float = float(os.getenv("SMOOTHING_FACTOR", "5.0"))
    meal_tau_minutes: float = float(os.getenv("MEAL_TAU_MINUTES", "40.0"))
    insulin_absorption_rate: float = float(os.getenv("INSULIN_KA", "0.03"))
    insulin_bioavailability: float = float(os.getenv("INSULIN_BIOAVAILABILITY", "1.0"))
    forecast_minutes: int = int(os.getenv("FORECAST_MINUTES", "120"))
    display_window_hours: int = int(os.getenv("DISPLAY_WINDOW_HOURS", "12"))
    calibration_min_points: int = int(os.getenv("CALIBRATION_MIN_POINTS", "8"))
    calibration_lookback_days: int = int(os.getenv("CALIBRATION_LOOKBACK_DAYS", "7"))
    calibration_blend: float = float(os.getenv("CALIBRATION_BLEND", "0.35"))
    state_estimator_mode: str = os.getenv("STATE_ESTIMATOR_MODE", "legacy").strip().lower()
    forecast_anchor_slope_window_minutes: float = float(os.getenv("FORECAST_ANCHOR_SLOPE_WINDOW_MINUTES", "30"))
    forecast_anchor_min_scale: float = float(os.getenv("FORECAST_ANCHOR_MIN_SCALE", "0.35"))
    forecast_anchor_max_drop_mgdl_per_min: float = float(os.getenv("FORECAST_ANCHOR_MAX_DROP_MGDL_PER_MIN", "0.6"))
    exercise_hr_baseline_bpm: float = float(os.getenv("EXERCISE_HR_BASELINE_BPM", "70.0"))
    exercise_fast_tau_minutes: float = float(os.getenv("EXERCISE_FAST_TAU_MINUTES", "5.0"))
    exercise_slow_tau_minutes: float = float(os.getenv("EXERCISE_SLOW_TAU_MINUTES", "20.0"))
    exercise_recovery_tau_minutes: float = float(os.getenv("EXERCISE_RECOVERY_TAU_MINUTES", "90.0"))
    exercise_clearance_gain: float = float(os.getenv("EXERCISE_CLEARANCE_GAIN", "0.45"))
    exercise_sensitivity_gain: float = float(os.getenv("EXERCISE_SENSITIVITY_GAIN", "0.30"))
    exercise_hepatic_suppression_gain: float = float(os.getenv("EXERCISE_HEPATIC_SUPPRESSION_GAIN", "0.12"))
    regime_model_enabled: bool = os.getenv("REGIME_MODEL_ENABLED", "0").strip() not in {"0", "false", "False"}
    regime_recent_meal_minutes: float = float(os.getenv("REGIME_RECENT_MEAL_MINUTES", "150"))
    regime_recent_insulin_minutes: float = float(os.getenv("REGIME_RECENT_INSULIN_MINUTES", "210"))
    regime_quiet_minutes: float = float(os.getenv("REGIME_QUIET_MINUTES", "240"))
    forecast_model_enabled: bool = os.getenv("FORECAST_MODEL_ENABLED", "1").strip() not in {"0", "false", "False"}
    forecast_model_min_points: int = int(os.getenv("FORECAST_MODEL_MIN_POINTS", "12"))
    forecast_model_l2: float = float(os.getenv("FORECAST_MODEL_L2", "4.0"))
    forecast_model_lookback_days: int = int(os.getenv("FORECAST_MODEL_LOOKBACK_DAYS", "14"))
    simple_forecast_model_enabled: bool = os.getenv("SIMPLE_FORECAST_MODEL_ENABLED", "1").strip() not in {"0", "false", "False"}
    simple_forecast_as_primary: bool = os.getenv("SIMPLE_FORECAST_AS_PRIMARY", "1").strip() not in {"0", "false", "False"}
    simple_forecast_retrain_from_external_enabled: bool = os.getenv("SIMPLE_FORECAST_RETRAIN_FROM_EXTERNAL_ENABLED", "1").strip() not in {"0", "false", "False"}
    simple_forecast_min_points: int = int(os.getenv("SIMPLE_FORECAST_MIN_POINTS", "96"))
    simple_forecast_l2: float = float(os.getenv("SIMPLE_FORECAST_L2", "6.0"))
    simple_forecast_max_samples: int = int(os.getenv("SIMPLE_FORECAST_MAX_SAMPLES", "0"))
    takagi_sugeno_ml_enabled: bool = os.getenv("TAKAGI_SUGENO_ML_ENABLED", "1").strip() not in {"0", "false", "False"}
    takagi_sugeno_ml_as_primary: bool = os.getenv("TAKAGI_SUGENO_ML_AS_PRIMARY", "1").strip() not in {"0", "false", "False"}
    jump_aware_forecast_enabled: bool = os.getenv("JUMP_AWARE_FORECAST_ENABLED", "1").strip() not in {"0", "false", "False"}
    jump_force_blend_when_detected: bool = os.getenv("JUMP_FORCE_BLEND_WHEN_DETECTED", "1").strip() not in {"0", "false", "False"}
    jump_slope_threshold: float = float(os.getenv("JUMP_SLOPE_THRESHOLD", "1.10"))
    jump_delta15_threshold: float = float(os.getenv("JUMP_DELTA15_THRESHOLD", "18.0"))
    jump_blend_base_ts_weight: float = float(os.getenv("JUMP_BLEND_BASE_TS_WEIGHT", "0.45"))
    jump_blend_slope_gain: float = float(os.getenv("JUMP_BLEND_SLOPE_GAIN", "0.35"))
    jump_blend_delta_gain: float = float(os.getenv("JUMP_BLEND_DELTA_GAIN", "0.25"))
    jump_blend_max_ts_weight: float = float(os.getenv("JUMP_BLEND_MAX_TS_WEIGHT", "0.90"))
    jump_ts_weight_bonus: float = float(os.getenv("JUMP_TS_WEIGHT_BONUS", "0.22"))
    jump_slope_ref: float = float(os.getenv("JUMP_SLOPE_REF", "1.0"))
    jump_delta15_ref: float = float(os.getenv("JUMP_DELTA15_REF", "20.0"))
    adaptive_forecast_bias_enabled: bool = os.getenv("ADAPTIVE_FORECAST_BIAS_ENABLED", "1").strip() not in {"0", "false", "False"}
    adaptive_forecast_bias_lookback_hours: int = int(os.getenv("ADAPTIVE_FORECAST_BIAS_LOOKBACK_HOURS", "24"))
    adaptive_forecast_bias_min_points: int = int(os.getenv("ADAPTIVE_FORECAST_BIAS_MIN_POINTS", "6"))
    adaptive_forecast_bias_blend: float = float(os.getenv("ADAPTIVE_FORECAST_BIAS_BLEND", "0.45"))
    adaptive_forecast_bias_max_adjustment_mg_dl: float = float(os.getenv("ADAPTIVE_FORECAST_BIAS_MAX_ADJUSTMENT_MG_DL", "45.0"))
    external_forecast_model_enabled: bool = os.getenv("EXTERNAL_FORECAST_MODEL_ENABLED", "0").strip() not in {"0", "false", "False"}
    external_forecast_weight: float = float(os.getenv("EXTERNAL_FORECAST_WEIGHT", "0.35"))
    external_history_priors_enabled: bool = os.getenv("EXTERNAL_HISTORY_PRIORS_ENABLED", "0").strip() not in {"0", "false", "False"}
    external_history_ratio_reference_g_per_unit: float = float(os.getenv("EXTERNAL_HISTORY_RATIO_REFERENCE_G_PER_UNIT", "8.0"))
    external_history_min_paired_events: int = int(os.getenv("EXTERNAL_HISTORY_MIN_PAIRED_EVENTS", "12"))
    external_history_prior_blend: float = float(os.getenv("EXTERNAL_HISTORY_PRIOR_BLEND", "0.35"))
    external_physiology_priors_enabled: bool = os.getenv("EXTERNAL_PHYSIOLOGY_PRIORS_ENABLED", "0").strip() not in {"0", "false", "False"}
    adaptive_params_enabled: bool = os.getenv("ADAPTIVE_PARAMS_ENABLED", "1").strip() not in {"0", "false", "False"}
    adaptive_params_min_points: int = int(os.getenv("ADAPTIVE_PARAMS_MIN_POINTS", "16"))
    adaptive_params_l2: float = float(os.getenv("ADAPTIVE_PARAMS_L2", "6.0"))
    adaptive_params_lookback_days: int = int(os.getenv("ADAPTIVE_PARAMS_LOOKBACK_DAYS", "14"))
    adaptive_params_blend: float = float(os.getenv("ADAPTIVE_PARAMS_BLEND", "0.25"))
    ukf_alpha: float = float(os.getenv("UKF_ALPHA", "0.15"))
    ukf_beta: float = float(os.getenv("UKF_BETA", "2.0"))
    ukf_kappa: float = float(os.getenv("UKF_KAPPA", "0.0"))
    ukf_measurement_noise_mgdl: float = float(os.getenv("UKF_MEASUREMENT_NOISE_MGDL", "6.0"))
    ukf_process_noise_s: float = float(os.getenv("UKF_PROCESS_NOISE_S", "8000.0"))
    ukf_process_noise_ip: float = float(os.getenv("UKF_PROCESS_NOISE_IP", "2500.0"))
    ukf_process_noise_ii: float = float(os.getenv("UKF_PROCESS_NOISE_II", "2500.0"))
    ukf_process_noise_g: float = float(os.getenv("UKF_PROCESS_NOISE_G", "18000.0"))
    ukf_process_noise_h: float = float(os.getenv("UKF_PROCESS_NOISE_H", "3500.0"))
    bayes_particle_count: int = int(os.getenv("BAYES_PARTICLE_COUNT", "80"))
    bayes_measurement_noise_mgdl: float = float(os.getenv("BAYES_MEASUREMENT_NOISE_MGDL", "12.0"))
    bayes_state_noise_scale: float = float(os.getenv("BAYES_STATE_NOISE_SCALE", "0.03"))
    bayes_param_noise_scale: float = float(os.getenv("BAYES_PARAM_NOISE_SCALE", "0.015"))

    def resolve(self) -> "AppConfig":
        self.glucose_json_path = (self.base_dir / self.glucose_json_path).resolve()
        self.outputs_dir = (self.base_dir / self.outputs_dir).resolve()
        self.plots_dir = (self.base_dir / self.plots_dir).resolve()
        self.metrics_path = (self.base_dir / self.metrics_path).resolve()
        self.state_path = (self.base_dir / self.state_path).resolve()
        self.series_path = (self.base_dir / self.series_path).resolve()
        self.forecast_history_path = (self.base_dir / self.forecast_history_path).resolve()
        self.calibrated_params_path = (self.base_dir / self.calibrated_params_path).resolve()
        self.forecast_model_path = (self.base_dir / self.forecast_model_path).resolve()
        self.simple_forecast_model_path = (self.base_dir / self.simple_forecast_model_path).resolve()
        self.takagi_sugeno_dynamic_model_path = (self.base_dir / self.takagi_sugeno_dynamic_model_path).resolve()
        self.takagi_sugeno_ml_benchmark_path = (self.base_dir / self.takagi_sugeno_ml_benchmark_path).resolve()
        self.external_forecast_model_path = (self.base_dir / self.external_forecast_model_path).resolve()
        self.external_history_report_path = (self.base_dir / self.external_history_report_path).resolve()
        self.external_physiology_priors_path = (self.base_dir / self.external_physiology_priors_path).resolve()
        self.personal_model_prototype_path = (self.base_dir / self.personal_model_prototype_path).resolve()
        self.personal_model_residual_prototype_path = (self.base_dir / self.personal_model_residual_prototype_path).resolve()
        self.adaptive_params_path = (self.base_dir / self.adaptive_params_path).resolve()
        self.extract_script = (self.base_dir / self.extract_script).resolve()
        return self


def load_config() -> AppConfig:
    return AppConfig().resolve()
