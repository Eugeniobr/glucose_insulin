import json
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from datetime import datetime
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares


# ============================================================
# 1) PARÂMETROS DO MODELO
# ============================================================

@dataclass
class UltradianParams:
    Vp: float = 3.0     # L
    Vi: float = 11.0    # L
    Vg: float = 10.0    # L

    E: float = 0.2      # 1/min
    tp: float = 6.0     # min
    ti: float = 100.0   # min
    td: float = 12.0    # min

    Rm: float = 209.0   # mU/min
    a1: float = 6.67
    C1: float = 300.0   # mg/L
    C2: float = 144.0   # mg/L
    C3: float = 100.0   # mg/L
    C4: float = 80.0    # mU/L
    C5: float = 26.0    # mU/L

    Ub: float = 72.0    # mg/min
    U0: float = 4.0     # mg/min
    Um: float = 94.0    # mg/min
    Rg: float = 180.0   # mg/min

    alpha: float = 7.5
    beta: float = 1.77

    @property
    def kappa(self):
        return (1.0 / self.C4) * (1.0 / self.Vi + 1.0 / (self.E * self.ti))


# ============================================================
# 2) FUNÇÕES DO MODELO
# ============================================================

def f1(G, p: UltradianParams):
    return p.Rm / (1.0 + np.exp(-G / (p.Vg * p.C1) + p.a1))


def f2(G, p: UltradianParams):
    return p.Ub * (1.0 - np.exp(-G / (p.C2 * p.Vg)))


def f3(Ii, p: UltradianParams):
    x = p.kappa * np.maximum(Ii, 1e-12)
    return (1.0 / (p.C3 * p.Vg)) * (
        p.U0 + (p.Um - p.U0) / (1.0 + x ** (-p.beta))
    )


def f4(h3, p: UltradianParams):
    return p.Rg / (1.0 + np.exp(p.alpha * (h3 / (p.C5 * p.Vp) - 1.0)))


# ============================================================
# 3) CONVERSÕES DE UNIDADES
# ============================================================

def glucose_mgdl_to_total_mg(g_mg_dl, p: UltradianParams):
    # mg/dL -> mg total
    # mg/L = 10 * mg/dL
    # total = conc(mg/L) * Vg(L)
    return 10.0 * np.asarray(g_mg_dl) * p.Vg


def total_mg_to_glucose_mgdl(G_total, p: UltradianParams):
    return np.asarray(G_total) / p.Vg / 10.0


def total_insulin_to_uU_ml(I_total, volume_L):
    # mU/L numericamente = µU/mL
    return np.asarray(I_total) / volume_L


# ============================================================
# 4) LEITURA DO JSON
# ============================================================

def load_glucose_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    timestamps = []
    glucose = []

    for row in data:
        if "timestamp" not in row:
            continue
        if "value_in_mg_per_dl" not in row:
            continue

        ts = datetime.fromisoformat(row["timestamp"])
        g = float(row["value_in_mg_per_dl"])

        timestamps.append(ts)
        glucose.append(g)

    if len(timestamps) < 5:
        raise ValueError("Poucos pontos no JSON.")

    t0 = timestamps[0]
    t_minutes = np.array([(ts - t0).total_seconds() / 60.0 for ts in timestamps], dtype=float)
    g_mg_dl = np.array(glucose, dtype=float)

    # garante ordenação
    order = np.argsort(t_minutes)
    t_minutes = t_minutes[order]
    g_mg_dl = g_mg_dl[order]

    return t_minutes, g_mg_dl, timestamps


# ============================================================
# 5) PROBLEMA INVERSO
# ============================================================

class GlucoseDrivenInverseModel:
    """
    Usa G_obs(t) como conhecida e reconstrói:
      Ip(t), Ii(t), h1(t), h2(t), h3(t)

    Assumimos I_G(t)=0 no modelo.
    """

    def __init__(self, t_data, g_data_mg_dl, p: UltradianParams, smoothing_s=None):
        self.p = p
        self.t_data = np.asarray(t_data, dtype=float)
        self.g_data_mg_dl = np.asarray(g_data_mg_dl, dtype=float)

        if np.any(np.diff(self.t_data) <= 0):
            raise ValueError("t_data deve ser estritamente crescente.")

        # spline suavizada
        # se smoothing_s=None, define automaticamente um valor moderado
        if smoothing_s is None:
            smoothing_s = len(self.t_data) * 4.0

        self.g_spline_mgdl = UnivariateSpline(
            self.t_data,
            self.g_data_mg_dl,
            s=smoothing_s,
            k=3
        )

        self.g_spline_total = lambda t: glucose_mgdl_to_total_mg(self.g_spline_mgdl(t), self.p)
        self.dgdt_spline_total = lambda t: 10.0 * self.p.Vg * self.g_spline_mgdl.derivative()(t)

    def rhs_internal(self, t, x):
        Ip, Ii, h1, h2, h3 = x
        G = self.g_spline_total(t)

        dIp = f1(G, self.p) - ((Ip / self.p.Vp) - (Ii / self.p.Vi)) * self.p.E - Ip / self.p.tp
        dIi = ((Ip / self.p.Vp) - (Ii / self.p.Vi)) * self.p.E - Ii / self.p.ti
        dh1 = (Ip - h1) / self.p.td
        dh2 = (h1 - h2) / self.p.td
        dh3 = (h2 - h3) / self.p.td

        return [dIp, dIi, dh1, dh2, dh3]

    def simulate_internal(self, x0, t_eval=None):
        if t_eval is None:
            t_eval = self.t_data

        sol = solve_ivp(
            fun=lambda t, x: self.rhs_internal(t, x),
            t_span=(t_eval[0], t_eval[-1]),
            y0=x0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol

    def glucose_residual(self, x0, t_fit=None, penalty_weight=1e-4):
        if t_fit is None:
            t_fit = self.t_data

        sol = self.simulate_internal(x0, t_eval=t_fit)

        Ip = sol.y[0]
        Ii = sol.y[1]
        h1 = sol.y[2]
        h2 = sol.y[3]
        h3 = sol.y[4]
        t = sol.t

        G = self.g_spline_total(t)
        dG_obs = self.dgdt_spline_total(t)

        # Equação do modelo para G com I_G(t)=0
        dG_model = f4(h3, self.p) - f2(G, self.p) - f3(Ii, self.p) * G

        resid_g = dG_obs - dG_model

        # penalização leve nos estados iniciais
        resid_penalty = penalty_weight * np.asarray(x0)

        return np.concatenate([resid_g, resid_penalty])

    def fit(self, x0_guess, bounds=None):
        if bounds is None:
            lower = np.zeros(5)
            upper = np.array([1e5, 1e5, 1e5, 1e5, 1e5], dtype=float)
            bounds = (lower, upper)

        res = least_squares(
            fun=lambda x: self.glucose_residual(x),
            x0=np.asarray(x0_guess, dtype=float),
            bounds=bounds,
            method="trf",
            max_nfev=500,
            verbose=1,
        )

        sol = self.simulate_internal(res.x, t_eval=self.t_data)

        return {
            "opt_result": res,
            "x0_est": res.x,
            "t": sol.t,
            "Ip": sol.y[0],
            "Ii": sol.y[1],
            "h1": sol.y[2],
            "h2": sol.y[3],
            "h3": sol.y[4],
            "G_obs_mg_dl": self.g_data_mg_dl,
            "G_smooth_mg_dl": self.g_spline_mgdl(sol.t),
        }


# ============================================================
# 6) PLOTS
# ============================================================

def plot_results(result, p: UltradianParams):
    t = result["t"]
    G_obs = result["G_obs_mg_dl"]
    G_smooth = result["G_smooth_mg_dl"]

    Ip_uU_ml = total_insulin_to_uU_ml(result["Ip"], p.Vp)
    Ii_uU_ml = total_insulin_to_uU_ml(result["Ii"], p.Vi)
    h3_uU_ml = total_insulin_to_uU_ml(result["h3"], p.Vp)

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    axes[0].plot(t, G_obs, "o", ms=4, label="Glicose observada")
    axes[0].plot(t, G_smooth, "-", lw=2, label="Glicose suavizada")
    axes[0].set_ylabel("Glicose (mg/dL)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, Ip_uU_ml, lw=2, label="Insulina plasmática estimada")
    axes[1].set_ylabel("Ip (µU/mL)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, Ii_uU_ml, lw=2, label="Insulina remota estimada")
    axes[2].set_ylabel("Ii (µU/mL)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(t, h3_uU_ml, lw=2, label="h3 estimado")
    axes[3].set_ylabel("h3")
    axes[3].set_xlabel("Tempo desde a 1ª medição (min)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# 7) RESUMO NUMÉRICO
# ============================================================

def summarize_result(result, p: UltradianParams):
    t = result["t"]
    Ip = total_insulin_to_uU_ml(result["Ip"], p.Vp)
    Ii = total_insulin_to_uU_ml(result["Ii"], p.Vi)
    G = result["G_obs_mg_dl"]

    print("\n===== RESUMO =====")
    print(f"Número de pontos de glicose: {len(G)}")
    print(f"Janela total: {t[-1] - t[0]:.2f} min")
    print(f"Glicose min/max: {np.min(G):.2f} / {np.max(G):.2f} mg/dL")
    print(f"Ip estimada min/max: {np.min(Ip):.2f} / {np.max(Ip):.2f} µU/mL")
    print(f"Ii estimada min/max: {np.min(Ii):.2f} / {np.max(Ii):.2f} µU/mL")
    print("Estados iniciais estimados [Ip0, Ii0, h10, h20, h30]:")
    print(result["x0_est"])


# ============================================================
# 8) PIPELINE PRINCIPAL
# ============================================================

def run_from_json(json_path):
    p = UltradianParams()

    t_data, g_data_mg_dl, timestamps = load_glucose_json(json_path)

    # chute inicial
    # você pode ajustar isso depois
    x0_guess = np.array([
        100.0,  # Ip0 total
        800.0,  # Ii0 total
        100.0,  # h10
        100.0,  # h20
        100.0,  # h30
    ])

    inv = GlucoseDrivenInverseModel(
        t_data=t_data,
        g_data_mg_dl=g_data_mg_dl,
        p=p,
        smoothing_s=len(t_data) * 5.0
    )

    result = inv.fit(x0_guess=x0_guess)

    summarize_result(result, p)
    plot_results(result, p)

    return result, t_data, g_data_mg_dl, timestamps


# ============================================================
# 9) EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    json_path = "dados_glicose_final.json"
    result, t_data, g_data_mg_dl, timestamps = run_from_json(json_path)