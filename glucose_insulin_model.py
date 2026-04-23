import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from scipy.integrate import solve_ivp


# ============================================================
# 1) PARÂMETROS DO MODELO
# ============================================================

@dataclass
class UltradianParams:
    # Volumes
    Vp: float = 3.0    # L   plasma insulin volume
    Vi: float = 11.0   # L   remote insulin volume
    Vg: float = 10.0   # L   glucose volume

    # Exchange / decay time constants
    E: float = 0.2     # 1/min
    tp: float = 6.0    # min
    ti: float = 100.0  # min
    td: float = 12.0   # min

    # Secretion / utilization parameters
    Rm: float = 209.0  # mU/min
    a1: float = 6.67
    C1: float = 300.0  # mg/L
    C2: float = 144.0  # mg/L
    C3: float = 100.0  # mg/L
    C4: float = 80.0   # mU/L
    C5: float = 26.0   # mU/L

    Ub: float = 72.0   # mg/min
    U0: float = 4.0    # mg/min
    Um: float = 94.0   # mg/min
    Rg: float = 180.0  # mg/min

    alpha: float = 7.5
    beta: float = 1.77

    @property
    def kappa(self) -> float:
        # Forma padrão do modelo de Sturis et al.
        return (1.0 / self.C4) * (1.0 / self.Vi + 1.0 / (self.E * self.ti))


# ============================================================
# 2) INPUT EXTERNO DE GLICOSE
# ============================================================

def glucose_input_constant(t: float, rate: float = 0.0) -> float:
    """
    Entrada externa constante de glicose I_G(t), em mg/min.
    Ex.: infusão intravenosa constante.
    """
    return rate


def glucose_input_meal(
    t: float,
    meal_time: float = 60.0,
    meal_size_mg: float = 40000.0,
    tau: float = 40.0
) -> float:
    """
    Aproximação simples de uma refeição:
    input tipo gamma/exponencial iniciando em meal_time.
    A integral total é aproximadamente meal_size_mg.
    """
    if t < meal_time:
        return 0.0

    x = t - meal_time
    # kernel ~ x * exp(-x/tau), normalizado para integral = meal_size_mg
    return meal_size_mg * (x / tau**2) * np.exp(-x / tau)


def glucose_input_multiple_meals(t: float, meals) -> float:
    """
    Soma de várias refeições.
    meals = [
        {"meal_time": 60, "meal_size_mg": 40000, "tau": 40},
        {"meal_time": 360, "meal_size_mg": 50000, "tau": 50},
        ...
    ]
    """
    total = 0.0
    for m in meals:
        total += glucose_input_meal(
            t,
            meal_time=m["meal_time"],
            meal_size_mg=m["meal_size_mg"],
            tau=m.get("tau", 40.0),
        )
    return total


# ============================================================
# 3) FUNÇÕES f1, f2, f3, f4
# ============================================================

def f1(G: float, p: UltradianParams) -> float:
    """
    Produção de insulina plasmática dependente da glicose.
    G é quantidade total de glicose (mg).
    """
    return p.Rm / (1.0 + np.exp(-G / (p.Vg * p.C1) + p.a1))


def f2(G: float, p: UltradianParams) -> float:
    """
    Utilização de glicose independente de insulina.
    """
    return p.Ub * (1.0 - np.exp(-G / (p.C2 * p.Vg)))


def f3(Ii: float, p: UltradianParams) -> float:
    """
    Utilização de glicose dependente da insulina remota.
    Ii é quantidade total de insulina remota (mU).
    """
    x = p.kappa * Ii
    # proteção numérica para evitar x=0 em potência negativa
    x = max(x, 1e-12)
    return (1.0 / (p.C3 * p.Vg)) * (
        p.U0 + (p.Um - p.U0) / (1.0 + x**(-p.beta))
    )


def f4(h3: float, p: UltradianParams) -> float:
    """
    Produção hepática de glicose, regulada pela insulina com atraso.
    """
    return p.Rg / (1.0 + np.exp(p.alpha * (h3 / (p.C5 * p.Vp) - 1.0)))


# ============================================================
# 4) SISTEMA DE EDOs
# ============================================================

def ultradian_rhs(t, y, p: UltradianParams, glucose_input_func):
    """
    y = [Ip, Ii, G, h1, h2, h3]
      Ip : insulina plasmática total (mU)
      Ii : insulina remota total   (mU)
      G  : glicose total           (mg)
      h1,h2,h3 : estados do filtro de atraso
    """
    Ip, Ii, G, h1, h2, h3 = y

    IG = glucose_input_func(t)

    dIp = (
        f1(G, p)
        - ((Ip / p.Vp) - (Ii / p.Vi)) * p.E
        - Ip / p.tp
    )

    dIi = (
        ((Ip / p.Vp) - (Ii / p.Vi)) * p.E
        - Ii / p.ti
    )

    dG = (
        f4(h3, p)
        + IG
        - f2(G, p)
        - f3(Ii, p) * G
    )

    dh1 = (Ip - h1) / p.td
    dh2 = (h1 - h2) / p.td
    dh3 = (h2 - h3) / p.td

    return [dIp, dIi, dG, dh1, dh2, dh3]


# ============================================================
# 5) FUNÇÕES AUXILIARES
# ============================================================

def amounts_to_concentrations(sol_y, p: UltradianParams):
    """
    Converte quantidades totais em concentrações.
    Retorna:
      insulin_plasma_uU_ml
      insulin_remote_uU_ml
      glucose_mg_dl
    """
    Ip = sol_y[0]
    Ii = sol_y[1]
    G = sol_y[2]

    # Insulina:
    # mU/L = microU/mL numericamente
    insulin_plasma_uU_ml = Ip / p.Vp
    insulin_remote_uU_ml = Ii / p.Vi

    # Glicose:
    # mg/L -> mg/dL divide por 10
    glucose_mg_dl = (G / p.Vg) / 10.0

    return insulin_plasma_uU_ml, insulin_remote_uU_ml, glucose_mg_dl


def simulate_ultradian(
    p: UltradianParams,
    t_span=(0.0, 1000.0),
    y0=None,
    glucose_input_func=None,
    dt=0.5
):
    if glucose_input_func is None:
        glucose_input_func = lambda t: 0.0

    if y0 is None:
        # condição inicial razoável, perto de basal
        # G ~ 100 mg/dL => 1000 mg/L * 10 L = 10000 mg
        G0 = 100.0 * 10.0 * p.Vg / 10.0  # simplifica para 100 * Vg
        # mas isso daria 1000 mg com Vg=10; o correto em mg total é:
        G0 = 100.0 * 10.0 * p.Vg  # 100 mg/dL = 1000 mg/L
        Ip0 = 15.0 * p.Vp         # ~15 microU/mL
        Ii0 = 15.0 * p.Vi
        h10 = Ip0
        h20 = Ip0
        h30 = Ip0
        y0 = [Ip0, Ii0, G0, h10, h20, h30]

    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    sol = solve_ivp(
        fun=lambda t, y: ultradian_rhs(t, y, p, glucose_input_func),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"Falha na integração: {sol.message}")

    return sol


def plot_ultradian_solution(sol, p: UltradianParams, glucose_input_func=None):
    t = sol.t
    insulin_plasma, insulin_remote, glucose = amounts_to_concentrations(sol.y, p)

    if glucose_input_func is None:
        glucose_input = np.zeros_like(t)
    else:
        glucose_input = np.array([glucose_input_func(tt) for tt in t])

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    axes[0].plot(t, insulin_plasma, label="Plasma insulin")
    axes[0].set_ylabel("Insulina plasmática\n(µU/mL)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, insulin_remote, label="Remote insulin")
    axes[1].set_ylabel("Insulina remota\n(µU/mL)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, glucose, label="Glucose")
    axes[2].set_ylabel("Glicose\n(mg/dL)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(t, glucose_input, label="I_G(t)")
    axes[3].set_ylabel("Entrada de glicose\n(mg/min)")
    axes[3].set_xlabel("Tempo (min)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.suptitle("Modelo ultradiano de insulina-glicose")
    plt.tight_layout()
    plt.show()


# ============================================================
# 6) EXEMPLOS DE USO
# ============================================================

if __name__ == "__main__":
    p = UltradianParams()

    # --------------------------------------------------------
    # Exemplo A: infusão constante de glicose
    # --------------------------------------------------------
    infusion_rate = 216.0  # mg/min, valor de teste
    glucose_input_func = lambda t: glucose_input_constant(t, rate=infusion_rate)

    sol = simulate_ultradian(
        p=p,
        t_span=(0.0, 1000.0),
        glucose_input_func=glucose_input_func,
        dt=0.5
    )
    plot_ultradian_solution(sol, p, glucose_input_func)

    # --------------------------------------------------------
    # Exemplo B: três refeições
    # --------------------------------------------------------
    meals = [
        {"meal_time": 60.0,  "meal_size_mg": 40000.0, "tau": 35.0},
        {"meal_time": 360.0, "meal_size_mg": 50000.0, "tau": 40.0},
        {"meal_time": 720.0, "meal_size_mg": 45000.0, "tau": 45.0},
    ]

    meal_input_func = lambda t: glucose_input_multiple_meals(t, meals)

    sol_meals = simulate_ultradian(
        p=p,
        t_span=(0.0, 1200.0),
        glucose_input_func=meal_input_func,
        dt=0.5
    )
    plot_ultradian_solution(sol_meals, p, meal_input_func)