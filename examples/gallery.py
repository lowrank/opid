"""Classical PDE solution gallery."""
import time, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opid._backend import EvolutionEquation

MODELS = {
    "KdV–Burgers":
        EvolutionEquation(["0.5*D[D[u]] - u*D[u] - (1.0/6)*D[D[D[u]]]"]),
    "Kuramoto–Sivashinsky":
        EvolutionEquation(["-D[D[u]] - 0.05*D[D[D[D[u]]]] - 0.5*D[u]*D[u]"]),
    "Ginzburg–Landau":
        EvolutionEquation(["D[D[u]] - u^3 + u"]),
    "FitzHugh–Nagumo":
        EvolutionEquation(["D[D[u]] + u^2 - u^3"]),
    "Fisher–KPP":
        EvolutionEquation(["D[D[u]] + u - u^2"]),
    "Nonlin. Schrödinger (Re)":
        EvolutionEquation([
            "-D[D[v]] - (u^2+v^2)*v",
            "D[D[u]] + (u^2+v^2)*u",
        ]),
    "Swift–Hohenberg":
        EvolutionEquation(["u - u^3 - (2*D[D[u]] + D[D[D[D[u]]]])"]),
    "Eikonal":
        EvolutionEquation(["0.25*D[D[u]] + Abs[D[u]]"]),
    "Porous Media":
        EvolutionEquation(["3*u^2*D[D[u]] + 6*u*D[u]*D[u]"]),
}

dt, T = 0.001, 1.0
t_span = np.arange(dt, T + dt, dt)
x_axis = np.linspace(0, 2*np.pi, 1001)[:-1]
Nx = len(x_axis)

# Single-component initial condition (shared by all models)
y0_real = (
    0.6 + 0.3*np.sin(x_axis) + 0.15*np.sin(2*x_axis)
    + 0.1*np.cos(3*x_axis) + 0.02*np.cos(4*x_axis)
)
# Two-component IC for NLS: same real part, zero imaginary part
y0_nls = np.concatenate([y0_real, np.zeros(Nx)])

n = len(MODELS)
cols = 3
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True)
axes = axes.flatten()

for idx, (name, model) in enumerate(MODELS.items()):
    print(f"[{idx+1}/{n}] {name}", flush=True)
    t0 = time.time()
    model.wrap(name.replace(" ", "_"))
    dt_compile = time.time() - t0

    y0 = y0_nls if model.complex else y0_real

    t0 = time.time()
    sol = model.solve(y0, t_span, rtol=1e-6, atol=1e-6)
    dt_solve = time.time() - t0

    for k in range(5):
        axes[idx].plot(x_axis, sol[k*200, :Nx], linewidth=0.6)
    axes[idx].set_title(f"{name}\n{dt_compile:.0f}s + {dt_solve:.0f}s", fontsize=9)
    axes[idx].set_xlim(0, 2*np.pi)

for j in range(n, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "docs", "example.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
