"""Иллюстрации для учебно-методического пособия.

Графики, которых нет в отчётах лабораторных работ: теоретические схемы и
примеры выполнения ЛР 1–2 (пересчитаны для варианта 3 по условиям).
Остальные графики пособие берёт напрямую из lab_NN/report/figures/.

Запуск из каталога metodichka/:
    uv run --with numpy --with matplotlib python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

# Палитра и фон — как в ноутбуках ЛР 3–7, чтобы иллюстрации читались единой системой
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": True,
    "legend.edgecolor": GRID,
    "legend.fontsize": 10,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- теория
def activation_functions():
    z = np.linspace(-4, 4, 801)
    funcs = [
        ("hardlim", np.where(z >= 0, 1.0, 0.0), r"$f(z)=1$ при $z\geq 0$, иначе 0"),
        ("purelin", z, r"$f(z)=z$"),
        ("logsig", 1 / (1 + np.exp(-z)), r"$f(z)=1/(1+e^{-z})$"),
        ("tansig", np.tanh(z), r"$f(z)=\tanh z$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
    for ax, (name, y, formula) in zip(axes, funcs):
        if name == "hardlim":
            ax.plot(z[z < 0], y[z < 0], color=PALETTE[0], lw=2.5)
            ax.plot(z[z >= 0], y[z >= 0], color=PALETTE[0], lw=2.5)
            ax.plot([0], [1], "o", color=PALETTE[0])
            ax.plot([0], [0], "o", mfc=SURFACE, color=PALETTE[0])
        else:
            ax.plot(z, y, color=PALETTE[0], lw=2.5)
        ax.set_title(f"{name}\n{formula}", fontsize=11)
        ax.set_xlabel("z")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(0, color=AXIS, lw=0.8)
        if name == "purelin":
            ax.set_ylim(-4, 4)
        else:
            ax.set_ylim(-1.25, 1.25)
    save(fig, "activation")


def separability():
    pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    tasks = [("И (AND)", [0, 0, 0, 1], [(1, 1, -1.5)]),
             ("ИЛИ (OR)", [0, 1, 1, 1], [(1, 1, -0.5)]),
             ("исключающее ИЛИ (XOR)", [0, 1, 1, 0], [(1, 1, -0.5), (1, 1, -1.5)])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))
    xs = np.linspace(-0.5, 1.5, 10)
    for ax, (title, t, lines) in zip(axes, tasks):
        t = np.array(t)
        for cls, col, mk in [(0, PALETTE[1], "o"), (1, PALETTE[0], "s")]:
            m = t == cls
            ax.scatter(pts[m, 0], pts[m, 1], s=150, c=col, marker=mk, edgecolor=INK, zorder=3,
                       label=f"класс {cls}")
        for i, (w1, w2, b) in enumerate(lines):
            ax.plot(xs, -(w1 * xs + b) / w2, "--", color=PALETTE[2] if len(lines) == 1 else MUTED, lw=2)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        sub = "одна прямая разделяет классы" if len(lines) == 1 else "нужны две прямые (скрытый слой)"
        ax.set_title(f"{title}\n{sub}", fontsize=11)
    axes[0].legend(loc="upper left", fontsize=9)
    save(fig, "separability")


def gd_valley():
    """Градиентный спуск на вытянутой квадратичной поверхности E = (λ1 w1² + λ2 w2²)/2."""
    lam = np.array([1.0, 18.0])  # λmax = 18, критический шаг 2/18 ≈ 0.111
    w0 = np.array([-4.5, 1.6])

    def run(eta, steps, mom=0.0):
        w, v, path = w0.copy(), np.zeros(2), [w0.copy()]
        for _ in range(steps):
            v = mom * v - eta * lam * w
            w = w + v
            path.append(w.copy())
        return np.array(path)

    W1, W2 = np.meshgrid(np.linspace(-5, 5, 300), np.linspace(-2.2, 2.2, 300))
    E = 0.5 * (lam[0] * W1 ** 2 + lam[1] * W2 ** 2)
    cases = [(r"$\eta=0{,}02$: медленно", run(0.02, 40), PALETTE[0]),
             (r"$\eta=0{,}105$: «зигзаг»", run(0.105, 40), PALETTE[1]),
             (r"$\eta=0{,}02$, момент 0,8", run(0.02, 40, mom=0.8), PALETTE[2])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
    for ax, (title, p, col) in zip(axes, cases):
        ax.contour(W1, W2, E, levels=np.geomspace(0.05, 60, 12), colors=AXIS, linewidths=0.8)
        ax.plot(p[:, 0], p[:, 1], "-o", color=col, ms=3, lw=1.5)
        ax.plot(0, 0, "*", color=INK, ms=12)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("$w_1$ (пологое)")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2.2, 2.2)
    axes[0].set_ylabel("$w_2$ (крутое)")
    save(fig, "gd_valley")


def hex_positions(rows, cols):
    pos = []
    for r in range(rows):
        for c in range(cols):
            pos.append((c + 0.5 * (r % 2), r * np.sqrt(3) / 2))
    return np.array(pos)


def som_topology():
    rows = cols = 5
    grid = np.array([(c, r) for r in range(rows) for c in range(cols)], float)
    hexp = hex_positions(rows, cols)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, pos, title in [(axes[0], grid, "gridtop: 4 соседа"), (axes[1], hexp, "hextop: 6 соседей")]:
        d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
        adj = (d < 1.01) & (d > 0)
        # linkdist — длина кратчайшего пути по связям
        n = len(pos)
        link = np.full((n, n), np.inf)
        np.fill_diagonal(link, 0)
        link[adj] = 1
        for k in range(n):
            link = np.minimum(link, link[:, [k]] + link[[k], :])
        c = 12
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    ax.plot(*pos[[i, j]].T, color=AXIS, lw=1.2, zorder=1)
        cols_d = [PALETTE[1], PALETTE[3], PALETTE[0], "#9ec5f4", "#cde2fb"]
        for i in range(n):
            dist = int(link[c, i])
            ax.scatter(*pos[i], s=380, color=cols_d[min(dist, 4)], edgecolor=INK, zorder=2)
            ax.text(*pos[i], str(dist), ha="center", va="center", fontsize=9, zorder=3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)
    fig.suptitle("Расстояние linkdist от центрального нейрона на решётке 5×5", y=0.02, fontsize=11,
                 color=INK2)
    save(fig, "som_topology")


def selection_probs():
    f = np.array([9.0, 7.5, 6.0, 5.2, 4.1, 3.3, 2.0, 1.4, 0.8, 0.2])
    N = len(f)
    p_prop = f / f.sum()
    a = 1.8  # линейное ранжирование: a ∈ [1; 2], b = 2 − a
    i = np.arange(1, N + 1)
    p_rank = (a - (a - (2 - a)) * (i - 1) / (N - 1)) / N
    p_tour = {k: ((N - i + 1) ** k - (N - i) ** k) / N ** k for k in (2, 3)}
    x = np.arange(N)
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(x - 1.5 * w, p_prop, w, label="пропорциональный (рулетка)", color=PALETTE[0])
    ax.bar(x - 0.5 * w, p_rank, w, label="линейное ранжирование, a = 1,8", color=PALETTE[2])
    ax.bar(x + 0.5 * w, p_tour[2], w, label="турнир, k = 2", color=PALETTE[3])
    ax.bar(x + 1.5 * w, p_tour[3], w, label="турнир, k = 3", color=PALETTE[1])
    ax.set_xticks(x, [f"{j}\nf={v:g}" for j, v in zip(i, f)])
    ax.set_xlabel("номер особи в порядке убывания приспособленности")
    ax.set_ylabel("вероятность выбора")
    ax.legend(ncol=2, fontsize=9)
    save(fig, "selection")


def real_crossover():
    p1, p2 = np.array([1.0, 3.0]), np.array([4.0, 1.5])
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
    d = hi - lo
    for alpha, col, ls in [(0.5, PALETTE[3], "--"), (0.0, PALETTE[0], "-")]:
        a_lo, a_hi = lo - alpha * d, hi + alpha * d
        ax.add_patch(plt.Rectangle(a_lo, *(a_hi - a_lo), fill=alpha == 0, alpha=0.18 if alpha == 0 else 1,
                                   facecolor=col, edgecolor=col, ls=ls, lw=2))
    ax.plot(*np.array([p1, p2]).T, color=PALETTE[2], lw=2.5)
    ax.scatter(*np.array([p1, p2]).T, s=120, color=INK, zorder=3)
    ax.annotate("$p^1$", p1, xytext=(-22, 6), textcoords="offset points", fontsize=12)
    ax.annotate("$p^2$", p2, xytext=(8, -4), textcoords="offset points", fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[0], alpha=0.3),
               plt.Rectangle((0, 0), 1, 1, fill=False, ec=PALETTE[3], ls="--", lw=2),
               Line2D([0], [0], color=PALETTE[2], lw=2.5)]
    ax.legend(handles, ["BLX-0 (промежуточная, d = 0)", "BLX-0,5", "арифметический / линейный"],
              loc="upper right", fontsize=9)
    ax.set_xlim(-1.5, 6.5)
    ax.set_ylim(0, 4.6)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    save(fig, "real_crossover")


def test_functions():
    fs = [
        ("1) де Йонга 2 (Розенброка)", lambda x, y: 100 * (x ** 2 - y) ** 2 + (1 - x) ** 2, 2.048, True),
        ("2) Шаффера F6", lambda x, y: 0.5 + (np.sin(np.sqrt(x ** 2 + y ** 2)) ** 2 - 0.5)
         / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2, 100, False),
        ("3) Шаффера F7", lambda x, y: (x ** 2 + y ** 2) ** 0.25
         * (np.sin(50 * (x ** 2 + y ** 2) ** 0.1) ** 2 + 1), 100, False),
        ("4) Голдстейна — Прайса", lambda x, y: (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y
                                                                            + 6 * x * y + 3 * y ** 2))
         * (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)), 2, True),
        ("5) Изома (окрестность (π; π))", lambda x, y: -np.cos(x) * np.cos(y)
         * np.exp(-((x - np.pi) ** 2 + (y - np.pi) ** 2)), (np.pi, 4), False),
        ("6) Бохачевского 1", lambda x, y: x ** 2 + 2 * y ** 2 - 0.3 * np.cos(3 * np.pi * x)
         - 0.4 * np.cos(4 * np.pi * y) + 0.7, 2, False),
        ("7) Бохачевского 2", lambda x, y: x ** 2 + 2 * y ** 2 - 0.3 * np.cos(3 * np.pi * x)
         * np.cos(4 * np.pi * y) + 0.3, 2, False),
        ("8) Растригина", lambda x, y: 20 + x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2
         - 10 * np.cos(2 * np.pi * y), 5.12, False),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4))
    for ax, (title, fn, r, logscale) in zip(axes.flat, fs):
        if isinstance(r, tuple):
            c, h = r
            xs = np.linspace(c - h, c + h, 500)
        else:
            xs = np.linspace(-r, r, 700)
        X, Y = np.meshgrid(xs, xs)
        Z = fn(X, Y)
        if logscale:
            Z = np.log10(Z - Z.min() + 1e-3)
        ax.contourf(X, Y, Z, levels=30, cmap="Blues_r")
        ax.set_title(title + ("\n(лог. шкала)" if logscale else ""), fontsize=11)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, "test_functions")


# ------------------------------------------------------- пример ЛР 1
class Perceptron:
    """Однослойный персептрон (правило Розенблатта), как в lab_01.ipynb."""

    def __init__(self, n_in, n_out=1, lr=1.0):
        self.W = np.zeros((n_out, n_in))
        self.b = np.zeros(n_out)
        self.lr = lr
        self.errors = []

    def predict(self, X):
        return (X @ self.W.T + self.b >= 0).astype(int)

    def fit(self, X, T, max_epochs=100):
        T = T.reshape(len(X), -1)
        for _ in range(max_epochs):
            err = 0
            for x, t in zip(X, T):
                e = t - self.predict(x[None])[0]
                self.W += self.lr * np.outer(e, x)
                self.b += self.lr * e
                err += int(np.any(e != 0))
            self.errors.append(err)
            if err == 0:
                break
        return self


def lab1_example():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
    # задание 1
    P = np.array([[-0.5, -0.5, 0.3, -0.1], [-0.5, 0.5, -0.5, 1.0]])
    T = np.array([0, 0, 1, 1])
    X = P.T
    net = Perceptron(2).fit(X, T)
    ax = axes[0]
    for cls, col, mk in [(0, PALETTE[1], "o"), (1, PALETTE[0], "s")]:
        m = T == cls
        ax.scatter(X[m, 0], X[m, 1], s=130, c=col, marker=mk, edgecolor=INK, zorder=3, label=f"класс {cls}")
    xs = np.linspace(-1.5, 1.5, 10)
    (w1, w2), b = net.W[0], net.b[0]
    ax.plot(xs, -(w1 * xs + b) / w2, "--", color=PALETTE[2], lw=2, label="граница")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.set_title(f"Задание 1: {len(net.errors)} эпохи")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc="upper right", fontsize=9)
    # задание 2 — XOR
    Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    Tx = np.array([0, 1, 1, 0])
    netx = Perceptron(2).fit(Xx, Tx, max_epochs=50)
    ax = axes[1]
    ax.plot(np.arange(1, len(netx.errors) + 1), netx.errors, "-o", ms=3, color=PALETTE[7])
    ax.set_ylim(-0.2, 4.2)
    ax.set_title("Задание 2: XOR не сходится")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("ошибок из 4")
    # задание 3 — 4 класса, 2 нейрона
    rng = np.random.default_rng(42)
    centers = [(-2, -2), (-2, 2), (2, -2), (2, 2)]  # коды 00, 01, 10, 11
    X4 = np.vstack([rng.normal(c, 0.3, (25, 2)) for c in centers])
    cls4 = np.repeat(np.arange(4), 25)
    T4 = np.column_stack([(cls4 >> 1) & 1, cls4 & 1])
    net4 = Perceptron(2, n_out=2).fit(X4, T4)
    ax = axes[2]
    g = np.linspace(-4, 4, 400)
    GX, GY = np.meshgrid(g, g)
    pred = net4.predict(np.column_stack([GX.ravel(), GY.ravel()]))
    code = (pred[:, 0] * 2 + pred[:, 1]).reshape(GX.shape)
    cm = matplotlib.colors.ListedColormap([PALETTE[1], PALETTE[0], PALETTE[2], PALETTE[6]])
    ax.contourf(GX, GY, code, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=cm, alpha=0.18)
    for k in range(4):
        m = cls4 == k
        ax.scatter(X4[m, 0], X4[m, 1], s=22, color=cm(k), edgecolor=INK, lw=0.4,
                   label=f"класс {k} ({k >> 1}{k & 1})")
    for j, ls in [(0, "--"), (1, ":")]:
        (w1, w2), b = net4.W[j], net4.b[j]
        if abs(w2) > 1e-9:
            ax.plot(g, -(w1 * g + b) / w2, ls, color=INK, lw=1.8)
        else:
            ax.axvline(-b / w1, ls=ls, color=INK, lw=1.8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title(f"Задание 3: 4 класса, {len(net4.errors)} эпохи")
    ax.set_xlabel("$x_1$")
    ax.legend(fontsize=8.5, loc="upper left", ncol=1, bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    save(fig, "lab1_example")
    print("  задание 1: W =", net.W, "b =", net.b, "эпох", len(net.errors))
    print("  задание 3: эпох", len(net4.errors))


# ------------------------------------------------------- пример ЛР 2
W_VAR3 = 1.5 * np.pi  # вариант 3: x(t) = sin(1,5πt), t ∈ [0; 5]


def delays(x, k):
    """Матрица задержанных входов с нулевыми начальными условиями (как в условии ЛР 2)."""
    n = len(x)
    P = np.zeros((k, n))
    for j in range(1, k + 1):
        P[j - 1, j:] = x[: n - j]
    return P


def newlind(P, T):
    """Аналог newlind: веса и смещение по МНК через псевдообратную матрицу."""
    Pb = np.vstack([P, np.ones(P.shape[1])])
    Wb = T @ np.linalg.pinv(Pb)
    return Wb[:-1], Wb[-1]


def lab2_example():
    dt, k = 0.05, 5
    t = np.arange(0, 5 + dt / 2, dt)
    x = np.sin(W_VAR3 * t)
    P = delays(x, k)
    W, b = newlind(P, x)
    y = W @ P + b
    e = y - x
    print(f"  ЛР2: N = {len(t)}, W = {np.round(W, 4)}, b = {b:.1e}, MSE = {np.mean(e ** 2):.2e}, "
          f"2cos(ωΔt) = {2 * np.cos(W_VAR3 * dt):.4f}")
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
    axes[0].plot(t, x, "*", color=PALETTE[0], ms=6, label="эталонный сигнал $x(t)$")
    axes[0].plot(t, y, "-", color=PALETTE[1], lw=1.6, label="прогноз сети $y(t)$")
    axes[0].set_ylabel("амплитуда")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title(r"$x(t)=\sin(1{,}5\pi t)$, $\Delta t=0{,}05$, $k=5$")
    axes[1].plot(t, e, "-", color=PALETTE[2], lw=1.6)
    axes[1].set_ylabel("$e = y - x$")
    axes[1].set_xlabel("t, с")
    fig.tight_layout()
    save(fig, "lab2_prediction")

    # исследование: чистый сигнал (MSE от Δt) и зашумлённый (MSE от k)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    dts = np.array([0.005, 0.01, 0.02, 0.03, 0.04, 0.05])
    for kk, mk, col in [(3, "o", PALETTE[0]), (5, "s", PALETTE[1]), (7, "^", PALETTE[2])]:
        mse = []
        for d in dts:
            tt = np.arange(0, 5 + d / 2, d)
            xx = np.sin(W_VAR3 * tt)
            Pk = delays(xx, kk)
            Wk, bk = newlind(Pk, xx)
            mse.append(np.mean((Wk @ Pk + bk - xx) ** 2))
        axes[0].loglog(dts, mse, mk, color=col, ms=8 - kk / 2, mfc="none", mew=1.8, label=f"k = {kk}")
    theory = [np.sin(W_VAR3 * d) ** 2 / len(np.arange(0, 5 + d / 2, d)) for d in dts]
    axes[0].loglog(dts, theory, "-", color=INK2, lw=1, label=r"$\sin^2(\omega\Delta t)/N$")
    axes[0].set_xticks(dts, [f"{d:g}".replace(".", ",") for d in dts])
    axes[0].minorticks_off()
    axes[0].set_xlabel(r"шаг дискретизации $\Delta t$, с")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("чистый сигнал: ошибка не зависит от k")
    axes[0].legend()
    ks = np.arange(2, 11)
    for d, col in [(0.01, PALETTE[0]), (0.02, PALETTE[1]), (0.05, PALETTE[2])]:
        tt = np.arange(0, 5 + d / 2, d)
        xc = np.sin(W_VAR3 * tt)
        res = []
        for kk in ks:
            m = []
            for s in range(30):
                xn = xc + np.random.default_rng(s).normal(0, 0.05, len(tt))
                Pk = delays(xn, kk)
                Wk, bk = newlind(Pk, xn)
                m.append(np.mean((Wk @ Pk + bk - xc)[kk:] ** 2))
            res.append(np.mean(m))
        axes[1].plot(ks, res, "-o", color=col, ms=5, label=rf"$\Delta t = {d:g}$".replace(".", "{,}"))
    axes[1].set_xlabel("размер окна k")
    axes[1].set_ylabel("MSE относительно чистого сигнала")
    axes[1].set_title(r"сигнал с шумом $\sigma = 0{,}05$ (30 реализаций)")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "lab2_study")


if __name__ == "__main__":
    activation_functions()
    separability()
    gd_valley()
    som_topology()
    selection_probs()
    real_crossover()
    test_functions()
    lab1_example()
    lab2_example()
