"""Примеры выполнения ЛР 5 и ЛР 7 (часть А) для варианта 3 по Скобцову.

ЛР 5: максимум f(t) = (t + 1,3)·sin(0,5πt + 1), t ∈ [0; 7] — простой ГА из lab_05.ipynb.
ЛР 7, часть А: минимум 7-й функции Шаффера на [−100; 100]² — вещественный ГА из lab_07.ipynb.
Алгоритмы перенесены из ноутбуков без изменений; меняется только целевая функция.

Запуск из каталога metodichka/:
    uv run --with numpy --with scipy --with matplotlib python scripts/examples_variant3.py
Графики сохраняются в figures/, числа для таблиц печатаются в консоль.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import INK, INK2, PALETTE, plt, save  # noqa: E402  (общий стиль графиков)

OUT_JSON = Path(__file__).resolve().parent.parent / "results" / "examples_variant3.json"

# ================================================================ ЛР 5
A, B = 0.0, 7.0


def f5(t):
    return (t + 1.3) * np.sin(0.5 * np.pi * t + 1)


_t = np.linspace(A, B, 700_001)
_i = int(np.argmax(f5(_t)))
_opt = minimize_scalar(lambda x: -f5(x), bounds=(_t[_i] - 1e-3, _t[_i] + 1e-3), method="bounded",
                       options=dict(xatol=1e-12))
T_STAR, F_STAR = float(_opt.x), float(-_opt.fun)
TOL = 1e-3


def decode(pop, a=A, b=B):
    L = pop.shape[1]
    return a + (pop @ (2 ** np.arange(L - 1, -1, -1))) * (b - a) / (2 ** L - 1)


def gray_decode(pop, a=A, b=B):
    """Код Грея → двоичный код (b_i = g_1 xor ... xor g_i) → вещественное значение."""
    return decode(np.bitwise_xor.accumulate(pop, axis=1), a, b)


def run_ga(pop_size=60, L=20, p_c=0.85, p_m=0.01, n_gen=200, elitism=True, gray=False, seed=0, track=False):
    """Простой ГА из lab_05.ipynb: рулетка со сдвигом, одноточечный кроссинговер, побитовая мутация."""
    decode = gray_decode if gray else globals()["decode"]
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, (pop_size, L), dtype=np.int8)
    t = decode(pop); fit = f5(t)
    best_i = int(np.argmax(fit)); best_chrom, best_fit = pop[best_i].copy(), fit[best_i]
    hist_best, hist_mean, snaps, hit_gen = [best_fit], [fit.mean()], [t.copy()] if track else None, None
    if abs(decode(best_chrom[None])[0] - T_STAR) < TOL:
        hit_gen = 0
    for g in range(1, n_gen + 1):
        w = fit - fit.min() + 1e-9
        pop = pop[rng.choice(pop_size, pop_size, p=w / w.sum())]
        n_pairs = pop_size // 2
        do = rng.random(n_pairs) < p_c
        cut = rng.integers(1, L, n_pairs)
        mask = (np.arange(L)[None, :] >= cut[:, None]) & do[:, None]
        a, b = pop[0:2 * n_pairs:2].copy(), pop[1:2 * n_pairs:2].copy()
        pop[0:2 * n_pairs:2] = np.where(mask, b, a)
        pop[1:2 * n_pairs:2] = np.where(mask, a, b)
        pop ^= (rng.random(pop.shape) < p_m).astype(np.int8)
        t = decode(pop); fit = f5(t)
        if elitism:
            w_i = int(np.argmin(fit))
            if fit[w_i] < best_fit:
                pop[w_i], t[w_i], fit[w_i] = best_chrom, decode(best_chrom[None])[0], best_fit
        cur = int(np.argmax(fit))
        if fit[cur] > best_fit:
            best_fit, best_chrom = fit[cur], pop[cur].copy()
        if hit_gen is None and abs(decode(best_chrom[None])[0] - T_STAR) < TOL:
            hit_gen = g
        hist_best.append(fit.max()); hist_mean.append(fit.mean())
        if track:
            snaps.append(t.copy())
    best_t = decode(best_chrom[None])[0]
    return dict(best_t=best_t, best_f=best_fit, hist_best=np.array(hist_best), hist_mean=np.array(hist_mean),
                snaps=snaps, hit_gen=hit_gen, success=abs(best_t - T_STAR) < TOL)


def lab5():
    res = {"t_star": T_STAR, "f_star": F_STAR}
    main = run_ga(seed=0, track=True)
    res["main"] = dict(t=main["best_t"], f=main["best_f"], err=abs(main["best_t"] - T_STAR), hit=main["hit_gen"])

    runs = [run_ga(seed=s) for s in range(50)]
    hits = [r["hit_gen"] for r in runs if r["success"]]
    fails = [(round(r["best_t"], 4), round(r["best_f"], 4)) for r in runs if not r["success"]]
    res["reliability"] = dict(success=sum(r["success"] for r in runs), median_hit=float(np.median(hits)),
                              max_hit=int(max(hits)), fails=fails)

    grid = {"pop_size": [10, 20, 60, 100], "L": [8, 12, 20, 24], "p_c": [0.0, 0.5, 0.85, 0.95],
            "p_m": [0.0, 0.001, 0.01, 0.05, 0.2]}
    rows = []
    for par, vals in grid.items():
        for v in vals:
            rr = [run_ga(**{par: v}, seed=100 + s) for s in range(30)]
            hg = [r["hit_gen"] for r in rr if r["success"]]
            rows.append(dict(par=par, value=v, success=100 * np.mean([r["success"] for r in rr]),
                             worst_f=float(min(r["best_f"] for r in rr)),
                             median_hit=float(np.median(hg)) if hg else None))
    res["params"] = rows

    el = {flag: np.mean([run_ga(p_m=0.05, elitism=flag, seed=200 + s)["hist_best"] for s in range(30)], axis=0)
          for flag in (True, False)}
    res["elitism_last"] = {str(k): float(v[-1]) for k, v in el.items()}

    # двоичный код против кода Грея: неудачи двоичного ГА — хэмминговый обрыв у t = 4,375
    res["gray"] = {}
    for gray in (False, True):
        for n, runs in ((60, 50), (20, 30)):
            rr = [run_ga(pop_size=n, gray=gray, seed=(0 if n == 60 else 100) + s) for s in range(runs)]
            hg = [r["hit_gen"] for r in rr if r["success"]]
            res["gray"][f"{'gray' if gray else 'binary'}_N{n}"] = dict(
                success=int(sum(r["success"] for r in rr)), runs=runs, median_hit=float(np.median(hg)),
                traps=sorted({round(float(r["best_t"]), 4) for r in rr if not r["success"]}))

    # графики
    tp = np.linspace(A, B, 600)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for ax, g in zip(axes, (0, 10, 200)):
        ax.plot(tp, f5(tp), color=INK2, lw=1.8)
        ax.scatter(main["snaps"][g], f5(main["snaps"][g]), s=22, color=PALETTE[1], edgecolor="white", lw=0.5,
                   zorder=3)
        ax.set_title(f"поколение {g}")
        ax.set_xlabel("t")
    axes[0].set_ylabel("f(t)")
    fig.tight_layout()
    save(fig, "lab5_population")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    gens = np.arange(len(main["hist_best"]))
    axes[0].plot(gens, main["hist_best"], color=PALETTE[0], lw=2, label="лучшее")
    axes[0].plot(gens, main["hist_mean"], color=PALETTE[1], lw=1.4, label="среднее")
    axes[0].axhline(F_STAR, color=INK, lw=0.8, ls="--")
    axes[0].set_xlabel("поколение"); axes[0].set_ylabel("f"); axes[0].legend(loc="lower right")
    axes[0].set_title("основной запуск")
    axes[1].plot(el[True], color=PALETTE[0], lw=2, label="с элитизмом")
    axes[1].plot(el[False], color=PALETTE[3], lw=1.6, label="без элитизма")
    axes[1].set_ylim(5.55, 5.72)
    axes[1].set_xlabel("поколение"); axes[1].legend(loc="lower right")
    axes[1].set_title(r"лучшая особь поколения, $P_m = 0{,}05$")
    fig.tight_layout()
    save(fig, "lab5_convergence")
    return res


# ================================================================ ЛР 7, часть А
def schaffer_f7(X):
    X = np.atleast_2d(X)
    r2 = np.sum(X ** 2, axis=1)
    return r2 ** 0.25 * (np.sin(50 * r2 ** 0.1) ** 2 + 1)


B_F7 = np.array([[-100.0, 100.0], [-100.0, 100.0]])
DEF_R = dict(pop_size=60, n_gen=150, p_c=0.9, p_m=0.2, alpha=0.5, sigma0=0.1, k=3, elite=2)


def real_ga(func=schaffer_f7, bounds=B_F7, pop_size=60, n_gen=150, p_c=0.9, p_m=0.2, alpha=0.5, sigma0=0.1, k=3,
            elite=2, seed=0, track=False):
    """Вещественный ГА из lab_07.ipynb: турнир, BLX-α, гауссова мутация с убывающим шагом, элитизм."""
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    P = rng.uniform(lo, hi, (pop_size, len(lo)))
    F = func(P)
    hb, snaps = [F.min()], [P.copy()] if track else None
    for t in range(n_gen):
        order = np.argsort(F)
        elites = P[order[:elite]].copy()
        cand = rng.integers(0, pop_size, (pop_size, k))
        par = P[cand[np.arange(pop_size), np.argmin(F[cand], axis=1)]]
        p1, p2 = par[0::2], par[1::2]
        d = np.abs(p1 - p2)
        lo_i, hi_i = np.minimum(p1, p2) - alpha * d, np.maximum(p1, p2) + alpha * d
        c1, c2 = rng.uniform(lo_i, hi_i), rng.uniform(lo_i, hi_i)
        keep = rng.random(len(p1)) >= p_c
        c1[keep], c2[keep] = p1[keep], p2[keep]
        kids = np.vstack([c1, c2])
        sigma = sigma0 * (hi - lo) * (1 - t / n_gen) + 1e-4
        m = rng.random(kids.shape) < p_m
        kids = np.clip(kids + m * rng.normal(0, 1, kids.shape) * sigma, lo, hi)
        kids[:elite] = elites
        P, F = kids, func(kids)
        hb.append(F.min())
        if track:
            snaps.append(P.copy())
    i = int(np.argmin(F))
    return dict(x=P[i], f=F[i], hist_best=np.array(hb), snaps=snaps)


def binary_ga(func, bounds, pop_size=60, n_gen=150, p_c=0.9, L=20, k=3, seed=0):
    """Бинарный ГА из lab_07.ipynb (как в ЛР 6) для сравнения при равном бюджете."""
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    w = 2 ** np.arange(L - 1, -1, -1)

    def dec(P):
        return np.column_stack([lo[j] + (P[:, j * L:(j + 1) * L] @ w) * (hi[j] - lo[j]) / (2 ** L - 1)
                                for j in range(len(lo))])

    n = L * len(lo)
    P = rng.integers(0, 2, (pop_size, n), dtype=np.int8)
    F = func(dec(P))
    for _ in range(n_gen):
        cand = rng.integers(0, pop_size, (pop_size, k))
        par = P[cand[np.arange(pop_size), np.argmin(F[cand], axis=1)]]
        p1, p2 = par[0::2], par[1::2]
        m = rng.integers(0, 2, p1.shape).astype(bool) & (rng.random(len(p1)) < p_c)[:, None]
        K = np.vstack([np.where(m, p2, p1), np.where(m, p1, p2)])
        K ^= (rng.random(K.shape) < 1 / n).astype(np.int8)
        KF = func(dec(K))
        allP, allF = np.vstack([P, K]), np.r_[F, KF]
        o = np.argsort(allF)[:pop_size]
        P, F = allP[o], allF[o]
    return dict(f=F.min())


def lab7a():
    res = {}
    main = real_ga(**DEF_R, seed=0, track=True)
    res["main"] = dict(x=main["x"].tolist(), f=float(main["f"]), evals=DEF_R["pop_size"] * (DEF_R["n_gen"] + 1))
    hit = next((g for g, v in enumerate(main["hist_best"]) if v < 1e-3), None)
    res["main"]["gen_below_1e-3"] = hit

    cmp = {}
    for enc, runner in [("binary", lambda s: binary_ga(schaffer_f7, B_F7, seed=700 + s)),
                        ("real", lambda s: real_ga(schaffer_f7, B_F7, **DEF_R, seed=700 + s))]:
        fb = np.array([runner(s)["f"] for s in range(20)])
        cmp[enc] = dict(median=float(np.median(fb)), best=float(fb.min()), worst=float(fb.max()))
    res["real_vs_binary"] = cmp

    grid = {"alpha": [0.0, 0.25, 0.5, 1.0], "sigma0": [0.0, 0.02, 0.1, 0.3], "p_m": [0.0, 0.05, 0.2, 0.5],
            "pop_size": [20, 40, 60, 120]}
    rows = []
    for par, vals in grid.items():
        for v in vals:
            fb = np.array([real_ga(**{**DEF_R, "n_gen": 50, par: v}, seed=300 + s)["f"] for s in range(20)])
            rows.append(dict(par=par, value=v, median=float(np.median(fb)), worst=float(fb.max()),
                             success=float(np.mean(fb < 1e-2) * 100)))
    res["params"] = rows

    # популяция на линиях уровня: масштаб осей уменьшается вместе с разбросом популяции
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, g, r in zip(axes, (0, 10, 150), (100, 10, 1)):
        xs = np.linspace(-r, r, 500)
        X1, X2 = np.meshgrid(xs, xs)
        Z = schaffer_f7(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)
        ax.contourf(X1, X2, Z, levels=25, cmap="Blues_r")
        P = main["snaps"][g]
        ax.scatter(P[:, 0], P[:, 1], s=16, color=PALETTE[1], edgecolor="white", lw=0.5, zorder=3)
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_aspect("equal"); ax.grid(False)
        ax.set_title(f"поколение {g}, $[-{r};\\ {r}]^2$")
        ax.tick_params(labelsize=9)
    fig.tight_layout()
    save(fig, "lab7a_population")
    return res


if __name__ == "__main__":
    out = {"lab5": lab5(), "lab7a": lab7a()}
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
