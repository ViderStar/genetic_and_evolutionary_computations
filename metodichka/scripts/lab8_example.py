"""Пример выполнения ЛР 8 (вариант 3): CMA-ES, подбор гиперпараметров, NSGA-II.

Часть А: CMA-ES и IPOP-CMA-ES против случайного поиска и вещественного ГА из ЛР 6 на функции
         Шаффера F7; проверка своей реализации по библиотеке pycma.
Часть Б: подбор (C, gamma) для SVC на breast_cancer: сетка, случайный поиск, свой ГА,
         Optuna TPE и Optuna CMA-ES при бюджете 30 обучений.
Часть В: своя реализация NSGA-II против pymoo на ZDT1 и многокритериальный отбор признаков.

Запуск из каталога metodichka/:
    uv run --with numpy --with scipy --with scikit-learn --with optuna --with cmaes --with cma \
        --with pymoo --with matplotlib python scripts/lab8_example.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "listings"))
from cma_es import cmaes, ipop_cmaes  # noqa: E402
from examples_variant3 import real_ga, schaffer_f7, B_F7, DEF_R  # noqa: E402
from make_figures import INK, INK2, PALETTE, plt, save  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "results" / "lab8_example.json"
RUNS = 20
BUDGET = 9060                      # как у вещественного ГА ЛР 6: 60 особей x 151 поколение


def median_curve(hists, grid):
    """Медиана по запускам кривой «лучшее f от числа вычислений» на общей сетке."""
    curves = []
    for h in hists:
        e, v = np.asarray(h)[:, 0], np.asarray(h)[:, 1]
        idx = np.searchsorted(e, grid, side="right") - 1
        curves.append(np.where(idx >= 0, v[np.clip(idx, 0, None)], np.nan))
    return np.nanmedian(np.array(curves), axis=0)


# ======================================================================== часть А
def part_a():
    res, hists = {}, {}
    grid = np.linspace(60, BUDGET, 300)

    rs = []
    for s in range(RUNS):
        rng = np.random.default_rng(500 + s)
        F = schaffer_f7(rng.uniform(-100, 100, (BUDGET, 2)))
        best = np.minimum.accumulate(F)
        rs.append(np.c_[np.arange(1, BUDGET + 1), best])
    hists["случайный поиск"] = rs

    ga = []
    for s in range(RUNS):
        r = real_ga(**DEF_R, seed=700 + s)
        gens = np.arange(len(r["hist_best"]))
        ga.append(np.c_[(gens + 1) * DEF_R["pop_size"], np.minimum.accumulate(r["hist_best"])])
    hists["вещественный ГА (ЛР 6)"] = ga

    cm = []
    for s in range(RUNS):
        x0 = np.random.default_rng(s).uniform(-100, 100, 2)
        _, _, h = cmaes(schaffer_f7, x0, 60.0, B_F7, BUDGET, seed=s)
        cm.append(h)
    hists["CMA-ES"] = cm

    ip = [ipop_cmaes(schaffer_f7, B_F7, BUDGET, seed=s)[2] for s in range(RUNS)]
    hists["IPOP-CMA-ES"] = ip

    for name, hs in hists.items():
        final = np.array([h[-1][1] for h in hs])
        hit = [next((e for e, v in h if v < 1e-3), None) for h in hs]
        hit = [e for e in hit if e is not None]
        res[name] = dict(median=float(np.median(final)), worst=float(final.max()),
                         success=float(np.mean(final < 1e-3) * 100),
                         evals_to_1e3=float(np.median(hit)) if hit else None)

    import cma
    lib = []
    for s in range(RUNS):
        x0 = np.random.default_rng(s).uniform(-100, 100, 2)
        es = cma.CMAEvolutionStrategy(x0, 60.0, {"bounds": [-100, 100], "seed": s + 1, "verbose": -9,
                                                 "maxfevals": BUDGET})
        es.optimize(lambda v: float(schaffer_f7(np.atleast_2d(v))[0]))
        lib.append(es.result.fbest)
    lib = np.array(lib)
    res["pycma (без перезапусков)"] = dict(median=float(np.median(lib)), worst=float(lib.max()),
                                           success=float(np.mean(lib < 1e-3) * 100), evals_to_1e3=None)

    # рисунок: эллипсы CMA-ES на функции Розенброка (слева) и сходимость на F7 (справа)
    def rosen(X):
        return 100 * (X[:, 0] ** 2 - X[:, 1]) ** 2 + (1 - X[:, 0]) ** 2
    trace = []
    cmaes(rosen, np.array([-1.5, 1.8]), 0.4, np.array([[-2.048, 2.048]] * 2), 1500, seed=3, trace=trace)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axes[0]
    g1 = np.linspace(-2, 2, 400)
    X1, X2 = np.meshgrid(g1, np.linspace(-1, 3, 400))
    Z = np.log10(rosen(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape) + 1e-3)
    ax.contourf(X1, X2, Z, levels=25, cmap="Blues_r")
    t = np.linspace(0, 2 * np.pi, 100)
    for k, g in enumerate((0, 12, 24, 48, 66, 102)):
        if g >= len(trace):
            continue
        m, sig, C = trace[g]
        vals, vecs = np.linalg.eigh(C)
        ell = m[:, None] + sig * vecs @ (np.sqrt(vals)[:, None] * np.vstack([np.cos(t), np.sin(t)]))
        ax.plot(*ell, color=PALETTE[1], lw=1.6)
        ax.plot(*m, "o", color=PALETTE[1], ms=3)
        ax.annotate(str(g), m, xytext=(4, 4), textcoords="offset points", fontsize=9, color=INK)
    ax.plot(1, 1, "*", color=INK, ms=12)
    ax.set_title("CMA-ES на функции Розенброка")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.grid(False)
    ax = axes[1]
    styles = {"случайный поиск": (INK2, ":"), "вещественный ГА (ЛР 6)": (PALETTE[2], "-"),
              "CMA-ES": (PALETTE[1], "--"), "IPOP-CMA-ES": (PALETTE[0], "-")}
    for name, hs in hists.items():
        col, ls = styles[name]
        ax.semilogy(grid, median_curve(hs, grid), ls, color=col, lw=2, label=name)
    ax.set_xlabel("вычислений функции"); ax.set_ylabel("лучшее f (медиана 20 запусков)")
    ax.set_title("функция Шаффера F7")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "lab8_cmaes")
    return res


# ======================================================================== часть Б
def part_b():
    import optuna
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    lo, hi = np.array([-2.0, -6.0]), np.array([4.0, 0.0])        # log10 C, log10 gamma

    def model(p):
        return make_pipeline(StandardScaler(), SVC(C=10 ** p[0], gamma=10 ** p[1]))

    def score(p):
        return cross_val_score(model(p), Xtr, ytr, cv=cv).mean()

    def test_acc(p):
        return model(p).fit(Xtr, ytr).score(Xte, yte)

    T, SEEDS = 30, 10
    runs = {m: [] for m in ("сетка 6x5", "случайный поиск", "ГА (N = 6, 5 поколений)", "Optuna TPE",
                            "Optuna CMA-ES")}

    gl = np.linspace(lo[0], hi[0], 6)
    gg = np.linspace(lo[1], hi[1], 5)
    pts = [np.array([a, b]) for a in gl for b in gg]
    runs["сетка 6x5"].append([(p, score(p)) for p in pts])

    for s in range(SEEDS):
        rng = np.random.default_rng(s)
        runs["случайный поиск"].append([(p, score(p)) for p in rng.uniform(lo, hi, (T, 2))])

        # маленький вещественный ГА: турнир 2, BLX-0,5, гауссова мутация, одна элита
        P = rng.uniform(lo, hi, (6, 2)); F = np.array([score(p) for p in P]); hist = list(zip(P, F))
        for g in range(5):
            cand = rng.integers(0, 6, (6, 2))
            par = P[cand[np.arange(6), F[cand].argmax(axis=1)]]
            a, b = par[0::2], par[1::2]
            d = np.abs(a - b)
            kids = rng.uniform(np.minimum(a, b) - 0.5 * d, np.maximum(a, b) + 0.5 * d, (2, 3, 2)).reshape(6, 2)
            kids += (rng.random(kids.shape) < 0.3) * rng.normal(0, 0.1 * (hi - lo), kids.shape)
            kids = np.clip(kids, lo, hi)
            kids[0] = P[F.argmax()]
            KF = np.array([F.max()] + [score(p) for p in kids[1:]])
            hist += list(zip(kids[1:], KF[1:]))
            P, F = kids, KF
        runs["ГА (N = 6, 5 поколений)"].append(hist[:T])

        for name, sampler in (("Optuna TPE", optuna.samplers.TPESampler(seed=s)),
                              ("Optuna CMA-ES", optuna.samplers.CmaEsSampler(seed=s))):
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(lambda tr: score(np.array([tr.suggest_float("logC", lo[0], hi[0]),
                                                      tr.suggest_float("logG", lo[1], hi[1])])),
                           n_trials=T)
            runs[name].append([(np.array([tr.params["logC"], tr.params["logG"]]), tr.value)
                               for tr in study.trials])

    res = {}
    for name, rr in runs.items():
        best = [max(h, key=lambda z: z[1]) for h in rr]
        cvb = np.array([b[1] for b in best])
        te = np.array([test_acc(b[0]) for b in best])
        curves = np.array([np.maximum.accumulate([z[1] for z in h]) for h in rr])
        trials_to = [int(np.argmax(c >= 0.98)) + 1 if (c >= 0.98).any() else None for c in curves]
        trials_to = [t for t in trials_to if t is not None]
        res[name] = dict(cv_median=float(np.median(cvb)), cv_worst=float(cvb.min()),
                         test_median=float(np.median(te)), curve=np.median(curves, axis=0).tolist(),
                         reach_098=f"{len(trials_to)}/{len(rr)}",
                         trials_to_098=float(np.median(trials_to)) if trials_to else None)

    # карта точности на сетке 31 x 31 для рисунка
    g1, g2 = np.linspace(lo[0], hi[0], 31), np.linspace(lo[1], hi[1], 31)
    Zacc = np.array([[score(np.array([a, b])) for a in g1] for b in g2])
    res["landscape_max"] = float(Zacc.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw=dict(width_ratios=[1, 1.2]))
    ax = axes[0]
    cs = ax.contourf(g1, g2, Zacc, levels=np.linspace(0.6, 1.0, 21), cmap="Blues", extend="min")
    for name, col, mk in (("Optuna TPE", PALETTE[1], "o"), ("случайный поиск", INK2, "x")):
        P = np.array([z[0] for z in runs[name][0]])
        ax.scatter(P[:, 0], P[:, 1], s=18, color=col, marker=mk, label=name, zorder=3)
    ax.set_xlabel(r"$\lg C$"); ax.set_ylabel(r"$\lg \gamma$"); ax.grid(False)
    ax.set_title("точность CV и 30 проб одного запуска", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.03)
    ax = axes[1]
    cols = {"сетка 6x5": INK2, "случайный поиск": "#9ec5f4", "ГА (N = 6, 5 поколений)": PALETTE[2],
            "Optuna TPE": PALETTE[1], "Optuna CMA-ES": PALETTE[0]}
    for name in runs:
        ax.plot(np.arange(1, T + 1), res[name]["curve"], color=cols[name], lw=2, label=name)
    ax.set_ylim(0.9, 0.99)
    ax.set_xlabel("число обучений модели"); ax.set_ylabel("лучшая точность CV (медиана)")
    ax.set_title("SVC на breast_cancer", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    save(fig, "lab8_hpo")
    for v in res.values():
        if isinstance(v, dict):
            v.pop("curve", None)
    return res


# ======================================================================== часть В
def fronts_of(F):
    """Быстрая недоминируемая сортировка: список фронтов (массивов индексов)."""
    dom = (F[:, None] <= F[None]).all(-1) & (F[:, None] < F[None]).any(-1)   # dom[i, j]: i доминирует j
    cnt = dom.sum(0)
    fronts, cur = [], np.where(cnt == 0)[0]
    while len(cur):
        fronts.append(cur)
        cnt = cnt - dom[cur].sum(0)
        cnt[np.concatenate(fronts)] = -1
        cur = np.where(cnt == 0)[0]
    return fronts


def crowding(F):
    """Расстояние скученности: сумма нормированных сторон «кубоида» соседей."""
    n, d = F.shape
    dist = np.zeros(n)
    for j in range(F.shape[1]):
        o = np.argsort(F[:, j])
        span = F[o[-1], j] - F[o[0], j] or 1.0
        dist[o[0]] = dist[o[-1]] = np.inf
        dist[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / span
    return dist


def nsga2(evaluate, init, vary, pop_size, n_gen, rng):
    P = init(pop_size)
    F = evaluate(P)
    for _ in range(n_gen):
        rank, crowd = np.empty(len(P), int), np.empty(len(P))
        for r, fr in enumerate(fronts_of(F)):
            rank[fr], crowd[fr] = r, crowding(F[fr])
        cand = rng.integers(0, len(P), (pop_size, 2))              # бинарный турнир скученности
        a, b = cand[:, 0], cand[:, 1]
        better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
        Q = vary(P[np.where(better, a, b)])
        R, FR = np.vstack([P, Q]), np.vstack([F, evaluate(Q)])
        keep = []
        for fr in fronts_of(FR):                                   # элитарный отбор (mu + lambda)
            if len(keep) + len(fr) <= pop_size:
                keep += list(fr)
            else:
                keep += list(fr[np.argsort(-crowding(FR[fr]))[:pop_size - len(keep)]])
                break
        P, F = R[keep], FR[keep]
    return P, F


def sbx_pm(rng, lo=0.0, hi=1.0, eta_c=15, eta_m=20, pc=0.9):
    """Операторы по Дебу: имитация двоичного кроссинговера (SBX) и полиномиальная мутация."""
    def vary(par):
        a, b = par[0::2], par[1::2]
        u = rng.random(a.shape)
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
        do = (rng.random((len(a), 1)) < pc) & (rng.random(a.shape) < 0.5)
        beta = np.where(do, beta, 1.0)
        c1, c2 = 0.5 * ((1 + beta) * a + (1 - beta) * b), 0.5 * ((1 - beta) * a + (1 + beta) * b)
        swap = rng.random(a.shape) < 0.5                  # обмен генами между потомками
        K = np.vstack([np.where(swap, c2, c1), np.where(swap, c1, c2)])
        u = rng.random(K.shape)
        delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
        K = K + (rng.random(K.shape) < 1 / K.shape[1]) * delta * (hi - lo)
        return np.clip(K, lo, hi)
    return vary


def zdt1(X):
    f1 = X[:, 0]
    g = 1 + 9 * X[:, 1:].mean(axis=1)
    return np.c_[f1, g * (1 - np.sqrt(f1 / g))]


def part_c():
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.indicators.hv import HV
    from pymoo.indicators.igd import IGD
    from pymoo.optimize import minimize
    from pymoo.problems import get_problem
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    res = {}
    prob = get_problem("zdt1")
    pf = prob.pareto_front()
    igd, hv = IGD(pf), HV(ref_point=np.array([1.1, 1.1]))
    own, lib = [], []
    for s in range(5):
        rng = np.random.default_rng(s)
        t0 = time.perf_counter()
        _, F = nsga2(zdt1, lambda n: rng.random((n, 30)), sbx_pm(rng), 100, 200, rng)
        own.append((igd(F), hv(F), time.perf_counter() - t0))
        t0 = time.perf_counter()
        r = minimize(prob, NSGA2(pop_size=100), ("n_gen", 200), seed=s + 1, verbose=False)
        lib.append((igd(r.F), hv(r.F), time.perf_counter() - t0))
        if s == 0:
            F_own, F_lib = F, r.F
    for name, arr in (("своя NSGA-II", own), ("pymoo NSGA2", lib)):
        a = np.array(arr)
        res[name] = dict(igd=float(np.median(a[:, 0])), hv=float(np.median(a[:, 1])),
                         time=float(np.median(a[:, 2])))
    res["hv_true_front"] = float(hv(pf))

    # многокритериальный отбор признаков: (ошибка CV, число признаков)
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cache = {}

    def lr():
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

    def err(mask):
        key = mask.tobytes()
        if key not in cache:
            cache[key] = 1.0 if not mask.any() else 1 - cross_val_score(lr(), Xtr[:, mask], ytr, cv=cv).mean()
        return cache[key]

    def evaluate(P):
        return np.array([[err(p.astype(bool)), p.sum()] for p in P], float)

    rng = np.random.default_rng(3)
    n = X.shape[1]

    def init(k):
        return (rng.random((k, n)) < rng.uniform(0.05, 0.5, (k, 1))).astype(np.int8)

    def vary(par):
        a, b = par[0::2], par[1::2]
        m = rng.random(a.shape) < 0.5                                  # однородный кроссинговер
        K = np.vstack([np.where(m, a, b), np.where(m, b, a)])
        return K ^ (rng.random(K.shape) < 1 / n).astype(np.int8)       # побитовая мутация

    t0 = time.perf_counter()
    P, F = nsga2(evaluate, init, vary, 40, 25, rng)
    res["fs_time"] = time.perf_counter() - t0
    res["fs_evals"] = len(cache)
    first = fronts_of(F)[0]
    pts = sorted({(int(F[i, 1]), float(F[i, 0])): P[i] for i in first if F[i, 1] > 0}.items())
    table = []
    for (k, e), mask in pts:
        mask = mask.astype(bool)
        te = lr().fit(Xtr[:, mask], ytr).score(Xte[:, mask], yte)
        table.append(dict(k=k, cv_acc=1 - e, test_acc=te))
    full_cv = 1 - err(np.ones(n, bool))
    full_te = lr().fit(Xtr, ytr).score(Xte, yte)
    res["fs_front"] = table
    res["fs_all"] = dict(k=n, cv_acc=full_cv, test_acc=full_te)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    ax = axes[0]
    ax.plot(pf[:, 0], pf[:, 1], "-", color=INK2, lw=1.5, label="истинный фронт")
    ax.scatter(F_lib[:, 0], F_lib[:, 1], s=14, color=PALETTE[0], label="pymoo NSGA2")
    ax.scatter(F_own[:, 0], F_own[:, 1], s=14, color=PALETTE[1], marker="x", label="своя NSGA-II")
    ax.set_xlabel("$f_1$"); ax.set_ylabel("$f_2$"); ax.set_title("ZDT1, 100 особей x 200 поколений")
    ax.legend(fontsize=9)
    ax = axes[1]
    ks = [r["k"] for r in table]
    ax.step(ks, [r["cv_acc"] for r in table], where="post", color=PALETTE[0], lw=2, label="точность CV (фронт)")
    ax.plot(ks, [r["cv_acc"] for r in table], "o", color=PALETTE[0], ms=4)
    ax.plot(ks, [r["test_acc"] for r in table], "s", color=PALETTE[1], ms=4, label="точность на тесте")
    ax.axhline(full_cv, color=INK2, ls="--", lw=1, label=f"все {n} признаков, CV")
    ax.set_xlabel("число признаков"); ax.set_ylabel("точность")
    ax.set_title("отбор признаков для логистической регрессии", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    save(fig, "lab8_pareto")
    return res


if __name__ == "__main__":
    t0 = time.perf_counter()
    out = {"part_a": part_a()}
    print("A done", time.perf_counter() - t0)
    out["part_b"] = part_b()
    print("B done", time.perf_counter() - t0)
    out["part_c"] = part_c()
    print("C done", time.perf_counter() - t0)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
