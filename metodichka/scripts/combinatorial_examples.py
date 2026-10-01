"""Примеры к подразделам о задаче о рюкзаке и о минимальном покрытии (раздел 7 теории).

Сравниваются способы учёта ограничений в ГА (по Скобцову): штрафы трёх видов, восстановление
(случайное и жадное) и декодер с порядковым представлением. Эталон — точное решение: динамическое
программирование для рюкзака и MILP (scipy.optimize.milp, решатель HiGHS) для покрытия.

Запуск из каталога metodichka/:
    uv run --with numpy --with scipy --with matplotlib python scripts/combinatorial_examples.py
"""
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
RUNS = 20


# ================================================================== общий ГА
def tournament(fit, rng, k=3):
    idx = rng.integers(0, len(fit), (len(fit), k))
    return idx[np.arange(len(fit)), np.argmax(fit[idx], axis=1)]


def ga(evaluate, init, crossover, mutate, rng, N=100, G=200, pc=0.9, elite=2):
    """Поколенческий ГА с турниром и элитой. evaluate(pop) -> (fit, feasible, objective, pop)."""
    pop = init(N)
    fit, feas, obj, pop = evaluate(pop)
    best, curve = -np.inf, []

    def track():
        nonlocal best
        if feas.any():
            best = max(best, obj[feas].max())
        curve.append(best)

    track()
    for _ in range(G - 1):
        par = pop[tournament(fit, rng)]
        kids = par.copy()
        for i in range(0, N - 1, 2):
            if rng.random() < pc:
                kids[i], kids[i + 1] = crossover(par[i], par[i + 1])
        kids = mutate(kids)
        kfit, kfeas, kobj, kids = evaluate(kids)
        el = np.argsort(fit)[-elite:]                     # элиты переходят без изменений
        pop = np.vstack([kids[: N - elite], pop[el]])
        fit = np.concatenate([kfit[: N - elite], fit[el]])
        feas = np.concatenate([kfeas[: N - elite], feas[el]])
        obj = np.concatenate([kobj[: N - elite], obj[el]])
        track()
    final_best_feasible = bool(feas[np.argmax(fit)])      # допустима ли лучшая по фитнесу особь
    return best, np.array(curve), final_best_feasible


def uniform_cx(rng):
    def cx(a, b):
        m = rng.random(a.shape) < 0.5
        return np.where(m, a, b), np.where(m, b, a)
    return cx


def bitflip(rng, pm):
    return lambda P: P ^ (rng.random(P.shape) < pm)


# ================================================================== рюкзак
def knap_instance(n=100, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.integers(10, 101, n)
    c = np.maximum(w + rng.integers(-10, 11, n), 1)      # слабая корреляция ценности с весом
    return w, c, int(0.25 * w.sum())


def knap_dp(w, c, W):
    best = np.zeros(W + 1, dtype=np.int64)
    for wi, ci in zip(w, c):
        best[wi:] = np.maximum(best[wi:], best[:-wi] + ci)   # правая часть считается по старым значениям
    return int(best[-1])


def knap_greedy(w, c, W):
    load = value = 0
    for i in np.argsort(-c / w, kind="stable"):
        if load + w[i] <= W:
            load, value = load + w[i], value + c[i]
    return int(value)


def knap_penalty(w, c, W, kind):
    rho = (c / w).max()

    def evaluate(P):
        excess = np.maximum(P @ w - W, 0)
        pen = {"log": np.log2(1 + rho * excess), "lin": rho * excess, "sq": (rho * excess) ** 2}[kind]
        val = (P @ c).astype(float)
        return val - pen, excess == 0, val, P
    return evaluate


def knap_repair(w, c, W, rng, greedy, p_write=0.05):
    by_ratio = np.argsort(c / w, kind="stable")           # сначала удаляются предметы с малым c/w

    def evaluate(P):
        R = P.copy()
        for r in R:
            over = r @ w - W
            if over <= 0:
                continue
            order = by_ratio if greedy else rng.permutation(len(w))
            inside = order[r[order]]
            k = np.searchsorted(np.cumsum(w[inside]), over)   # сколько предметов вынуть
            r[inside[: k + 1]] = False
        val = (R @ c).astype(float)
        write = rng.random(len(P)) < p_write              # часть восстановленных особей заменяет исходные
        P = np.where(write[:, None], R, P)
        return val, np.ones(len(P), bool), val, P
    return evaluate


def knap_decoder(w, c, W):
    base = list(np.argsort(-c / w, kind="stable"))        # базовый список L — «жадный» порядок

    def decode(e):
        L, load, value = base.copy(), 0, 0
        for g in e:
            i = L.pop(g)
            if load + w[i] <= W:
                load, value = load + w[i], value + c[i]
        return value

    def evaluate(E):
        val = np.array([decode(e) for e in E], float)
        return val, np.ones(len(E), bool), val, E
    return evaluate


def run_knapsack():
    w, c, W = knap_instance()
    n = len(w)
    t0 = time.perf_counter()
    opt = knap_dp(w, c, W)
    t_dp = time.perf_counter() - t0
    res = dict(n=n, W=W, sum_w=int(w.sum()), optimum=opt, dp_seconds=t_dp, greedy=knap_greedy(w, c, W))
    methods = {}
    for name in ("log", "lin", "sq", "repair_random", "repair_greedy", "decoder", "decoder_seed"):
        bests, curves, final_ok = [], [], []
        for s in range(RUNS):
            rng = np.random.default_rng(s)
            if name.startswith("decoder"):
                hi = n - np.arange(n)                      # ген i принимает значения 0 .. n-i-1

                def init(N, seed_greedy=name == "decoder_seed"):
                    E = (rng.random((N, n)) * hi).astype(int)
                    if seed_greedy:
                        E[0] = 0                           # нулевые гены декодируются в жадное решение
                    return E

                def cx(a, b):
                    p = rng.integers(1, n)
                    return np.r_[a[:p], b[p:]], np.r_[b[:p], a[p:]]

                def mut(E):
                    m = rng.random(E.shape) < 1 / n
                    return np.where(m, (rng.random(E.shape) * hi).astype(int), E)
                ev = knap_decoder(w, c, W)
            else:
                init = lambda N: rng.random((N, n)) < 0.5
                cx, mut = uniform_cx(rng), bitflip(rng, 1 / n)
                ev = (knap_penalty(w, c, W, name) if name in ("log", "lin", "sq")
                      else knap_repair(w, c, W, rng, greedy=name == "repair_greedy"))
            b, curve, ok = ga(ev, init, cx, mut, rng)
            bests.append(b), curves.append(curve), final_ok.append(ok)
        bests = np.array(bests)
        feas = np.isfinite(bests)
        methods[name] = dict(
            median_pct=float(np.median(100 * bests[feas] / opt)) if feas.any() else None,
            worst_pct=float(np.min(100 * bests[feas] / opt)) if feas.any() else None,
            optimum_found=int((bests == opt).sum()), runs=RUNS, runs_with_feasible=int(feas.sum()),
            final_best_infeasible=int(RUNS - sum(final_ok)),
            curve=np.median(np.where(np.isfinite(curves), curves, np.nan), axis=0).tolist())
        print("рюкзак", name, {k: v for k, v in methods[name].items() if k != "curve"})
    res["methods"] = methods
    return res


# ================================================================== минимальное покрытие
def scp_instance(m=200, n=1000, density=0.02, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random((m, n)) < density
    for i in range(m):                                    # каждую строку покрывают не менее двух столбцов
        if A[i].sum() < 2:
            A[i, rng.choice(n, 2, replace=False)] = True
    for j in np.where(~A.any(axis=0))[0]:                 # и каждый столбец что-то покрывает
        A[rng.integers(m), j] = True
    return A, rng.integers(1, 101, n)


def scp_exact(A, cost):
    t0 = time.perf_counter()
    r = milp(cost, integrality=np.ones(len(cost)), bounds=Bounds(0, 1),
             constraints=LinearConstraint(A.astype(float), lb=1, ub=np.inf))
    return int(round(r.fun)), time.perf_counter() - t0


class Cover:
    def __init__(self, A, cost):
        self.A, self.cost = A, cost
        self.rows_of = [np.flatnonzero(A[:, j]) for j in range(A.shape[1])]
        self.cols_of = [np.flatnonzero(A[i]) for i in range(A.shape[0])]
        self.by_cost_desc = np.argsort(-cost, kind="stable")

    def repair(self, x):
        """Добавить дешёвые столбцы для непокрытых строк, затем удалить избыточные (Beasley, Chu)."""
        x = x.copy()
        cnt = self.A[:, x].sum(axis=1)
        for i in np.flatnonzero(cnt == 0):
            if cnt[i] > 0:
                continue
            cand = self.cols_of[i]
            gain = np.array([(cnt[self.rows_of[j]] == 0).sum() for j in cand])
            j = cand[np.argmin(self.cost[cand] / gain)]
            x[j] = True
            cnt[self.rows_of[j]] += 1
        for j in self.by_cost_desc[x[self.by_cost_desc]]:
            if (cnt[self.rows_of[j]] >= 2).all():
                x[j] = False
                cnt[self.rows_of[j]] -= 1
        return x

    def greedy(self):
        return self.repair(np.zeros(self.A.shape[1], bool))


def run_cover():
    A, cost = scp_instance()
    opt, t_milp = scp_exact(A, cost)
    cov = Cover(A, cost)
    g = cov.greedy()
    res = dict(m=A.shape[0], n=A.shape[1], density=float(A.mean()), optimum=opt, milp_seconds=t_milp,
               greedy=int(cost[g].sum()), greedy_cols=int(g.sum()))
    print("покрытие: оптимум", opt, "за", round(t_milp, 2), "с; жадный", res["greedy"])
    P_row, At = cost.max(), A.T.astype(np.int32)

    def ev_penalty(P):
        unc = (P.astype(np.int32) @ At == 0).sum(axis=1)
        val = -(P @ cost).astype(float)
        return val - P_row * unc, unc == 0, val, P

    def ev_repair(P):
        P = np.array([cov.repair(x) for x in P])          # восстановленная особь заменяет исходную
        val = -(P @ cost).astype(float)
        return val, np.ones(len(P), bool), val, P

    methods = {}
    n = A.shape[1]
    for name, ev, p0 in (("penalty", ev_penalty, 0.5), ("repair", ev_repair, 0.02)):
        bests, curves, secs = [], [], []
        for s in range(10):
            rng = np.random.default_rng(s)
            t0 = time.perf_counter()
            b, curve, _ = ga(ev, lambda N: rng.random((N, n)) < p0, uniform_cx(rng), bitflip(rng, 1 / n), rng,
                             N=100, G=200)
            secs.append(time.perf_counter() - t0)
            bests.append(-b), curves.append(-curve)
        bests = np.array(bests)
        methods[name] = dict(median=float(np.median(bests)), best=float(bests.min()), worst=float(bests.max()),
                             optimum_found=int((bests == opt).sum()), runs=10, seconds=float(np.median(secs)),
                             curve=np.median(np.where(np.isfinite(curves), curves, np.nan), axis=0).tolist())
        print("покрытие", name, {k: v for k, v in methods[name].items() if k != "curve"})
    res["methods"] = methods
    return res


# ================================================================== маленькие примеры для текста
def toy_examples():
    w, c, W = np.array([16, 15, 15, 8, 6, 5]), np.array([34, 30, 31, 15, 11, 8]), 30
    best = max((np.array(x) for x in product([0, 1], repeat=6)), key=lambda x: (x @ w <= W) * (x @ c))
    x = np.array([1, 1, 0, 1, 0, 0])
    rho, exc = (c / w).max(), max(0, x @ w - W)
    pen = dict(log=np.log2(1 + rho * exc), lin=rho * exc, sq=(rho * exc) ** 2)
    base = list(np.argsort(-c / w, kind="stable"))

    def decode(e):                                        # гены с единицы, как в тексте
        L, load, val, taken = base.copy(), 0, 0, []
        for g in e:
            i = L.pop(g - 1)
            if load + w[i] <= W:
                load, val = load + w[i], val + c[i]
                taken.append(int(i) + 1)
        return int(val), taken
    return dict(knap=dict(w=w.tolist(), c=c.tolist(), W=W, ratio=(c / w).round(3).tolist(),
                          x=x.tolist(), x_weight=int(x @ w), x_value=int(x @ c), rho=float(rho),
                          fitness={k: float(x @ c - v) for k, v in pen.items()},
                          optimum=best.tolist(), opt_value=int(best @ c), opt_weight=int(best @ w),
                          greedy=knap_greedy(w, c, W), base_list=[int(i) + 1 for i in base],
                          decode_111111=decode([1] * 6), decode_211111=decode([2, 1, 1, 1, 1, 1])))


def plot(knap, cover):
    from make_figures import INK2, PALETTE, plt, save
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    # логарифмический штраф допустимых решений не дал, поэтому на рисунке его нет
    names = {"lin": "штраф линейный", "sq": "штраф квадратичный",
             "repair_random": "восстановление случайное", "repair_greedy": "восстановление жадное",
             "decoder": "декодер", "decoder_seed": "декодер + жадная особь"}
    styles = {"lin": (PALETTE[3], "--"), "sq": (PALETTE[2], "-."),
              "repair_random": (PALETTE[4], "--"), "repair_greedy": (PALETTE[1], "-"), "decoder": (PALETTE[0], "-"),
              "decoder_seed": (PALETTE[0], ":")}
    ev = np.arange(1, 201) * 100
    for k, name in names.items():
        col, ls = styles[k]
        a1.plot(ev, 100 * np.array(knap["methods"][k]["curve"]) / knap["optimum"], color=col, ls=ls, lw=1.8,
                label=name)
    a1.set_ylim(80, 100.5)
    a1.set_xlabel("вычислений фитнес-функции"); a1.set_ylabel("% от оптимума")
    a1.set_title(f"Рюкзак: {knap['n']} предметов, медиана {RUNS} запусков", fontsize=11)
    a1.legend(fontsize=8.5, loc="lower right")
    for k, name, col in (("penalty", "ГА со штрафом", INK2), ("repair", "ГА с восстановлением", PALETTE[1])):
        a2.plot(ev, cover["methods"][k]["curve"], color=col, lw=2, label=name)
    a2.axhline(cover["greedy"], color=PALETTE[0], ls="--", lw=1.5, label="жадный алгоритм")
    a2.axhline(cover["optimum"], color=PALETTE[2], ls="-.", lw=1.5, label="оптимум (MILP)")
    a2.set_yscale("log")
    a2.set_ylim(cover["optimum"] * 0.9, np.nanmax(cover["methods"]["penalty"]["curve"]) * 1.2)
    a2.set_xlabel("вычислений фитнес-функции"); a2.set_ylabel("стоимость покрытия")
    a2.set_title(f"Покрытие: {cover['m']} строк × {cover['n']} столбцов, медиана 10 запусков", fontsize=11)
    a2.legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "t7_constraints")


def main():
    out = dict(toy=toy_examples(), knapsack=run_knapsack(), cover=run_cover())
    (ROOT / "results" / "combinatorial_examples.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    plot(out["knapsack"], out["cover"])


if __name__ == "__main__":
    main()
