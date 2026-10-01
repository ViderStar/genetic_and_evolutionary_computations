"""Пример выполнения ЛР 9 (вариант 3): мини-FunSearch для онлайн-упаковки в контейнеры.

Сравниваются классические эвристики, параметрическая эвристика с параметрами от CMA-ES (ЛР 8)
и эвристики, выведенные эволюционным поиском с языковой моделью в роли оператора мутации.

Запуск из каталога metodichka/ (бэкенд cli — Claude Code, api — Anthropic SDK):
    uv run --with numpy --with scipy --with matplotlib --with anthropic python scripts/lab9_example.py --backend cli --dist weibull
    ... --smoke — одна короткая волна для проверки; --plot — рисунок по results/lab9_weibull.json и lab9_bimodal.json
"""
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "listings"))
import funsearch as fs  # noqa: E402
from cma_es import cmaes  # noqa: E402

CAP = 100
TRAIN = dict(n_items=1000, seeds=list(range(5)))           # по чему идёт поиск
TEST = dict(n_items=5000, seeds=list(range(100, 105)))     # проверка обобщения на больших наборах
TITLES = {"weibull": "Вейбулла: масштаб 45, форма 3", "bimodal": "смесь 0,5 N(30; 5) + 0,5 N(65; 5)"}

BEST_FIT_CODE = '''import numpy as np

def priority(item, bins, capacity):
    return -(bins - item)'''


def param_priority(theta):
    """Гипотеза человека: мелкие ненулевые остатки бесполезны — штрафуем их, точное попадание поощряем."""
    a, t, b = theta

    def pr(item, bins, capacity):
        r = bins - item
        return -r - a * ((r > 0) & (r < t)) + b * (r == 0)
    return pr


def tune_param(train, budget=240, seed=0):  # noqa: D103
    bounds = np.array([[0.0, 100.0], [0.0, 50.0], [0.0, 100.0]])
    f = lambda X: np.array([fs.excess(train, param_priority(x), CAP) for x in X])
    x, fx, hist = cmaes(f, np.array([10.0, 10.0, 10.0]), 15.0, bounds, budget, seed=seed, lam=8)
    return x, fx, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cli", "api"], default="cli")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--waves", type=int, default=15)
    ap.add_argument("--per-wave", type=int, default=4)
    ap.add_argument("--dist", default="weibull", choices=["weibull", "bimodal", "uniform", "normal"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plot", action="store_true", help="только построить рисунок по сохранённым результатам")
    args = ap.parse_args()
    if args.plot:
        plot()
        return
    DIST = args.dist
    if args.smoke:
        args.waves, args.per_wave = 1, 2

    train = fs.make_instances(DIST, TRAIN["n_items"], TRAIN["seeds"], CAP)
    test = fs.make_instances(DIST, TEST["n_items"], TEST["seeds"], CAP)
    res = {"baselines": {}}
    for name, f in (("First Fit", fs.first_fit), ("Best Fit", fs.best_fit), ("Worst Fit", fs.worst_fit)):
        res["baselines"][name] = dict(train=fs.excess(train, f, CAP), test=fs.excess(test, f, CAP))
    print(res["baselines"])

    t0 = time.perf_counter()
    theta, ftr, _ = tune_param(train)
    res["cmaes_param"] = dict(theta=theta.tolist(), train=ftr, test=fs.excess(test, param_priority(theta), CAP),
                              seconds=time.perf_counter() - t0)
    print("CMA-ES:", res["cmaes_param"])

    backend = fs.llm_claude_cli if args.backend == "cli" else fs.llm_anthropic
    llm = partial(backend, model=args.model, effort=args.effort)
    t0 = time.perf_counter()
    db, history, stats = fs.funsearch([BEST_FIT_CODE], llm, DIST, TRAIN["n_items"], TRAIN["seeds"], CAP,
                                      n_waves=args.waves, per_wave=args.per_wave, rng=np.random.default_rng(3))
    stats["seconds"] = time.perf_counter() - t0
    best_score, best_code = db[0]
    test_score = fs.evaluate_code(best_code, DIST, TEST["n_items"], TEST["seeds"], CAP, timeout=300)
    res["funsearch"] = dict(model=args.model, effort=args.effort, backend=args.backend, history=history,
                            train=best_score, test=test_score, code=best_code, **stats,
                            top5=[dict(train=s, code=c) for s, c in db[:5]],
                            all=[dict(train=s, code=c) for s, c in db])
    print("FunSearch:", {k: v for k, v in res["funsearch"].items() if k not in ("code", "top5", "history")})
    print(best_code)
    if not args.smoke:
        out = ROOT / "results" / f"lab9_{DIST}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def plot():
    from make_figures import INK2, PALETTE, plt, save
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, dist in zip(axes, ("weibull", "bimodal")):
        res = json.loads((ROOT / "results" / f"lab9_{dist}.json").read_text(encoding="utf-8"))
        h = res["funsearch"]["history"]
        ax.step(np.arange(1, len(h) + 1), h, where="post", color=PALETTE[1], lw=2.2, label="мини-FunSearch")
        for name, col, ls in (("First Fit", INK2, ":"), ("Best Fit", PALETTE[0], "--")):
            ax.axhline(res["baselines"][name]["train"], color=col, ls=ls, lw=1.5, label=name)
        ax.axhline(res["cmaes_param"]["train"], color=PALETTE[2], ls="-.", lw=1.5, label="параметр. + CMA-ES")
        ax.set_xlabel("вызовов языковой модели")
        ax.set_title(TITLES[dist], fontsize=11)
    axes[0].set_ylabel("превышение границы $L_1$, %")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save(fig, "lab9_progress")


if __name__ == "__main__":
    main()
