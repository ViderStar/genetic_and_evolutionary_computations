import numpy as np

A, B = 0.0, 7.0                       # отрезок поиска
f = lambda t: (t - 1.3) * np.sin(0.5 * np.pi * t + 1)


def decode(pop, a=A, b=B):
    """Бинарные строки (N, L) → вещественные значения t по формуле (5.2)."""
    L = pop.shape[1]
    return a + pop @ (2 ** np.arange(L - 1, -1, -1)) * (b - a) / (2 ** L - 1)


def simple_ga(N=60, L=20, p_c=0.85, p_m=0.01, n_gen=200, seed=0):
    rng = np.random.default_rng(seed)
    pop = rng.integers(0, 2, (N, L), dtype=np.int8)
    fit = f(decode(pop))
    best = pop[fit.argmax()].copy()
    for g in range(n_gen):
        # селекция: рулетка со сдвигом приспособленности
        w = fit - fit.min() + 1e-9
        pop = pop[rng.choice(N, N, p=w / w.sum())]
        # одноточечный кроссинговер для пар (0,1), (2,3), ...
        for i in range(0, N - 1, 2):
            if rng.random() < p_c:
                k = rng.integers(1, L)
                pop[[i, i + 1], k:] = pop[[i + 1, i], k:]
        # побитовая мутация
        pop ^= (rng.random(pop.shape) < p_m).astype(np.int8)
        fit = f(decode(pop))
        # элитизм: лучшая найденная особь заменяет худшую
        if f(decode(best[None]))[0] > fit.max():
            pop[fit.argmin()] = best
            fit = f(decode(pop))
        best = pop[fit.argmax()].copy()
    t = decode(best[None])[0]
    return t, f(t)


t, ft = simple_ga()
print(f"t = {t:.6f}, f(t) = {ft:.6f}")   # t = 4.488825, f(t) = 3.127117
