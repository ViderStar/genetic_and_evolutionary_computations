import numpy as np


def cmaes(func, x0, sigma0, bounds, budget, seed=0, lam=None, trace=None):
    """(mu/mu_w, lambda)-CMA-ES по учебнику Н. Хансена (2016): минимизация func.

    func принимает матрицу (lam, n) и возвращает вектор значений.
    Возвращает лучшую точку, её значение и историю (число вычислений, лучшее f);
    в список trace, если он передан, записываются (m, sigma, C) каждого поколения.
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds[:, 0], bounds[:, 1]
    n = len(x0)
    lam = lam or 4 + int(3 * np.log(n))                   # размер выборки
    mu = lam // 2                                         # число родителей
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))   # веса рекомбинации
    w /= w.sum()
    mueff = 1 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)        # скорости обучения
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))   # E||N(0, I)||

    m, sigma = np.asarray(x0, float), sigma0
    C, B, D = np.eye(n), np.eye(n), np.ones(n)            # C = B diag(D^2) B^T
    pc, ps = np.zeros(n), np.zeros(n)                     # пути эволюции
    best_x, best_f, evals, hist, g = None, np.inf, 0, [], 0
    while evals < budget:
        if trace is not None:
            trace.append((m.copy(), sigma, C.copy()))
        z = rng.standard_normal((lam, n))
        y = z @ (B * D).T                                 # y ~ N(0, C)
        X = np.clip(m + sigma * y, lo, hi)
        F = func(X)
        evals += lam
        order = np.argsort(F)
        if F[order[0]] < best_f:
            best_f, best_x = F[order[0]], X[order[0]].copy()
        hist.append((evals, best_f))
        y_sel = y[order[:mu]]
        yw = w @ y_sel                                    # взвешенный шаг среднего
        m = m + sigma * yw
        # адаптация шага: путь ps в отбелённых координатах
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ ((B.T @ yw) / D))
        g += 1
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * g)) / chi_n < 1.4 + 2 / (n + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        # ранг-1 и ранг-mu обновления ковариационной матрицы
        C = ((1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (y_sel.T * w) @ y_sel)
        sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
        C = (C + C.T) / 2
        D2, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D2, 1e-30))
        if sigma * D.max() < 1e-14 * (hi - lo).max():    # распределение выродилось
            break
    return best_x, best_f, np.array(hist)


def ipop_cmaes(func, bounds, budget, seed=0, sigma_frac=0.3):
    """IPOP-CMA-ES: перезапуски из случайной точки с удвоением популяции."""
    rng = np.random.default_rng(seed)
    lam, used, best_x, best_f, hist, k = 4 + int(3 * np.log(len(bounds))), 0, None, np.inf, [], 0
    while used < budget:
        x0 = rng.uniform(bounds[:, 0], bounds[:, 1])
        x, f, h = cmaes(func, x0, sigma_frac * np.ptp(bounds, axis=1).max(), bounds, budget - used,
                        seed=seed * 1000 + k, lam=lam)
        hist += [(used + e, min(v, best_f)) for e, v in h]      # лучшее с учётом прошлых запусков
        if f < best_f:
            best_x, best_f = x, f
        used, lam, k = used + int(h[-1, 0]), 2 * lam, k + 1
    return best_x, best_f, np.array(hist)


if __name__ == "__main__":
    sphere = lambda X: np.sum(X ** 2, axis=1)
    bounds = np.array([[-5.0, 5.0]] * 10)
    x, fx, _ = cmaes(sphere, np.full(10, 3.0), 2.0, bounds, budget=6000)
    print(f"сфера, n = 10: f = {fx:.1e}")                  # порядка 1e-15 и меньше
