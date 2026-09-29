import numpy as np


def read_tsp(path):
    """Координаты городов из раздела NODE_COORD_SECTION файла TSPLIB."""
    rows = [l.split() for l in open(path) if l.split() and l.split()[0].isdigit()]
    return np.array([[float(r[1]), float(r[2])] for r in rows])


def dist_matrix(xy):
    """Расстояния EUC_2D: евклидовы, округлённые до целого."""
    return np.floor(np.hypot(*(xy[:, None] - xy[None]).transpose(2, 0, 1)) + 0.5)


def tour_len(tour, D):
    return D[tour, np.roll(tour, -1)].sum()


def cx(p1, p2):
    """Циклический кроссинговер: циклы позиций берутся попеременно."""
    n = len(p1)
    pos1 = np.empty(n, int); pos1[p1] = np.arange(n)   # позиция города в p1
    c1, c2 = p1.copy(), p2.copy()
    seen, cyc = np.zeros(n, bool), 0
    for start in range(n):
        if seen[start]:
            continue
        i = start
        while not seen[i]:                              # обход одного цикла
            seen[i] = True
            if cyc % 2:                                 # нечётные циклы — обмен
                c1[i], c2[i] = p2[i], p1[i]
            i = pos1[p2[i]]
        cyc += 1
    return c1, c2


def two_opt(tour, D):
    """Локальный поиск 2-opt: лучшая замена пары рёбер ищется векторно среди всех пар."""
    n, t = len(tour), tour.copy()
    I, J = np.triu_indices(n, 2)
    keep = ~((I == 0) & (J == n - 1))             # рёбра, смежные через замыкание тура
    I, J = I[keep], J[keep]
    while True:
        a, b, c, d = t[I], t[I + 1], t[J], t[(J + 1) % n]
        gain = D[a, c] + D[b, d] - D[a, b] - D[c, d]   # формула (7.4) для всех пар сразу
        m = int(np.argmin(gain))
        if gain[m] > -1e-9:                        # улучшающих замен нет
            return t
        t[I[m] + 1:J[m] + 1] = t[I[m] + 1:J[m] + 1][::-1]   # инверсия отрезка b..c


p1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) - 1
p2 = np.array([4, 5, 2, 1, 8, 7, 6, 9, 3]) - 1
print((cx(p1, p2)[0] + 1).tolist())     # [1, 5, 2, 4, 8, 6, 7, 9, 3]
