"""Мини-FunSearch: эволюция эвристики онлайн-упаковки в контейнеры с помощью LLM (ЛР 9)."""
import ast
import json
import re
import subprocess
import sys
import tempfile

import numpy as np

# ------------------------------------------------------------------ задача и оценка
def make_instances(dist, n_items, seeds, capacity=100):
    """Наборы размеров предметов: список целочисленных массивов."""
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        if dist == "weibull":                        # Weibull(k = 3, lambda = 45), как в FunSearch
            x = 45 * rng.weibull(3, n_items)
        elif dist == "uniform":
            x = rng.uniform(20, 100, n_items)
        elif dist == "normal":
            x = rng.normal(50, 15, n_items)
        elif dist == "bimodal":
            x = np.where(rng.random(n_items) < 0.5, rng.normal(30, 5, n_items), rng.normal(65, 5, n_items))
        else:
            raise ValueError(dist)
        out.append(np.clip(np.round(x), 1, capacity).astype(int))
    return out


def pack(items, priority, capacity=100):
    """Онлайн-упаковка: предмет идёт в допустимый контейнер с наибольшим приоритетом.

    Кандидаты — открытые контейнеры, где предмет помещается, и один новый (пустой) контейнер.
    Возвращает число использованных контейнеров.
    """
    free = []                                        # свободное место в открытых контейнерах
    for item in items:
        fits = [i for i, r in enumerate(free) if r >= item]
        cand = np.array([free[i] for i in fits] + [capacity], dtype=float)
        k = int(np.argmax(priority(float(item), cand, float(capacity))))
        if k < len(fits):
            free[fits[k]] -= item
        else:
            free.append(capacity - item)
    return len(free)


def excess(instances, priority, capacity=100):
    """Среднее превышение нижней границы L1 = ceil(sum / C), %: меньше — лучше."""
    res = []
    for items in instances:
        lb = int(np.ceil(items.sum() / capacity))
        res.append(100 * (pack(items, priority, capacity) - lb) / lb)
    return float(np.mean(res))


def first_fit(item, bins, capacity):
    return -np.arange(len(bins), dtype=float)       # самый ранний открытый контейнер


def best_fit(item, bins, capacity):
    return -(bins - item)                            # наименьший остаток после укладки


def worst_fit(item, bins, capacity):
    p = bins - item                                  # наибольший остаток среди открытых
    p[-1] = -np.inf                                  # новый контейнер — только если иначе нельзя
    return p


# ------------------------------------------------------------------ проверка кода от LLM
ALLOWED_IMPORTS = {"numpy", "math"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "input",
                   "os", "sys", "subprocess", "socket", "shutil", "pathlib", "builtins", "getattr"}
FORBIDDEN_ATTRS = {"load", "save", "savez", "savetxt", "loadtxt", "fromfile", "tofile", "memmap",
                   "genfromtxt", "system", "popen"}


def extract_function(text):
    """Достаёт из ответа модели определение функции priority* и переименовывает его в priority."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S) or [text]
    for code in blocks:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("priority")]
        if funcs:
            funcs[-1].name = "priority"                  # последняя версия — новая эвристика
            keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
                    or (isinstance(n, ast.FunctionDef) and not n.name.startswith("priority_v"))]
            return ast.unparse(ast.Module(body=keep, type_ignores=[]))
    return None


def is_safe(code):
    """Статическая проверка: только разрешённые импорты и никаких опасных имён."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] not in ALLOWED_IMPORTS for a in node.names):
            return False
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
            return False
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRS):
            return False
    return True


RUNNER = """
import json, math, sys
import numpy as np
sys.path.insert(0, {path!r})
from funsearch import excess, make_instances
ns = {{"np": np, "numpy": np, "math": math}}
exec(sys.stdin.read(), ns)
inst = make_instances({dist!r}, {n_items}, {seeds}, {capacity})
print(json.dumps(excess(inst, ns["priority"], {capacity})))
"""


def evaluate_code(code, dist, n_items, seeds, capacity=100, timeout=60):
    """Оценка кода в отдельном процессе с тайм-аутом; None — если код упал, завис или небезопасен."""
    if code is None or not is_safe(code):
        return None
    runner = RUNNER.format(path=str(__import__("pathlib").Path(__file__).parent), dist=dist,
                           n_items=n_items, seeds=list(seeds), capacity=capacity)
    try:
        out = subprocess.run([sys.executable, "-c", runner], input=code, capture_output=True, text=True,
                             timeout=timeout)
        score = json.loads(out.stdout.strip().splitlines()[-1])
        return score if np.isfinite(score) else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, IndexError, ValueError):
        return None


# ------------------------------------------------------------------ языковая модель
SYSTEM = ("Ты эксперт по алгоритмам и эвристикам комбинаторной оптимизации. "
          "Отвечай только кодом на Python в одном блоке ```python```.")

PROMPT = """Задача: онлайн-упаковка в контейнеры. Предметы с целыми размерами приходят по одному,
ёмкость контейнера {capacity}. Для очередного предмета эвристика получает item — его размер,
bins — свободное место в контейнерах, куда он помещается (последний элемент — новый пустой
контейнер), capacity — ёмкость, и возвращает массив приоритетов той же длины, что bins.
Предмет кладётся в контейнер с наибольшим приоритетом. Цель — как можно меньше контейнеров.
Оценка — превышение нижней границы числа контейнеров, в процентах (меньше — лучше).

{examples}
Напиши функцию priority_v{k}(item, bins, capacity) — улучшенную версию предыдущих.
Используй только numpy (как np) и math. Функция должна быть быстрой и детерминированной."""


def llm_anthropic(prompt, model="claude-opus-5-5", effort="medium"):
    """Вызов через Anthropic SDK (ключ ANTHROPIC_API_KEY или профиль `ant auth login`)."""
    import anthropic
    client = anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            model=model, max_tokens=16000, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        print("LLM error:", type(e).__name__)
        return "", 0.0
    if resp.stop_reason == "refusal":
        return "", 0.0
    return "".join(b.text for b in resp.content if b.type == "text"), None


def llm_claude_cli(prompt, model="claude-opus-5-5", effort="medium"):
    """Вызов через Claude Code CLI в неинтерактивном режиме, без инструментов и MCP."""
    cmd = ["claude", "-p", prompt, "--model", model, "--effort", effort, "--tools", "",
           "--system-prompt", SYSTEM, "--output-format", "json", "--no-session-persistence",
           "--strict-mcp-config", "--setting-sources", ""]
    try:
        with tempfile.TemporaryDirectory() as cwd:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=cwd)
        d = json.loads(out.stdout)
        return ("" if d.get("is_error") else d.get("result", "")), d.get("total_cost_usd", 0.0)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return "", 0.0


# ------------------------------------------------------------------ эволюционный цикл
def funsearch(seed_codes, llm, dist, n_items, seeds, capacity=100, n_waves=10, per_wave=4,
              k_shot=2, db_size=20, rng=None, log=print):
    """База программ + подсказка из k_shot лучших по турниру + параллельные вызовы LLM."""
    from concurrent.futures import ThreadPoolExecutor
    rng = rng or np.random.default_rng(0)
    db = []                                          # (оценка, код)
    for code in seed_codes:
        db.append((evaluate_code(code, dist, n_items, seeds, capacity), code))
    history, cost, calls, valid = [], 0.0, 0, 0

    def make_prompt():
        pool = sorted(db)[:db_size]
        idx = sorted({min(rng.integers(0, len(pool), 2)) for _ in range(k_shot)})   # турнир из двух
        chosen = sorted((pool[i] for i in idx), key=lambda z: -z[0])                  # от худшей к лучшей
        ex = ""
        for v, (score, code) in enumerate(chosen):
            ex += f"# версия {v}, оценка {score:.3f} %\n" + code.replace("def priority(", f"def priority_v{v}(") + "\n\n"
        return PROMPT.format(capacity=capacity, examples=ex, k=len(chosen))

    for wave in range(n_waves):
        prompts = [make_prompt() for _ in range(per_wave)]
        with ThreadPoolExecutor(per_wave) as ex:
            answers = list(ex.map(llm, prompts))
        for text, c in answers:
            calls += 1
            cost += c or 0.0
            code = extract_function(text)
            score = evaluate_code(code, dist, n_items, seeds, capacity)
            if score is not None:
                valid += 1
                db.append((score, code))
            history.append(min(s for s, _ in db))
        log(f"волна {wave + 1}: лучшее {history[-1]:.3f} %, корректных {valid}/{calls}, стоимость ${cost:.2f}")
    return sorted(db), history, dict(calls=calls, valid=valid, cost=cost)
