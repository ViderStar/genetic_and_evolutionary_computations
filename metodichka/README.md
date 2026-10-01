# Учебно-методическое пособие «Генетические и эволюционные вычисления: нейронные сети и генетические алгоритмы»

Авторы: А. В. Лебедевич, С. Н. Нестеренков. БГУИР, кафедра ПОИТ, 2026.

Пособие построено по образцу [Скобцов, Лапицкая, Нестеренков, 2018](../materials/Skobcov_Lapitskaya_Nesterenkov_018.pdf):
теоретическая часть, затем лабораторный практикум. Оформление — СТП БГУИР: A4, Times 14 pt, поля 30/15/20/20 мм.

- **Теория (разделы 1–8):** основы машинного обучения и модель нейрона; персептрон, ADALINE, прогнозирование;
  многослойный персептрон и обратное распространение; карты Кохонена; простой ГА, кодирование, код Грея,
  теорема схем; операторы селекции, скрещивания, мутации и редукции, вещественное кодирование;
  учёт ограничений (штрафы, восстановление, декодеры) на задачах о рюкзаке и о минимальном покрытии,
  коммивояжёр (PMX, OX, CX, ER, 2-opt, меметический ГА); современные методы —
  CMA-ES, байесовская оптимизация и TPE, NSGA-II, эволюция программ с языковой моделью (FunSearch).
  В каждом блоке — подраздел «Реализация на Python»: код на NumPy и сравнение с scikit-learn, PyTorch,
  MiniSom, SciPy, DEAP, `cma`, Optuna, `pymoo`.
- **Практикум (ЛР 1–9):** цель, постановка, варианты (ЛР 5–7 — по Скобцову, ЛР 1–4 — по заданиям курса,
  ЛР 8–9 — новые), требования к программе, порядок выполнения, пример выполнения для варианта 3,
  контрольные вопросы. Основная среда — Python; MATLAB и JavaScript по желанию.
- **Приложения:** справочник функций Python; листинги: операторы для задачи коммивояжёра (CX, векторный 2-opt),
  CMA-ES и IPOP-CMA-ES, мини-FunSearch.

## Сборка

```bash
latexmk -pdf main.tex
```

Нужен pdfLaTeX (TinyTeX) с пакетами `extsizes`, `tempora`, `newtx`, `tikz`, `titlesec`, `tocloft`, `listings`.
Графики ЛР 3–7 берутся напрямую из `../lab_NN/report/figures/`, поэтому после перезапуска ноутбуков
пособие подхватывает новые картинки при следующей сборке.

Иллюстрации, которых нет в отчётах (теоретические схемы, пример ЛР 1 и пример ЛР 2 для варианта 3 по условию),
генерирует скрипт:

```bash
uv run --with numpy --with matplotlib python scripts/make_figures.py
```

Примеры ЛР 5 и ЛР 6 (часть Б, вещественный ГА) пересчитаны для варианта 3 по Скобцову — (t + 1,3)·sin(0,5πt + 1)
и функция Шаффера F7 — теми же алгоритмами, что в ноутбуках; числа сохраняются в `results/examples_variant3.json`:

```bash
uv run --with numpy --with scipy --with matplotlib python scripts/examples_variant3.py
```

Пример ЛР 8 (CMA-ES на F7, подбор гиперпараметров SVC, NSGA-II на ZDT1 и отбор признаков) —
`results/lab8_example.json` и рисунки `figures/lab8_*.png`:

```bash
uv run --with numpy --with scipy --with scikit-learn --with optuna --with cmaes --with cma --with pymoo --with matplotlib python scripts/lab8_example.py
```

Пример ЛР 9 (мини-FunSearch для онлайн-упаковки) вызывает языковую модель: `--backend cli` — Claude Code CLI
в неинтерактивном режиме без инструментов и MCP, `--backend api` — Anthropic SDK (ключ только в переменной
окружения `ANTHROPIC_API_KEY`). Один прогон — 60 вызовов `claude-opus-5-5`, около 5 мин и 3 долларов США;
результаты — `results/lab9_<распределение>.json`, рисунок строится по обоим файлам:

```bash
uv run --with numpy --with scipy --with matplotlib python scripts/lab9_example.py --backend cli --dist weibull
uv run --with numpy --with scipy --with matplotlib python scripts/lab9_example.py --backend cli --dist bimodal
uv run --with numpy --with matplotlib python scripts/lab9_example.py --plot
```

`--smoke` — одна короткая волна для проверки окружения.

Эксперимент раздела 7 теории (штрафы, восстановление и декодер для задачи о рюкзаке, ГА для задачи
о минимальном покрытии против жадного алгоритма и точного MILP) — `results/combinatorial_examples.json`
и рисунок `figures/t7_constraints.png`:

```bash
uv run --with numpy --with scipy --with matplotlib python scripts/combinatorial_examples.py
```

Листинги лежат в `listings/` и запускаются как есть (`uv run --with numpy python listings/tsp_cx.py`);
в пособие входят `tsp_cx.py`, `cma_es.py` и `funsearch.py`, `simple_ga.py` — дополнительный пример простого ГА.
Модуль CMA-ES назван `cma_es.py`, а не `cmaes.py`, чтобы не перекрывать пакет `cmaes`, который использует Optuna.
Список источников упорядочивается по первому упоминанию скриптом `scripts/sort_bibliography.py`
(после сборки, затем пересобрать).

## Что заполнить перед изданием

Помечено прочерками: УДК, ББК, авторский знак, рецензенты, ISBN (оборот титула), выходные данные
(последняя страница), строка о рекомендации УМО на титуле (закомментирована в `sections/00_title.tex`).

## Структура

```
metodichka/
├── main.tex, main.pdf         # сборка и готовое пособие
├── preamble_manual.tex        # оформление по СТП БГУИР
├── sections/                  # 00_title, 01_intro, t1–t8 (теория), l0–l9 (практикум), references, appendix
├── figures/                   # иллюстрации, созданные скриптами
├── results/                   # числа примеров ЛР 5, 6 (часть Б), 8, 9 и раздела 7 теории
├── listings/                  # код приложения Б
└── scripts/                   # make_figures.py, examples_variant3.py, combinatorial_examples.py,
                               # lab8_example.py, lab9_example.py, sort_bibliography.py
```
