# Генетические и эволюционные вычисления

Лабораторные работы по дисциплине «Генетические и эволюционные вычисления».

**Студент**: Лебедевич Артём Владимирович, группа 556301  
**Преподаватель**: Нестеренков С. Н.  
**Университет**: БГУИР, факультет КСиС, кафедра ПОИТ

## Лабораторные работы

| № | Тема | Метод |
|---|------|-------|
| 1 | [Классификация с помощью персептрона](lab_01/) | Perceptron (NumPy) |
| 2 | [Прогнозирование с помощью линейной НС](lab_02/) | ADALINE, псевдообратная матрица |
| 3 | [Аппроксимация функции двухслойной НС](lab_03/) | MLP + Backpropagation (NumPy) |
| 4 | [Кластеризация с помощью сети Кохонена](lab_04/) | Self-Organizing Map (NumPy) |

## Структура проекта

```
genetic_and_evolutionary_computations/
├── README.md
├── lab_01/                         # Персептрон
│   ├── lab_01.ipynb
│   ├── report.tex
│   ├── README.md
│   └── outputs/
├── lab_02/                         # Линейная НС
│   ├── lab_02.ipynb
│   ├── report.tex
│   ├── README.md
│   └── outputs/
├── lab_03/                         # Двухслойная НС
│   ├── lab_03.ipynb
│   ├── report.tex
│   ├── README.md
│   └── outputs/
└── lab_04/                         # Сеть Кохонена
    ├── lab_04.ipynb
    ├── report.tex
    ├── README.md
    └── outputs/
```

## Технологии

- **Python 3.10+**
- **NumPy** — все нейронные сети реализованы с нуля
- **Matplotlib** — статические графики
- **Plotly** — интерактивные визуализации (сохраняются в `outputs/`)
- **scikit-learn** — метрики качества (silhouette score)

## Запуск

```bash
pip install numpy matplotlib plotly scikit-learn
jupyter notebook
```
