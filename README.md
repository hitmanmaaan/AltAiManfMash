# AltAiManfMash — токовая диагностика двигателя на нейроморфном чипе «Алтай»

Ищем механические дефекты асинхронного двигателя (подшипник, дисбаланс, расцентровка) **по току статора**
и запускаем классификатор на эмуляторе российского нейроморфного процессора **AltAI («Алтай»)**
через Kaspersky Neuromorphic Platform (KNP).

```
ток 3 фаз, 25,6 кГц ──► спектр / огибающая / вейвлеты ──► признаки в дБ над нормой
                                                              │
               диагноз каждые 10 с ◄── 9 нейронов AltAILIF ◄──┘ 42 бинарных спайка
               (тип дефекта + степень)   веса −1/0/+1, 60 синапсов
```

## Что получилось

**Реальные записи (45 испытаний).** Питание 48,5 Гц от частотного преобразователя (ШИМ 3 кГц), реальная скорость
вала ~1452 об/мин. 45 записей — это 6 режимов работы (3 нагрузки × 2 варианта питания); внутри одного режима
записи стандартными признаками токовой диагностики неразличимы, разметки в данных нет.

**Синтетические дефекты на «Алтае».** В реальные записи внедрены симптомы 6 дефектов × 3 степени, сеть обучена
на записях 1–30 и проверена на 45 «слепых» файлах из записей 31–45:

| | Результат |
|---|---|
| Тип дефекта верно | **34 / 45 (76 %)**: норма 9/9, сильные 12/12, средние 11/12, слабые 2/12 |
| Степень верно | 82 % |
| Путаница типов | нет — все ошибки это пропуски слабых дефектов |
| Совпадение с эталоном | все спайки эмулятора «Алтая» совпали с numpy-эталоном бит-в-бит (2964/2964) |

**Сравнение методов** (`analysis/compare_methods.csv`, доля обнаруженных синтетических дефектов):
модуляционные дефекты лучше всего ловит огибающая вектора Парка (85 %), ударные — огибающая ВЧ
с выбором полосы по частотам дефекта (59 %). Подробности и рисунки — в `analysis/figures/`.

## Структура

```
analysis/            Python: весь анализ, результаты (*.csv) и рисунки (figures/)
  mcsa.py              загрузка записей, спектры, частоты подшипника NSK 6205
  survey.py            CSV → npy, обзор записей
  speeds.py            частота питания и скорость вала по каждому испытанию
  envelope.py          спектр огибающей (1 фаза, вектор Парка, ВЧ + кюртограмма)
  wavelet.py           CWT-скалограмма, вейвлет-пакеты, шумоподавление
  compare_methods.py   проверка всех методов на синтетике с известным ответом
  synth/               синтетические дефекты, обучение тернарной сети, запуск на «Алтае»
matlab_demo/         MATLAB-демо по секциям: start_here.m (свой README внутри)
knp-lab/             нейроморфная часть
  examples/            C++-программы для KNP (классификатор, мост MATLAB, тестовая сеть)
  patches/             правки KNP v1.0.0 и v2.0.0, без которых «Алтай» не запускается
  matlab_bridge/       мост MATLAB ↔ KNP через CSV и docker exec
  chip_io/             папка обмена с контейнерами
  setup_docker.sh      сборка обоих контейнеров с нуля
data/                данные (в git не входят, см. data/README.md)
```

## Как запустить

### 1. Python-окружение

Нужен Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Данные

Распаковать 45 CSV-файлов испытаний в `data/Датасеты/csv/` (подробно — [data/README.md](data/README.md)), затем:

```bash
cd analysis
python survey.py      # CSV → data/npy/*.npy (один раз)
python speeds.py      # speeds.csv — нужен почти всем остальным скриптам
python make_mat.py    # data/mat/*.mat — только для MATLAB-демо
```

### 3. Анализ реальных записей

Все скрипты запускаются из папки `analysis/` и пишут результат в CSV рядом с собой.

```bash
python features.py            # признаки токовой диагностики по испытаниям
python fingerprint.py         # «отпечаток» спектра и кластеры
python nominal_groups.py      # отклонения внутри группы с одинаковым режимом
python order_diff.py          # порядковые спектры (нормированные на скорость вала)
python order_diff_cond.py
python narrowband_groups.py
python impulsiveness.py       # эксцесс, пик-фактор, спектр огибающей ВЧ
python classify_peaks.py      # пики: механика / сеть / фиксированные
python plot_overview.py 1     # обзорный спектр испытания 1 → figures/
python plot_groups.py
```

### 4. Огибающая и вейвлеты

```bash
python test_methods.py        # самопроверка методов на чистой синтетике
python envelope.py            # → envelope.csv
python wavelet.py             # → wavelet.csv
python pwm_lines.py           # откуда импульсы 4–5 кГц (ответ: ШИМ преобразователя)
python compare_methods.py --rerun   # сравнение методов → compare_methods.csv (долго, 6 процессов)
python plot_methods.py        # рисунки figures/m_*.png
```

Порог чувствительности: `python compare_methods.py --sens 3` (порог = max(чистые) − 3 дБ).

### 5. Синтетические дефекты и обучение сети

```bash
cd analysis/synth
python gen_features.py        # 8550 примеров → features.npz
python train_ternary.py       # тернарная сеть под ограничения «Алтая» → weights.csv, model.npz
python gen_blind.py           # 45 слепых CSV → data/synthetic/csv + labels.csv
```

Обученные `weights.csv` и `model.npz` уже лежат в репозитории — этот шаг можно пропустить.

### 6. Запуск на «Алтае» (нужен Docker)

Один раз собрать контейнеры (клонирует KNP, накатывает патчи, скачивает пакеты «Алтая» из релизов KNP, собирает):

```bash
bash knp-lab/setup_docker.sh
```

После перезагрузки достаточно `docker start knp-run knp-lab`. Дальше из `analysis/synth/`:

```bash
python chip_eval.py altai-v1                # весь тест на эмуляторе + сверка с эталоном бит-в-бит
python infer_blind.py altai-v1              # 45 слепых файлов → data/synthetic/blind_results_altai-v1.csv
python diagnose_file.py syn_07.csv altai-v1 # диагноз одного файла
```

Бэкенды: `altai-v1` — golden model чипа «Алтай» из KNP v1.0.0 (контейнер `knp-lab`);
`cpu-v2` — тот же нейрон AltAILIF на CPU-бэкенде KNP v2.0.0 (контейнер `knp-run`).

### 7. MATLAB-демо

Открыть `matlab_demo/start_here.m` и запускать по секциям (Cmd+Enter). Номер испытания — переменная `TRIAL`.
Секции 7–9 требуют Wavelet Toolbox, секции 10–13 — запущенные контейнеры. Подробно — [matlab_demo/README.md](matlab_demo/README.md).

### 8. Отчёт в Word (по желанию)

```bash
cd analysis
python test_methods.py > test_methods.log
npm install docx@9
node make_report.js           # → Отчет_Огибающая_Вейвлет.docx в корне
```

## Заметки про KNP и «Алтай»

- Бэкенд «Алтая» из релиза v2.0.0 падает при загрузке любой сети. Рабочий вариант — пакет v1.0.0 с одной
  правкой ядра: пакет собран против ядра с методом `Backend::select_device`, которого нет в публичном теге
  v1.0.0 (виртуальная таблица сдвинута). Метод перенесён из v2 — см. `knp-lab/patches/knp-v1.0.0.patch`.
- В v2.0.0 добавлен недостающий `SynapticResourceSTDPAltAILIFNeuron`, на который ссылается заголовок пакета
  `knp-altai-backend-dev` (`knp-lab/patches/knp-v2.0.0.patch`).
- Пакеты бэкенда — x86-64, поэтому на Apple Silicon контейнеры работают через эмуляцию `linux/amd64`.
- Путь к `docker` в MATLAB-скриптах — `/usr/local/bin/docker` (MATLAB не видит `PATH` терминала); поправить при другой установке.
