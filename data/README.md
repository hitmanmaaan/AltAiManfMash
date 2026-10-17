# Данные

Данные в репозиторий не входят: исходные записи весят ~10 ГБ и выдаются преподавателем.
Все скрипты ждут их по путям ниже (относительно корня репозитория).

| Путь | Что это | Откуда берётся |
|---|---|---|
| `data/Датасеты/csv/current_1.csv … current_45.csv` | 45 испытаний: ток статора трёх фаз (`current_R, current_S, current_T`), 25,6 кГц, ~60 с | Выдаются преподавателем (`Data_Set_bonus.zip` и др.), распаковать сюда |
| `data/npy/current_N.npy` | То же в float32, массив `(N, 3)` — быстро читается | `python analysis/survey.py` |
| `data/mat/current_N.mat` | То же для MATLAB: переменные `R`, `S`, `T`, `fs` | `python analysis/make_mat.py` |
| `data/synthetic/csv/syn_01.csv … syn_45.csv` | «Слепые» файлы с внедрёнными дефектами, формат как у исходных | `python analysis/synth/gen_blind.py` |
| `data/synthetic/labels.csv` | Ключ ответов к слепым файлам | `gen_blind.py` (лежит в репозитории) |
| `data/synthetic/blind_results_<backend>.csv` | Диагнозы сети по слепым файлам | `infer_blind.py` (лежит в репозитории) |

Записи 1–45 вероятно взяты из открытого датасета KAIST (Mendeley Data, `ztmf3m7h5x`): тот же стенд,
15 состояний × 3 нагрузки = 45 записей, с метками дефектов.
