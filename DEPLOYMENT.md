# Сборка и автономный запуск

Для готового решения нужны только Python-код и выпущенный checkpoint `models/solution.pt`. Исходные датасеты, CUDA, Telegram-выгрузка и журналы обучения в Docker-образ не включаются. Интернет нужен при сборке образа для установки пакетов; готовый контейнер выполняет синтез без сети.

Окружения разделены: готовый CPU inference проверяется на Python 3.11 с `requirements.txt`; CUDA-исследования выполнялись на Python 3.14 с `requirements-train.txt`. Последний фиксирует NumPy 2.5.4, требующий Python ≥ 3.12, поэтому его не следует устанавливать поверх CPU-окружения 3.11. Наличие указанных CUDA wheels для Linux/Windows и Python 3.14 проверено в [официальном индексе PyTorch](https://download.pytorch.org/whl/cu130/torch/); ограничение NumPy указано в [метаданных выпуска](https://pypi.org/pypi/numpy/2.5.4/json). Наличие wheels не заменяет проверку установки/выполнения; фактически использованные версии записаны в конфигурации каждого запуска.

## Docker, Linux x86-64

Из корня репозитория:

```bash
docker build --platform linux/amd64 -t neural-holography:submission .
mkdir -p results
docker run --rm --network none \
  --mount type=bind,src="$(pwd)/examples/input_set1",dst=/input,readonly \
  --mount type=bind,src="$(pwd)/results",dst=/output \
  neural-holography:submission \
  --inputs /input/image1.png /input/image2.png /input/image3.png /input/image4.png \
  --output /output/run1
```

Команда использует первый включённый в репозиторий комплект. Для своих данных замените путь `examples/input_set1` каталогом с четырьмя PNG 128×128, 8 бит, Grayscale (`L`). Порядок входов соответствует расстояниям 24,1 / 25,3 / 26,5 / 27,7 мм. Каталог результата `results/run1` до запуска отсутствует; программа не перезаписывает предыдущие результаты. Родительский каталог `results` создаётся заранее. На Linux при необходимости можно добавить `--user "$(id -u):$(id -g)"`, чтобы выходные файлы принадлежали текущему пользователю. Команды проверки всех пяти комплектов приведены в [README](README.md).

PowerShell:

```powershell
docker build --platform linux/amd64 -t neural-holography:submission .
New-Item -ItemType Directory -Force results | Out-Null
docker run --rm --network none --mount "type=bind,source=$((Get-Location).Path)/examples/input_set1,target=/input,readonly" --mount "type=bind,source=$((Get-Location).Path)/results,target=/output" neural-holography:submission --inputs /input/image1.png /input/image2.png /input/image3.png /input/image4.png --output /output/run1
```

Контейнер вызывает `python -m src.predict --checkpoint /app/models/solution.pt --device cpu`. Выход: `phase.png`, четыре `reconstruction_*.png`, копии входов `target_*.png`, `metrics.json`. Метрики и протокол нормировки записываются в JSON. Маска перечитывается из PNG до физического восстановления.

## Python без Docker

Использовать отдельное окружение CPython 3.11:

```bash
python3.11 -m venv .venv-runtime
.venv-runtime/bin/python -m pip install -r requirements-runtime.txt
.venv-runtime/bin/python -m pip check
.venv-runtime/bin/python -m src.predict --checkpoint models/solution.pt --device cpu --inputs examples/input_set1/image1.png examples/input_set1/image2.png examples/input_set1/image3.png examples/input_set1/image4.png --output results/python_run1
```

На Windows исполняемый файл окружения — `.venv-runtime\Scripts\python.exe`. Не использовать старое исследовательское `.venv-gpu` как единственную проверку переносимости.

`requirements-runtime.txt` фиксирует CPU PyTorch 2.6.0, NumPy 1.26.4 и Pillow 11.3.0. Наличие Linux/Windows CPython 3.11 CPU wheels проверено по [официальному индексу PyTorch](https://download.pytorch.org/whl/cpu/torch/); CPU-команда установки опубликована в [документации предыдущих версий](https://pytorch.org/get-started/previous-versions/). Версии [NumPy](https://pypi.org/project/numpy/1.26.4/) и [Pillow](https://pypi.org/project/pillow/11.3.0/) поддерживают Python 3.11. `requirements-train.txt` отдельно сохраняет наблюдавшееся CUDA-окружение исследований; оно не требуется для inference.

## Экспорт выбранной модели

Выпускается **R72 / `transport_spatial_complex`, шаг 5000**: `outputs/r72_spatial_frequency_control/step_005000.pt`. Выбор зафиксирован после резервной валидации и **до открытия official test**; исходный SHA и правило выбора сохранены в [frozen_winner.json](experiments/final/frozen_winner.json). Для готового inference используются включённые веса `models/solution.pt`; исследовательские checkpoint в Git не входят.

После повторения обучения экспортировать тот же шаг в новый каталог можно так:

```bash
python -m scripts.export_solution --checkpoint outputs/r72_spatial_frequency_control/step_005000.pt --output-dir outputs/reproduced_model
```

Экспортёр читает доверенный локальный исследовательский checkpoint, оставляет CPU-тензоры float32 (complex64 для комплексных весов), минимальную конфигурацию inference и номер шага; состояние оптимизатора не переносится. `models/metadata.json` содержит SHA-256 исходного и готового файла, параметры физики, протокол, число параметров и хэши исходников. Проверяются строгая загрузка весов, конечный выход сети и загрузка результата с `torch.load(..., weights_only=True)`. Уже созданный экспорт защищён от перезаписи; для намеренной замены предусмотрен `--overwrite`. Оригинальный checkpoint никогда не перезаписывается.

Тесты экспортёра проверяют отсутствие состояния обучения, целостность источника, точность float32 inference, сохранение мнимой части комплексных тензоров и отказ при некорректных весах/метаданных. Диагностические checkpoint с `diagnostic_only=True` или `fixed_train_quartets > 0` экспортёр отклоняет. Историческая проверка промежуточного R31 дала побайтово одинаковые фазу и четыре реконструкции до/после экспорта: [отчёт R31](experiments/portable_export_r31_validation.json).

Итоговый экспорт **R72** проверен в Python 3.11 / PyTorch 2.6.0+cpu: на `examples/input_set1` все пять выходных PNG побайтово совпали с результатами исходного checkpoint, метрики каждой плоскости и средние идентичны. Переносимые веса загружаются с `weights_only=True` и не содержат optimizer state. Подтверждение с SHA обоих файлов: [export_validation.json](experiments/final/export_validation.json); описание выпущенных весов — [models/metadata.json](models/metadata.json). Результаты всех пяти фиксированных комплектов доступны в [галерее](docs/demo/index.html).

## Восстановление датасетов

Датасеты нужны для повторного обучения, но не для запуска готовой модели. Скрипт использует стандартную библиотеку Python и проверяет опубликованные контрольные суммы архивов и распакованные файлы:

```bash
python scripts/download_data.py --root data
python scripts/download_data.py --root data --verify-only
```

Можно выбрать часть наборов: `--datasets mnist fashionmnist`. Сохраняются исходные train/test файлы; загрузка и проверка checksum не означают использование test при подборе модели. Повторный запуск использует уже проверенные файлы. Крупные архивы и распакованные наборы не помещаются в Git.

Источники: [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [резервное зеркало того же CIFAR10-архива](https://zenodo.org/records/10089977), MNIST через HTTPS-зеркало OSSCI из [исходников torchvision](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/datasets/mnist.py), [репозиторий FashionMNIST](https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion). Контрольные суммы закреплены по torchvision v0.21.0: [CIFAR](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/datasets/cifar.py), [MNIST/FashionMNIST](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/datasets/mnist.py).

## Статус проверки

Проверены доступность официальных Linux/Windows CPU wheels, контрольные суммы всех трёх локальных датасетов и распакованных файлов, реальные небольшие загрузки MNIST/FashionMNIST, отказ при повреждённом распакованном файле и его восстановление. Локальный отчёт проверки наборов — `outputs/dataset_checksum_verification.json`.

В CPU-окружении **Python 3.11.18 / PyTorch 2.6.0+cpu / NumPy 1.26.4 / Pillow 11.3.0** прошли **190 тестов** полного набора на рабочей машине и **173 теста** через `python -m scripts.run_offline_tests` без загрузки датасетов. Последний исключает только три явно перечисленных модуля, которым нужны исходные обучающие данные; остальные ошибки приводят к ненулевому коду выхода. Команды, журналы и контрольные суммы: [host_verification.json](experiments/final/host_verification.json).

На **Intel Core i3-12100F, Windows, 4 потока, batch=1** медиана одного прохода R72 составила **20,12 мс**, вычислительного конвейера с четырьмя восстановлениями — **21,64 мс**. Это 50 измерений после 10 прогревов, без дискового ввода-вывода, загрузки весов и расчёта метрик. Один полный запуск CLI в новом Python-процессе занял **1,872 с**, включая чтение и запись PNG; кеш ОС не очищался. Условия, все измерения и SHA модели: [benchmark_cpu.json](experiments/final/benchmark_cpu.json).

На рабочей Windows-машине Docker Engine отсутствует; успешный Docker-запуск локально не заявляется. [Workflow](.github/workflows/docker-smoke.yml) настроен на push в ветку `dzprht/neural-holography-20261108` и ручной запуск. Он собирает образ, запускает 173 теста без датасетов, проверяет все 20 входных PNG и выполняет inference с `--network none` на **всех пяти** комплектах `examples/input_set1..5`. Затем проверяет **25 выходных PNG** и конечность метрик, сохраняет отдельные результаты и логи каждого комплекта. Фактический статус контейнерной проверки конкретного опубликованного commit доступен в [GitHub Actions этой ветки](https://github.com/memily0/neural-holography/actions/workflows/docker-smoke.yml?query=branch%3Adzprht%2Fneural-holography-20261108); успешные проверки на рабочей машине не заменяют этот запуск.
