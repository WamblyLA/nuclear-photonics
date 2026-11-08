# Датасеты

Локальная структура данных совместима с `torchvision.datasets` при `root="data"`:

| Датасет | Каталог | Обучение / тест | Формат |
| --- | --- | --- | --- |
| CIFAR10 | `cifar-10-batches-py/` | 50 000 / 10 000 | Исходные Python batches |
| MNIST | `MNIST/raw/` | 60 000 / 10 000 | IDX, изображения и метки |
| FashionMNIST | `FashionMNIST/raw/` | 60 000 / 10 000 | IDX, изображения и метки |

Для каждого датасета сохранены обе исходные выборки. MNIST и FashionMNIST распакованы из `.gz`; исходные архивы также сохранены. Архив CIFAR10 находится в `cifar-10-python.tar.gz`. Загруженные архивы проверяются по MD5 из источников.

Источники:

- CIFAR10: https://www.cs.toronto.edu/~kriz/cifar.html
- Зеркало архива CIFAR10, использованное для загрузки: https://zenodo.org/records/10089977
- MNIST: https://ossci-datasets.s3.amazonaws.com/mnist/ (зеркало, используемое torchvision).
- FashionMNIST: https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion
- Контрольные суммы MNIST и FashionMNIST: https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py

Данные и архивы исключены из Git через `.gitignore`.
