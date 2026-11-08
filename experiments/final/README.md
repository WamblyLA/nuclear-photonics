# Финальная оценка

Все обучения завершены до открытия свежего резерва. Из 133 сохранённых допустимых checkpoint выбраны максимум три запуска по заранее заданным правилам; до резерва зафиксированы SHA R72, R62, R71 и фиксированного R31. На резерве выбран R72. До official test зафиксирован ровно один кандидат и тот же baseline.

- [Исходный план](../final_validation_plan.json), [качество и guards](../night_selection_policy.json).
- [Аудит отсутствия пересечений источников](pre_freeze_data_audit.json).
- [Фиксация кандидатов](frozen_candidates.json), [проверка одинаковых состояний checkpoint-алиасов](checkpoint_alias_verification.json).
- [Выбор победителя до test](frozen_winner.json), [итоговые числа](evaluation_summary.json).
- [Резерв: кандидаты против baseline](reserve_vs_baseline.json), [попарные различия финалистов](reserve_finalist_comparison.json).
- [Официальные test split: R72 против R31](test_comparison.json).
- [Обучение выбранной ветки](training_lineage.json), [конфигурация](selected_training_config.json).
- [Точный экспорт](export_validation.json), [host-тесты](host_verification.json), [скорость CPU](benchmark_cpu.json).

В каталогах selection и reserve хранятся все три финалиста и baseline; в test — только заранее выбранный R72 и R31. В каждом архиве есть metrics.json, попримерный NPZ, точный tuple manifest, provenance, контрольные суммы и snapshot кода evaluator. archive_index.json документирует байтовое сохранение чисел/исходников и ограниченное обезличивание путей в metadata. Изображения, исходные датасеты и промежуточные веса в архив оценки не включены.

Selection: 256 четвёрок. Свежий source-disjoint резерв: 512, seed 30042; он использован для выбора, поэтому не называется test. Official dataset test: 1024, seed 20261108; это test split CIFAR10/MNIST/FashionMNIST, не скрытая выборка организаторов. Все итоговые оценки выполнены CPU PyTorch 2.6.0+cpu, batch 16, четыре потока, после PNG roundtrip фазы и амплитуды.

Доверительные интервалы — 2000 парных стратифицированных bootstrap-повторов целых четвёрок. Они условны по эмпирическим комплектам: изображения переиспользуются между четвёрками, поэтому это не независимые image-level/population intervals. Поправка на многочисленные исследовательские сравнения не вводилась. Победитель не менялся по результатам test.

Пример воспроизведения финального попарного расчёта:

```bash
python -m src.compare_candidates --reference experiments/final/test/r31_peak_fourier --candidates experiments/final/test/r72_spatial_frequency_control --output outputs/recomputed_test_comparison.json --repetitions 2000 --seed 20261108
```

Сильное наложение объектов между глубинами сохраняется. Основной доказанный результат — улучшение метрик относительно фиксированного baseline при соблюдении протокола, а не достижение точного разделения четырёх изображений.
