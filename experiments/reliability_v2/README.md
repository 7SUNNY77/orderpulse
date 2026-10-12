# experiments/reliability_v2

Результаты этапа Reliability & Queue Evaluation. Создаются командой `python -m orderpulse reliability`
(конфигурация — `configs/reliability.yaml`) и `python -m orderpulse tail-audit`. Интерпретация —
`reports/reliability_report.md` и `reports/dataset_tail_audit.md`.

| Файл | Содержание |
|---|---|
| `mvp_v1_snapshot/` | Снимок MVP v1 до этапа: конфиги, метаданные, метрики, модели, прогнозы, `manifest.json` с sha256. **Локальный, в репозиторий не входит** (модели и прогнозы по заказам) |
| `config.json` | Конфигурация запуска, окна, проверки доступности меток по окнам, ranking-метрики v1 на test, чувствительность к post-cutoff заказам |
| `experiments_log.json` | Список проведённых экспериментов с размерами выборок и временем |
| `topk_test_v1.csv` | Top-K в день (K = 5…100) для фиксированных прогнозов v1 на **просмотренном** test: модель, random-ожидание, lift, micro/macro |
| `daily_test_v1.csv` | Дневные таблицы n_d, m_d, k_d, h_d, E[h_d] для test v1 |
| `random_mc_test_v1.csv` | Monte Carlo случайного отбора (500 повторений, seed 2024+K) |
| `bootstrap_test_v1.csv` | Парный bootstrap CatBoost vs LogReg на test v1 (дни/недели, 1 000 повторений, seed 7+K) |
| `stability_test_v1.csv` | v1 по месяцам и неделям test |
| `ablation_dev.csv` | Календарные варианты A/B/C × окна W1–W3 × CatBoost/LogReg (+ правило, константа) |
| `topk_dev.csv`, `daily_dev.csv` | K-кривые и дневные таблицы на development-окнах (вариант A) и pooled |
| `bootstrap_dev.csv` | Парный bootstrap разниц на pooled development |
| `calibration.json` | Проверка sigmoid-калибровки: схема, Brier, Logloss, корзины |
| `tail_audit.json` | Числа аудита конца датасета |
| `figures/` | Precision/Recall/Lift vs K, найденные задержки vs проверки, reliability diagram |
| `dev_predictions/*.parquet` | Прогнозы окон по заказам (не коммитятся; воспроизводятся `python -m orderpulse reliability`) |
