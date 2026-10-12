# Аудит качества данных OrderPulse

Сгенерировано командой `python -m orderpulse quality` · 2026-10-12T22:21:37+00:00 · fingerprint данных `4179f0f63056937d`.

Отчёт создаётся автоматически; ручные правки будут перезаписаны. Решения по правилам — в `reports/decisions.md`.

## Итог

- Критических проблем: **0**
- Предупреждений: **17**
- Ожидаемых особенностей данных: **11**
- Проверок без замечаний: 36

## Размеры таблиц

| Таблица | Строк |
|---|---:|
| orders | 99,441 |
| order_items | 112,650 |
| order_payments | 103,886 |
| order_reviews | 99,224 |
| customers | 99,441 |
| sellers | 3,095 |
| products | 32,951 |
| geolocation | 1,000,163 |
| category_translation | 71 |

## Ключи и гранулярность

| Проверка | Статус | Число | Доля | Действие / примечание |
|---|---|---:|---:|---|
| orders: duplicates of (order_id) | 🟢 ok | 0 | 0.000% | pipeline stops if > 0; primary key of the table grain |
| customers: duplicates of (customer_id) | 🟢 ok | 0 | 0.000% | pipeline stops if > 0; primary key of the table grain |
| sellers: duplicates of (seller_id) | 🟢 ok | 0 | 0.000% | pipeline stops if > 0; primary key of the table grain |
| products: duplicates of (product_id) | 🟢 ok | 0 | 0.000% | pipeline stops if > 0; primary key of the table grain |
| order_items: duplicates of (order_id, order_item_id) | 🟢 ok | 0 | 0.000% | pipeline stops if > 0; primary key of the table grain |
| order_payments: duplicates of (order_id, payment_sequential) | 🟢 ok | 0 | 0.000% | pipeline stops if > 0; primary key of the table grain |
| order_reviews: duplicates of (review_id, order_id) | 🟢 ok | 0 | 0.000% | pipeline stops if > 0; primary key of the table grain |
| category_translation: duplicated category name | 🟢 ok | 0 | 0.000% | first English name kept; dimension must be unique |
| order_reviews: review_id linked to >1 order | 🔵 ожидаемо | 814 | 0.820% | review_id is not unique on its own; the grain is (review_id, order_id) |
| orders with more than one review | 🔵 ожидаемо | 547 | 0.550% | canonical review = latest review_creation_date, then answer timestamp, then review_id |
| orders with more than one payment record | 🔵 ожидаемо | 2,961 | 2.978% | payments are summed per order before joining |
| orders with more than one item | 🔵 ожидаемо | 9,803 | 9.858% | items are aggregated per order before joining |
| orders with more than one seller | 🔵 ожидаемо | 1,278 | 1.285% | counted once in order-level KPIs and once per seller in seller statistics |
| geolocation: points per zip prefix > 1 | 🔵 ожидаемо | 981,148 | 98.099% | aggregated to one row per prefix (median lat/lng) |
| customer_unique_id with several customer_id | 🔵 ожидаемо | 2,997 | 3.119% | repeat customers; customer_id is per order |

## Ссылочная целостность

| Проверка | Статус | Число | Доля | Действие / примечание |
|---|---|---:|---:|---|
| order_items.order_id -> orders | 🟢 ok | 0 | 0.000% | reported; orphans cannot enter order marts; orphan child rows |
| order_payments.order_id -> orders | 🟢 ok | 0 | 0.000% | reported; orphans cannot enter order marts; orphan child rows |
| order_reviews.order_id -> orders | 🟢 ok | 0 | 0.000% | reported; orphans cannot enter order marts; orphan child rows |
| orders.customer_id -> customers | 🟢 ok | 0 | 0.000% | reported; orphans cannot enter order marts; orphan child rows |
| order_items.product_id -> products | 🟢 ok | 0 | 0.000% | reported; orphans cannot enter order marts; orphan child rows |
| order_items.seller_id -> sellers | 🟢 ok | 0 | 0.000% | reported; orphans cannot enter order marts; orphan child rows |
| customer zip prefix without coordinates | 🟡 предупреждение | 279 | 0.281% | geo_missing indicator = 1; distance features are NULL |
| seller zip prefix without coordinates | 🟡 предупреждение | 7 | 0.226% | geo_missing indicator = 1; distance features are NULL |
| product category without English translation | 🟡 предупреждение | 13 | 0.040% | original Portuguese name is kept; categories: ['pc_gamer', 'portateis_cozinha_e_preparadores_de_alimentos'] |
| orders without items | 🟡 предупреждение | 775 | 0.779% | kept in mart_orders with NULL item value (not 0); excluded from ML |
| orders without payments | 🟡 предупреждение | 1 | 0.001% | kept; payment value NULL |
| orders without reviews | 🟡 предупреждение | 768 | 0.772% | kept; review metrics use reviewed orders only |

## Типизация

| Проверка | Статус | Число | Доля | Действие / примечание |
|---|---|---:|---:|---|
| orders.order_purchase_timestamp not castable to TIMESTAMP | 🟢 ok | 0 | 0.000% | value becomes NULL |
| orders.order_approved_at not castable to TIMESTAMP | 🟢 ok | 0 | 0.000% | value becomes NULL |
| orders.order_delivered_carrier_date not castable to TIMESTAMP | 🟢 ok | 0 | 0.000% | value becomes NULL |
| orders.order_delivered_customer_date not castable to TIMESTAMP | 🟢 ok | 0 | 0.000% | value becomes NULL |
| orders.order_estimated_delivery_date not castable to TIMESTAMP | 🟢 ok | 0 | 0.000% | value becomes NULL |
| order_items.price not castable to DECIMAL(18,2) | 🟢 ok | 0 | 0.000% | value becomes NULL |
| order_items.freight_value not castable to DECIMAL(18,2) | 🟢 ok | 0 | 0.000% | value becomes NULL |
| order_items.order_item_id not castable to INTEGER | 🟢 ok | 0 | 0.000% | value becomes NULL |
| order_payments.payment_value not castable to DECIMAL(18,2) | 🟢 ok | 0 | 0.000% | value becomes NULL |
| order_payments.payment_installments not castable to INTEGER | 🟢 ok | 0 | 0.000% | value becomes NULL |
| order_reviews.review_score not castable to INTEGER | 🟢 ok | 0 | 0.000% | value becomes NULL |
| order_reviews.review_creation_date not castable to TIMESTAMP | 🟢 ok | 0 | 0.000% | value becomes NULL |
| products.product_weight_g not castable to DOUBLE | 🟢 ok | 0 | 0.000% | value becomes NULL |
| geolocation.geolocation_lat not castable to DOUBLE | 🟢 ok | 0 | 0.000% | value becomes NULL |

## Суммы

| Проверка | Статус | Число | Доля | Действие / примечание |
|---|---|---:|---:|---|
| negative item price | 🟢 ok | 0 | 0.000% | flagged, kept |
| negative freight | 🟢 ok | 0 | 0.000% | flagged, kept |
| negative payment | 🟢 ok | 0 | 0.000% | flagged, kept |
| zero item price | 🟢 ok | 0 | 0.000% | flagged, kept |
| zero freight (free shipping) | 🔵 ожидаемо | 383 | 0.340% | flagged, kept; free shipping is a valid business case |
| zero payment records | 🔵 ожидаемо | 9 | 0.009% | flagged, kept; mostly vouchers |
| payment_type = not_defined | 🟡 предупреждение | 3 | 0.003% | flagged, kept |
| payment_installments = 0 | 🟡 предупреждение | 2 | 0.002% | flagged, kept |

## Временная последовательность

| Проверка | Статус | Число | Доля | Действие / примечание |
|---|---|---:|---:|---|
| purchase_missing | 🟢 ok | 0 | 0.000% | purchase date is the default time axis |
| approved_before_purchase | 🟢 ok | 0 | 0.000% | excluded from ML (temporal_error) |
| carrier_before_purchase | 🟡 предупреждение | 166 | 0.167% | flag only; carrier date is not used in features or target |
| carrier_before_approval | 🟡 предупреждение | 1,359 | 1.367% | flag only; carrier date is not used in features or target |
| delivered_before_carrier | 🟡 предупреждение | 23 | 0.023% | flag only; target uses delivered vs promised date |
| delivered_before_approval | 🟡 предупреждение | 61 | 0.061% | excluded from ML (temporal_error): prediction time is unreliable |
| delivered_before_purchase | 🟢 ok | 0 | 0.000% | excluded from ML (temporal_error) |
| estimated_before_purchase | 🟢 ok | 0 | 0.000% | excluded from ML (temporal_error) |
| estimated_before_approval | 🟡 предупреждение | 10 | 0.010% | kept; promised_days <= 0 is a legitimate feature value |
| delivered_date_not_delivered_status | 🟡 предупреждение | 6 | 0.006% | outside the delivered cohort (status is authoritative) |
| delivered_status_no_date | 🟡 предупреждение | 8 | 0.008% | outside the delivered cohort; NOT treated as on time |
| delivered_status_not_approved | 🟡 предупреждение | 14 | 0.014% | excluded from ML (not_approved) |

## Геолокация

| Проверка | Статус | Число | Доля | Действие / примечание |
|---|---|---:|---:|---|
| points outside Brazil bounding box | 🟡 предупреждение | 42 | 0.004% | dropped before the per-prefix median |

## Покрытие и срез данных

| Проверка | Статус | Число | Доля | Действие / примечание |
|---|---|---:|---:|---|
| orders purchased on/after data_cutoff 2018-09-03 | 🔵 ожидаемо | 20 | 0.020% | excluded from ML splits by time boundaries; sparse tail after the effective snapshot |
| deliveries on/after data_cutoff 2018-09-03 | 🔵 ожидаемо | 53 | 0.053% | outcomes after the effective snapshot are treated as not observable |

## Статусы заказов

| Статус | Заказов | С подтверждением | С передачей перевозчику | С доставкой | С обещанной датой |
|---|---:|---:|---:|---:|---:|
| delivered | 96,478 | 96,464 | 96,476 | 96,470 | 96,478 |
| shipped | 1,107 | 1,107 | 1,107 | 0 | 1,107 |
| canceled | 625 | 484 | 75 | 6 | 625 |
| unavailable | 609 | 609 | 0 | 0 | 609 |
| invoiced | 314 | 314 | 0 | 0 | 314 |
| processing | 301 | 301 | 0 | 0 | 301 |
| created | 5 | 0 | 0 | 0 | 5 |
| approved | 2 | 2 | 0 | 0 | 2 |

## Исходы (mart_orders.outcome)

| Исход | Заказов |
|---|---:|
| on_time | 89,936 |
| late | 6,534 |
| not_delivered | 1,729 |
| canceled | 625 |
| unavailable | 609 |
| delivered_missing_date | 8 |

## Временное покрытие

| Поле | Минимум | Максимум |
|---|---|---|
| purchase | 2016-09-04 21:15:19 | 2018-10-17 17:30:18 |
| approved | 2016-09-15 12:16:38 | 2018-09-03 17:40:06 |
| carrier | 2016-10-08 10:34:01 | 2018-09-11 19:48:28 |
| delivered | 2016-10-11 13:46:32 | 2018-10-17 13:22:46 |
| estimated | 2016-09-30 00:00:00 | 2018-11-12 00:00:00 |
| review_creation | 2016-10-02 00:00:00 | 2018-08-31 00:00:00 |

### По месяцам покупки

Доля задержек = опоздавшие / доставленные с обеими датами (когорта по месяцу покупки).

| Месяц | Заказов | Доставлено | Опоздало | Доля задержек |
|---|---:|---:|---:|---:|
| 2016-09 | 4 | 1 | 1 | 100.0% |
| 2016-10 | 324 | 265 | 2 | 0.8% |
| 2016-12 | 1 | 1 | 0 | 0.0% |
| 2017-01 | 800 | 750 | 22 | 2.9% |
| 2017-02 | 1,780 | 1,653 | 49 | 3.0% |
| 2017-03 | 2,682 | 2,546 | 116 | 4.6% |
| 2017-04 | 2,404 | 2,303 | 151 | 6.6% |
| 2017-05 | 3,700 | 3,545 | 106 | 3.0% |
| 2017-06 | 3,245 | 3,135 | 95 | 3.0% |
| 2017-07 | 4,026 | 3,872 | 108 | 2.8% |
| 2017-08 | 4,331 | 4,193 | 122 | 2.9% |
| 2017-09 | 4,285 | 4,150 | 182 | 4.4% |
| 2017-10 | 4,631 | 4,478 | 187 | 4.2% |
| 2017-11 | 7,544 | 7,288 | 904 | 12.4% |
| 2017-12 | 5,673 | 5,513 | 411 | 7.5% |
| 2018-01 | 7,269 | 7,069 | 403 | 5.7% |
| 2018-02 | 6,728 | 6,555 | 926 | 14.1% |
| 2018-03 | 7,211 | 7,003 | 1,328 | 19.0% |
| 2018-04 | 6,939 | 6,798 | 306 | 4.5% |
| 2018-05 | 6,873 | 6,749 | 443 | 6.6% |
| 2018-06 | 6,167 | 6,096 | 71 | 1.2% |
| 2018-07 | 6,292 | 6,156 | 208 | 3.4% |
| 2018-08 | 6,512 | 6,351 | 393 | 6.2% |
| 2018-09 | 16 | 0 | 0 |  |
| 2018-10 | 4 | 0 | 0 |  |

## Пропуски в исходных CSV (NULL или пустая строка)

- **orders**: `order_approved_at` 160, `order_delivered_carrier_date` 1,783, `order_delivered_customer_date` 2,965
- **order_reviews**: `review_comment_title` 87,658, `review_comment_message` 58,256
- **products**: `product_category_name` 610, `product_name_lenght` 610, `product_description_lenght` 610, `product_photos_qty` 610, `product_weight_g` 2, `product_length_cm` 2, `product_height_cm` 2, `product_width_cm` 2

## Пригодность для ML

Постановка: прогноз опоздания на момент `order_approved_at`; обучающая популяция — подтверждённые доставленные заказы с обещанной датой, без критических временных ошибок, с позициями.

- Доставленная когорта (status=delivered, есть дата доставки и обещанная дата): **96,470**, из них опоздали **6,534** (6.77%).
- Пригодно для ML (до временного разбиения): **96,395**, доля задержек 6.78%.

| Причина | Заказов | Доля |
|---|---:|---:|
| eligible | 96,395 | 96.94% |
| not_delivered | 2,825 | 2.84% |
| not_approved | 160 | 0.16% |
| temporal_error | 61 | 0.06% |

Исключения по временным периодам и доступности меток фиксируются при обучении (`artifacts/model_metadata.json`, `reports/model_card.md`).
