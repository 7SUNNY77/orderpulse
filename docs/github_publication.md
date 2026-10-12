# Публикация OrderPulse на GitHub

Документ для владельца репозитория. На момент подготовки создан только **локальный** репозиторий (ветка `main`).
Удалённый репозиторий не создан, push не выполнялся, GitHub Release не опубликован, workflow CI на GitHub
ещё ни разу не запускался.

## 1. Перед публикацией (действия владельца)

- [x] Лицензия кода — MIT (решение владельца), `LICENSE`. Лицензия данных Olist (CC BY-NC-SA 4.0) отдельна.
- [x] `OrderPulse_Technical_Specification.md` не публикуется (решение владельца), файл исключён в `.gitignore`.
- [ ] Проверить автора коммитов: `git log --format='%an <%ae>'`. Коммит создан с глобальной Git-identity этой машины.
      Если нужен другой адрес, например GitHub noreply, задайте его до push (см. раздел 6).

## 2. Параметры репозитория

| Поле | Значение |
|---|---|
| Имя | `orderpulse` |
| Видимость | public (решение владельца) |
| Описание | E-commerce delivery analytics and daily risk prioritization on the Olist dataset, with temporal backtesting and interpretable ML. |
| Website | не указывать: live demo нет |

Topics:

```
python  machine-learning  data-engineering  ecommerce  streamlit  duckdb  scikit-learn  catboost
explainable-ai  plotly  olist
```

Все темы соответствуют реализации: CatBoost используется как модель для сравнения, Plotly — для графиков,
`explainable-ai` — линейные вклады LogReg и SHAP для CatBoost.

## 3. Создание remote и push

Замените `<GITHUB_USERNAME>` на свой логин. Пустой репозиторий `orderpulse` создайте на github.com без README,
`.gitignore` и лицензии — иначе первый push конфликтует с локальной историей.

```bash
git remote add origin https://github.com/<GITHUB_USERNAME>/orderpulse.git
git remote -v
git push -u origin main
```

Вариант с GitHub CLI (создаёт репозиторий и делает push одной командой):

```bash
gh repo create <GITHUB_USERNAME>/orderpulse --public --source . --remote origin --push \
  --description "E-commerce delivery analytics and daily risk prioritization on the Olist dataset, with temporal backtesting and interpretable ML."
gh repo edit <GITHUB_USERNAME>/orderpulse --add-topic python,machine-learning,data-engineering,ecommerce,streamlit,duckdb,scikit-learn,catboost,explainable-ai,plotly,olist
```

## 4. Проверка после публикации

1. Вкладка **Actions**: workflow `CI` запустился на push и прошёл на `ubuntu-latest` и `windows-latest`.
   Локально workflow не исполнялся. Если Linux-установка упадёт на закреплённых версиях
   (`requirements-lock.txt` собран на Windows), исправьте lock-файл, а не тесты.
2. README открывается без битых картинок, ссылки на `reports/` и `docs/` работают.
3. В файлах репозитория нет `data/`, `*.duckdb`, `*.parquet`, `*.cbm`, `*.joblib`.
4. Чистый клон на другой машине: шаги раздела 3 README, затем `python -m pytest`.
5. Бейдж CI добавляйте в README только после первого зелёного запуска:
   `![CI](https://github.com/<GITHUB_USERNAME>/orderpulse/actions/workflows/ci.yml/badge.svg)`.

## 5. Тег v1.0.0 и GitHub Release

Тег ставится на проверенный коммит после зелёного CI.

```bash
git tag -a v1.0.0 -m "OrderPulse v1.0.0"
git push origin v1.0.0
gh release create v1.0.0 --title "OrderPulse v1.0.0" --notes-file <файл с текстом из раздела 7> --draft
```

Флаг `--draft` создаёт черновик: перед публикацией проверьте текст на github.com.

## 6. Автор коммитов (если нужно изменить)

Только для этого репозитория, без глобальных настроек:

```bash
git config user.name "<ИМЯ>"
git config user.email "<EMAIL>"
```

Уже созданный коммит при этом не меняется. Переписывание автора (`git commit --amend --reset-author`)
меняет историю — делайте это только до первого push и осознанно.

## 7. Черновик release notes

```markdown
## OrderPulse v1.0.0

Аналитика доставки и ежедневная очередь заказов для проверки по риску опоздания на открытых данных Olist.

### Что внутри
- DuckDB-конвейер raw → staging → marts: аудит качества данных, 17 проверок сверки, атомарная пересборка.
- Streamlit-приложение: обзор с KPI и знаменателями, проблемные сегменты с 95% ДИ, очередь риска
  (top-K в каждый день, K = 20 по умолчанию), карточка заказа, страница качества модели и данных.
- Основная модель — Logistic Regression с точными линейными вкладами признаков; CatBoost — для сравнения (SHAP).
- Временная проверка: признаки на момент подтверждения, временные выборки с зазорами, кодовая проверка
  доступности меток, случайный baseline для top-K, парный bootstrap.

### Главный результат
Три ретроспективных development-окна, K = 20 заказов в день: Logistic Regression находит 3,412 опоздания в день
против 1,925 при случайном отборе (precision 17,13%, lift 1,773).

### Ограничения
Историческая симуляция; модель обучена на доставленных заказах; риск-скор не откалиброван; найденная задержка
не означает предотвращённую; development-результат относится к backtest-моделям по тому же рецепту; исходный
test ранее просмотрен. Это портфолио-проект, не production-система.

### Данные и лицензия
Данные Olist (CC BY-NC-SA 4.0) не входят в репозиторий — см. README, «Получение данных».
Код — под лицензией MIT.
```
