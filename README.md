

**Preparation:**
1. Dataset address: https://www.kaggle.com/datasets/Cornell-University/arxiv
2. Download the json data:
```bash
mkdir -p ./data
curl -L -o ./data/arxiv.zip https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv
unzip ./data/arxiv.zip -d ./data
```
3.  Готуємо датасет: `uv run scripts/01_prepare_data.py`

**1.2. answers:**
1. Pinecone vs Qdrant vs Chroma

    a) За моделлю розгортання
    - **Pinecone** — переважно хмарний, повністю керований сервіс. Це найпростіший варіант для продакшену, бо не потребує адміністрації інфраструктури.
    - **Qdrant** — може працювати як self-hosted (Docker, Kubernetes) або у хмарній версії. Дає більше контролю над середовищем розгортання.
    - **Chroma** — найпростіший у запуску для локальної розробки або self-hosted використання. Часто обирають для прототипів та невеликих проєктів.

    b) За ліцензією
    - **Pinecone** — комерційний продукт з пропрієтарною ліцензією.
    - **Qdrant** — відкритий код, ліцензія Apache 2.0.
    - **Chroma** — відкритий код, ліцензія Apache 2.0.

    c) За продуктивністю
    - **Pinecone** — зазвичай найкраще підходить для великих production RAG-систем завдяки високій доступності, низькій затримці та простоті масштабування.
    - **Qdrant** — показує сильний баланс між продуктивністю, гнучкістю та контролем. Добре підходить для продакшену, якщо важливий self-hosting.
    - **Chroma** — добре працює для локальної розробки та середніх навантажень, але не завжди є найкращим вибором для дуже великих production сценаріїв.

    d) Коли обрати кожен
    - **Pinecone** — якщо потрібен швидкий, масштабований і простий у підтримці продакшен для RAG або векторного пошуку.
    - **Qdrant** — якщо хочете зберегти контроль над інфраструктурою, працювати з open-source стеком і мати гнучке розгортання.
    - **Chroma** — якщо потрібен простий старт для MVP, локальної розробки або невеликого проєкту.

    e) Короткий висновок
    - **Pinecone** — найпростіший продакшен-вибір.
    - **Qdrant** — найкращий компроміс між open-source, контролем і продуктивністю.
    - **Chroma** — найкращий для прототипів і локальної роботи.

2. Для задачі пошуку по наукових статтях було обрано модель `allenai/specter2_base`, а не універсальну `sentence-transformers/all-MiniLM-L6-v2`, оскільки вона спеціально створена для роботи з науковими текстами.

    З картки моделі на Hugging Face видно, що SPECTER2 призначена для генерації embeddingів для наукових задач. У описі моделі зазначено:

    > “SPECTER2 is capable of generating task specific embeddings for scientific tasks when paired with adapters. Given the combination of title and abstract of a scientific paper or a short textual query, the model can be used to generate effective embeddings to be used in downstream applications.”

    Крім того, у картці моделі вказано, що вона навчена для таких задач:
    - Classification
    - Regression
    - Proximity (Retrieval)
    - Adhoc Search

    Також модель навчалась на більш ніж 6 мільйонах triplets, що базуються на цитуваннях наукових статей, що робить її особливо придатною для пошуку схожих статей, рекомендацій та retrieval у науковій сфері.

    На відміну від цього, `sentence-transformers/all-MiniLM-L6-v2` є більш загальною sentence embedding-моделлю. Її картка описує її як модель для:
    - sentence and short paragraph encoding
    - sentence similarity
    - clustering
    - semantic search

    Тобто це хороший універсальний варіант для загальних текстових задач, але не спеціалізований під наукову доменну задачу.

    Джерела:
    - Hugging Face: https://huggingface.co/allenai/specter2_base
    - Hugging Face: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

3. Про метрику схожості для `allenai/specter2_base`

    У картці моделі `allenai/specter2_base` прямо не вказано назву конкретної метрики схожості, наприклад cosine similarity або dot product. Проте в описі моделі явно зазначено, що вона призначена для задач:

    - Proximity (Retrieval)
    - Adhoc Search
    - Nearest Neighbor Search

    Це означає, що модель розроблена для пошуку близьких документів за векторами, а не лише для загального представлення тексту.

    Чому це важливо при створенні індексу:
    - при створенні векторного індексу необхідно використовувати ту ж саму метрику схожості, що й під час пошуку запиту;
    - якщо метрика зміниться, результати пошуку можуть стати менш точними;
    - релевантні документи будуть повертатися рідше, а якість retrieval знизиться.

    На практиці для embedding-based search зазвичай використовують cosine similarity, оскільки вона добре працює для семантичного пошуку і дає стабільні результати для векторів текстів.

