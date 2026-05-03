# stance-detection-app

Приложение для определения правдивости новостных статей с использованием моделей машинного обучения.
Правдивость определяется схожестью заголовка с содержимым текста новости.
Создано на основе датасета:
https://www.kaggle.com/datasets/morfifinka/fake-real-news-ru/data
##  Используемые модели
- LinearSVC, Logistic Regression,Gradient Boosting,Random Forest
- По итогу эффективности выбран LinearSVC

##  Что сделано
- Проведен EDA (анализ данных)
- Обработаны наборы данных
- Выполнен feature engineering с TF-IDF и cosine-similarity-matrices
- Построены модели обучения
- Реализовано веб-приложение на Streamlit

##  Demo


##  Технологии
- Python
- scikit-learn
- TfidfVectorizer
- scipy
- Streamlit

##  Структура проекта
- `stance_detection_app.py` — приложение
- `svc_stance.pkl,tfidf_body_word.pkl, tfidf_hl_char.pkl,tfidf_hl_word.pkl,tfidf_shared.pkl` — обученные модели
- `requirements.txt` — зависимости
