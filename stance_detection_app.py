import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

# Заголовок
st.title("❓ Классификатор правдивости новости")

# Загрузка моделей
tfidf_hl_word = joblib.load('tfidf_hl_word.pkl')
tfidf_hl_char = joblib.load('tfidf_hl_char.pkl')
tfidf_body_word = joblib.load('tfidf_body_word.pkl')
tfidf_shared = joblib.load('tfidf_shared.pkl')

model = joblib.load('svc_stance.pkl')




# Ввод пользователя
headline = st.text_input("📝 Введите заголовок")
body = st.text_area("📘 Введите текст статьи")

# Кнопка
if st.button("🔍 Проверить"):
    if headline and body:
        # TF-IDF
        hl_w = tfidf_hl_word.transform([headline])
        hl_c = tfidf_hl_char.transform([headline])
        hl_s = tfidf_shared.transform([headline])


        body_v = tfidf_body_word.transform([body])
        body_s = tfidf_shared.transform([body])



        #  cosine функция
        cos = cosine_similarity(hl_s, body_s)[0][0]
        cos = np.array([[cos]])

        # Объединение
        X = hstack([hl_w, hl_c, body_v, cos])

        # Предсказание
        pred = model.predict(X)[0]

        if pred == 1:
            st.success("✅ Заголовок согласуется с текстом (Правда)")
        else:
            st.error("❌ Заголовок НЕ соответствует тексту (Фейк)")
    else:
        st.warning("Введите и заголовок, и текст")

st.caption("🏦 Определитель правдивости новостных статей  |  LinearSVC with TF-IDF | Курс ML, Итоговый проект Каренкиной Сабины | ")
