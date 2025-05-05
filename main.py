import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


with open('./models/gnb.pkl', 'rb') as f:
    gnb = pickle.load(f)
with open('./models/gbc.pkl', 'rb') as f:
    gbc = pickle.load(f)
with open('./models/bc.pkl', 'rb') as f:
    bc = pickle.load(f)
with open('./models/sc.pkl', 'rb') as f:
    sc = pickle.load(f)
with open('./models/xgbc.pkl', 'rb') as f:
    xgbc = pickle.load(f)
nn = tf.keras.models.load_model('./models/nn_model.keras')

models = {
    "Наивный Байес": gnb,
    "Градиентный бустинг": gbc,
    "Бэггинг": bc,
    "Стэкинг": sc,
    "XGBoost": xgbc,
    "Нейронная сеть": nn
}

def preprocess(df):
    X = df.values
    scaler = StandardScaler()
    return scaler.fit_transform(X)


st.set_page_config(page_title="РГР", layout="wide")

page = st.sidebar.radio("Навигация", ["О работе", "О датасете", "Визуализации", "Предсказание"])

if page == "О работе":
    st.markdown("<h1 style='text-align: center; font-size: 64px;'>РГР</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image("./images/image.png", width=400)

    with col2:
        st.markdown("""
            <div style='padding-left: 0px; font-size: 32px;'>
                <p style='margin: 0;'><strong>Выполнил:</strong> Биба Никита Владимирович</p>
                <p style='margin: 0;'><strong>Группа:</strong> ФИТ-231</p>
                <p style='margin-top: 16px;'><strong>Тема:</strong> Разработка web-интерфейса для предсказаний моделей машинного обучения с использованием Streamlit.</p>
            </div>
        """, unsafe_allow_html=True)

elif page == "О датасете":
    st.title("Описание датасета")
    st.write("Название: CSGO Dataset")
    st.write("""
        Этот датасет содержит информацию о .... Он используется для .... 
        Признаки включают: ....
    """)
    df = pd.read_csv('./datasets/csgo_processed.csv', sep=';')
    st.subheader("Первые 5 строк датасета:")
    st.dataframe(df.head())

elif page == "Визуализации":
    st.title("Визуализации зависимостей")

    df = pd.read_csv('./datasets/csgo_processed.csv', sep=';')
    
    # --- 1. Heatmap ---
    fig1, ax1 = plt.subplots(figsize=(9, 9))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, annot_kws={"size": 6}, ax=ax1)
    ax1.set_title("Корреляционная матрица", fontsize=14)

    # --- 2. Scatter plot ---
    fig2, ax2 = plt.subplots(figsize=(9, 9))
    sns.scatterplot(data=df, x='ct_score', y='t_score', ax=ax2)
    ax2.set_title("ct_score vs t_score", fontsize=14)

    # --- 3. Распределение t_money ---
    fig3, ax3 = plt.subplots(figsize=(9, 9))
    sns.histplot(df['t_money'], kde=True, ax=ax3)
    ax3.set_title("Распределение t_money", fontsize=14)

    # --- 4. Распределение ct_score ---
    fig4, ax4 = plt.subplots(figsize=(9, 9))
    sns.histplot(df['ct_score'], kde=True, ax=ax4)
    ax4.set_title("Распределение ct_score", fontsize=14)

    # --- Отображение в сетке 2x2 ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Heatmap")
        st.pyplot(fig1, use_container_width=False)

    with col2:
        st.subheader("2. ct_score vs t_score")
        st.pyplot(fig2, use_container_width=False)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("3. Распределение t_money")
        st.pyplot(fig3, use_container_width=False)

    with col4:
        st.subheader("4. Распределение ct_score")
        st.pyplot(fig4, use_container_width=False)

elif page == "Предсказание":
    st.title("Предсказание модели")

    model_name = st.selectbox("Выберите модель:", list(models.keys()))
    uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file, sep=";")
        st.write("Загруженные данные:")
        st.dataframe(input_df)

        processed = preprocess(input_df)

        model = models[model_name]
        if model_name == "Нейронная сеть":
            preds = model.predict(processed)
            labels = preds.argmax(axis=1)
        else:
            labels = model.predict(processed)

        def interpret_label(label):
            return "Бомба не заложена" if label == 0 else "Бомба заложена"

        interpreted = [interpret_label(label) for label in labels]

        st.subheader("Результаты предсказания:")
        for i, (label, text) in enumerate(zip(labels, interpreted)):
            color = "red" if label == 0 else "green"
            st.markdown(
                f"<p style='color:{color}; font-size:18px;'>Запись {i+1}: {text}</p>",
                unsafe_allow_html=True
            )
