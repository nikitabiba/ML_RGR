import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


with open('./models/gnb.pkl', 'rb') as f:
    gnb = pickle.load(f)
with open('./models/cbc.pkl', 'rb') as f:
    cbc = pickle.load(f)
with open('./models/bc.pkl', 'rb') as f:
    bc = pickle.load(f)
with open('./models/sc.pkl', 'rb') as f:
    sc = pickle.load(f)
with open('./models/xgbc.pkl', 'rb') as f:
    xgbc = pickle.load(f)
with open("./models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
nn = tf.keras.models.load_model('./models/nn_model.h5')

models = {
    "Наивный Байес": gnb,
    "CatBoost": cbc,
    "Бэггинг": bc,
    "Стэкинг": sc,
    "XGBoost": xgbc,
    "Нейронная сеть": nn
}

def preprocess(df):
    X = df.values
    return scaler.transform(X)


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
        Предментная область:\n
        Предметной область датасета является игра в жанре "шутер" - CSGO. Записи в датасета представляют из себя состояния игры в случайные моменты времени. Состояния игры фиксировались в среднем раз в 30 секунд. Целевым признаком датасета является факт заложенной бомбы, то есть заложена она(класс 1) или нет(класс 0).\n
        Признаки датасета:\n
        time_left - оставшееся до конца раунда время;\n
        ct_score - количество выигранных контр-террористами раундов;\n 
        t_score - количество выигранных террористами раундов;\n
        map - игровая карта;\n
        bomb_planted - заложена ли;\n
        ct_health - суммарные очки здоровья контр-террористов;\n
        t_health - суммарные очки здоровья террористов;\n
        ct_armor - суммарные очки брони контр-террористов;\n
        t_armor - суммарные очки брони террористов;\n
        ct_money - суммарные деньги контр-террористов;\n
        t_money - суммарные деньги террористов;\n
        ct_helmets - количество шлемов контр-террористов;\n
        t_helmets - количество шлемов террористов;\n
        ct_defuse_kits - количество наборов для обезвреживания бомбы контр-террористов;\n
        ct_players_alive - количество живых контр-террористов;\n
        t_players_alive - количество живых террористов;\n
        Описание процесса предобработки датасета:\n
        В процессе предобработки датасета были заполнены пропущенные значения. Они заполнялись по значению предыдущей записи, поскольку записи в датасете идут в порядке хода времени. Далее, категориальные значения были заменены на упорядоченные целые числа. Также, были удалены дубликаты. Выбросов в данных при исследовании распределния данных по квантилям обнаружено не было. Был добавлен новый признак t_healt>ct_health, который отражает факт того, больше ли очков здоровья у команды террористов, чем у команды контр-террористов. Также, были удалены данные, которых не может быть в реальности. Например, в игре не может быть в живых одновременно 6 игроков одной команды. Или, сумма очков здоровья у команды не может превышать количество живых игроков, умноженное на 100, поскольку у каждого игрока максимум 100 очков здоровья. После предобработки данные были сохранены в файл формата CSV. При обучении моделей данные также были стандартизированы для лучших результатов, а также была решена проблема дисбаланса классов.
    """)
    df = pd.read_csv('./datasets/csgo_processed.csv', sep=';')
    st.subheader("Первые 5 строк датасета:")
    st.dataframe(df.head())

elif page == "Визуализации":
    st.title("Визуализации зависимостей")

    df = pd.read_csv('./datasets/csgo_processed.csv', sep=';')
    
    fig1, ax1 = plt.subplots(figsize=(9, 9))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, annot_kws={"size": 6}, ax=ax1)
    ax1.set_title("Корреляционная матрица", fontsize=14)

    fig2, ax2 = plt.subplots(figsize=(9, 9))
    sns.scatterplot(data=df, x='ct_score', y='t_score', ax=ax2)
    ax2.set_title("ct_score vs t_score", fontsize=14)

    fig3, ax3 = plt.subplots(figsize=(9, 9))
    sns.histplot(df['t_money'], kde=True, ax=ax3)
    ax3.set_title("Распределение t_money", fontsize=14)

    fig4, ax4 = plt.subplots(figsize=(9, 9))
    sns.histplot(df['ct_score'], kde=True, ax=ax4)
    ax4.set_title("Распределение ct_score", fontsize=14)

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
    st.title("🔍 Предсказание модели")

    model_name = st.selectbox("Выберите модель:", list(models.keys()))
    uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file, sep=";")
        st.write("Загруженные данные:")
        st.dataframe(input_df)

    else:
        st.info("Или введите данные вручную ниже:")

        time_left = st.number_input("Время до конца раунда", min_value=0.0, max_value=175.0, value=100.0, step=0.01, format="%.2f")
        ct_score = st.number_input("Счёт спецназа по раундам", min_value=0, max_value=32, value=0)
        t_score = st.number_input("Счёт террористов по раундам", min_value=0, max_value=32, value=0)
        map_val = st.number_input("Карта", min_value=0, max_value=7, value=0)
        ct_health = st.number_input("Здоровье спецназа", min_value=0, max_value=500, value=100)
        t_health = st.number_input("Здоровье террористов", min_value=0, max_value=500, value=100)
        ct_armor = st.number_input("Броня спецназа", min_value=0, max_value=500, value=50)
        t_armor = st.number_input("Броня террористов", min_value=0, max_value=500, value=50)
        ct_money = st.number_input("Деньги спецназа", min_value=0, max_value=80000, value=1000)
        t_money = st.number_input("Деньги террористов", min_value=0, max_value=80000, value=1000)
        ct_helmets = st.number_input("Шлемы спецназа", min_value=0, max_value=5, value=3)
        t_helmets = st.number_input("Шлемы террористов", min_value=0, max_value=5, value=2)
        ct_defuse_kits = st.number_input("Дефуза спецназа", min_value=0, max_value=5, value=2)
        ct_players_alive = st.number_input("Живые спецназовцы", min_value=0, max_value=5, value=3)
        t_players_alive = st.number_input("Живые террористы", min_value=0, max_value=5, value=4)
        t_health_gt_ct = int(t_health > ct_health)

        manual_input = pd.DataFrame([{
            "time_left": time_left,
            "ct_score": ct_score,
            "t_score": t_score,
            "map": map_val,
            "ct_health": ct_health,
            "t_health": t_health,
            "ct_armor": ct_armor,
            "t_armor": t_armor,
            "ct_money": ct_money,
            "t_money": t_money,
            "ct_helmets": ct_helmets,
            "t_helmets": t_helmets,
            "ct_defuse_kits": ct_defuse_kits,
            "ct_players_alive": ct_players_alive,
            "t_players_alive": t_players_alive,
            "t_health > ct_health": t_health_gt_ct
        }])

        trigger_predict = st.button("Загрузить запись")
        input_df = manual_input if trigger_predict else None

        if trigger_predict:
            st.subheader("Введённая запись:")
            st.dataframe(input_df)

    if input_df is not None:
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
                f"<p style='color:{color}; font-size:18px;'>Запись {i}: {text}</p>",
                unsafe_allow_html=True
            )

