import streamlit as st
import datetime
import warnings
from paths import Path
from src.models.trainer import train_and_predict

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Anomaly detection",
                   page_icon="chart_with_upwards_trend", layout="wide")
st.markdown("<h1 style='text-align:center;'>Anomaly Detection</h1>", unsafe_allow_html=True)
st.write(datetime.datetime.now(tz=None))
tabs = ["Train And Predict", "About"]
page = st.sidebar.radio("Tabs", tabs)
if page == "Train And Predict":
    option = st.radio(
        'What model would you like to use for training?',
        ('Ad Exchange', 'AWS Cloud Watch', 'Known Cause', 'Traffic', 'Tweets'))
    with st.spinner("Training is in progress, please wait..."):
        if option == 'Ad Exchange':
            train_and_predict(Path.exchange_path,streamlit=True)
        elif option == 'AWS Cloud Watch':
            train_and_predict(Path.awscloudwatch_path,streamlit=True)
        elif option == 'Known Cause':
            train_and_predict(Path.knowncause_path,streamlit=True)
        elif option == 'Traffic':
            train_and_predict(Path.traffic_path,streamlit=True)
        elif option == 'Tweets':
            train_and_predict(Path.tweets_path,streamlit=True)


elif page == "About":
    st.header("Contact Info")
    st.markdown("""**mahmutyvz324@gmail.com**""")
    st.markdown("""**[LinkedIn](https://www.linkedin.com/in/mahmut-yavuz-687742168/)**""")
    st.markdown("""**[Github](https://github.com/mahmutyvz)**""")
    st.markdown("""**[Kaggle](https://www.kaggle.com/mahmutyavuz)**""")
st.set_option('deprecation.showPyplotGlobalUse', False)
