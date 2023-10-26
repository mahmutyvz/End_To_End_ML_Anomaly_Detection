import pandas as pd
import os
from src.features.feature_engineering import feature_engineering
from src.data.dataset_source import read_dataset
from src.data.preprocess_data import pipeline_build,drop_distinct
from paths import Path
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px
import streamlit as st
def train_and_predict(path,streamlit=False):
    """
    This function trains anomaly detection models on datasets from a specified path and generates scatter plots to visualize the anomalies.

    Parameters:

    path: The path to the directory containing datasets.
    streamlit (optional, default=False): If True, the function uses Streamlit to display the scatter plots. If False, the function uses Plotly Express for visualization.
    Returns:

    None. The function generates scatter plots to visualize anomalies.
    """
    data_list = os.listdir(path)
    model_list = [IsolationForest(
        random_state=Path.random_state,
        contamination=0.05,
        n_estimators=100,
        max_samples='auto'
    ),OneClassSVM(nu=0.01),LocalOutlierFactor(n_neighbors=10)]
    for data in data_list:
        if path.split('/')[-1] == 'realTraffic':
            title = data.split('_')[0].upper()
        elif path.split('/')[-1] == 'realTweets':
            title = data.split('.')[0].upper()
        elif path.split('/')[-1] == 'realKnownCause':
            title = data.split('.')[0].upper()
        else:
            if 'exchange' in data.split('_')[0]:
                title = data.split('_')[1].upper()
            elif 'ec2' in data.split('_')[0]:
                title = f'{data.split("_")[1].upper()}_{data.split("_")[2].upper()}'

        data = read_dataset(path+"/"+data)
        data = feature_engineering(data,Path.timestamp_column)
        data = drop_distinct(data)
        cat_cols = data.select_dtypes(include=['object']).columns.tolist()
        for model in model_list:
            pipe = pipeline_build(model,cat_cols)
            output = pd.Series(pipe.fit_predict(data)).apply(lambda x: "anomaly" if x == -1 else "not anomaly")
            df = data.reset_index().copy()
            df['output'] = output
            fig = px.scatter(df, x=Path.timestamp_column, y='value', color='output',
                             title=f'{title} with {type(model).__name__}',
                             color_discrete_map={"anomaly": "red", "not anomaly": "blue"})
            fig.update_layout(width=800, height=600)
            if streamlit:
                st.plotly_chart(fig)
            else:
                fig.show()
