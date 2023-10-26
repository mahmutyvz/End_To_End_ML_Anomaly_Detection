# -*- coding: utf-8 -*-
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def pipeline_build(model,cat_cols):
    """
    This function is used to create a data processing pipeline, typically employed in machine learning workflows to facilitate preprocessing steps.

    Parameters:

    model: The machine learning model. This parameter represents the model that will be used at the end of the pipeline.
    cat_cols: A list of columns containing categorical features. This parameter determines the preprocessing steps to handle categorical data.
    Returns:

    A Pipeline object representing the constructed data processing pipeline.
    """
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_cols)], remainder='passthrough')
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('algorithm', model)])
    return pipe

def drop_distinct(df):
    """
    This function takes a DataFrame (df) as input and removes columns where all values are the same, effectively dropping columns with constant values.

    Parameters:

    df: The input DataFrame.
    Returns:

    A modified DataFrame with columns removed if all values in the column are identical.
    """
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df = df.drop([col], 1)
    return df
