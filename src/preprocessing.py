import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

# 必要な説明変数の抽出
def get_data_df(df):
    return df[["Age", "Fare", "Pclass", "Sex", "SibSp", "Embarked"]]


# 必要な目的変数の抽出
def get_target_df(df):
    return df[["Survived"]].values


# traindataの前処理をする関数
def train_preprocessing(df):
    # 欠損処理
    df["Embarked"] = df.groupby(["Sex", "Pclass"])["Embarked"].transform(lambda x: x.fillna(x.mode()[0]))
    
    # 性別とクラスごとの中央値で欠損値を補完
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.median()))
    # Ageカラムの値を切り捨てて置き換える
    df["Age"] = np.floor(df["Age"])

    return df[["Age", "Fare", "Pclass", "Sex", "SibSp", "Embarked", "Survived"]]


# testdataの前処理をする関数
def test_preprocessing(df):
    # 欠損処理
    df["Embarked"] = df.groupby(["Sex", "Pclass"])["Embarked"].transform(lambda x: x.fillna(x.mode()[0]))
    
    # 性別とクラスごとの中央値で欠損値を補完
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(lambda x: x.fillna(x.median()))
    df["Fare"] = df.groupby(["Sex", "Pclass"])["Fare"].transform(lambda x: x.fillna(x.median()))
    # Ageカラムの値を切り捨てて置き換える
    df["Age"] = np.floor(df["Age"])

    return df[["Age", "Fare", "Pclass", "Sex", "SibSp", "Embarked"]]


# SurvivedとAgeの関係を可視化する関数
def visualize_survived_age(df):
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", kde=0, ax=ax)
    plt.title("Survived vs Age")
    plt.xlabel("Age")
    plt.ylabel("Count")
    st.pyplot(fig)


# SurvivedとFareの関係を可視化する関数
def visualize_survived_fare(df):
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Survived", y="Fare", ax=ax)
    plt.title("Survived vs Fare")
    plt.xlabel("Survived")
    plt.ylabel("Fare")
    plt.ylim(0, 200)
    st.pyplot(fig)