import streamlit as st
import pandas as pd
import pandas_profiling as pdp
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
import numpy as np
from catboost import CatBoost, Pool

from preprocessing import (
    get_target_df,
    get_data_df,
    train_preprocessing,
    test_preprocessing,
    visualize_survived_age,
    visualize_survived_fare
)


@st.cache_data
def load_data():
    # データの読み込み
    row_traindf = pd.read_csv('src/train.csv')
    row_testdf = pd.read_csv('src/test.csv')
    
    return row_traindf, row_testdf


# データの読み込み
row_traindf, row_testdf = load_data()

# 統計データを確認
pr = row_traindf.profile_report()
st_profile_report(pr)
pr = row_testdf.profile_report()
st_profile_report(pr)


# 前処理
traindf = train_preprocessing(row_traindf)
testdf = test_preprocessing(row_testdf)

#Survivedの値をラベルとして指定する
traindf["Survived"] = traindf["Survived"].replace({0: "Deceased", 1: "Survived"})
# SurvivedとAgeの関係を可視化
st.subheader("Survived vs Age")
visualize_survived_age(traindf)
# SurvivedとFareの関係を可視化
st.subheader("Survived vs Fare")
visualize_survived_fare(traindf)

# データを分ける
data_train = get_data_df(traindf)
target_train = get_target_df(traindf)
# 予測するデータ
X_test = get_data_df(testdf)

# trainとevalに分ける
X_train, X_eval, y_train, y_eval = train_test_split(data_train, target_train, test_size=0.3, shuffle=True, stratify=target_train, random_state=42)

# カテゴリカル変数の指定
cat_features = ["Embarked", "Sex"]

# データを専用の型に変換
cat_train = Pool(X_train, label=y_train, cat_features=cat_features)
cat_eval = Pool(X_eval, label=y_eval, cat_features=cat_features)

# パラメータ設定
params = {
    'num_boost_round': 10000,
    'depth': 5,
    'learning_rate': 0.01,
    'loss_function': 'Logloss',
    'early_stopping_rounds': 200,
    'random_seed': 42
}

# 学習
catb = CatBoost(params)
catb.fit(cat_train, eval_set=[cat_eval], verbose=0, use_best_model=True)

# 検証用データで予測
predictiondf = catb.predict(X_test, prediction_type="Class")

# PassengerIdを取得
PassengerId = np.array(row_testdf["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
ans = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictiondf})

# catboost_result.csvとして書き出し
ans.to_csv("my_result.csv", index=False)

print("Your submission was successfully saved!")