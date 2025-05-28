#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 모델 및 인코더 불러오기
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# 피처 추출 함수
def extract_features(df, window_size=50, step=25):
    features = []
    for start in range(0, len(df) - window_size + 1, step):
        window = df.iloc[start:start + window_size]
        x = window["Linear Acceleration x (m/s^2)"]
        y = window["Linear Acceleration y (m/s^2)"]
        z = window["Linear Acceleration z (m/s^2)"]
        abs_acc = window["Absolute acceleration (m/s^2)"]
        stats = lambda s: [s.mean(), s.std(), s.max(), s.min(), s.median()]
        feature_vector = stats(x) + stats(y) + stats(z) + stats(abs_acc)
        features.append(feature_vector)
    return np.array(features)

# UI
st.title("동작 인식 웹앱")
uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터 미리보기:", df.head())
    
    try:
        features = extract_features(df)
        preds = model.predict(features)
        labels = label_encoder.inverse_transform(preds)
        
        st.success("예측 성공!")
        st.write("예측 결과 (창 닫으면 사라집니다):")
        st.dataframe(pd.DataFrame({"예측 동작": labels}))
        
        # 간단한 분포 시각화
        st.bar_chart(pd.Series(labels).value_counts())
    except Exception as e:
        st.error(f" 오류 발생: {e}")

