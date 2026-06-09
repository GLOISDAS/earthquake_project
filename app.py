import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from sklearn.decomposition import PCA

import folium
from streamlit_folium import st_folium

# -----------------------------
# 파일 불러오기
# -----------------------------

model = joblib.load("earthquake_model.pkl")
scaler = joblib.load("earthquake_scaler.pkl")

df = pd.read_csv("earthquake_data.csv")

# -----------------------------
# 페이지 설정
# -----------------------------

st.set_page_config(
    page_title="세계 지진 위험도 분석",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 세계 지진 위험도 분석 시스템")

st.markdown("""
지진의 규모, 진원깊이, 영향도를 입력하면

- 머신러닝 군집 분석 결과
- 위험도 평가
- PCA 시각화
- 세계 지도 시각화

를 확인할 수 있습니다.
""")

# -----------------------------
# 입력
# -----------------------------

st.header("📊 지진 정보 입력")

col1, col2, col3 = st.columns(3)

with col1:
    magnitude = st.number_input(
        "규모 (Magnitude)",
        min_value=0.0,
        max_value=10.0,
        value=4.5,
        step=0.1
    )

with col2:
    depth = st.number_input(
        "진원깊이 (km)",
        min_value=0.0,
        value=50.0
    )

with col3:
    impact = st.number_input(
        "영향도",
        min_value=0.0,
        value=300.0
    )

# -----------------------------
# 위치 정보
# -----------------------------

st.header("📍 위치 정보")

col4, col5 = st.columns(2)

with col4:
    latitude = st.number_input(
        "위도 (Latitude)",
        min_value=-90.0,
        max_value=90.0,
        value=37.5
    )

with col5:
    longitude = st.number_input(
        "경도 (Longitude)",
        min_value=-180.0,
        max_value=180.0,
        value=127.0
    )

st.info(
    "위도와 경도는 지도 시각화를 위한 정보이며 군집 예측에는 사용되지 않습니다."
)

# -----------------------------
# 분석
# -----------------------------

if st.button("🔍 분석하기"):

    # 입력 데이터 생성

    user_df = pd.DataFrame(
        [[magnitude, depth, impact]],
        columns=["규모", "진원깊이", "영향도"]
    )

    # 스케일링

    user_scaled = scaler.transform(user_df)

    # 군집 예측

    cluster = int(model.predict(user_scaled)[0])

    # -----------------------------
    # 위험도 해석
    # -----------------------------

    if cluster == 0:
        risk = "🟢 낮은 위험"
        description = "규모와 영향도가 비교적 낮은 지진 그룹입니다."

    elif cluster == 1:
        risk = "🔴 높은 위험"
        description = "규모와 영향도가 높은 위험 지진 그룹입니다."

    elif cluster == 2:
        risk = "🟡 중간 위험"
        description = "진원 깊이가 깊은 중간 위험 지진 그룹입니다."

    else:
        risk = "알 수 없음"
        description = ""

    st.divider()

    st.header("📈 분석 결과")

    st.success(f"예측 군집 : Group {cluster}")

    st.markdown(f"## {risk}")

    st.write(description)

    # -----------------------------
    # PCA 시각화
    # -----------------------------

    st.subheader("📊 PCA 군집 시각화")

    features = df[["규모", "진원깊이", "영향도"]]

    scaled_features = scaler.transform(features)

    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(scaled_features)

    plot_df = pd.DataFrame(
        pca_data,
        columns=["PCA1", "PCA2"]
    )

    plot_df["Cluster"] = model.predict(scaled_features)

    user_pca = pca.transform(user_scaled)

    fig = px.scatter(
        plot_df,
        x="PCA1",
        y="PCA2",
        color=plot_df["Cluster"].astype(str),
        title="전체 지진 데이터 분포"
    )

    fig.add_scatter(
        x=[user_pca[0][0]],
        y=[user_pca[0][1]],
        mode="markers",
        marker=dict(
            size=18,
            symbol="star"
        ),
        name="현재 입력값"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "PCA(주성분 분석)를 이용하여 3차원 데이터를 2차원으로 축소하여 시각화했습니다."
    )

    # -----------------------------
    # 지도 시각화
    # -----------------------------

    #st.subheader("🌍 세계 지진 분포")

    m = folium.Map(
        location=[0, 0],
        zoom_start=2
    )

    sample_df = df.sample(
        min(300, len(df)),
        random_state=42
    )

    for _, row in sample_df.iterrows():

        cluster_value = row["cluster"]

        if cluster_value == 0:
            color = "green"

        elif cluster_value == 1:
            color = "red"

        else:
            color = "orange"

        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=3,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    # 사용자 위치

    folium.Marker(
        [latitude, longitude],
        popup="사용자 입력 위치",
        tooltip="⭐ 사용자",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    st_folium(
        m,
        width=1000,
        height=600
    )

    # -----------------------------
    # 군집 설명표
    # -----------------------------

    st.subheader("📌 군집 설명")

    cluster_info = pd.DataFrame({
        "군집": [
            "Group 0",
            "Group 1",
            "Group 2"
        ],
        "위험도": [
            "🟢 낮은 위험",
            "🔴 높은 위험",
            "🟡 중간 위험"
        ],
        "특징": [
            "규모와 영향도가 낮음",
            "규모와 영향도가 높음",
            "진원 깊이가 매우 깊음"
        ]
    })

    st.table(cluster_info)
