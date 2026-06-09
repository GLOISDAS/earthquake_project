import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from sklearn.decomposition import PCA

# ----------------------------------
# 파일 불러오기
# ----------------------------------

model = joblib.load("earthquake_model.pkl")
scaler = joblib.load("earthquake_scaler.pkl")

df = pd.read_csv("earthquake_data.csv")

# ----------------------------------
# 페이지 설정
# ----------------------------------

st.set_page_config(
    page_title="세계 지진 위험도 분석 시스템",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 세계 지진 위험도 분석 시스템")

st.markdown("""
지진의 규모, 진원깊이, 영향도를 입력하면

- 머신러닝 군집 분석 결과
- 위험도 평가
- PCA 시각화
- 세계 지진 분포 지도

를 확인할 수 있습니다.
""")

# ----------------------------------
# 지진 정보 입력
# ----------------------------------

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

# ----------------------------------
# 위치 정보
# ----------------------------------

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

# ----------------------------------
# 분석 버튼
# ----------------------------------

if st.button("🔍 분석하기"):

    try:

        # --------------------------
        # 입력 데이터
        # --------------------------

        user_df = pd.DataFrame(
            [[magnitude, depth, impact]],
            columns=["규모", "진원깊이", "영향도"]
        )

        user_scaled = scaler.transform(user_df)

        cluster = int(model.predict(user_scaled)[0])

        # --------------------------
        # 위험도 해석
        # --------------------------

        if cluster == 0:
            risk = "🟢 낮은 위험"
            description = "규모와 영향도가 비교적 낮은 지진 그룹입니다."

        elif cluster == 1:
            risk = "🔴 높은 위험"
            description = "규모와 영향도가 높은 위험 지진 그룹입니다."

        elif cluster == 2:
            risk = "🟡 중간 위험"
            description = "진원 깊이가 매우 깊은 지진 그룹입니다."

        else:
            risk = "알 수 없음"
            description = ""

        st.divider()

        st.header("📈 분석 결과")

        st.success(f"예측 군집 : Group {cluster}")

        st.markdown(f"## {risk}")

        st.write(description)

        # --------------------------
        # PCA 시각화
        # --------------------------

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

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.caption(
            "PCA(주성분 분석)를 이용하여 3차원 데이터를 2차원으로 축소하여 시각화했습니다."
        )

        # --------------------------
        # Plotly 세계 지도
        # --------------------------

        st.subheader("🌍 세계 지진 분포 지도")

        map_df = df.dropna(
            subset=["위도", "경도"]
        ).copy()

        map_df["Cluster"] = map_df["cluster"].astype(str)

        fig_map = px.scatter_geo(
            map_df.sample(
                min(1000, len(map_df)),
                random_state=42
            ),
            lat="위도",
            lon="경도",
            color="Cluster",
            hover_data=[
                "규모",
                "진원깊이",
                "영향도"
            ],
            title="세계 지진 데이터 분포"
        )

        fig_map.add_scattergeo(
            lat=[latitude],
            lon=[longitude],
            mode="markers",
            marker=dict(
                size=14,
                symbol="star"
            ),
            name="사용자 위치"
        )

        fig_map.update_layout(
            height=600,
            geo=dict(
                showland=True,
                showcountries=True,
                showocean=True
            )
        )

        st.plotly_chart(
            fig_map,
            width="stretch"
        )

        # --------------------------
        # 군집 설명
        # --------------------------

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

    except Exception as e:

        st.error(f"오류 발생: {e}")
