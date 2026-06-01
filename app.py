import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. 파일 경로 동적 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'lung_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# 2. 모델 및 스케일러 로드 함수
@st.cache_resource
def load_ml_components():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"파일을 불러오는 중 오류가 발생했습니다: {e}")
        return None, None

model, scaler = load_ml_components()

# 3. UI 구성 (사이드바 - 건강 지표 입력)
st.sidebar.header("📋 건강 지표 입력")

pregnancy = st.sidebar.number_input("임신 횟수", min_value=0, max_value=20, value=2, step=1)
glucose = st.sidebar.number_input("포도당 수치", min_value=0, max_value=300, value=100, step=1)
blood_pressure = st.sidebar.number_input("혈압 (mmHg)", min_value=0, max_value=200, value=70, step=1)
skin_thickness = st.sidebar.number_input("삼두근 피부 두께 (mm)", min_value=0, max_value=100, value=20, step=1)
insulin = st.sidebar.number_input("인슐린 수치", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
bmi = st.sidebar.number_input("체질량 지수 (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
dpf = st.sidebar.number_input("당뇨 내력 가중치", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.sidebar.number_input("나이", min_value=0, max_value=120, value=30, step=1)

# 💡 [보완] 총 11개 변수를 맞추기 위한 파생 변수 계산 생성 (가장 흔히 쓰이는 조합 3가지)
glucose_insulin = glucose * insulin  # 9번째 변수
bmi_age = bmi * age                  # 10번째 변수
glucose_age = glucose * age          # 11번째 변수

# 4. 메인 화면 구성
st.title("🏥 당뇨병 발병 예측 모델")
st.write("사용자의 건강 지표를 입력하면 당뇨 발생 확률을 분석합니다.")

st.subheader("📊 입력된 데이터 확인")

# 입력 데이터를 데이터프레임으로 변환 (총 11개 컬럼)
input_data = pd.DataFrame([{
    '임신 횟수': pregnancy,
    '포도당': glucose,
    '혈압': blood_pressure,
    '삼두근 피부 두께': skin_thickness,
    '인슐린': insulin,
    '체질량 지수': bmi,
    '당뇨 내력 가중치': dpf,
    '나이': age,
    '포도당_인슐린': glucose_insulin,
    'BMI_나이': bmi_age,          # 학습 당시 컬럼명과 다르면 수정이 필요할 수 있습니다.
    '포도당_나이': glucose_age       # 학습 당시 컬럼명과 다르면 수정이 필요할 수 있습니다.
}])

st.dataframe(input_data)

# 5. 예측 실행 버튼 클릭 이벤트
if st.button("결과 분석하기"):
    if model is None or scaler is None:
        st.error("⚠️ 모델 파일(lung_model.pkl)과 스케일러(scaler.pkl)가 로드되지 않았습니다.")
    else:
        try:
            # 11개의 피처가 스케일러와 모델에 그대로 들어갑니다.
            scaled_data = scaler.transform(input_data.values)
            
            prediction = model.predict(scaled_data)
            prediction_proba = model.predict_proba(scaled_data)
            
            st.markdown("---")
            st.subheader("💡 분석 결과")
            
            diabetes_prob = prediction_proba[0][1] * 100
            
            if prediction[0] == 1:
                st.error(f"분석 결과 **당뇨 발병 위험군**에 속합니다. (확률: {diabetes_prob:.2f}%)")
            else:
                st.success(f"분석 결과 **정상** 범주입니다. (당뇨 확률: {diabetes_prob:.2f}%)")
                
        except Exception as e:
            st.error(f"예측을 수행하는 동안 오류가 발생했습니다: {e}")
