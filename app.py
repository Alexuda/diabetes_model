import streamlit as st
import pandas as pd
import numpy as np

# 페이지 설정
st.set_page_config(page_title="당뇨병 예측 서비스", layout="centered")

st.title("🏥 당뇨병 발병 예측 모델")
st.markdown("사용자의 건강 지표를 입력하면 당뇨 발생 확률을 분석합니다.")

# 사이드바 또는 메인 화면에 입력창 생성
st.sidebar.header("📝 건강 지표 입력")

def get_user_input():
    preg = st.sidebar.number_input("임신 횟수", min_value=0, max_value=20, value=0, step=1)
    glucose = st.sidebar.number_input("포도당 수치", min_value=0, max_value=300, value=100)
    bp = st.sidebar.number_input("혈압 (mmHg)", min_value=0, max_value=200, value=70)
    skin = st.sidebar.number_input("삼두근 피부 두께 (mm)", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("인슐린 수치", min_value=0.0, max_value=1000.0, value=80.0)
    bmi = st.sidebar.number_input("체질량 지수 (BMI)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.sidebar.number_input("당뇨 내력 가중치", min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input("나이", min_value=1, max_value=120, value=30)
    
    # 데이터프레임 생성
    data = {
        '임신 횟수': preg,
        '포도당': glucose,
        '혈압': bp,
        '삼두근 피부 두께': skin,
        '인슐린': insulin,
        '체질량 지수': bmi,
        '당뇨 내력 가중치': dpf,
        '나이': age
    }
    return pd.DataFrame([data])

# 입력 데이터 받기
input_data = get_user_input()

# 파생 변수 계산
# 0으로 나누기 방지 처리 (인슐린이 0일 경우 대비)
input_data['포도당_인슐린_비율'] = input_data['포도당'] / (input_data['인슐린'] + 0.001)
input_data['신체위험지수'] = input_data['혈압'] + input_data['체질량 지수']
input_data['고령'] = (input_data['나이'] >= 50).astype(int)

# 데이터 확인용 (선택 사항)
st.subheader("📊 입력된 데이터 확인")
st.write(input_data)

# 분석 실행 버튼
if st.button("결과 분석하기"):
    # 주의: scaler와 log_model_eng는 사전에 학습된 객체가 로드되어 있어야 합니다.
    # 예: scaler = joblib.load('scaler.pkl')
    try:
        # 데이터 표준화
        input_scaled = scaler.transform(input_data)
        
        # 예측 및 확률 계산
        predicted = log_model_eng.predict(input_scaled)
        prob = log_model_eng.predict_proba(input_scaled)
        
        st.divider()
        
        # 결과 출력
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="분석 결과", value="당뇨 위험" if predicted[0] == 1 else "정상")
            
        with col2:
            st.metric(label="당뇨 확률", value=f"{prob[0][1]*100:.1f}%")
            
        if predicted[0] == 1:
            st.error("⚠️ 당뇨병 고위험군으로 분류되었습니다. 전문의와의 상담을 권장합니다.")
        else:
            st.success("✅ 현재 입력 데이터상으로는 정상 범위입니다.")
            
    except NameError:
        st.warning("모델 파일(`log_model_eng`)과 스케일러(`scaler`)가 로드되지 않았습니다. 모델 학습 코드를 먼저 실행해주세요.")