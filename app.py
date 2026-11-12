import pickle
import numpy as np
import streamlit as st
from Orange.data import Domain, Table

# 페이지 기본 설정
st.set_page_config(
    page_title="교통사고 사망자 예측 및 스쿨존 안전 대시보드",
    page_icon="🚦",
    layout="centered",
)

# 스타일 커스터마이징 (대형 신호등 및 안전 수칙 카드)
st.markdown(
    """
    <style>
    .traffic-box {
        background-color: #1e2129;
        border-radius: 28px;
        padding: 24px 32px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 28px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
        margin: 16px auto;
        max-width: 420px;
        border: 3px solid #323746;
    }
    .light {
        width: 82px;
        height: 82px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 16px;
        color: rgba(255, 255, 255, 0.35);
        background-color: #2c303a;
        transition: all 0.25s ease;
    }
    .light.green.active {
        background-color: #16a34a;
        color: #ffffff;
        box-shadow: 0 0 28px #22c55e, 0 0 55px rgba(34, 197, 94, 0.55);
        transform: scale(1.08);
    }
    .light.yellow.active {
        background-color: #eab308;
        color: #111827;
        box-shadow: 0 0 28px #facc15, 0 0 55px rgba(250, 204, 21, 0.55);
        transform: scale(1.08);
    }
    .light.red.active {
        background-color: #dc2626;
        color: #ffffff;
        box-shadow: 0 0 28px #ef4444, 0 0 55px rgba(239, 68, 68, 0.6);
        transform: scale(1.08);
    }
    .safety-card {
        border-radius: 16px;
        padding: 22px 24px;
        margin-top: 18px;
        border-left: 8px solid;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }
    .card-safe {
        background-color: #f0fdf4;
        border-left-color: #16a34a;
        color: #14532d;
    }
    .card-caution {
        background-color: #fefce8;
        border-left-color: #eab308;
        color: #713f12;
    }
    .card-danger {
        background-color: #fef2f2;
        border-left-color: #dc2626;
        color: #7f1d1d;
    }
    .safety-card h4 {
        margin-top: 0;
        margin-bottom: 10px;
        font-size: 19px;
    }
    .safety-card ul {
        margin-bottom: 0;
        padding-left: 20px;
        line-height: 1.75;
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_orange_model(model_path: str = "traffic_casualty_model.pkcls"):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict_casualties(model, sido: str, accidents: float, minor_injuries: float) -> float:
    """Orange3 모델의 도메인 구조에 맞춰 입력 테이블을 생성하고 예상 사망자 수를 반환합니다."""
    orig_domain = getattr(model, "original_domain", None) or model.domain
    pred_domain = Domain(orig_domain.attributes)

    # 모델의 원래 피처 순서(['시도', '사고건수', '중상자수', '경상자수', '부상신고자수'])에 맞춰 행 구성
    # 미입력 연속형 변수는 np.nan으로 전달하면 Orange3 내장 전처리기가 학습 평균으로 자동 대체함
    row = []
    for attr in pred_domain.attributes:
        if attr.name == "시도":
            row.append(sido)
        elif attr.name == "사고건수":
            row.append(float(accidents))
        elif attr.name == "경상자수":
            row.append(float(minor_injuries))
        else:
            row.append(np.nan)

    input_table = Table.from_list(pred_domain, [row])
    prediction = model(input_table)

    # 신경망 회귀 특성상 극소 음수가 나올 경우 0.0으로 보정
    return max(0.0, float(prediction[0]))


# 1. 모델 로드
try:
    model = load_orange_model("traffic_casualty_model.pkcls")
except FileNotFoundError:
    st.error("❌ `traffic_casualty_model.pkcls` 파일을 찾을 수 없습니다. 같은 폴더에 모델 파일을 업로드해 주세요.")
    st.stop()
except Exception as e:
    st.error(f"❌ 모델 로드 중 오류가 발생했습니다: {e}")
    st.stop()

# 모델에 학습된 시도 목록 추출 (기본: ['경기', '부산', '서울'])
orig_domain = getattr(model, "original_domain", None) or model.domain
sido_options = (
    list(orig_domain["시도"].values)
    if "시도" in orig_domain
    else ["서울", "부산", "경기"]
)
# 보기 편하도록 서울·부산·경기 순으로 정렬