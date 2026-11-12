import pickle
import numpy as np
import streamlit as st
from sklearn.neural_network import MLPRegressor

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


# Orange3 라이브러리 없이 .pkcls 파일을 직접 로드하기 위한 커스텀 Unpickler
class _DummyOrangeObject:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


def _dummy_make_variable(*args, **kwargs):
    return _DummyOrangeObject(*args, **kwargs)


class OrangeSafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Orange3 신경망 래퍼를 scikit-learn MLPRegressor로 직접 매핑
        if module == "Orange.regression.neural_network" and name == "MLPRegressorWCallback":
            return MLPRegressor
        if module == "Orange.data.variable" and name == "make_variable":
            return _dummy_make_variable
        if module.startswith("Orange"):
            return type(name, (_DummyOrangeObject,), {})
        # NumPy 1.x / 2.x 간 피클 경로 호환 처리
        if module.startswith("numpy._core"):
            try:
                return super().find_class(module, name)
            except ModuleNotFoundError:
                return super().find_class(
                    module.replace("numpy._core", "numpy.core", 1), name
                )
        return super().find_class(module, name)


@st.cache_resource
def load_orange_model(model_path: str = "traffic_casualty_model.pkcls"):
    with open(model_path, "rb") as f:
        raw_model = OrangeSafeUnpickler(f).load()

    skl_model = raw_model.skl_model
    attributes = getattr(raw_model.domain, "attributes", ())

    features = []
    for attr in attributes:
        feat_name = getattr(attr, "_name", "")
        normalizer = getattr(attr, "_compute_value", None)
        offset = (
            float(np.asarray(normalizer.offset).ravel()[0])
            if normalizer is not None and hasattr(normalizer, "offset")
            else 0.0
        )
        factor = (
            float(np.asarray(normalizer.factor).ravel()[0])
            if normalizer is not None and hasattr(normalizer, "factor")
            else 1.0
        )
        features.append(
            {
                "name": feat_name,
                "offset": offset,
                "factor": factor,
            }
        )

    return {"skl_model": skl_model, "features": features}


def predict_casualties(bundle: dict, sido: str, accidents: float, minor_injuries: float) -> float:
    """Orange3 전처리(원-핫 인코딩 + 평균 대체 + 표준화)를 동일하게 적용해 예상 사망자 수를 계산합니다."""
    scaled_row = []

    for feat in bundle["features"]:
        name = feat["name"]
        offset = feat["offset"]
        factor = feat["factor"]

        if name.startswith("시도="):
            target_sido = name.split
