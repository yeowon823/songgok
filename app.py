import pickle
import numpy as np
import streamlit as st

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


# Orange3 및 scikit-learn 설치 없이 .pkcls 가중치를 추출하기 위한 순수 파이썬 Unpickler
class _DummyObject:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


def _dummy_make_variable(*args, **kwargs):
    return _DummyObject(*args, **kwargs)


class PureNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "Orange.data.variable" and name == "make_variable":
            return _dummy_make_variable
        if module.startswith(("Orange", "sklearn")):
            return type(name, (_DummyObject,), {})
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
        raw_model = PureNumpyUnpickler(f).load()

    skl_model = raw_model.skl_model
    coefs = [np.asarray(w, dtype=np.float64) for w in skl_model.coefs_]
    intercepts = [np.asarray(b, dtype=np.float64) for b in skl_model.intercepts_]
    activation = getattr(skl_model, "activation", "relu")

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

    return {
        "coefs": coefs,
        "intercepts": intercepts,
        "activation": activation,
        "features": features,
    }


def _apply_activation(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "relu":
        return np.maximum(0.0, x)
    if activation == "tanh":
        return np.tanh(x)
    if activation == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return x


def predict_casualties(bundle: dict, sido: str, accidents: float, minor_injuries: float) -> float:
    """Orange3 전처리 및 다층 퍼셉트론(MLP) 순전파를 NumPy로 직접 계산합니다."""
    scaled_row = []

    for feat in bundle["features"]:
        name = feat["name"]
        offset = feat["offset"]
        factor = feat["factor"]

        if name.startswith("시도="):
            target_sido = name.split("=", 1)[1]
            raw_val = 1.0 if sido == target_sido else 0.0
            scaled_row.append((raw_val - offset) * factor)
        elif name == "사고건수":
            scaled_row.append((float(accidents) - offset) * factor)
        elif name == "경상자수":
            scaled_row.append((float(minor_injuries) - offset) * factor)
        else:
            # 미입력 변수(중상자
