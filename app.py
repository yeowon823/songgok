import pickle
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="교통사고 사망자 예측 및 스쿨존 안전 대시보드",
    page_icon="🚦",
    layout="centered",
)

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
        features.append({"name": feat_name, "offset": offset, "factor": factor})

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
            scaled_row.append(0.0)

    h = np.array([scaled_row], dtype=np.float64)
    coefs = bundle["coefs"]
    intercepts = bundle["intercepts"]
    activation = bundle["activation"]

    for W, b in zip(coefs[:-1], intercepts[:-1]):
        h = _apply_activation(h @ W + b, activation)

    out = h @ coefs[-1] + intercepts[-1]
    return max(0.0, float(out.ravel()[0]))


try:
    model_bundle = load_orange_model("traffic_casualty_model.pkcls")
except FileNotFoundError:
    st.error("❌ `traffic_casualty_model.pkcls` 파일을 찾을 수 없습니다. 같은 폴더에 업로드해 주세요.")
    st.stop()
except Exception as e:
    st.error(f"❌ 모델 로드 중 오류가 발생했습니다: {e}")
    st.stop()

extracted_sidos = [
    f["name"].split("=", 1)[1]
    for f in model_bundle["features"]
    if f["name"].startswith("시도=")
]
if not extracted_sidos:
    extracted_sidos = ["서울", "부산", "경기"]

preferred_order = [s for s in ["서울", "부산", "경기"] if s in extracted_sidos]
sido_options = preferred_order + [s for s in extracted_sidos if s not in preferred_order]

st.title("🚦 교통사고 사망자 실시간 예측 & 스쿨존 안전 알림")
st.caption("Orange3 신경망 예측 모델(`traffic_casualty_model.pkcls`) 기반 실시간 위험도 분석")

st.divider()

st.subheader("1️⃣ 지역 및 사고 발생 조건 입력")

col1, col2 = st.columns([1, 2])

with col1:
    selected_sido = st.selectbox(
        "📍 시도 선택",
        options=sido_options,
        index=0,
        help="모델에 학습된 시·도 지역을 선택합니다.",
    )

with col2:
    accident_count = st.slider(
        "🚗 사고건수 (건)",
        min_value=0,
        max_value=3000,
        value=350,
        step=10,
    )
    minor_injury_count = st.slider(
        "🩹 경상자수 (명)",
        min_value=0,
        max_value=4000,
        value=450,
        step=10,
    )

predicted_deaths = predict_casualties(
    bundle=model_bundle,
    sido=selected_sido,
    accidents=accident_count,
    minor_injuries=minor_injury_count,
)

st.divider()

st.subheader("2️⃣ 실시간 예측 결과 및 위험도 신호등")

if predicted_deaths < 2.5:
    status_key = "safe"
    status_emoji = "🟢"
    status_label = "안전"
    status_desc = "예상 사망자 수가 낮은 수준입니다. 기본 안전 수칙을 준수해 주세요."
    card_class = "card-safe"
    card_title = "🟢 [안전 단계] 어린이 보호구역 기본 안전 수칙"
    safety_rules = [
        "<b>시속 30km 이하 서행:</b> 스쿨존 진입 시 제한속도를 준수하고 전방을 주시하세요.",
        "<b>횡단보도 앞 일시정지:</b> 보행자가 없더라도 횡단보도 진입 전 속도를 줄여 확인하세요.",
        "<b>불법 주·정차 절대 금지:</b> 어린이 시야를 가리는 주·정차는 사고의 주요 원인입니다.",
        "<b>등·하교 시간대 집중 주의:</b> 08:00~09:00, 13:00~16:00에는 골목길 돌발 보행에 대비하세요.",
    ]
elif predicted_deaths < 6.0:
    status_key = "caution"
    status_emoji = "🟡"
    status_label = "주의"
    status_desc = "사고 규모 대비 사망 사고 위험이 증가하는 구간입니다. 각별한 감속 운전이 필요합니다."
    card_class = "card-caution"
    card_title = "🟡 [주의 단계] 어린이 보호구역 강화 안전 수칙"
    safety_rules = [
        "<b>신호등 없는 횡단보도 무조건 일시정지:</b> 보행자 횡단 여부와 관계없이 정지선 앞에 반드시 멈추세요.",
        "<b>교차로 우회전 시 완전 정지 후 서행:</b> 사각지대에 키가 작은 어린이가 있는지 재차 확인하세요.",
        "<b>급가속·급제동 및 추월 금지:</b> 보호구역 내 앞지르기는 치명적인 인명 피해로 이어집니다.",
        "<b>전조등 켜기 및 경적 자제:</b> 차량 접근을 시각적으로 알리되 어린이가 놀라지 않도록 경적은 자제하세요.",
    ]
else:
    status_key = "danger"
    status_emoji = "🔴"
    status_label = "위험"
    status_desc = "고위험 사고 구간으로 예측되었습니다. 스쿨존 및 보행자 밀집 구역 통행에 최고 수준의 주의가 요구됩니다."
    card_class = "card-danger"
    card_title = "🔴 [위험 단계] 어린이 보호구역 특별 경보 수칙"
    safety_rules = [
        "<b>스쿨존 우회 통행 권고:</b> 불필요한 보호구역 내 진입을 자제하고 간선도로로 우회하세요.",
        "<b>시속 20km 이하 초저속 주행:</b> 돌발 상황에 즉시 정지할 수 있도록 브레이크에 발을 올린 채 서행하세요.",
        "<b>통학버스 주변 절대 정지:</b> 어린이 통학버스가 정차해 점멸등이 켜지면 차선을 불문하고 일시정지하세요.",
        "<b>스마트폰·내비게이션 조작 금지:</b> 운전 중 시선 분산을 완전히 차단하고 보행자 움직임에만 집중하세요.",
    ]

m_col1, m_col2, m_col3 = st.columns(3)
m_col1.metric("선택 지역", selected_sido)
m_col2.metric("예상 사망자 수", f"{predicted_deaths:.2f} 명")
m_col3.metric("위험도 등급", f"{status_emoji} {status_label}")

green_cls = "active" if status_key == "safe" else ""
yellow_cls = "active" if status_key == "caution" else ""
red_cls = "active" if status_key == "danger" else ""

st.markdown(
    f"""
    <div class="traffic-box">
        <div class="light green {green_cls}">🟢 안전</div>
        <div class="light yellow {yellow_cls}">🟡 주의</div>
        <div class="light red {red_cls}">🔴 위험</div>
    </div>
    <p style="text-align: center; color: #6b7280; font-size: 14px; margin-top: 6px;">
        {status_desc}
    </p>
    """,
    unsafe_allow_html=True,
)

rules_html = "".join([f"<li>{rule}</li>" for rule in safety_rules])
st.markdown(
    f"""
    <div class="safety-card {card_class}">
        <h4>{card_title}</h4>
        <ul>{rules_html}</ul>
    </div>
    """,
    unsafe_allow_html=True,
)
