"""공통 분석 함수. 학습/검증/최종 평가를 고정하여 비교합니다."""
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

FEATURE_LABELS = {
    "age": "나이", "sex": "성별 코드", "bmi": "체질량지수",
    "bp": "평균 혈압", "s1": "총 혈청 콜레스테롤",
    "s2": "저밀도 지단백", "s3": "고밀도 지단백",
    "s4": "총 콜레스테롤 / HDL", "s5": "혈청 중성지방의 로그값",
    "s6": "혈당 지표",
}
MODEL_NAMES = ["평균값 기준", "선형회귀", "의사결정나무"]

def load_data():
    # 전체 데이터로 미리 표준화하지 않고 학습 단계에서만 표준화합니다.
    return load_diabetes(as_frame=True, scaled=False).frame.copy()

def split_data():
    df = load_data()
    development, test = train_test_split(df, test_size=0.2, random_state=42)
    train, valid = train_test_split(development, test_size=0.25, random_state=42)
    return train.copy(), valid.copy(), test.copy()

def make_model(name, depth=3):
    if name == "평균값 기준":
        return DummyRegressor(strategy="mean")
    if name == "선형회귀":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "의사결정나무":
        return DecisionTreeRegressor(max_depth=depth, random_state=42)
    raise ValueError(f"알 수 없는 모델: {name}")

def check_features(features):
    if not features or len(features) != len(set(features)):
        raise ValueError("입력 변수를 중복 없이 한 개 이상 선택하세요.")
    if not set(features).issubset(FEATURE_LABELS):
        raise ValueError("입력 변수에 목표값 또는 알 수 없는 열이 포함됐습니다.")

def prediction_table(frame, prediction):
    result = frame.copy()
    result.insert(0, "사례번호", result.index)
    result["예측값"] = prediction
    result["절대오차"] = (result["target"] - result["예측값"]).abs()
    return result.rename(columns={"target": "실제값"})

def compare_models(features, depth=3):
    check_features(features)
    train, valid, _ = split_data()
    records, predictions = [], {}
    for name in MODEL_NAMES:
        model = make_model(name, depth)
        model.fit(train[features], train["target"])
        pred = model.predict(valid[features])
        records.append({
            "모델": name,
            "학습 MAE": mean_absolute_error(train["target"], model.predict(train[features])),
            "검증 MAE": mean_absolute_error(valid["target"], pred),
            "검증 R2": r2_score(valid["target"], pred),
        })
        predictions[name] = prediction_table(valid, pred)
    return pd.DataFrame(records), predictions

def final_evaluate(features, model_name, depth=3):
    check_features(features)
    train, valid, test = split_data()
    development = pd.concat([train, valid])
    model = make_model(model_name, depth)
    model.fit(development[features], development["target"])
    pred = model.predict(test[features])
    baseline = make_model("평균값 기준")
    baseline.fit(development[features], development["target"])
    return {
        "모델": model_name, "변수": list(features), "나무 깊이": depth,
        "최종 MAE": float(mean_absolute_error(test["target"], pred)),
        "기준 MAE": float(mean_absolute_error(test["target"], baseline.predict(test[features]))),
        "최종 R2": float(r2_score(test["target"], pred)),
        "예측표": prediction_table(test, pred),
    }
