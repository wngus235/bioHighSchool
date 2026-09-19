"""실행: python -m streamlit run app.py"""
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from bioai_core import FEATURE_LABELS, MODEL_NAMES, split_data, compare_models, final_evaluate

st.set_page_config(page_title="BioAI 데이터 실험실", page_icon="🧬", layout="wide")
st.title("BioAI 데이터 실험실")
st.caption("공개 데이터로 질문하고, 비교하고, 설명하는 개인 프로젝트")
st.info("목표값은 1년 후 질병 진행도 지표입니다. 교육용 분석이며 개인 건강 진단에 사용하지 않습니다.")
train, valid, test = split_data()
for key, value in [("history", []), ("result", None), ("final", None)]:
    if key not in st.session_state:
        st.session_state[key] = value
locked = st.session_state.final is not None

with st.sidebar:
    st.header("실험 설정")
    features = st.multiselect("입력 변수", list(FEATURE_LABELS), default=["bmi", "bp", "s5"], disabled=locked)
    depth = st.slider("의사결정나무 최대 깊이", 1, 12, 3, disabled=locked)
    run = st.button("검증 실험 실행", type="primary", disabled=locked or not features)
    if not features:
        st.warning("입력 변수를 한 개 이상 선택하세요.")
    st.caption("나무 깊이는 의사결정나무에만 적용됩니다.")
    st.caption("최종 평가 후에는 이 세션의 실험 설정이 잠깁니다.")

if run:
    scores, preds = compare_models(features, depth)
    st.session_state.result = {"features": list(features), "depth": depth, "scores": scores, "preds": preds}
    for row in scores.to_dict("records"):
        st.session_state.history.append({"실험": len(st.session_state.history)//3 + 1, "변수": ", ".join(features), "깊이": depth, **row})

a, b, c = st.columns(3)
a.metric("학습 사례", len(train))
b.metric("검증 사례", len(valid))
c.metric("최종 평가 사례", len(test))
tabs = st.tabs(["데이터 탐색", "모델 비교", "최종 평가", "나의 발견"])
with tabs[0]:
    st.subheader("학습 데이터에서 관계 찾기")
    st.caption("탐색에는 학습 데이터만 사용합니다. sex는 범주 코드이며 s5는 로그 변환된 값입니다.")
    feature = st.selectbox("그래프로 볼 변수", list(FEATURE_LABELS), index=2,
                           format_func=lambda x: f"{x} · {FEATURE_LABELS[x]}")
    left, right = st.columns(2)
    left.plotly_chart(px.histogram(train, x=feature, nbins=20, title="변수 분포", color_discrete_sequence=["#087F8C"]), use_container_width=True)
    right.plotly_chart(px.scatter(train, x=feature, y="target", title="입력값과 목표값의 관계", color_discrete_sequence=["#2864B4"]), use_container_width=True)
    st.dataframe(train, width="stretch")
    with st.expander("변수 사전"):
        st.dataframe(pd.DataFrame(FEATURE_LABELS.items(), columns=["열", "의미"]), hide_index=True)
        st.write("scaled=False 원자료를 불러오고 선형회귀의 표준화는 학습 데이터에서만 수행합니다. 관계가 보여도 원인이라고 단정하지 않습니다.")

with tabs[1]:
    result = st.session_state.result
    if result is None:
        st.write("왼쪽에서 변수를 선택하고 검증 실험 실행을 누르세요.")
    else:
        st.subheader("검증 결과 비교")
        st.caption(f"실행한 설정: {', '.join(result['features'])} / 나무 깊이 {result['depth']}")
        if features != result["features"] or depth != result["depth"]:
            st.warning("설정이 바뀌었습니다. 현재 표는 이전 실험 결과입니다. 다시 실행해 갱신하세요.")
        st.dataframe(result["scores"].round(3), hide_index=True, width="stretch")
        st.plotly_chart(px.bar(result["scores"], x="모델", y="검증 MAE", color="모델", title="검증 MAE · 낮을수록 좋음"), use_container_width=True)
        name = st.selectbox("예측 결과를 볼 모델", MODEL_NAMES, index=1)
        prediction = result["preds"][name]
        fig = px.scatter(prediction, x="실제값", y="예측값", hover_data=["사례번호", "절대오차"], title="점선에 가까울수록 작은 오차")
        low = min(prediction["실제값"].min(), prediction["예측값"].min())
        high = max(prediction["실제값"].max(), prediction["예측값"].max())
        fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(dash="dash", color="#777777"))
        st.plotly_chart(fig, use_container_width=True)
        st.write("오차가 큰 검증 사례 3개")
        st.dataframe(prediction.sort_values("절대오차", ascending=False).head(3), hide_index=True)
        history = pd.DataFrame(st.session_state.history)
        with st.expander("실험 기록"):
            st.dataframe(history.round(3), hide_index=True)
        st.download_button("실험 기록 CSV 저장", history.to_csv(index=False).encode("utf-8-sig"), "experiment_log.csv", "text/csv")
        st.caption("기록은 현재 접속 세션에만 남습니다. 새로고침 전에 내려받으세요.")

with tabs[2]:
    st.subheader("설정을 확정하고 마지막으로 평가하기")
    st.write("검증 결과로 설정을 결정한 후, 학습·검증 데이터를 합쳐 다시 학습하고 남겨 둔 89개 사례를 평가합니다.")
    result = st.session_state.result
    if st.session_state.final is not None:
        final = st.session_state.final
        st.success(f"평가 완료: {final['모델']} / {', '.join(final['변수'])} / 깊이 {final['나무 깊이']}")
        x, y = st.columns(2)
        x.metric("최종 MAE", f"{final['최종 MAE']:.2f}")
        y.metric("평균값 기준 MAE", f"{final['기준 MAE']:.2f}")
        st.write(f"최종 R²: {final['최종 R2']:.3f} · R²는 음수가 될 수 있으며 정확도 %가 아닙니다.")
        st.dataframe(final["예측표"], hide_index=True)
        st.download_button("최종 예측 CSV 저장", final["예측표"].to_csv(index=False).encode("utf-8-sig"), "final_predictions.csv", "text/csv")
    elif result is None:
        st.write("먼저 검증 실험을 실행하세요.")
    else:
        st.caption(f"마지막 실행 설정 사용: {', '.join(result['features'])} / 깊이 {result['depth']}")
        final_name = st.selectbox("최종 모델", MODEL_NAMES, index=1)
        agree = st.checkbox("검증 결과로 설정을 확정했으며 평가 후 다시 조정하지 않겠습니다.")
        if st.button("최종 평가 실행", disabled=not agree):
            st.session_state.final = final_evaluate(result["features"], final_name, result["depth"])
            st.rerun()
    st.caption("학습용 세션 잠금입니다. 새 접속으로 초기화할 수 있지만 최종 결과를 보고 재조정하면 공정한 최종 평가가 아닙니다.")

with tabs[3]:
    st.subheader("나의 발견 기록")
    question = st.text_input("분석 질문", "세 개의 변수로도 전체 변수와 비슷하게 예측할 수 있을까?")
    hypothesis = st.text_area("실험 전 예상")
    finding = st.text_area("수치로 설명하는 관찰 결과")
    limitation = st.text_area("한계와 다음 질문")
    report = {"분석 질문": question, "실험 전 예상": hypothesis, "관찰 결과": finding, "한계": limitation}
    if st.session_state.final:
        report["최종 평가"] = {k:v for k,v in st.session_state.final.items() if k != "예측표"}
    st.download_button("프로젝트 기록 JSON 저장", json.dumps(report, ensure_ascii=False, indent=2), "my_report.json", "application/json")
    st.caption("제출 전 기록을 다운로드하세요. 입력 내용은 서버 파일에 자동 저장되지 않습니다.")
