# BioAI 데이터 분석 개인 프로젝트

공개 바이오 데이터를 파이썬으로 탐색하고 회귀 모델을 비교한 뒤 Streamlit 앱으로 공유합니다.
대상: 파이썬 변수와 함수의 기본 개념을 아는 입문자. 수업: 휴식 포함 240분.

## 빠른 실행

Python 3.12를 설치하고 VS Code에서 이 project 폴더를 엽니다.
터미널의 현재 폴더에 app.py와 requirements.txt가 보여야 합니다.

```bash
python -m venv .venv
```

Windows 명령 프롬프트에서는 `.venv\Scripts\activate`,
macOS/Linux에서는 `source .venv/bin/activate`를 실행합니다.
Windows PowerShell에서 활성화가 차단되면 보안 설정을 바꾸지 않고
VS Code 터미널을 명령 프롬프트로 변경합니다.

```bash
python -m pip install -r requirements.txt
python 01_explore.py
python 02_compare.py
python -m streamlit run 03_streamlit_basic.py
```

기본 앱을 종료하려면 터미널에서 Ctrl+C를 누릅니다.

```bash
python -m streamlit run app.py
```

설치 후 내장 데이터 분석은 인터넷 없이 가능합니다. 웹 배포에는 인터넷과 계정이 필요합니다.

## 파일 안내

| 파일 | 역할 |
|---|---|
| bioai_core.py | 데이터 읽기, 고정 분할, 모델 생성, 검증 및 최종 평가 |
| 01_explore.py | 학습 데이터 탐색과 HTML 그래프 2개 저장 |
| 02_compare.py | 세 모델 검증 성능 비교, CSV 저장 |
| 03_streamlit_basic.py | 첫 번째 Streamlit 화면 |
| app.py | 완성 앱, 검증 기록 및 최종 평가 |
| exercises/ | 빈칸 코딩 실습과 정답 |
| requirements.txt | 실습에 사용한 직접 의존성 버전 |
| DEPLOY.md | GitHub 업로드와 배포 안내 |
| DATA.md | 데이터 출처와 변수 설명 |
| outputs/ | 실행하면 자동 생성되는 결과 폴더 |

## 데이터와 실험 원칙

scikit-learn의 diabetes 데이터 442개 사례, 10개 입력 변수를 사용합니다.
당뇨병 유무 분류가 아니라 1년 후 질병 진행도 수치 예측입니다.
`scaled=False`로 불러와 학습 데이터에만 표준화기를 적합합니다.
학습 264개, 검증 89개, 최종 평가 89개로 고정합니다.
탐색은 학습 데이터, 설정 비교는 검증 데이터, 마지막 평가는 최종 평가 데이터를 사용합니다.
선형회귀와 의사결정나무를 평균값 기준 모델과 비교합니다.
MAE는 예측값과 실제값 차이의 절댓값 평균이며 작을수록 좋습니다.
R²는 정확도 %가 아니며 음수가 될 수 있습니다.

## 개인 프로젝트 작성 틀

- 프로젝트 이름:
- 분석 질문:
- 실험 전 예상:
- 실험 1의 변수와 검증 MAE:
- 실험 2의 변수와 검증 MAE:
- 최종 선택과 이유:
- 최종 MAE와 기준 모델 MAE:
- 질문에 대한 답:
- 분석의 한계:
- 앱 URL:

제출물: 저장소 링크, 앱 링크 또는 로컬 실행 화면, 그래프 2개,
experiment_log.csv, final_predictions.csv, my_report.json, 활동지.
앱의 입력과 기록은 현재 세션에만 남으므로 다운로드해 보관하세요.
최종 평가 잠금은 수업 진행을 돕는 장치이며 강제 시험 보안 기능이 아닙니다.

## 출처

- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
- https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository

이 프로젝트의 예측은 교육용입니다. 실제 개인의 건강 진단이나 치료 결정에 사용하지 않습니다.
