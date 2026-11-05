# 팀원 가이드

조원이 이 저장소를 받아서 확인하고, 각자 맡은 부분을 작업하는 방법입니다.

## 1. 처음 시작하기

```bash
git clone https://github.com/<계정>/kaist_dfmba8_jeonbanhoney.git
cd kaist_dfmba8_jeonbanhoney
python -m venv .venv
source .venv/bin/activate          # 윈도우: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                             # 테스트 30개가 통과하면 환경이 맞는 것
python -m jeoningu demo            # 합성 데이터로 전체 파이프라인이 도는지 확인
```

`demo` 는 데이터와 API 키 없이 돌아갑니다. 결과는 `demo_outputs/report.md` 에서 볼 수 있습니다(그림 포함).

## 2. 데이터는 git 에 올리지 않습니다

| 데이터 | 이유 | 공유 방법 |
|---|---|---|
| 유튜브 스크립트 (`data/raw/`) | 영상 자막이라 저작권이 있음 | 한 명이 `collect` 로 수집한 뒤 팀 드라이브로 CSV 를 공유 |
| 수업 Case 1 데이터 (주가, FF 팩터) | 수업 자료(라이선스) | 각자 수업에서 받은 파일을 쓰고, `config/local.toml` 에 경로를 적음 |
| API 키 | 비밀 값 | 각자 발급하거나 환경변수로만 사용. 파일로 저장했다면 `.gitignore` 에 걸리는지 확인 |

`.gitignore` 가 `data/`, `outputs/`, `config/local.toml`, 키 파일을 막아 둡니다. 그래도 커밋 전에 `git status` 로 올라갈 파일을 한 번 확인하세요.

`config/local.toml` 예시:

```toml
[paths]
korea_stocks = "/내/경로/Korea_Stocks_2000_2025.csv"
ff_korea     = "/내/경로/FF_Factors_Korea.csv"
```

## 3. 역할 분담 제안

팀 상황에 맞게 바꾸세요. 모듈이 나뉘어 있어서 서로 다른 파일을 건드리게 짤 수 있습니다.

| 역할 | 하는 일 | 주로 보는 곳 |
|---|---|---|
| 수집 | 전체 기간 영상·자막 수집, 게스트 영상 확인 | `collect/`, `text/clean.py` |
| 종목 매칭 | 종목명 사전과 별칭 점검, 오탐 줄이기 | `text/mentions.py`, `config.toml` 의 `mentions` |
| 라벨링 | `label-sample` 로 문장 100개, 2명 이상이 −2~+2 라벨 | `analysis/validation.py` |
| 톤 점수화 | LLM 채점과 사전 점수 비교, 사전 보강 | `text/tone.py`, `text/lexicon_ko.py` |
| 수익률 분석 | 포트폴리오·회귀 결과 확인, 점검, 해석 | `analysis/` |
| 발표 | 결과를 발표 자료에 반영 | `docs/` |

## 4. 작업 흐름

1. 작업마다 브랜치를 만듭니다: `git switch -c feature/<이름>-<작업>` (예: `feature/minsu-label-sample`)
2. 작게 커밋합니다. 메시지는 무엇을 왜 바꿨는지 한 줄로.
3. GitHub 에서 Pull Request 를 열고, 팀원 한 명 이상이 확인한 뒤 `main` 에 합칩니다.
4. 합친 뒤에는 `git switch main && git pull` 로 최신 상태를 받습니다.

`main` 에 바로 push 하지 않는 것을 권합니다. 테스트(`pytest`)가 깨지면 합치지 마세요. GitHub Actions 가 push 와 PR 마다 테스트를 돌립니다.

## 5. 결과를 팀에 공유하기

`outputs/` 는 git 에 올라가지 않습니다. 분석을 돌린 사람이 `outputs/report.md` 와 `outputs/figures/` 를 팀 드라이브에 올리거나, PR 설명에 요약과 그림을 붙여 주세요. 결과가 바뀌면 어떤 설정(`config.toml` 의 값, 톤 방법, 데이터 범위)으로 돌렸는지도 같이 적습니다.

## 6. 바꾸면 안 되는 약속 (사전에 정한 것)

- **판정 규칙**(`analysis/decision.py`)과 `meaningful_alpha` 는 결과를 본 뒤에 바꾸지 않습니다. 바꾸려면 이유를 팀에 먼저 공유하세요.
- **타이밍**은 월 t 의 발언 → 월 t+1 의 수익률입니다. 같은 달 수익률을 쓰는 변경은 미래 정보를 쓰는 것이라 금지입니다.
- 게스트 영상은 표본에서 제외하는 것이 기본입니다.
- 결과가 유의하지 않아도 그대로 보고합니다(지지 / 기각(월 단위) / 판정 불가).
