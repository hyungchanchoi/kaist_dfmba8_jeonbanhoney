# kaist_dfmba8_jeonbanhoney

전인구경제연구소 유튜브 발언의 **톤**이 이후 **초과수익률**과 어떤 관계인지 검증하는 프로젝트입니다.
이른바 '전반꿀'("전인구 반대로 하면 꿀") 밈이 데이터로 맞는지 보는 KAIST BAF.50022 Financial Data Analytics 기말 프로젝트입니다.

> 연구용 프로젝트이며 투자 조언이 아닙니다. 공개 영상의 발언 텍스트를 분석하며, 특정인을 평가하거나 비방하려는 목적이 아닙니다.

## 연구 질문과 가설

> 전인구가 긍정적으로 말한 종목·시장은 이후 초과수익률이 낮은가?

| 가설 | 내용 |
|---|---|
| H0 | 발언 톤과 이후 수익률은 관계가 없다 (알파 = 0) |
| H1 (종목) | 긍정−부정 포트폴리오의 CAPM 알파가 0보다 작다 |
| H2 (시장) | 월별 시장 톤이 높을수록 다음 달 코스피 초과수익률이 낮다 |

판정 규칙(결과를 보기 전에 정해 둠): **지지**(알파가 유의한 음수이고 사전 추세로 설명 안 됨) / **기각(월 단위)**(신뢰구간이 0을 포함하고 의미 있는 크기의 음의 효과는 배제) / **판정 불가**(신뢰구간이 너무 넓음).
자세한 설계는 [docs/design.md](docs/design.md), 발표 자료는 [docs/project_proposal.pptx](docs/project_proposal.pptx) 입니다.
**팀원은 [CONTRIBUTING.md](CONTRIBUTING.md) 를 먼저 보세요**(시작하는 법, 데이터 공유 방법, 역할 분담, 작업 흐름).
최종 발표까지의 역할과 일정은 [docs/project_plan.md](docs/project_plan.md) 에 있습니다.

## 파이프라인

```
collect ──► preprocess ──► mentions ──► tone ──► analyze ──► report
 (영상·자막)   (정리·문장 분리)  (종목·시장 언급)  (톤 점수화)  (H1·H2·점검)   (표·그림·report.md)
                                                    ▲
prices (Yahoo Finance 월 수익률) ────────────────────┘
```

| 단계 | 하는 일 | 수업 근거 | 코드 |
|---|---|---|---|
| collect | YouTube Data API 로 채널 영상·메타데이터, youtube-transcript-api 로 자막 수집(이어받기, 차단 대응) | Case 3 (수집·파싱) | `collect/youtube.py` |
| prices | Yahoo Finance(yfinance)에서 종목과 코스피 지수(^KS11)의 일별 수정종가를 받아 월 수익률 CSV 로 저장 | | `returns/data.py` |
| preprocess | 자막 태그 제거, 문장 분리, 게스트 영상 표시, KST 월 | week 6 전처리, Case 3 regex | `text/clean.py` |
| mentions | `config/universe.csv` 의 종목명·코스피가 든 문장 추출, 전망 문장 표시 | Case 3 (regex 문장 추출) | `text/mentions.py` |
| tone | 사전 점수 / LLM 채점 / (번역본) VADER·FinBERT | Case 4, week 9, sentiment 실습 | `text/tone.py` |
| analyze | 톤 상위·하위 포트폴리오, H−L CAPM 알파, 시장 회귀, 사전 추세 점검, 검정력 | Case 1 (포트폴리오 정렬, 알파) | `analysis/` |
| report | `outputs/` 에 표, 그림, `report.md` | | `report.py` |

타이밍 규칙: **월 t 의 발언 톤 → 월 t+1 의 초과수익률**(미래 정보를 쓰지 않음). 게스트 대담 영상은 분석에서 제외하고, 톤은 전망 문장만으로 계산합니다.

## 빠른 시작

```bash
git clone <이 저장소 주소>
cd kaist_dfmba8_jeonbanhoney
python -m venv .venv && source .venv/bin/activate     # 윈도우: .venv\Scripts\activate
pip install -e ".[dev]"                                # LLM 채점까지 쓰려면 ".[dev,llm]"
```

### 0) 합성 데이터로 전체 점검 (데이터·키 필요 없음)

```bash
python -m jeoningu demo --planted-alpha -0.03        # 심어 둔 효과를 찾는지
python -m jeoningu demo --seed 2 --out demo_outputs/null   # 효과가 없으면 찾지 않는지
pytest                                               # 단위 테스트 + 종단 테스트
```

### 1) 설정

수익률은 Yahoo Finance 에서 받으므로 따로 데이터 파일을 준비할 필요가 없습니다. 두 가지만 확인하세요.

1. **분석할 종목 목록**: [config/universe.csv](config/universe.csv) (Yahoo 티커, 종목명, 별칭). 여기에 없는 종목은 분석에서 빠집니다.
   영상에 자주 나오는데 빠진 종목이 있으면 한 줄씩 추가하세요(예: `005930.KS,삼성전자,삼전`). 코스피는 `.KS`, 코스닥은 `.KQ` 입니다.
2. **무위험수익률**: Yahoo 에 없습니다. 기본은 `config.toml` 의 `returns.rf_annual`(연 0%) 상수입니다. 한국 단기금리 월별 CSV(열 `ym`, `rf`)가 있으면
   `config/local.toml` 의 `paths.rf_monthly` 에 경로를 적으세요(이 파일은 git 에 올라가지 않음). H-L 알파에는 거의 영향이 없고 시장 초과수익률에만 영향을 줍니다.

비밀 값은 환경변수로 줍니다(`.env.example` 참고): `YOUTUBE_API_KEY`, 그리고 LLM 채점을 쓸 때 `ANTHROPIC_API_KEY` 또는 `OPENAI_API_KEY`.

### 2) 실행

```bash
python -m jeoningu collect --max-videos 30   # 먼저 소량으로 시험. 이후 옵션 없이 전체 기간 수집
python -m jeoningu prices                    # Yahoo Finance 월 수익률(인터넷 필요). 월 수익률이 40% 를 넘는 달은 경고 출력
python -m jeoningu preprocess
python -m jeoningu mentions
python -m jeoningu tone --method dictionary  # 또는 llm (config 의 tone.llm_model 설정 필요)
python -m jeoningu analyze                   # outputs/report.md 생성
# 한 번에: python -m jeoningu all
```

LLM 채점은 `config/config.toml` 의 `tone.llm_provider`, `tone.llm_model` 을 채운 뒤 `--method llm` 으로 실행합니다.
같은 문장은 `data/interim/llm_cache.jsonl` 에 캐시되어 다시 요청하지 않습니다.

### 3) 타당성 검증

```bash
python -m jeoningu label-sample --n 100      # data/labels/human_labels.csv 생성
# 팀원 2명 이상이 label_a, label_b 열에 -2~+2 를 채움
python -m jeoningu validate                  # 라벨러 간 일치도(kappa)와 모델-사람 상관
```

## 폴더 구조

```
config/        config.toml(기본), universe.csv(분석 종목), local.toml(개인 설정, git 제외)
data/          raw / interim / processed / labels  (git 제외: 저작권 스크립트, 받아 온 수익률)
docs/          설계 문서, 발표 자료
src/jeoningu/  collect/ text/ returns/ analysis/ report.py pipeline.py cli.py demo.py
tests/         단위 테스트와 합성 데이터 종단 테스트
outputs/       분석 결과(git 제외)
```

## 데이터와 저작권

- 스크립트(`data/raw`)는 유튜브 영상의 자막이라 **저장소에 올리지 않습니다.** 수집 코드만 공유합니다.
- 주가는 Yahoo Finance(yfinance, 비공식 API)에서 받습니다. 받아 온 CSV 는 `data/raw` 에 두고 **저장소에 올리지 않습니다.** 야후의 이용 약관은 개인·교육 목적 사용을 전제로 하니 재배포하지 마세요.
  각자 `prices` 명령으로 다시 받을 수 있지만, 야후 데이터는 시점에 따라 조금씩 달라질 수 있어서 결과를 공유할 때 받은 날짜를 적어 두세요.
- API 키 파일(`api.txt` 등)은 `.gitignore` 로 막아 두었지만, 커밋 전에 `git status` 로 한 번 더 확인하세요.

## 한계 (결과를 읽을 때)

1. **월 단위 분석**: 월 t 발언 → 월 t+1 수익률로 설계해서 영상 직후 며칠의 반응은 못 봅니다. Yahoo 에서 일별 가격을 받으므로 일별로 넓히는 것은 가능하지만 현재 코드는 월 단위입니다. 해외 연구(Engelberg, Sasseville & Williams 2012)는 야간 급등과 이후 반전을 보고합니다.
2. **언급 종목이 적음**: 30편 파일럿에서 단독 영상의 종목은 사실상 삼성전자·SK하이닉스였습니다. 그래서 시장 수준(H2) 분석을 병행합니다.
3. **검정력**: 매달 5 vs 5 종목이라면 월 2% 안팎보다 작은 알파는 구분하기 어렵습니다(`analysis/power.py`). '유의하지 않음'과 '효과 없음'을 구분해 보고합니다.
4. **한국어 톤 사전**은 검증 전 파일럿 버전이고, 전망이 아니라 '일어난 일'의 단어를 세어 톤을 놓칠 수 있습니다. LLM 채점과 사람 라벨로 확인하세요.
5. **LLM 의 사후 지식**: 모델이 이후 시장 결과를 알고 있을 수 있어 프롬프트에서 모른다고 가정하게 하지만 완전히 막지는 못합니다.
6. 같은 종목이 여러 달 이어서 언급되면 관측치가 독립이 아닙니다(HAC 표준오차로 일부만 보정).
7. **Yahoo Finance 데이터의 한계**
   - **생존 편향**: 상장폐지된 종목은 거의 없습니다. 종목 목록(`universe.csv`)도 현재 상장된 종목 중심입니다.
   - **액면분할·분할상장 미반영**: 수업 데이터와 비교했을 때(43개 종목, 3,428 종목-월) 97% 는 0.5%p 이내로 일치했지만, 에코프로 2021년 5월(-45.7% vs -5.5%), SK텔레콤 2021년 11월(-21.8% vs +2.1%)처럼 기업 행위가 있던 달에 큰 차이가 났습니다. `prices` 가 월 수익률 40% 초과 달을 경고하지만 그보다 작은 오류는 잡지 못합니다.
   - **시장 대용**: 코스피 지수(^KS11)는 배당이 빠진 가격 지수입니다(수업의 시장 팩터와 월 수익률 상관 0.98).
   - **무위험수익률 없음**: 상수 또는 직접 만든 CSV 를 써야 합니다.
   - **크기 정보 없음**: 시가총액 대신 월 거래대금을 크기 대용으로 씁니다(검정력 계산에만 사용).

## 라이선스

아직 정하지 않았습니다. 공개 저장소로 올리기 전에 `LICENSE` 를 추가하세요.
