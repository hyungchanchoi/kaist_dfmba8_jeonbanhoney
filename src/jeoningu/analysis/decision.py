"""판정 규칙 (발표 10번 슬라이드). 결과를 보기 전에 정해 둔 기준입니다.

  지지        : 알파가 음수이고 5% 에서 유의하며, 사전 추세로는 설명되지 않음
  기각(월 단위): 신뢰구간이 0 을 포함하고, '의미 있는 크기'(meaningful)의 음의 효과는 배제됨
  판정 불가   : 신뢰구간이 너무 넓어 위 둘 다 말할 수 없음
  (그 밖에) 유의하게 양수이면 '반대 방향', 유의한 음수지만 사전 추세가 크면 '지지 보류'

'기각' 은 월 단위 분석에서만 말할 수 있습니다. 영상 직후 며칠의 반응은 이 데이터로는 알 수 없습니다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    label: str
    reason: str


def decide(alpha: float, ci_low: float, ci_high: float, p: float, n_months: int,
           meaningful: float = 0.02, min_months: int = 24, pretrend_p: float | None = None,
           pretrend_alpha: float = 0.05) -> Decision:
    if n_months < min_months:
        return Decision("판정 불가", f"분석 가능한 달이 {n_months}개로 기준({min_months}개)보다 적음")
    pretrend_bad = pretrend_p is not None and pretrend_p < pretrend_alpha
    if alpha < 0 and p < 0.05:
        if pretrend_bad:
            return Decision("지지 보류(사전 추세)", "알파는 유의한 음수지만 발언 전 달에도 H−L 이 유의함")
        return Decision("지지", f"추정치 {alpha:.4f} (p={p:.3f}) 가 유의한 음수이고 사전 추세로 설명되지 않음")
    if alpha > 0 and p < 0.05:
        return Decision("반대 방향", f"추정치 {alpha:.4f} (p={p:.3f}) 가 유의한 양수: 긍정 톤이 이후 더 높음")
    if ci_low > -meaningful:
        return Decision(
            "기각(월 단위)",
            f"신뢰구간 [{ci_low:.4f}, {ci_high:.4f}] 이 0 을 포함하고 월 {meaningful:.1%} 이하의 음의 효과는 배제")
    return Decision("판정 불가", f"신뢰구간 [{ci_low:.4f}, {ci_high:.4f}] 이 너무 넓어 효과가 없다고도 있다고도 말할 수 없음")
