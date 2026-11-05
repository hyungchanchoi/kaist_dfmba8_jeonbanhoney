"""월 표기 변환 도구. 수익률 데이터는 정수 YYYYMM(예: 201810), 텍스트 쪽은 'YYYY-MM' 문자열을 씁니다."""
from __future__ import annotations


def ym_add(ym: int, k: int) -> int:
    """YYYYMM 정수에 k 개월을 더합니다. 예: ym_add(201812, 1) == 201901."""
    y, m = divmod(int(ym), 100)
    idx = y * 12 + (m - 1) + k
    return (idx // 12) * 100 + (idx % 12) + 1


def month_str_to_ym(s: str) -> int:
    """'2018-10' → 201810."""
    y, m = s.split("-")
    return int(y) * 100 + int(m)


def ym_to_month_str(ym: int) -> str:
    """201810 → '2018-10'."""
    y, m = divmod(int(ym), 100)
    return f"{y:04d}-{m:02d}"


def ym_range(start_ym: int, end_ym: int) -> list[int]:
    out = []
    cur = start_ym
    while cur <= end_ym:
        out.append(cur)
        cur = ym_add(cur, 1)
    return out
