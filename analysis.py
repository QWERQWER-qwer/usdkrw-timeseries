"""
analysis.py
원/달러(USD/KRW) 2022~2025 시계열 분석 스크립트.

실행:  python analysis.py
전제:  data/usdkrw_2022_2025.csv 가 있어야 한다. 없으면 먼저 `python make_dataset.py` 실행.
출력:  images/ 에 그래프 png 6개, 콘솔에 핵심 수치, stats.json 에 리포트용 수치.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.seasonal import seasonal_decompose

# ── 한글 폰트 설정 ────────────────────────────────────────────────
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
try:
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False   # 마이너스 부호 깨짐 방지
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# 색상(브랜드 중립, 명도 대비 확보)
C_LINE = "#2b6cb0"   # 종가
C_MA20 = "#dd6b20"   # 20일선
C_MA60 = "#718096"   # 60일선
C_POS  = "#c53030"   # 상승(원화 약세)
C_NEG  = "#2f855a"   # 하락(원화 강세)


def won(x, _=None):
    return f"{x:,.0f}"


# ── 1) 데이터 로드 & 기본 정보 ────────────────────────────────────
df = pd.read_csv("data/usdkrw_2022_2025.csv", parse_dates=["Date"], index_col="Date")
df = df.sort_index()

print("=" * 60)
print("[1] 기본 정보")
print(f"  기간      : {df.index.min().date()} ~ {df.index.max().date()}")
print(f"  데이터 수 : {len(df)} 개(거래일)")
print(f"  컬럼      : {list(df.columns)}")
print(f"  결측치    : {int(df['Close'].isna().sum())} 개")
# 달력일 기준으로 빠진 날(주말/공휴일) 개수 — '누락'이 아니라 비거래일임을 확인
cal_days = pd.date_range(df.index.min(), df.index.max(), freq="D")
print(f"  달력일 대비 비거래일(주말/공휴일 등): {len(cal_days) - len(df)} 일")

# ── 2) 정제 방침 ──────────────────────────────────────────────────
# ECB 기준환율은 거래일에만 존재. 주말/공휴일 '빈 날'은 결측이 아니라 비거래일이므로
# 인위적으로 채우지 않고 '거래일 기준'으로 분석한다(기법의 표준 관행).
# 이상치: 큰 일간 변동은 대부분 실제 시장 이벤트이므로 제거하지 않고 유지하되,
#         변화율 기준으로 어떤 날이 극단적이었는지 별도로 식별해 해석에 활용한다.
close = df["Close"]

# ── 3) 시계열 분석 기법 ──────────────────────────────────────────
df["MA20"] = close.rolling(20).mean()      # 단기 추세(약 1개월)
df["MA60"] = close.rolling(60).mean()      # 중기 추세(약 3개월)
df["ret"] = close.pct_change()             # 일간 변화율
df["vol20"] = df["ret"].rolling(20).std() * np.sqrt(252)  # 20일 연율화 변동성

# 월별 집계
monthly_mean = close.resample("ME").mean()
monthly_ret = close.resample("ME").last().pct_change()     # 월말 대비 월간 변화율
# 연 x 월 피벗(월간 변화율, %)
mr = monthly_ret.copy()
mr.index = pd.MultiIndex.from_arrays([mr.index.year, mr.index.month], names=["year", "month"])
pivot = (mr * 100).unstack("month")

# ── 4) 핵심 수치(인사이트 근거) ──────────────────────────────────
stats = {}
stats["n"] = int(len(df))
stats["start"] = {"date": str(close.index[0].date()), "val": round(float(close.iloc[0]), 2)}
stats["end"] = {"date": str(close.index[-1].date()), "val": round(float(close.iloc[-1]), 2)}
stats["overall_change_pct"] = round(float(close.iloc[-1] / close.iloc[0] - 1) * 100, 2)
stats["min"] = {"date": str(close.idxmin().date()), "val": round(float(close.min()), 2)}
stats["max"] = {"date": str(close.idxmax().date()), "val": round(float(close.max()), 2)}
stats["mean"] = round(float(close.mean()), 2)

# 연도별 요약
yearly = []
for y, g in close.groupby(close.index.year):
    yearly.append({
        "year": int(y),
        "mean": round(float(g.mean()), 2),
        "year_end": round(float(g.iloc[-1]), 2),
        "min": round(float(g.min()), 2),
        "max": round(float(g.max()), 2),
    })
# 연말 대비 연간 변화율
for i in range(1, len(yearly)):
    yearly[i]["yoy_pct"] = round((yearly[i]["year_end"] / yearly[i - 1]["year_end"] - 1) * 100, 2)
yearly[0]["yoy_pct"] = None
stats["yearly"] = yearly

# 일간 최대 상승/하락
r = df["ret"].dropna()
up = r.idxmax(); dn = r.idxmin()
stats["max_daily_up"] = {"date": str(up.date()), "pct": round(float(r.max()) * 100, 2), "level": round(float(close.loc[up]), 2)}
stats["max_daily_down"] = {"date": str(dn.date()), "pct": round(float(r.min()) * 100, 2), "level": round(float(close.loc[dn]), 2)}

# 변동성: 최고 시점
v = df["vol20"].dropna()
stats["vol_mean_pct"] = round(float(v.mean()) * 100, 2)
stats["vol_max"] = {"date": str(v.idxmax().date()), "pct": round(float(v.max()) * 100, 2)}

# 1400원 상회 거래일 비중
stats["days_above_1400_pct"] = round(float((close >= 1400).mean()) * 100, 1)

# 월(1~12)별 평균 변화율(계절성 관찰용)
month_avg = (monthly_ret.groupby(monthly_ret.index.month).mean() * 100).round(2)
stats["month_of_year_avg_ret_pct"] = {int(k): float(v) for k, v in month_avg.items()}

with open("stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("\n[2] 핵심 수치")
print(json.dumps(stats, ensure_ascii=False, indent=2))
print("\n[3] 월간 변화율 피벗(%) — 연 x 월")
print(pivot.round(1).to_string())

# ── 5) 시각화 ────────────────────────────────────────────────────
# 주요 관찰 시점(데이터에서 확인된 고점/저점/급변) — 주석용
# (날짜, 라벨, x오프셋, y오프셋) — 겹침 방지를 위해 위치를 개별 지정
events = [
    ("2022-09-28", "2022-09 고점", 0, 24),
    ("2022-11-11", "2022-11 급락", -18, -34),
    ("2024-12-27", "2024-12 급등", -60, -30),
    ("2025-04-08", "2025-04 고점", 24, 34),
    ("2025-05-05", "2025-05 급락", 6, -40),
]

def mark_events(ax):
    for d, label, dx, dy in events:
        ts = pd.Timestamp(d)
        if ts in close.index:
            y = close.loc[ts]
            ax.annotate(label, xy=(ts, y), xytext=(dx, dy), textcoords="offset points",
                        ha="center", fontsize=8, color="#1a202c",
                        arrowprops=dict(arrowstyle="->", color="#1a202c", lw=0.8))

# 그림 1 — 종가 + 이동평균(추세)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(close.index, close, color=C_LINE, lw=1.0, label="종가(USD/KRW)")
ax.plot(df.index, df["MA20"], color=C_MA20, lw=1.4, label="20일 이동평균")
ax.plot(df.index, df["MA60"], color=C_MA60, lw=1.4, label="60일 이동평균")
ax.axhline(1400, color="#a0aec0", ls="--", lw=1, alpha=0.8)
ax.text(close.index[5], 1405, "1,400원", color="#718096", fontsize=8, va="bottom")
mark_events(ax)
ax.yaxis.set_major_formatter(FuncFormatter(won))
ax.set_title("원/달러 환율 추세 (2022~2025) — 종가와 이동평균", fontsize=13, weight="bold")
ax.set_xlabel("날짜"); ax.set_ylabel("환율 (원/USD)")
ax.legend(loc="upper left", framealpha=0.9)
fig.tight_layout(); fig.savefig("images/01_trend_ma.png"); plt.close(fig)

# 그림 2 — 20일 연율화 변동성
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(v.index, v * 100, color="#805ad5", lw=1.2)
ax.fill_between(v.index, v * 100, color="#805ad5", alpha=0.15)
ax.axvspan(pd.Timestamp("2022-09-01"), pd.Timestamp("2022-11-30"), color=C_POS, alpha=0.08)
ax.axvspan(pd.Timestamp("2025-04-01"), pd.Timestamp("2025-05-31"), color=C_POS, alpha=0.08)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.set_title("변동성(20일 연율화 표준편차) — 불안정했던 시기", fontsize=13, weight="bold")
ax.set_xlabel("날짜"); ax.set_ylabel("연율화 변동성")
fig.tight_layout(); fig.savefig("images/02_volatility.png"); plt.close(fig)

# 그림 3 — 월간 변화율 히트맵(연 x 월)
fig, ax = plt.subplots(figsize=(10, 3.6))
data = pivot.values
vmax = np.nanmax(np.abs(data))
im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(12)); ax.set_xticklabels([f"{m}월" for m in range(1, 13)])
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=7,
                    color="#1a202c" if abs(val) < vmax * 0.6 else "white")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cb.set_label("월간 변화율(%)  ← 원화 강세 | 원화 약세 →")
ax.set_title("월간 변화율 히트맵 (양수=원화 약세, 음수=원화 강세)", fontsize=12, weight="bold")
fig.tight_layout(); fig.savefig("images/03_monthly_heatmap.png"); plt.close(fig)

# 그림 4 (보너스) — 시계열 분해(추세/계절성/잔차)
# 영업일 기준 주기성 확인을 위해 월(약 21거래일) 주기로 분해
s = close.copy()
result = seasonal_decompose(s, model="additive", period=21, extrapolate_trend="period")
fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
axes[0].plot(s.index, s, color=C_LINE, lw=0.9); axes[0].set_ylabel("원본")
axes[1].plot(result.trend.index, result.trend, color=C_MA60, lw=1.2); axes[1].set_ylabel("추세")
axes[2].plot(result.seasonal.index, result.seasonal, color=C_MA20, lw=0.8); axes[2].set_ylabel("계절성")
axes[3].plot(result.resid.index, result.resid, color="#805ad5", lw=0.6); axes[3].set_ylabel("잔차")
axes[0].set_title("시계열 분해 (추세 / 계절성 / 잔차, 주기=21거래일)", fontsize=13, weight="bold")
for a in axes:
    a.yaxis.set_major_formatter(FuncFormatter(won))
axes[3].set_xlabel("날짜")
fig.tight_layout(); fig.savefig("images/04_decompose.png"); plt.close(fig)

# 그림 5 (보너스) — 베이스라인 예측 백테스트(마지막 30거래일)
H = 30
train, test = close.iloc[:-H], close.iloc[-H:]
naive = pd.Series(train.iloc[-1], index=test.index)          # 직전값 유지
ma_base = pd.Series(train.tail(20).mean(), index=test.index)  # 최근 20일 평균 연장
mae_naive = float((test - naive).abs().mean())
mae_ma = float((test - ma_base).abs().mean())
stats_fc = {"H": H, "mae_naive": round(mae_naive, 2), "mae_ma": round(mae_ma, 2)}
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(close.index[-90:], close.iloc[-90:], color=C_LINE, lw=1.2, label="실제")
ax.plot(test.index, naive, color=C_POS, ls="--", lw=1.4, label=f"직전값 유지 (MAE {mae_naive:.1f}원)")
ax.plot(test.index, ma_base, color=C_NEG, ls="--", lw=1.4, label=f"20일 평균 연장 (MAE {mae_ma:.1f}원)")
ax.axvline(train.index[-1], color="#a0aec0", ls=":", lw=1)
ax.yaxis.set_major_formatter(FuncFormatter(won))
ax.set_title(f"베이스라인 예측 백테스트 — 마지막 {H}거래일", fontsize=13, weight="bold")
ax.set_xlabel("날짜"); ax.set_ylabel("환율 (원/USD)")
ax.legend(loc="upper left", framealpha=0.9)
fig.tight_layout(); fig.savefig("images/05_forecast_baseline.png"); plt.close(fig)

# 그림 6 — 일간 변화율(관찰: 급등락 분포)
fig, ax = plt.subplots(figsize=(11, 4))
colors = np.where(r.values >= 0, C_POS, C_NEG)
ax.bar(r.index, r.values * 100, color=colors, width=2.0)
ax.axhline(0, color="#4a5568", lw=0.8)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0f}%"))
ax.set_title("일간 변화율 (%) — 빨강=원화 약세, 초록=원화 강세", fontsize=13, weight="bold")
ax.set_xlabel("날짜"); ax.set_ylabel("일간 변화율")
fig.tight_layout(); fig.savefig("images/06_daily_returns.png"); plt.close(fig)

stats["forecast"] = stats_fc
with open("stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("\n[4] 예측 백테스트")
print(json.dumps(stats_fc, ensure_ascii=False, indent=2))
print("\n저장된 그래프: images/01~06 png 6개")
print("완료.")
