# -*- coding: utf-8 -*-
"""
Alpha Vantage：抓取 近20周普通周线 + 当天最新价（GLOBAL_QUOTE）
注意：
- 周线用 TIME_SERIES_WEEKLY（“普通周线”，不做复权；免费端点）
- 最新价用 GLOBAL_QUOTE（轻量报价；免费端点）
- 免费版有速率限制（常见每分钟5次、每天~25次），故每次请求后 sleep 以规避节流
- 兼容：美股/ETF/港股（港股用 1919.HK 这类写法）
"""

import os
import time
import requests
import pandas as pd
from typing import Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ====== Server酱配置（方糖服务号）======
# 建议：在系统里配置环境变量 SERVERCHAN_SENDKEY
# 没配的话，可以直接把你的 SendKey 写到默认值里
# ====== Server酱配置（方糖服务号）======
SERVERCHAN_SENDKEY = os.environ.get("SERVERCHAN_SENDKEY")

def push_serverchan(title: str, content: str):
    if not SERVERCHAN_SENDKEY:
        print("未配置 Server酱 SendKey（环境变量 SERVERCHAN_SENDKEY），跳过推送。")
        return
    url = f"https://sctapi.ftqq.com/{SERVERCHAN_SENDKEY}.send"
    ...


# ============== 1) API Key ==============
# 优先读取环境变量 ALPHAVANTAGE_API_KEY；如未设置，可在此处填入固定值。
API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY")  # 不再写默认值

def _get_api_key() -> str:
    if API_KEY:
        return API_KEY
    raise ValueError("未配置 Alpha Vantage API Key（环境变量 ALPHAVANTAGE_API_KEY）")




# ============== 2) 目标标的（可增删） ==============
SYMBOL_NORMALIZE = {
    "appl": "AAPL",  # 手误自动校正
    "aapl": "AAPL",
    "cony": "CONY",
    "nvdy": "NVDY",
}
targets_raw = ["appl", "cony", "nvdy", "UGL", "SPYI", "JEPQ"]


def normalize_symbol(s: str) -> str:
    s2 = s.strip()
    key = s2.lower()
    if key in SYMBOL_NORMALIZE:
        return SYMBOL_NORMALIZE[key]
    if key.isdigit():
        return f"{s2}.HK"      # 纯数字默认按港股代码处理
    return s2.upper()

symbols = [normalize_symbol(s) for s in targets_raw]

# ============== 3) 稳定版请求器（禁用系统代理 + 自动重试） ==============
BASE = "https://www.alphavantage.co/query"

def make_session(disable_proxy=True):
    s = requests.Session()
    s.trust_env = not disable_proxy           # 不从系统环境读取代理
    s.proxies = {"http": None, "https": None} if disable_proxy else {}
    retry = Retry(
        total=5, connect=5, read=5, status=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"Connection": "close", "User-Agent": "av-weekly/1.0"})
    return s

SESSION = make_session(disable_proxy=True)

def _get_api_key() -> str:
    env_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if env_key:
        return env_key
    if API_KEY and API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY":
        return API_KEY
    raise ValueError("未配置 Alpha Vantage API Key：请设置环境变量 ALPHAVANTAGE_API_KEY 或在文件顶部填入 API_KEY")

def av_get(params: dict, min_sleep: float = 12.0):
    """
    与 Alpha Vantage 通信的统一入口：
    - 自动带 apikey
    - (connect, read) 超时分别 15s / 90s
    - 自动重试 429/5xx/超时
    - 每次请求后 sleep 以降低限速触发概率
    """
    p = dict(params)
    p["apikey"] = _get_api_key()
    try:
        resp = SESSION.get(BASE, params=p, timeout=(15, 90), allow_redirects=True)
        ctype = resp.headers.get("Content-Type", "")
        data = resp.json() if ctype.startswith("application/json") else resp.text
        time.sleep(min_sleep)
        return data
    except requests.exceptions.ReadTimeout:
        raise RuntimeError("读超时：网络慢或被限速，可提高 min_sleep 或开启稳定代理。")
    except requests.exceptions.ProxyError:
        raise RuntimeError("代理错误：请禁用系统代理或在代码中显式关闭 proxies。")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"连接失败：{e}")

# ============== 4) 功能函数：周线(普通) + 最新价 ==============
def fetch_weekly_20(symbol: str) -> pd.DataFrame:
    """
    拉取普通周线 TIME_SERIES_WEEKLY，取最近 20 条
    返回：index=周末交易日（datetime），列=[open, high, low, close, volume, symbol]
    """
    data = av_get({"function": "TIME_SERIES_WEEKLY", "symbol": symbol}, min_sleep=12.0)
    if not isinstance(data, dict):
        raise RuntimeError(f"{symbol} 周线返回非JSON：{str(data)[:120]} ...")

    # Alpha Vantage 周线键名固定为 "Weekly Time Series"
    ts = data.get("Weekly Time Series") or data.get("Weekly Adjusted Time Series")
    if not ts:
        # 记录可能的错误信息
        msg = None
        for k in ("Note", "Information", "Error Message"):
            if k in data:
                msg = data[k]; break
        raise RuntimeError(f"{symbol} 周线不可用或被限速：{msg}")

    rows = []
    for date_str, v in ts.items():
        rows.append({
            "date": pd.to_datetime(date_str),
            "open": float(v.get("1. open", "nan")),
            "high": float(v.get("2. high", "nan")),
            "low": float(v.get("3. low", "nan")),
            "close": float(v.get("4. close", "nan")),
            "volume": float(v.get("5. volume", "nan")),
        })
    df = pd.DataFrame(rows).sort_values("date").set_index("date")
    df = df.tail(20)  # 近20周
    df.insert(0, "symbol", symbol)
    return df

def fetch_latest_quote(symbol: str) -> pd.DataFrame:
    """
    拉取 GLOBAL_QUOTE（轻量报价）：最新价、涨跌、当日开高低等
    返回：单行 DataFrame
    """
    data = av_get({"function": "GLOBAL_QUOTE", "symbol": symbol}, min_sleep=12.0)
    if not isinstance(data, dict):
        raise RuntimeError(f"{symbol} 最新价返回非JSON：{str(data)[:120]} ...")
    q = data.get("Global Quote")
    if not q:
        msg = None
        for k in ("Note", "Information", "Error Message"):
            if k in data:
                msg = data[k]; break
        raise RuntimeError(f"{symbol} 最新价不可用或被限速：{msg}")

    row = {
        "symbol": symbol,
        "price": float(q.get("05. price", "nan")),
        "change": float(q.get("09. change", "nan")),
        "change_percent": q.get("10. change percent", None),
        "volume": float(q.get("06. volume", "nan")),
        "latest_trading_day": q.get("07. latest trading day", None),
        "prev_close": float(q.get("08. previous close", "nan")),
        "open": float(q.get("02. open", "nan")),
        "high": float(q.get("03. high", "nan")),
        "low": float(q.get("04. low", "nan")),
    }
    return pd.DataFrame([row])

# ============== 5) 批量抓取（返回 DataFrame/字典，不落盘） ==============
def fetch_all_symbols(symbols_list):
    """
    批量抓取：近20周周线 + 最新报价
    返回：
      - weekly_dict: {symbol -> 周线 DataFrame(index=date, cols=[symbol, open, high, low, close, volume])}
      - latest_df:   最新价汇总 DataFrame（多行，每行一个标的）
      - wide_df:     宽表（index=date, 列=各symbol的周收盘价）
    """
    weekly_dict = {}
    quote_list = []

    for sym in symbols_list:
        try:
            print(f"抓取周线(20周)：{sym}")
            df_w = fetch_weekly_20(sym)
            weekly_dict[sym] = df_w
        except Exception as e:
            print(f"[周线] {sym} 失败：{e}")

        try:
            print(f"抓取最新价：{sym}")
            df_q = fetch_latest_quote(sym)
            quote_list.append(df_q)
        except Exception as e:
            print(f"[最新价] {sym} 失败：{e}")

    latest_df = pd.concat(quote_list, ignore_index=True) if quote_list else pd.DataFrame()

    # 统一构建周收盘宽表
    if weekly_dict:
        wide_df = pd.concat(
            [df["close"].rename(sym) for sym, df in weekly_dict.items()], axis=1
        ).sort_index()
    else:
        wide_df = pd.DataFrame()

    return weekly_dict, latest_df, wide_df


def results_as_dict(weekly_dict, latest_df, wide_df):
    """
    将 DataFrame 结果转为易序列化的字典/列表结构（日期转字符串）。
    """
    def df_to_records(df: pd.DataFrame):
        if df is None or df.empty:
            return []
        out = df.reset_index()
        if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        return out.to_dict(orient="records")

    weekly = {}
    for sym, df in (weekly_dict or {}).items():
        out = df.reset_index()
        if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
            out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        weekly[sym] = out.to_dict(orient="records")

    latest = [] if latest_df is None or latest_df.empty else latest_df.to_dict(orient="records")
    wide = df_to_records(wide_df)

    return {"symbols": symbols, "weekly": weekly, "latest": latest, "weekly_close_wide": wide}


# ============== 6) 监测与策略计算（不落盘） ==============
def _safe_last_ma(series: pd.Series, window: int):
    if series is None or series.empty or len(series) < window:
        return None
    return series.rolling(window).mean().iloc[-1]


def classify_trend(close_series: pd.Series, current_price: float) -> dict:
    """
    趋势判断：基于 MA4/MA8/MA16 与当前价。
    返回：{"status", "ma4", "ma8", "ma16"}
    """
    s = close_series.dropna().astype(float)
    ma4 = _safe_last_ma(s, 4)
    ma8 = _safe_last_ma(s, 8)
    ma16 = _safe_last_ma(s, 16)
    if ma4 is None or ma8 is None or ma16 is None:
        return {"status": "数据不足", "ma4": ma4, "ma8": ma8, "ma16": ma16}

    status = "不明确"
    # 强/弱趋势
    if ma4 > ma8 > ma16 and current_price > ma4:
        status = "强势上升"
    elif ma4 < ma8 < ma16 and current_price < ma4:
        status = "强势下降"
    else:
        # 横盘震荡：MA 之间差值<5%，且当前价在 MA8 ±8% 内
        try:
            spread = (max(ma4, ma8, ma16) - min(ma4, ma8, ma16)) / ma8 if ma8 else 1.0
            near_ma8 = abs(current_price - ma8) / ma8 if ma8 else 1.0
        except ZeroDivisionError:
            spread, near_ma8 = 1.0, 1.0
        if spread < 0.05 and near_ma8 <= 0.08:
            status = "横盘震荡"
        else:
            if ma4 > ma8:
                status = "弱势上升"
            elif ma4 < ma8:
                status = "弱势下降"

    return {"status": status, "ma4": ma4, "ma8": ma8, "ma16": ma16}


def price_position(close_series: pd.Series, current_price: float, lookback: int = 16) -> dict:
    s = close_series.dropna().astype(float)
    if s.empty:
        return {"pos_pct": None, "level": "数据不足", "hi": None, "lo": None}
    s = s.tail(lookback)
    hi = float(s.max())
    lo = float(s.min())
    rng = hi - lo
    if hi <= 0 or rng <= 0:
        return {"pos_pct": None, "level": "数据不足", "hi": hi, "lo": lo}
    pos = (current_price - lo) / rng
    if pos > 0.70:
        level = "高位"
    elif pos < 0.30:
        level = "低位"
    else:
        level = "中位"
    return {"pos_pct": float(pos), "level": level, "hi": hi, "lo": lo}


def peak_drawdown_buy_signal(close_series: pd.Series, current_price: float) -> dict:
    """
    近4个月(16周)高点下跌检测；返回评分与是否触发买入信号。
    回撤≥15%：+3；回撤≥25%：再+2；当前价<MA16：+2；当前价<MA8：+1；下降趋势：+2。
    """
    s = close_series.dropna().astype(float)
    s16 = s.tail(16)
    if s16.empty:
        return {"score": 0, "buy": False, "drawdown": None, "details": {}}
    peak = float(s16.max())
    dd = None
    if peak > 0:
        dd = (peak - current_price) / peak
    # MAs
    ma8 = _safe_last_ma(s, 8)
    ma16 = _safe_last_ma(s, 16)
    # Trend simple
    trend = classify_trend(s, current_price)["status"]
    score = 0
    details = {}
    if dd is not None:
        if dd >= 0.15:
            score += 3
            details["dd>=15%"] = 3
        if dd >= 0.25:
            score += 2
            details["dd>=25%"] = 2
    if ma16 is not None and current_price < ma16:
        score += 2
        details["px<MA16"] = 2
    if ma8 is not None and current_price < ma8:
        score += 1
        details["px<MA8"] = 1
    if trend in ("强势下降", "弱势下降"):
        score += 2
        details["downtrend"] = 2
    return {"score": score, "buy": score >= 5, "drawdown": dd, "details": details}


def volatility_20w(close_series: pd.Series) -> Optional[float]:
    """20周波动率：近20周收益率标准差（百分比）。"""
    s = close_series.dropna().astype(float)
    rets = s.pct_change().dropna().tail(20)
    if rets.empty:
        return None
    return float(rets.std() * 100.0)


def dynamic_take_profit(cost: Optional[float], current_price: float, close_series: pd.Series, trend_status: str) -> dict:
    """
    动态止盈：根据波动率与趋势调整阈值。返回阈值与建议卖出比例。
    """
    if cost is None or cost <= 0:
        return {"available": False, "reason": "未提供成本价"}

    vol_pct = volatility_20w(close_series)
    if vol_pct is None:
        vol_factor = 1.0
    else:
        if vol_pct > 8.0:
            vol_factor = 1.3
        elif vol_pct < 4.0:
            vol_factor = 0.8
        else:
            vol_factor = 1.0

    if trend_status == "强势上升":
        trend_factor = 1.2
    elif trend_status in ("强势下降", "弱势下降"):
        trend_factor = 0.8
    else:
        trend_factor = 1.0

    base_thr = [0.20, 0.35, 0.50]
    base_sell = [0.10, 0.15, 0.25]
    final_thr = [round(t * vol_factor * trend_factor, 4) for t in base_thr]

    profit_ratio = (current_price - cost) / cost
    actions = []
    total_to_sell = 0.0
    for t, s in zip(final_thr, base_sell):
        if profit_ratio >= t:
            actions.append({"threshold": t, "sell_ratio": s})
            total_to_sell += s

    return {
        "available": True,
        "vol_pct": vol_pct,
        "vol_factor": vol_factor,
        "trend_factor": trend_factor,
        "final_thresholds": final_thr,
        "profit_ratio": profit_ratio,
        "plan": actions,
        "total_sell_ratio": total_to_sell,
    }


def averaging_recommendation(cost: Optional[float], current_price: float, weekly_close: pd.Series, weekly_volume: pd.Series,
                             support_near_pct: float = 0.02) -> dict:
    """智能补仓：基于浮亏、支撑与量比。"""
    if cost is None or cost <= 0:
        return {"available": False, "reason": "未提供成本价"}

    loss = (cost - current_price) / cost
    s_close = weekly_close.dropna().astype(float)
    s_vol = weekly_volume.dropna().astype(float)
    if s_close.empty or s_vol.empty:
        return {"available": False, "reason": "数据不足"}

    lo8 = float(s_close.tail(8).min())
    lo16 = float(s_close.tail(16).min())
    cur_vol = float(s_vol.iloc[-1]) if len(s_vol) else None
    avg8_vol = float(s_vol.tail(8).mean()) if len(s_vol) >= 1 else None
    vol_ratio = (cur_vol / avg8_vol) if (cur_vol is not None and avg8_vol and avg8_vol > 0) else None

    def near_support(price: float, support: float, tol: float) -> bool:
        if not support or support <= 0:
            return False
        return abs(price - support) / support <= tol

    near_8 = near_support(current_price, lo8, support_near_pct)
    near_16 = near_support(current_price, lo16, support_near_pct)

    decision = {"action": "不加仓", "size": 0.0}

    if loss >= 0.25:
        decision = {"action": "重新评估投资逻辑", "size": 0.0}
    elif loss >= 0.15 and (near_16 or (vol_ratio is not None and vol_ratio < 0.5)):
        decision = {"action": "加仓", "size": 0.30}
    elif loss >= 0.08 and (near_8 or (vol_ratio is not None and vol_ratio < 0.7)):
        decision = {"action": "加仓", "size": 0.20}

    return {
        "available": True,
        "loss_ratio": float(loss),
        "support8": lo8,
        "support16": lo16,
        "near8": near_8,
        "near16": near_16,
        "volume_ratio": vol_ratio,
        "decision": decision,
    }


def risk_control(weekly_close: pd.Series, cost: Optional[float], current_price: float, trend_status: str) -> dict:
    s = weekly_close.dropna().astype(float)
    s20 = s.tail(20)
    if s20.empty:
        return {"available": False, "reason": "数据不足"}
    hi = float(s20.max())
    lo = float(s20.min())
    mdd = (hi - lo) / hi if hi > 0 else None
    if mdd is None:
        return {"available": False, "reason": "数据不足"}

    if mdd > 0.35:
        risk = "高风险"; pos_range = "10%-15%"
    elif mdd >= 0.20:
        risk = "中风险"; pos_range = "15%-20%"
    else:
        risk = "低风险"; pos_range = "20%-25%"

    stop = None
    warn = None
    if cost is not None and cost > 0:
        loss = (cost - current_price) / cost
        if loss >= 0.30:
            stop = "浮亏≥30%，强制止损"
        elif loss >= 0.20 and trend_status in ("强势下降", "弱势下降"):
            warn = "浮亏≥20%且处于下降趋势，风险警告"

    return {
        "available": True,
        "mdd": float(mdd),
        "risk_level": risk,
        "position_suggestion": pos_range,
        "stop_loss": stop,
        "warning": warn,
    }


def analyze_symbol(symbol: str, weekly_df: pd.DataFrame, latest_row: pd.Series, cost: Optional[float] = None) -> dict:
    """综合分析单个标的，输出字典结果。"""
    close = weekly_df["close"]
    volume = weekly_df["volume"] if "volume" in weekly_df.columns else pd.Series([], dtype=float)
    current_price = float(latest_row.get("price", close.iloc[-1]))

    trend = classify_trend(close, current_price)
    pos = price_position(close, current_price, lookback=16)
    buy_sig = peak_drawdown_buy_signal(close, current_price)
    take_profit = dynamic_take_profit(cost, current_price, close, trend["status"])
    avg_rec = averaging_recommendation(cost, current_price, close, volume)
    risk = risk_control(close, cost, current_price, trend["status"])

    return {
        "symbol": symbol,
        "current_price": current_price,
        "trend": trend,
        "position": pos,
        "buy_signal": buy_sig,
        "take_profit": take_profit,
        "averaging": avg_rec,
        "risk": risk,
    }


def summarize_result(res: dict) -> str:
    """构造简短中文摘要。"""
    sym = res["symbol"]
    trend = res["trend"]["status"]
    pos = res["position"]
    pos_str = "未知"
    if pos.get("pos_pct") is not None:
        pos_pct = round(pos["pos_pct"] * 100, 1)
        pos_str = f"{pos['level']}({pos_pct}%)"
    buy = res["buy_signal"]
    dd = buy.get("drawdown")
    dd_str = f"{round(dd*100,1)}%" if dd is not None else "-"
    buy_phrase = "→关注买入" if buy.get("buy") else ""

    # 盈亏（只要提供了成本就展示）
    tp = res.get("take_profit", {})
    pl_phrase = ""
    if tp.get("available") and tp.get("profit_ratio") is not None:
        pr = tp.get("profit_ratio")
        pr_pct = round(pr * 100, 1)
        if pr >= 0:
            pl_phrase = f"，浮盈{pr_pct}%"
        else:
            pl_phrase = f"，浮亏{abs(pr_pct)}%"

    # 补仓
    avg = res["averaging"]
    avg_phrase = ""
    if avg.get("available"):
        loss = avg.get("loss_ratio")
        if loss is not None:
            loss_pct = round(loss * 100, 1)
            if avg["decision"]["action"] == "加仓":
                size_pct = int(avg["decision"]["size"] * 100)
                avg_phrase = f"，浮亏{loss_pct}%→建议加仓{size_pct}%"
            elif avg["decision"]["action"] == "重新评估投资逻辑":
                avg_phrase = f"，浮亏{loss_pct}%→建议重新评估"

    # 风险
    risk = res["risk"]
    risk_phrase = ""
    if risk.get("available"):
        risk_phrase = f"，{risk['risk_level']}→仓位{risk['position_suggestion']}"

    return f"{sym}：{trend}({pos_str})，4个月回撤{dd_str}{buy_phrase}{pl_phrase}{avg_phrase}{risk_phrase}"

# if __name__ == "__main__":
#     # 运行脚本：返回 DataFrame 结果，不写入磁盘
#     weekly_dict, latest_df, wide_df = fetch_all_symbols(symbols)

#     # 打印预览
#     print("\n=== 最新价(汇总) 预览 ===")
#     if not latest_df.empty:
#         print(latest_df.head(10))
#         print(f"行数: {len(latest_df)}  列数: {len(latest_df.columns)}")
#     else:
#         print("(空)")

#     print("\n=== 周线(近20周) 样例 ===")
#     for sym, df in list(weekly_dict.items())[:3]:
#         print(f"-- {sym} --")
#         print(df.tail(5))

#     print("\n=== 周收盘(宽表) 预览 ===")
#     if not wide_df.empty:
#         print(wide_df.tail(5))
#         print(f"行数: {len(wide_df)}  列数: {len(wide_df.columns)}")
#     else:
#         print("(空)")

#     # 同时准备一个便于序列化的字典（需要可直接用于后续分析/存储时使用）
#     RESULTS = {
#         "weekly": weekly_dict,           # {symbol -> DataFrame}
#         "latest_quotes": latest_df,      # DataFrame
#         "weekly_close_wide": wide_df,    # DataFrame
#     }
#     RESULTS_DICT = results_as_dict(weekly_dict, latest_df, wide_df)

#     # ========== 监测逻辑演示：请在此处填入持仓成本 ==========
#     # 例：COST_BASIS = {"AAPL": 120.0, "CONY": 20.0, "NVDY": 25.0}
#     COST_BASIS = {"AAPL": 262.08, "CONY": 7.1, "NVDY": 16.51, "UGL": 55.75, "SPYI": 51.96, "JEPQ": 56.06}
#     # 将成本键归一化为规范代码（大小写不敏感，也支持纯数字转 .HK）
#     COST_BASIS_NORM = {normalize_symbol(k): float(v) for k, v in COST_BASIS.items()}

#     # 将 latest_df 转为索引便于查找
#     latest_map = {row["symbol"]: row for _, row in latest_df.iterrows()} if not latest_df.empty else {}

#     print("\n=== 监测结果（摘要） ===")
#     ANALYSIS = {}
#     for sym in symbols:
#         wdf = weekly_dict.get(sym)
#         if wdf is None or wdf.empty:
#             print(f"{sym}：无周线数据")
#             continue
#         lrow = latest_map.get(sym, pd.Series({}))
#         cost = COST_BASIS_NORM.get(sym)
#         res = analyze_symbol(sym, wdf, lrow, cost)
#         ANALYSIS[sym] = res
#         print(summarize_result(res))


if __name__ == "__main__":
    # 运行脚本：返回 DataFrame 结果，不写入磁盘
    weekly_dict, latest_df, wide_df = fetch_all_symbols(symbols)

    # 打印预览
    print("\n=== 最新价(汇总) 预览 ===")
    if not latest_df.empty:
        print(latest_df.head(10))
        print(f"行数: {len(latest_df)} 列数: {len(latest_df.columns)}")
    else:
        print("(空)")

    print("\n=== 周线(近20周) 样例 ===")
    for sym, df in list(weekly_dict.items())[:3]:
        print(f"-- {sym} --")
        print(df.tail(5))

    print("\n=== 周收盘(宽表) 预览 ===")
    if not wide_df.empty:
        print(wide_df.tail(5))
        print(f"行数: {len(wide_df)} 列数: {len(wide_df.columns)}")
    else:
        print("(空)")

    # 同时准备一个便于序列化的字典（需要可直接用于后续分析/存储时使用）
    RESULTS = {
        "weekly": weekly_dict,          # {symbol -> DataFrame}
        "latest_quotes": latest_df,     # DataFrame
        "weekly_close_wide": wide_df,   # DataFrame
    }
    RESULTS_DICT = results_as_dict(weekly_dict, latest_df, wide_df)

    # ========== 监测逻辑：持仓成本 ========== 
    COST_BASIS = {
        "AAPL": 262.08,
        "CONY": 7.1,
        "NVDY": 16.51,
        "UGL": 55.75,
        "SPYI": 51.96,
        "JEPQ": 56.06,
    }
    # 将成本键归一化为规范代码（大小写不敏感，也支持纯数字转 .HK）
    COST_BASIS_NORM = {normalize_symbol(k): float(v) for k, v in COST_BASIS.items()}

    # 将 latest_df 转为索引便于查找
    latest_map = {row["symbol"]: row for _, row in latest_df.iterrows()} if not latest_df.empty else {}

    print("\n=== 监测结果（摘要） ===")
    ANALYSIS = {}
    summary_lines = []  # 用来收集每只票的摘要，后面发给微信

    for sym in symbols:
        wdf = weekly_dict.get(sym)
        if wdf is None or wdf.empty:
            line = f"{sym}：无周线数据"
            print(line)
            summary_lines.append(line)
            continue

        lrow = latest_map.get(sym, pd.Series({}))
        cost = COST_BASIS_NORM.get(sym)

        res = analyze_symbol(sym, wdf, lrow, cost)
        ANALYSIS[sym] = res

        line = summarize_result(res)
        print(line)
        summary_lines.append(line)

    # ====== 统一推送到微信（Server酱）======
    if summary_lines:
        # 标题可以自定义，这里顺便带上时间
        title = time.strftime("持仓监测 %Y-%m-%d %H:%M", time.localtime())
        # 正文：每只票一行，中间空一行
        body = "\n\n".join(summary_lines)

        push_serverchan(title, body)
    else:
        print("没有可推送的结果，不推送。")
