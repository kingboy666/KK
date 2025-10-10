#!/usr/bin/env python3
# railway_bot.py
# K-line Momentum + ATR adaptive SL/TP using exchange-side conditional orders (OKX via ccxt)
# NOTE: Test in sandbox first. Adjust per your ccxt/OKX version.

import os
import time
import math
import logging
import traceback
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
import numpy as np

# ----------------------------
# Logging
# ----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logger = logging.getLogger("railway_kmomentum")
logger.setLevel(LOG_LEVEL)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# 中文日志过滤器（按需将常见英文片段替换为中文）
LOG_LANG = os.getenv("LOG_LANG", "zh").lower()
if LOG_LANG in ("zh", "zh-cn", "chinese"):
    class CnLogFilter(logging.Filter):
        MAP = {
            "Starting K-line Momentum bot": "启动K线动量机器人（交易所侧条件单）",
            "Starting main monitor loop. Symbols:": "开始主监控循环。交易对：",
            "fetch_ohlcv error": "获取K线错误",
            "fetch_ohlcv failed for": "获取K线失败：",
            "insufficient data for timeframe": "该周期数据不足，跳过",
            "ATR invalid": "ATR无效，跳过",
            "SIGNAL BUY": "信号 买入",
            "SIGNAL SELL": "信号 卖出",
            "Placing market entry": "执行市价入场",
            "Attached TP/SL to entry order via ccxt params.": "已在下单时同步附加交易所侧止盈/止损",
            "Entry executed:": "入场成交：",
            "Entry TP/SL attached at order creation; skipping separate conditional orders.": "入场时已附加TP/SL；跳过后续单独创建",
            "position opened and protective orders placed.": "持仓已建立且保护单已处理",
            "failed to open position": "开仓失败",
            "market order creation raised": "创建市价单异常",
            "conditional create (try1) failed": "条件单创建（方案1）失败",
            "conditional create (try2) failed": "条件单创建（方案2）失败",
            "All conditional-order attempts failed": "所有条件单创建方案均失败",
            "detected existing exchange TP/SL; skip re-attach.": "检测到交易所已有保护单；不再重复附加",
            "no existing exchange TP/SL detected; will consider first attach if not attempted.": "未检测到保护单；若未尝试过将首次附加",
            "first attach TP/SL planned:": "计划首次附加TP/SL：",
            "attached exchange-side TP/SL after entry (once).": "已在入场后附加TP/SL（一次性）",
            "dynamic check: enabled": "动态检查：已启用",
            "dynamically updated exchange-side TP/SL (old canceled).": "已动态更新交易所侧TP/SL（已先撤旧单）",
            "dynamic TP/SL update failed": "动态更新TP/SL失败",
            "dynamic TP/SL disabled by config; skipping update.": "已通过配置禁用动态TP/SL；跳过更新",
            "local SL hit for long": "本地止损触发（多单）",
            "local TP hit for long": "本地止盈触发（多单）",
            "local SL hit for short": "本地止损触发（空单）",
            "local TP hit for short": "本地止盈触发（空单）",
            "Free USDT balance is zero or couldn't fetch. Sleeping.": "可用USDT余额为0或获取失败。休眠中……",
            "fetch_balance attempt": "获取余额重试",
            "Unable to reliably fetch balance": "无法稳定获取余额",
            "order amount clamped to exchange max amount": "下单数量已按交易所上限收敛，避免 51202",
            "trailing refs:": "跟踪止损参考：",
            "local SL/TP fallback check error": "本地SL/TP兜底检查错误",
            "error closing position on reverse": "反向信号平仓错误",
            "Fatal exception": "致命异常",
            "KeyboardInterrupt received. Exiting.": "接收到中断信号，退出。",
        }
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
                for en, zh in self.MAP.items():
                    if en in msg:
                        msg = msg.replace(en, zh)
                record.msg = msg
                record.args = ()
            except Exception:
                pass
            return True
    handler.addFilter(CnLogFilter())

# ----------------------------
# Basic config (can edit)
# ----------------------------
# Default global timeframe (per-symbol override allowed)
GLOBAL_TIMEFRAME = "15m"

# Symbols configuration: for each symbol specify timeframe, leverage, atr SL multiplier, atr TP multiplier
# Modify these values to tune each symbol independently.
SYMBOL_CONFIG = {
    # symbol: (timeframe, leverage, atr_mult_sl, atr_mult_tp)
    'FIL/USDT:USDT':  ('15m', 30, 2.2, 4.4),
    'ZRO/USDT:USDT':  ('5m',  30, 2.5, 5.0),
    'WIF/USDT:USDT':  ('5m',  30, 2.5, 5.0),
    'WLD/USDT:USDT':  ('15m', 25, 2.5, 5.0),
    'BTC/USDT:USDT':  ('15m', 25, 1.5, 3.0),
    'ETH/USDT:USDT':  ('15m', 25, 1.6, 3.2),
    'SOL/USDT:USDT':  ('15m', 30, 1.8, 3.6),
    'DOGE/USDT:USDT': ('5m',  40, 2.0, 4.0),
    'XRP/USDT:USDT':  ('15m', 25, 1.7, 3.4),
    'PEPE/USDT:USDT': ('5m',  30, 2.8, 5.6),
    'ARB/USDT:USDT':  ('15m', 30, 2.0, 4.0),
}

# Strategy parameters
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
BODY_RATIO_THRESHOLD = float(os.getenv("BODY_RATIO_THRESHOLD", "0.6"))  # entity / range
CONFIRM_CANDLES = int(os.getenv("CONFIRM_CANDLES", "2"))  # require 2 consecutive momentum candles
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))  # fraction of account balance risk per trade
MIN_ORDER_USDT = float(os.getenv("MIN_ORDER_USDT", "1.0"))
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.0008"))  # slippage / fees approx
SANDBOX = os.getenv("SANDBOX", "false").lower() in ("1", "true", "yes")
DYNAMIC_TPSL = os.getenv("DYNAMIC_TPSL", "true").lower() in ("1", "true", "yes")
MIN_TPSL_UPDATE_INTERVAL_SEC = int(os.getenv("MIN_TPSL_UPDATE_INTERVAL_SEC", "60"))
# 跟踪止损配置
TRAIL_SL = os.getenv("TRAIL_SL", "true").lower() in ("1","true","yes")
TRAIL_MULT = float(os.getenv("TRAIL_MULT", "2.0"))  # 跟踪距离 = ATR * TRAIL_MULT
TRAIL_UPDATE_INTERVAL_SEC = int(os.getenv("TRAIL_UPDATE_INTERVAL_SEC", "60"))
# 按交易对禁用动态更新（逗号分隔）
DYNAMIC_TPSL_DISABLED = set(s.strip() for s in os.getenv("DYNAMIC_TPSL_DISABLED", "").split(",") if s.strip())
def is_dynamic_enabled(symbol: str) -> bool:
    return DYNAMIC_TPSL and (symbol not in DYNAMIC_TPSL_DISABLED)

# Fetch / timing
BARS_LIMIT = 200
MAIN_LOOP_INTERVAL = 10  # seconds between ticks (will sleep shorter if waiting for next candle) 

# ----------------------------
# Exchange init
# ----------------------------
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_API_SECRET = os.getenv("OKX_API_SECRET") or os.getenv("OKX_SECRET_KEY")
OKX_API_PASSPHRASE = os.getenv("OKX_API_PASSPHRASE") or os.getenv("OKX_PASSPHRASE")

if not OKX_API_KEY or not OKX_API_SECRET or not OKX_API_PASSPHRASE:
    logger.warning("OKX API credentials missing in env. Set OKX_API_KEY/OKX_API_SECRET/OKX_API_PASSPHRASE before running.")

exchange_opts = {
    'apiKey': OKX_API_KEY,
    'secret': OKX_API_SECRET,
    'password': OKX_API_PASSPHRASE,
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
}
if SANDBOX:
    # ccxt OKX sandbox sometimes requires different urls; keep SANDBOX flag for your manual adaptation
    logger.info("SANDBOX flag is true - ensure your OKX sandbox credentials are used.")
exchange = ccxt.okx(exchange_opts)
exchange.verbose = False



# ----------------------------
# Helpers: indicators & data fetch
# ----------------------------
def fetch_ohlcv(symbol, timeframe, limit=BARS_LIMIT, since=None):
    """Fetch ohlcv robustly; return DataFrame indexed by datetime."""
    attempts = 0
    while attempts < 6:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not bars:
                return None
            df = pd.DataFrame(bars, columns=['ts','open','high','low','close','volume'])
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
            df.set_index('datetime', inplace=True)
            return df
        except Exception as e:
            attempts += 1
            logger.debug(f"fetch_ohlcv error {symbol} attempt {attempts}: {e}")
            time.sleep(1 + attempts)
    logger.debug(f"fetch_ohlcv failed for {symbol}")
    return None

def compute_atr(df, period=ATR_PERIOD):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def body_ratio_series(df):
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, 1e-9)
    return (body / rng).fillna(0)

# ----------------------------
# Position & order tracking
# ----------------------------
positions = {}  # symbol -> position dict (local state)
# position example: {
#   'side':'long'/'short',
#   'entry_price': float,
#   'qty': float,
#   'sl_order_id': 'algo123',
#   'tp_order_id': 'algo124',
#   'opened_at': datetime,
#   'atr_sl_value': float,
# }

# 运行统计：累计已实现盈亏、平仓数与胜率
stats = {
    'realized_pnl': 0.0,
    'closed_count': 0,
    'wins': 0,
    'losses': 0,
}
last_summary_ts = 0

def compute_unrealized_pnl(symbol: str, p: dict, last_px: float) -> float:
    """
    线性USDT本位合约 PnL:
      多单: (last - entry) * 张数 * contractSize
      空单: (entry - last) * 张数 * contractSize
    若取不到 contractSize, 回退为现货近似 (不推荐, 但保证不中断)。
    """
    try:
        qty = float(p.get('qty') or 0)
        entry = float(p.get('entry_price') or 0)
        if qty <= 0 or entry <= 0 or last_px is None:
            return 0.0
        # 尝试获取合约张价值
        cs = 1.0
        try:
            mkt = exchange.market(symbol)
            cs = float(mkt.get('contractSize') or 1.0)
        except Exception:
            cs = 1.0
        if p.get('side') == 'buy':
            return (float(last_px) - entry) * qty * cs
        else:
            return (entry - float(last_px)) * qty * cs
    except Exception:
        return 0.0

def update_stats_on_close(symbol: str, p: dict, close_px: float, reason: str):
    global stats
    pnl = compute_unrealized_pnl(symbol, p, close_px)
    stats['realized_pnl'] += pnl
    stats['closed_count'] += 1
    if pnl >= 0:
        stats['wins'] += 1
    else:
        stats['losses'] += 1
    winrate = (stats['wins'] / stats['closed_count'] * 100.0) if stats['closed_count'] > 0 else 0.0
    logger.info(f"{symbol} 已平仓（原因：{reason}），本单已实现盈亏={pnl:.6f}，累计已实现盈亏={stats['realized_pnl']:.6f}，胜率={winrate:.2f}%（{stats['wins']}/{stats['closed_count']}）")

def log_summary(free_usdt: float):
    # 输出账户与持仓看板
    try:
        winrate = (stats['wins'] / stats['closed_count'] * 100.0) if stats['closed_count'] > 0 else 0.0
        logger.info(f"账户看板：可用USDT（合约）={free_usdt:.4f}，累计已实现盈亏={stats['realized_pnl']:.6f}，总平仓={stats['closed_count']}，胜率={winrate:.2f}%（胜/总={stats['wins']}/{stats['closed_count']}）")
        if positions:
            for sym, p in positions.items():
                try:
                    tk = exchange.fetch_ticker(sym)
                    last_px = tk.get('last') if tk and 'last' in tk else None
                except Exception:
                    last_px = None
                upnl = compute_unrealized_pnl(sym, p, last_px if last_px is not None else p.get('entry_price'))
                # 计算名义≈USDT（张数×contractSize×最新价）
                try:
                    mkt = exchange.market(sym)
                    cs = float(mkt.get('contractSize') or 1.0)
                except Exception:
                    cs = 1.0
                notional_usdt = (float(p.get('qty') or 0) * cs * float(last_px or p.get('entry_price') or 0)) if (last_px or p.get('entry_price')) else 0.0
                logger.info(f"持仓：{sym} 方向={p.get('side')} 张数={p.get('qty')} 名义≈{notional_usdt:.6f}USDT 入场={p.get('entry_price')} 最新={last_px} 未实现盈亏={upnl:.6f} SL={p.get('sl_price')} TP={p.get('tp_price')}")
        else:
            logger.info("当前无持仓。")
    except Exception as e:
        logger.debug(f"输出账户看板时异常：{e}")

# ----------------------------
# Account helpers
# ----------------------------
def sync_positions_from_exchange():
    """
    同步交易所当前SWAP持仓到本地 positions（仅在本地不存在该symbol持仓时建立）。
    使程序在重启或手动开仓后也能正确显示持仓与继续管理保护单。
    """
    try:
        # 优先使用 ccxt 统一接口
        ex_positions = []
        try:
            ex_positions = exchange.fetch_positions(params={'instType': 'SWAP'}) or []
        except Exception:
            # 某些版本不接受该参数，退回无参
            try:
                ex_positions = exchange.fetch_positions() or []
            except Exception:
                ex_positions = []

        for px in ex_positions:
            try:
                sym = px.get('symbol')
                if not sym:
                    # 兼容：从 info.instId 转换
                    info = px.get('info') or {}
                    inst_id = info.get('instId') or info.get('inst_id')
                    if inst_id and hasattr(exchange, 'markets_by_id'):
                        m = exchange.markets_by_id.get(inst_id)
                        sym = m.get('symbol') if m else None
                if not sym:
                    continue

                # 只关心非零合约数
                contracts = safe_float(px.get('contracts') or px.get('positionAmt') or px.get('contractsAbs') or 0)
                if contracts <= 0:
                    continue

                side_ccxt = (px.get('side') or '').lower()  # 'long'/'short'（ccxt标准）
                pos_side = 'buy' if side_ccxt == 'long' else ('sell' if side_ccxt == 'short' else None)
                if not pos_side:
                    continue

                entry_price = safe_float(px.get('entryPrice') or px.get('avgCostPrice') or px.get('markPrice') or 0)
                # 若本地已有，不覆盖数量与方向（避免干扰当前单管理）
                if sym in positions:
                    continue

                # 初始化本地结构（简化，TP/SL价格稍后由动态模块计算与覆盖）
                positions[sym] = {
                    'side': pos_side,
                    'entry_price': entry_price,
                    'qty': contracts,
                    'sl_order': None,
                    'tp_order': None,
                    'opened_at': datetime.now(timezone.utc).isoformat(),
                    'atr_sl_value': None,
                    'atr_val': None,
                    'sl_price': None,
                    'tp_price': None,
                    'tp_sl_attached': False,
                    'tp_sl_attempted': False,
                    'tp_sl_updated_at': None,
                    'highest_since_entry': entry_price if pos_side == 'buy' else None,
                    'lowest_since_entry': entry_price if pos_side == 'sell' else None,
                    'trail_last_update_at': None,
                    'cfg': SYMBOL_CONFIG.get(sym) or (GLOBAL_TIMEFRAME, 20, 2.0, 3.0),
                }
                # 尝试标记已存在的保护单
                try:
                    mark_existing_protection(sym, positions[sym])
                except Exception:
                    pass
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"同步交易所持仓失败：{e}")

def safe_float(v):
    try:
        return float(v)
    except:
        return 0.0

def get_free_balance_usdt():
    """Try to get free USDT from swap account via ccxt."""
    attempts = 0
    while attempts < 4:
        try:
            bal = exchange.fetch_balance(params={"type":"swap"})
            # ccxt version differences handled carefully:
            if isinstance(bal, dict):
                # try known places
                if 'USDT' in bal.get('free', {}):
                    return safe_float(bal['free']['USDT'])
                if 'USDT' in bal.get('total', {}):
                    return safe_float(bal['total']['USDT'])
                # older ccxt may return nested dict
                info = bal.get('info', {})
                # try to extract from info
                if isinstance(info, dict):
                    acct = info.get('data') or info.get('balances') or info
                    # best-effort search
                    def search_usdt(x):
                        if not isinstance(x, list): return None
                        for it in x:
                            if isinstance(it, dict):
                                if it.get('ccy') == 'USDT':
                                    return safe_float(it.get('avail') or it.get('availBal') or it.get('availBalance') or it.get('available'))
                        return None
                    found = None
                    for k in ('data','balances','list',''):
                        arr = info.get(k) if isinstance(info, dict) else None
                        if not arr:
                            arr = info if isinstance(info, list) else None
                        if arr:
                            found = search_usdt(arr)
                            if found is not None:
                                return found
            return 0.0
        except Exception as e:
            attempts += 1
            logger.warning(f"fetch_balance attempt {attempts} error: {e}")
            time.sleep(1 + attempts)
    logger.error("Unable to reliably fetch balance")
    return 0.0

# ----------------------------
# Exchange-side protection helpers (detect and reconcile once)
# ----------------------------
def fetch_open_reduce_only(symbol, pos_side):
    """
    Return list of exchange-side protective orders for symbol+posSide.
    Some OKX algo TP/SL may not expose reduceOnly; detect via info.algoId/ordType as well.
    """
    try:
        open_orders = exchange.fetch_open_orders(symbol, params={'tdMode': 'cross'})
        result = []
        for o in (open_orders or []):
            try:
                params = o.get('info') or {}
                # detect reduceOnly or algo-like orders
                ro = o.get('reduceOnly')
                if ro is None:
                    ro = str(params.get('reduceOnly', '')).lower() in ('true','1')
                ord_type = (o.get('type') or params.get('ordType') or params.get('orderType') or '').lower()
                has_algo = bool(params.get('algoId') or params.get('algoClOrdId'))
                ps = (params.get('posSide') or params.get('positionSide') or '').lower()
                # treat as protection if reduceOnly or algo order, and posSide matches
                if (ro or has_algo or ord_type in ('conditional','trigger','oco','stop','take_profit','tp','sl')) and ps == str(pos_side).lower():
                    result.append(o)
            except Exception:
                continue
        return result
    except Exception as e:
        logger.debug(f"{symbol} fetch_open_reduce_only error: {e}")
        return []

def okx_inst_id(symbol: str) -> str:
    # OKX 合约 instId 形如 BTC-USDT-SWAP
    try:
        s = symbol.replace("/", "-")
        if not s.endswith("-SWAP"):
            s = f"{s}-SWAP"
        return s
    except Exception:
        return symbol

def fetch_okx_algo_protections(symbol: str, pos_side: str):
    """
    读取 OKX 待触发的计划委托/止盈止损（algo）并按 posSide 过滤。
    返回列表元素统一为 dict，含 algoId、posSide、cTime。
    """
    out = []
    try:
        if hasattr(exchange, 'private_get_trade_orders_algo_pending'):
            inst_id = okx_inst_id(symbol)
            # 常见 ordType: conditional/oco
            resp = exchange.private_get_trade_orders_algo_pending({'instType': 'SWAP', 'instId': inst_id})
            data = (resp or {}).get('data') or resp or []
            if isinstance(data, list):
                for it in data:
                    try:
                        ps = str(it.get('posSide') or it.get('positionSide') or '').lower()
                        if ps == str(pos_side).lower():
                            out.append({
                                'algoId': it.get('algoId') or it.get('algo_id'),
                                'posSide': ps,
                                'cTime': it.get('cTime') or it.get('ctime') or it.get('createTime') or '0',
                                'info': it
                            })
                    except Exception:
                        continue
    except Exception:
        pass
    return out

def cancel_okx_algos(symbol: str, algos: list) -> int:
    """
    使用 OKX 原生接口撤销 algo 计划委托/止盈止损。
    """
    canceled = 0
    if not algos:
        return 0
    inst_id = okx_inst_id(symbol)
    # 尝试逐个撤销，兼容不同 ccxt 版本的参数格式
    for a in algos:
        aid = a.get('algoId') or (a.get('info') or {}).get('algoId')
        if not aid:
            continue
        try:
            if hasattr(exchange, 'private_post_trade_cancel_algos'):
                # 两种可能的 payload 结构，逐一尝试
                try:
                    exchange.private_post_trade_cancel_algos({'algoId': [aid], 'instId': inst_id})
                except Exception:
                    exchange.private_post_trade_cancel_algos({'algoIds': [{'algoId': aid, 'instId': inst_id}]})
                canceled += 1
            else:
                # 兜底：尝试常规 cancel_order（多数情况下对 algo 无效）
                try:
                    exchange.cancel_order(aid, symbol, params={})
                    canceled += 1
                except Exception:
                    pass
        except Exception:
            continue
    return canceled

def cancel_all_protection(symbol, pos_side):
    """
    撤掉同 symbol + posSide 的所有保护单（reduceOnly 及 OKX 条件单），返回撤单总数。
    """
    try:
        # 1) 通过统一 open_orders 方式撤普通 reduceOnly/条件单
        orders = fetch_open_reduce_only(symbol, pos_side) or []
        canceled = 0
        for o in orders:
            try:
                oid = o.get('id') or (o.get('info') or {}).get('algoId') or (o.get('info') or {}).get('order_id')
                if oid:
                    try:
                        exchange.cancel_order(oid, symbol, params={})
                        canceled += 1
                    except Exception:
                        pass
            except Exception:
                continue
        # 2) 专门清理 OKX algo（计划委托/止盈止损）
        try:
            algo_list = fetch_okx_algo_protections(symbol, pos_side)
            canceled += cancel_okx_algos(symbol, algo_list)
        except Exception:
            pass
        if canceled > 0:
            logger.info(f"{symbol} posSide={pos_side} 已清理保护单数量={canceled}")
        return canceled
    except Exception as e:
        logger.debug(f"{symbol} cancel_all_protection error: {e}")
        return 0

def enforce_protection_limit(symbol, pos_side, keep=2):
    """
    强制限制保护单数量，默认最多保留 keep 个（通常 SL+TP=2）。返回撤销数量。
    同时考虑：
      - 普通 reduceOnly/条件单（fetch_open_orders）
      - OKX algo 计划委托/止盈止损（orders-algo-pending）
    """
    try:
        # 拉取两类保护单并合并视图
        normal_orders = fetch_open_reduce_only(symbol, pos_side) or []
        algo_orders = fetch_okx_algo_protections(symbol, pos_side) or []

        # 统一构造 (kind, id, ts) 列表，kind in {'normal','algo'}
        items = []
        for o in normal_orders:
            try:
                oid = o.get('id') or (o.get('info') or {}).get('order_id') or (o.get('info') or {}).get('algoId')
                ts = o.get('timestamp') or 0
                items.append(('normal', oid, int(ts), o))
            except Exception:
                continue
        for a in algo_orders:
            try:
                oid = a.get('algoId')
                ts = int(a.get('cTime') or 0)
                items.append(('algo', oid, ts, a))
            except Exception:
                continue

        if len(items) <= keep:
            return 0

        # 时间从旧到新
        items_sorted = sorted(items, key=lambda x: (x[2] or 0))
        to_cancel = items_sorted[0:max(0, len(items_sorted)-keep)]
        canceled = 0
        # 分别用不同方式撤销
        for kind, oid, _ts, obj in to_cancel:
            if not oid:
                continue
            try:
                if kind == 'normal':
                    exchange.cancel_order(oid, symbol, params={})
                    canceled += 1
                else:
                    canceled += cancel_okx_algos(symbol, [obj])
            except Exception:
                continue

        if canceled > 0:
            logger.info(f"{symbol} posSide={pos_side} 保护单超额，已强制收敛撤销={canceled}，保留最新{keep}个")
        return canceled
    except Exception as e:
        logger.debug(f"{symbol} enforce_protection_limit error: {e}")
        return 0

def mark_existing_protection(symbol, p):
    """
    If exchange already has reduceOnly protective orders for this position, mark attached and set ids.
    Returns True if found any.
    """
    pos_side_flag = 'long' if p.get('side') == 'buy' else 'short'
    existing = fetch_open_reduce_only(symbol, pos_side_flag)
    logger.debug(f"{symbol} posSide={pos_side_flag} open protection orders detected: {len(existing)}")
    if existing:
        # Try to map ids
        for o in existing:
            oid = o.get('id') or (o.get('info') or {}).get('algoId') or (o.get('info') or {}).get('order_id')
            if not oid:
                continue
            # 先尽力放到本地结构里，无法区分SL/TP也问题不大，标记为已附加即可
            if not p.get('sl_order_id'):
                p['sl_order_id'] = oid
            elif not p.get('tp_order_id'):
                p['tp_order_id'] = oid
        p['tp_sl_attached'] = True
        p['tp_sl_attempted'] = True
        return True
    return False

# ----------------------------
# Qty calculation respecting OKX contract size and limits
# ----------------------------
def compute_order_amount(symbol, price, notional_usdt):
    """
    Compute order amount for OKX swap:
    - Do NOT multiply by leverage; leverage affects margin, not amount.
    - Use market.contractSize and limits.amount min/max to size correctly.
    - Returns a float/int amount already rounded to exchange precision, or None if below minimum.
    """
    try:
        market = None
        if hasattr(exchange, 'market'):
            try:
                market = exchange.market(symbol)
                if not market or not market.get('active', True):
                    # ensure markets loaded if needed
                    exchange.load_markets(reload=False)
                    market = exchange.market(symbol)
            except Exception:
                try:
                    exchange.load_markets(reload=True)
                    market = exchange.market(symbol)
                except Exception:
                    market = None

        if price is None or price <= 0:
            return None

        # base quantity needed for target notional
        base_qty = notional_usdt / price

        # contract sizing
        amount = base_qty
        contract_size = None
        if isinstance(market, dict):
            contract_size = market.get('contractSize')
            if contract_size and contract_size > 0:
                # amount should be number of contracts (integer)
                amount = math.floor(base_qty / contract_size)

        # apply min/max limits
        min_amt = None
        max_amt = None
        if isinstance(market, dict):
            limits = market.get('limits') or {}
            amt_limits = limits.get('amount') or {}
            min_amt = amt_limits.get('min')
            max_amt = amt_limits.get('max')

        # if contractSize present, ensure integer
        if contract_size and contract_size > 0:
            amount = max(0, int(amount))

        # enforce min
        if min_amt is not None:
            if amount < min_amt:
                # if we can bump to min, do so; else return None to skip
                amount = min_amt

        # enforce max
        clamped = False
        if max_amt is not None and amount > max_amt:
            amount = max_amt
            clamped = True

        # precision rounding via ccxt helper if available
        try:
            amount = float(exchange.amount_to_precision(symbol, amount))
        except Exception:
            # fallback: keep as-is
            pass

        if amount is None or amount <= 0:
            return None

        return amount, clamped
    except Exception as e:
        logger.warning(f"compute_order_amount error for {symbol}: {e}")
        return None

# ----------------------------
# Order placement: entry + exchange conditional SL/TP
# ----------------------------
def place_market_entry_with_exchange_tp_sl(symbol, side, notional_usdt, leverage, sl_price, tp_price):
    """
    1) place market order to open position (using notional_usdt * leverage as approximate notional)
    2) immediately place conditional SL and TP on exchange with reduceOnly flags
    Returns dict with entry info and created orders (if succeed).
    NOTE: OKX/ccxt parameter names vary; we try a few common parameter sets and handle errors.
    """
    try:
        logger.info(f"Placing market entry {symbol} {side} notional {notional_usdt} with leverage {leverage}")
        # fetch price for qty calc
        ticker = exchange.fetch_ticker(symbol)
        entry_est_price = ticker['last'] if ticker and 'last' in ticker else None
        if entry_est_price is None:
            # fallback: place market order and rely on exchange fill
            entry_est_price = 0.0

        # Calculate qty correctly for OKX swaps (do NOT multiply by leverage)
        # ensure we have a valid price for sizing
        price_for_size = entry_est_price
        if not price_for_size or price_for_size <= 0:
            try:
                _tk = exchange.fetch_ticker(symbol)
                price_for_size = _tk.get('last') if _tk and 'last' in _tk else None
            except Exception:
                price_for_size = None
        qty_info = compute_order_amount(symbol, price_for_size, notional_usdt)
        if not qty_info:
            logger.warning(f"{symbol} computed amount below minimum or invalid; skipping entry.")
            return {'entry_order': None, 'filled_qty': 0, 'entry_price': entry_est_price, 'sl_order': None, 'tp_order': None}
        qty, was_clamped = qty_info
        if was_clamped:
            logger.warning(f"{symbol} order amount clamped to exchange max amount to avoid 51202.")

        # Place market order (try attach TP/SL at creation for OKX via ccxt)
        pos_side = 'long' if side == 'buy' else 'short'
        attached_ok = False
        try:
            order = exchange.create_order(
                symbol,
                'market',
                side,
                qty,
                None,
                params={
                    'tdMode': 'cross',
                    'posSide': pos_side,
                    'takeProfit': {'triggerPrice': float(tp_price), 'price': '-1'},
                    'stopLoss': {'triggerPrice': float(sl_price), 'price': '-1'},
                },
            )
            attached_ok = True
            logger.info("Attached TP/SL to entry order via ccxt params.")
        except Exception as e:
            logger.warning(f"attach TP/SL at entry failed: {e}; falling back to plain market + separate conditionals")
            try:
                order = exchange.create_market_order(symbol, side, qty, params={'tdMode': 'cross', 'posSide': pos_side})
            except Exception as e2:
                logger.warning(f"market order creation raised: {e2}; trying create_order('market') fallback")
                order = exchange.create_order(symbol, 'market', side, qty, None, params={'tdMode': 'cross', 'posSide': pos_side})

        # Determine filled qty and avg price
        filled_qty = None
        entry_price = None
        if isinstance(order, dict):
            filled_qty = safe_float(order.get('filled', 0.0)) or safe_float(order.get('amount', 0.0))
            entry_price = safe_float(order.get('average') or order.get('price') or entry_est_price)
        else:
            filled_qty = qty
            entry_price = entry_est_price

        if filled_qty == 0:
            # fallback: use qty we attempted
            filled_qty = qty

        logger.info(f"Entry executed: filled_qty={filled_qty}, entry_price={entry_price}")

        # Place conditional SL and TP orders - try several param variations known in ccxt/OKX world
        sl_side = 'sell' if side == 'buy' else 'buy'
        tp_side = sl_side

        client_base = f"auto_{int(time.time()*1000)}"
        sl_order = None
        tp_order = None

        if attached_ok:
            logger.info("Entry TP/SL attached at order creation; skipping separate conditional orders.")
            result = {
                'entry_order': order,
                'filled_qty': filled_qty,
                'entry_price': entry_price,
                'sl_order': None,
                'tp_order': None,
                'tp_sl_attached': True,
                'tp_sl_updated_at': datetime.now(timezone.utc).isoformat(),
            }
            return result

        # attempt variation 1: common param names
        sl_params_try = {
            'triggerPrice': float(sl_price),
            'reduceOnly': True,
            'closeOnTrigger': True,  # if supported

            'tdMode': 'cross',
            'posSide': pos_side,
            # optionally: 'triggerType':'last_price'
        }
        tp_params_try = {
            'triggerPrice': float(tp_price),
            'reduceOnly': True,
            'closeOnTrigger': True,

            'tdMode': 'cross',
            'posSide': pos_side,
        }
        # try create_order with type='market' and params as conditional
        try:
            sl_order = exchange.create_order(symbol, type='market', side=sl_side, amount=filled_qty, price=None, params=sl_params_try)
            tp_order = exchange.create_order(symbol, type='market', side=tp_side, amount=filled_qty, price=None, params=tp_params_try)
            logger.info(f"Placed conditional SL/TP (try1) for {symbol}")
        except Exception as e1:
            logger.warning(f"conditional create (try1) failed: {e1}; trying alternative params")
            # attempt variation 2: different param keys
            alt_sl_params = {
                'stopPrice': float(sl_price),
                'reduceOnly': True,
                'orderType': 'market',
                'tdMode': 'cross',
                'posSide': pos_side,

            }
            alt_tp_params = {
                'stopPrice': float(tp_price),
                'reduceOnly': True,
                'orderType': 'market',
                'tdMode': 'cross',
                'posSide': pos_side,

            }
            try:
                sl_order = exchange.create_order(symbol, type='market', side=sl_side, amount=filled_qty, price=None, params=alt_sl_params)
                tp_order = exchange.create_order(symbol, type='market', side=tp_side, amount=filled_qty, price=None, params=alt_tp_params)
                logger.info(f"Placed conditional SL/TP (try2) for {symbol}")
            except Exception as e2:
                logger.warning(f"conditional create (try2) failed: {e2}; trying okx algo endpoint fallback")
                # Some ccxt versions expose privatePost... endpoints; try raw call (best-effort)
                try:
                    # This is a heuristic fallback and may need adjustment per ccxt version
                    if hasattr(exchange, 'private_post_trade_batch_orders'):
                        # example payload; requires exact keys for OKX REST; user should validate in sandbox
                        sl_payload = {
                            "instId": symbol.replace("/", "-"),
                            "tdMode": "cross",
                            "side": sl_side.upper(),
                            "ordType": "conditional",
                            "sz": str(filled_qty),
                            "px": str(sl_price),
                            "triggerPx": str(sl_price),
                            "reduceOnly": "true",
                        }
                        # raw call - may not exist
                        resp_sl = exchange.private_post_trade_batch_orders({"orders_data":[sl_payload]})
                        sl_order = resp_sl
                    else:
                        raise Exception("raw fallback not supported in this ccxt build")
                except Exception as e3:
                    logger.error(f"All conditional-order attempts failed for {symbol}: {e3}")
                    sl_order = None
                    # we will fallback to local-monitoring as protection

        # Build return structure
        result = {
            'entry_order': order,
            'filled_qty': filled_qty,
            'entry_price': entry_price,
            'sl_order': sl_order,
            'tp_order': tp_order,
            'tp_sl_attached': True if (sl_order or tp_order) else False,
        }
        return result

    except Exception as exc:
        logger.error(f"place_market_entry_with_exchange_tp_sl exception: {exc}")
        logger.error(traceback.format_exc())
        raise

# ----------------------------
# Momentum detection (K-line based)
# ----------------------------
def detect_kline_momentum(df):
    """
    df: OHLCV dataframe with chronological index
    returns: 'buy' / 'sell' / None
    logic: require CONFIRM_CANDLES consecutive momentum candles,
    each with entity_ratio > BODY_RATIO_THRESHOLD and direction same,
    and final candle closes beyond previous high/low.
    """
    if len(df) < CONFIRM_CANDLES + 1:
        return None
    br = body_ratio_series(df)
    # check last CONFIRM_CANDLES candles
    last = df.iloc[-(CONFIRM_CANDLES+1):].copy()  # include one previous for breakout comparison
    br_last = br.iloc[-(CONFIRM_CANDLES+1):]
    # indices
    # earlier candle for breakout baseline:
    prev_high = last['high'].iloc[-(CONFIRM_CANDLES+1)]
    prev_low = last['low'].iloc[-(CONFIRM_CANDLES+1)]
    # examine the next CONFIRM_CANDLES candles
    directions = []
    for i in range(1, CONFIRM_CANDLES+1):
        row = last.iloc[i]
        body_ratio = br_last.iloc[i]
        is_bull = row['close'] > row['open'] and body_ratio >= BODY_RATIO_THRESHOLD
        is_bear = row['close'] < row['open'] and body_ratio >= BODY_RATIO_THRESHOLD
        if is_bull:
            directions.append('bull')
        elif is_bear:
            directions.append('bear')
        else:
            directions.append('none')
    # If all bull and final close > prev_high -> buy
    if all(d == 'bull' for d in directions):
        final_close = last['close'].iloc[-1]
        if final_close > prev_high:
            return 'buy'
    if all(d == 'bear' for d in directions):
        final_close = last['close'].iloc[-1]
        if final_close < prev_low:
            return 'sell'
    return None

# ----------------------------
# Utility: calculate sl/tp using ATR-based multipliers
# ----------------------------
def calc_sl_tp(entry_price, side, atr_value, atr_mult_sl, atr_mult_tp):
    sl_dist = atr_value * atr_mult_sl
    tp_dist = atr_value * atr_mult_tp
    if side == 'buy':
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist
    return sl_price, tp_price, sl_dist, tp_dist

# ----------------------------
# Monitor loop & main
# ----------------------------
def monitor_and_run():
    logger.info("Starting main monitor loop. Symbols: " + ", ".join(SYMBOL_CONFIG.keys()))
    # main loop: per tick, check each symbol with its configured timeframe
    while True:
        start_ts = time.time()
        try:
            # 同步交易所当前持仓到本地（解决重启/手动开仓后看板不显示的问题）
            sync_positions_from_exchange()
            # fetch balance and calculate allocation per symbol (equal allocation of free USDT)
            free_usdt = get_free_balance_usdt()
            if free_usdt <= 0:
                logger.warning("Free USDT balance is zero or couldn't fetch. Sleeping.")
                time.sleep(15)
                continue
            alloc_per_symbol = max(MIN_ORDER_USDT, free_usdt / len(SYMBOL_CONFIG))
        except Exception as e:
            logger.error(f"balance fetch error: {e}")
            alloc_per_symbol = MIN_ORDER_USDT

        for symbol, cfg in SYMBOL_CONFIG.items():
            timeframe, leverage, atr_mult_sl, atr_mult_tp = cfg
            tf = timeframe if timeframe else GLOBAL_TIMEFRAME
            try:
                # fetch recent OHLCV
                df = fetch_ohlcv(symbol, tf, limit=BARS_LIMIT)
                if df is None or len(df) < ATR_PERIOD + CONFIRM_CANDLES + 2:
                    logger.debug(f"{symbol} insufficient data for timeframe {tf}. Skipping.")
                    continue
                atr = compute_atr(df, ATR_PERIOD)
                current_atr = atr.iloc[-1]
                if math.isnan(current_atr) or current_atr <= 0:
                    logger.debug(f"{symbol} ATR invalid: {current_atr}. Skip.")
                    continue
                # detect signal
                signal = detect_kline_momentum(df)
                pos = positions.get(symbol)
                if pos:
                    # there is an open local position; check for manual close conditions (reverse momentum)
                    # For safety, also query exchange positions/liquidation? (left as enhancement)
                    # Check reverse signal: if long and 'sell' detected => close
                    if pos['side'] == 'buy' and signal == 'sell':
                        logger.info(f"{symbol} local long sees reverse momentum -> close by market")
                        try:
                            # market close (reduceOnly)
                            qty = pos.get('qty')
                            if qty and float(qty) > 0:
                                exchange.create_market_order(symbol, 'sell', qty, params={'reduceOnly': True, 'tdMode': 'cross', 'posSide': 'long'})
                            # 统计已实现盈亏
                            try:
                                tkc = exchange.fetch_ticker(symbol)
                                close_px = tkc.get('last') if tkc and 'last' in tkc else pos.get('entry_price')
                            except Exception:
                                close_px = pos.get('entry_price')
                            update_stats_on_close(symbol, pos, close_px, "反向信号（多→空）")
                            # cancel conditional orders if any
                            if pos.get('sl_order_id'):
                                try:
                                    exchange.cancel_order(pos['sl_order_id'], symbol, params={})
                                except Exception:
                                    pass
                            if pos.get('tp_order_id'):
                                try:
                                    exchange.cancel_order(pos['tp_order_id'], symbol, params={})
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.error(f"{symbol} error closing position on reverse: {e}")
                        positions.pop(symbol, None)
                        continue
                    if pos['side'] == 'sell' and signal == 'buy':
                        logger.info(f"{symbol} local short sees reverse momentum -> close by market")
                        try:
                            qty = pos.get('qty')
                            if qty and float(qty) > 0:
                                exchange.create_market_order(symbol, 'buy', qty, params={'reduceOnly': True, 'tdMode': 'cross', 'posSide': 'short'})
                            try:
                                tkc = exchange.fetch_ticker(symbol)
                                close_px = tkc.get('last') if tkc and 'last' in tkc else pos.get('entry_price')
                            except Exception:
                                close_px = pos.get('entry_price')
                            update_stats_on_close(symbol, pos, close_px, "反向信号（空→多）")
                            if pos.get('sl_order_id'):
                                try:
                                    exchange.cancel_order(pos['sl_order_id'], symbol, params={})
                                except Exception:
                                    pass
                            if pos.get('tp_order_id'):
                                try:
                                    exchange.cancel_order(pos['tp_order_id'], symbol, params={})
                                except Exception:
                                    pass
                        except Exception as e:
                            logger.error(f"{symbol} error closing position on reverse: {e}")
                        positions.pop(symbol, None)
                        continue

                # If no position and signal exists -> open
                if (not pos) and signal in ('buy','sell'):
                    side = 'buy' if signal == 'buy' else 'sell'
                    # determine entry notional: use alloc_per_symbol
                    notional = alloc_per_symbol
                    # determine entry price estimate
                    ticker = exchange.fetch_ticker(symbol)
                    entry_price_est = ticker.get('last') if ticker and 'last' in ticker else df['close'].iloc[-1]
                    # compute sl/tp with ATR
                    sl_price, tp_price, sl_dist, tp_dist = calc_sl_tp(entry_price_est, side, current_atr, atr_mult_sl, atr_mult_tp)
                    logger.info(f"{symbol} SIGNAL {signal.upper()} @ est {entry_price_est:.6f} ATR={current_atr:.6f} SL={sl_price:.6f} TP={tp_price:.6f}")
                    try:
                        res = place_market_entry_with_exchange_tp_sl(symbol, side, notional, leverage, sl_price, tp_price)
                        filled_qty = res.get('filled_qty')
                        entry_price = res.get('entry_price')
                        sl_order = res.get('sl_order')
                        tp_order = res.get('tp_order')
                        # store position local state
                        positions[symbol] = {
                            'side': 'buy' if side == 'buy' else 'sell',
                            'entry_price': entry_price,
                            'qty': filled_qty,
                            'sl_order': sl_order,
                            'tp_order': tp_order,
                            'opened_at': datetime.now(timezone.utc).isoformat(),
                            'atr_sl_value': current_atr * atr_mult_sl,
                            'atr_val': current_atr,
                            'sl_price': sl_price,
                            'tp_price': tp_price,
                            'tp_sl_attached': bool(res.get('tp_sl_attached', False)),
                            'tp_sl_attempted': bool(res.get('tp_sl_attached', False)),
                            'tp_sl_updated_at': (res.get('tp_sl_updated_at') or (datetime.now(timezone.utc).isoformat() if res.get('tp_sl_attached') else None)),
                            # 跟踪止损锚点
                            'highest_since_entry': entry_price if side == 'buy' else None,
                            'lowest_since_entry': entry_price if side == 'sell' else None,
                            'trail_last_update_at': None,
                            'cfg': cfg
                        }
                        logger.info(f"{symbol} position opened and protective orders placed.")
                    except Exception as e:
                        logger.error(f"{symbol} failed to open position: {e}")
                        logger.error(traceback.format_exc())
                        # continue to next symbol

                # Optionally: monitor existing conditional orders status, if they triggered -> cleanup
                if positions.get(symbol):
                    # attach/update exchange-side TP/SL exactly-once; avoid duplicates; dynamic update cancels previous first
                    p = positions[symbol]

                    # Detect if exchange already has reduceOnly protection; if yes, mark and skip attach
                    if not p.get('tp_sl_attached'):
                        try:
                            if mark_existing_protection(symbol, p):
                                logger.debug(f"{symbol} detected existing exchange TP/SL; skip re-attach.")
                            else:
                                logger.debug(f"{symbol} no existing exchange TP/SL detected; will consider first attach if not attempted.")
                        except Exception as e:
                            logger.debug(f"{symbol} protection detection error: {e}")

                    try:
                        # compute desired SL/TP based on latest ATR but original entry price and per-symbol config
                        atr_latest = compute_atr(df, ATR_PERIOD).iloc[-1]
                        desired_sl, desired_tp, _, _ = calc_sl_tp(p['entry_price'], p['side'], atr_latest, p['cfg'][2], p['cfg'][3])

                        # Trailing Stop: 只收紧、不放松
                        if TRAIL_SL:
                            # 获取最新价格，更新峰值/谷值
                            try:
                                tnow = exchange.fetch_ticker(symbol)
                                last_px_ts = tnow.get('last') if tnow and 'last' in tnow else None
                            except Exception:
                                last_px_ts = None
                            if last_px_ts:
                                if p.get('side') == 'buy':
                                    prev_high = p.get('highest_since_entry') or p['entry_price']
                                    if last_px_ts > prev_high:
                                        p['highest_since_entry'] = last_px_ts
                                elif p.get('side') == 'sell':
                                    prev_low = p.get('lowest_since_entry') or p['entry_price']
                                    if last_px_ts < prev_low:
                                        p['lowest_since_entry'] = last_px_ts

                            # 以 ATR*TRAIL_MULT 作为跟踪距离
                            trail_dist = float(atr_latest) * float(TRAIL_MULT)
                            trail_sl_candidate = None
                            if p.get('side') == 'buy' and p.get('highest_since_entry'):
                                trail_sl = float(p['highest_since_entry']) - trail_dist
                                trail_sl_candidate = trail_sl
                                # 只上移，不下移
                                desired_sl = max(desired_sl, trail_sl)
                            elif p.get('side') == 'sell' and p.get('lowest_since_entry'):
                                trail_sl = float(p['lowest_since_entry']) + trail_dist
                                trail_sl_candidate = trail_sl
                                # 只下移，不上移
                                desired_sl = min(desired_sl, trail_sl)
                            logger.debug(f"{symbol} trailing refs: side={p.get('side')}, highest={p.get('highest_since_entry')}, lowest={p.get('lowest_since_entry')}, atr={atr_latest}, trail_mult={TRAIL_MULT}, trail_dist={trail_dist}, trail_sl_candidate={trail_sl_candidate}, desired_sl={desired_sl}")

                        # 1) Attach once if not yet attached and not attempted
                        if (not p.get('tp_sl_attached')) and (not p.get('tp_sl_attempted')):
                            pos_side_flag = 'long' if p['side'] == 'buy' else 'short'
                            try:
                                logger.info(f"{symbol} first attach TP/SL planned: posSide={pos_side_flag}, qty={p['qty']}, SL={desired_sl}, TP={desired_tp}")
                                # 附加前先清理一遍，防止历史保护单残留
                                try:
                                    cancel_all_protection(symbol, pos_side_flag)
                                except Exception:
                                    pass
                                sl_params_once = {'triggerPrice': float(desired_sl), 'reduceOnly': True, 'closeOnTrigger': True, 'tdMode': 'cross', 'posSide': pos_side_flag}
                                tp_params_once = {'triggerPrice': float(desired_tp), 'reduceOnly': True, 'closeOnTrigger': True, 'tdMode': 'cross', 'posSide': pos_side_flag}
                                sl_o = exchange.create_order(symbol, type='market', side=('sell' if p['side']=='buy' else 'buy'), amount=p['qty'], price=None, params=sl_params_once)
                                tp_o = exchange.create_order(symbol, type='market', side=('sell' if p['side']=='buy' else 'buy'), amount=p['qty'], price=None, params=tp_params_once)
                                p['sl_order'] = sl_o; p['tp_order'] = tp_o
                                if isinstance(sl_o, dict): p['sl_order_id'] = sl_o.get('id') or sl_o.get('algoId') or sl_o.get('order_id')
                                if isinstance(tp_o, dict): p['tp_order_id'] = tp_o.get('id') or tp_o.get('algoId') or tp_o.get('order_id')
                                p['tp_sl_attached'] = True
                                p['sl_price'] = desired_sl; p['tp_price'] = desired_tp
                                logger.info(f"{symbol} attached exchange-side TP/SL after entry (once). sl_id={p.get('sl_order_id')} tp_id={p.get('tp_order_id')}")
                            except Exception as ex_attach_once:
                                logger.warning(f"{symbol} attach TP/SL after entry failed: {ex_attach_once}")
                            finally:
                                p['tp_sl_attempted'] = True

                        # 2) Dynamic update: if enabled and prices changed materially, cancel old and re-place new
                        elif is_dynamic_enabled(symbol) and p.get('tp_sl_attached'):
                            # throttle updates to avoid flapping
                            try:
                                last_upd_iso = p.get('tp_sl_updated_at') or p.get('opened_at')
                                last_upd_dt = None
                                if last_upd_iso:
                                    try:
                                        # best-effort parse ISO; timezone-aware preferred
                                        last_upd_dt = datetime.fromisoformat(last_upd_iso.replace('Z','+00:00'))
                                    except Exception:
                                        last_upd_dt = None
                                now_dt = datetime.now(timezone.utc)
                                allow_update = True
                                gap_sec = None
                                if last_upd_dt:
                                    gap_sec = (now_dt - last_upd_dt).total_seconds()
                                    # 动态更新与跟踪止损的节流：取更严格的间隔
                                    min_gap = max(MIN_TPSL_UPDATE_INTERVAL_SEC, TRAIL_UPDATE_INTERVAL_SEC)
                                    allow_update = gap_sec >= min_gap
                            except Exception:
                                allow_update = True
                                gap_sec = None

                            eps = max(1e-8, abs(p.get('sl_price', desired_sl)) * 0.001)  # 放宽阈值，避免微小波动频繁更新
                            changed = (abs(desired_sl - p.get('sl_price', desired_sl)) > eps) or (abs(desired_tp - p.get('tp_price', desired_tp)) > eps)
                            logger.debug(f"{symbol} dynamic check: enabled, changed={changed}, allow_update={allow_update}, gap_sec={gap_sec}, eps={eps}, old(SL/TP)=({p.get('sl_price')}/{p.get('tp_price')}), new=({desired_sl}/{desired_tp})")

                            if allow_update and changed:
                                # 动态更新前，先做一次全面清理，避免叠加
                                try:
                                    pos_side_flag = 'long' if p['side'] == 'buy' else 'short'
                                    cancel_all_protection(symbol, pos_side_flag)
                                except Exception:
                                    pass
                                # cancel previous protective orders if we have ids; 若无id，则从交易所侧筛选并取消
                                canceled_any = False
                                ids = [p.get('sl_order_id'), p.get('tp_order_id')]
                                for oid in ids:
                                    if oid:
                                        try:
                                            exchange.cancel_order(oid, symbol, params={})
                                            canceled_any = True
                                        except Exception:
                                            pass
                                if not canceled_any:
                                    try:
                                        pos_side_flag = 'long' if p['side'] == 'buy' else 'short'
                                        for o in fetch_open_reduce_only(symbol, pos_side_flag):
                                            oid = o.get('id') or (o.get('info') or {}).get('algoId') or (o.get('info') or {}).get('order_id')
                                            if oid:
                                                try:
                                                    exchange.cancel_order(oid, symbol, params={})
                                                except Exception:
                                                    pass
                                    except Exception:
                                        pass

                                pos_side_flag = 'long' if p['side'] == 'buy' else 'short'
                                try:
                                    sl_params_upd = {'triggerPrice': float(desired_sl), 'reduceOnly': True, 'closeOnTrigger': True, 'tdMode': 'cross', 'posSide': pos_side_flag}
                                    tp_params_upd = {'triggerPrice': float(desired_tp), 'reduceOnly': True, 'closeOnTrigger': True, 'tdMode': 'cross', 'posSide': pos_side_flag}
                                    sl_o = exchange.create_order(symbol, type='market', side=('sell' if p['side']=='buy' else 'buy'), amount=p['qty'], price=None, params=sl_params_upd)
                                    tp_o = exchange.create_order(symbol, type='market', side=('sell' if p['side']=='buy' else 'buy'), amount=p['qty'], price=None, params=tp_params_upd)
                                    p['sl_order'] = sl_o; p['tp_order'] = tp_o
                                    if isinstance(sl_o, dict): p['sl_order_id'] = sl_o.get('id') or sl_o.get('algoId') or sl_o.get('order_id')
                                    if isinstance(tp_o, dict): p['tp_order_id'] = tp_o.get('id') or tp_o.get('algoId') or tp_o.get('order_id')
                                    p['sl_price'] = desired_sl; p['tp_price'] = desired_tp
                                    p['tp_sl_updated_at'] = datetime.now(timezone.utc).isoformat()
                                    logger.info(f"{symbol} dynamically updated exchange-side TP/SL (old canceled).")
                                except Exception as ex_upd:
                                    logger.warning(f"{symbol} dynamic TP/SL update failed: {ex_upd}")

                        else:
                            if not is_dynamic_enabled(symbol) and p.get('tp_sl_attached'):
                                logger.info(f"{symbol} dynamic TP/SL disabled by config; skipping update.")
                        # continue with local fallback checks below
                        ticker_now = exchange.fetch_ticker(symbol)
                        last_px = ticker_now.get('last') if ticker_now and 'last' in ticker_now else None
                        if last_px is None:
                            last_px = df['close'].iloc[-1]
                        if last_px:
                            if p.get('side') == 'buy':
                                # long: hit SL if last <= sl_price; hit TP if last >= tp_price
                                if p.get('sl_price') and last_px <= float(p['sl_price']):
                                    logger.info(f"{symbol} local SL hit for long @ {last_px} <= {p['sl_price']} -> closing position")
                                    qty = p.get('qty')
                                    if qty and float(qty) > 0:
                                        try:
                                            exchange.create_market_order(symbol, 'sell', qty, params={'reduceOnly': True, 'tdMode': 'cross', 'posSide': 'long'})
                                        except Exception as e:
                                            logger.error(f"{symbol} local SL close error: {e}")
                                    update_stats_on_close(symbol, p, last_px, "本地止损（多）")
                                    # best-effort cancel protective orders
                                    for oid in (p.get('sl_order_id'), p.get('tp_order_id')):
                                        if oid:
                                            try:
                                                exchange.cancel_order(oid, symbol, params={})
                                            except Exception:
                                                pass
                                    positions.pop(symbol, None)
                                elif p.get('tp_price') and last_px >= float(p['tp_price']):
                                    logger.info(f"{symbol} local TP hit for long @ {last_px} >= {p['tp_price']} -> closing position")
                                    qty = p.get('qty')
                                    if qty and float(qty) > 0:
                                        try:
                                            exchange.create_market_order(symbol, 'sell', qty, params={'reduceOnly': True, 'tdMode': 'cross', 'posSide': 'long'})
                                        except Exception as e:
                                            logger.error(f"{symbol} local TP close error: {e}")
                                    update_stats_on_close(symbol, p, last_px, "本地止盈（多）")
                                    for oid in (p.get('sl_order_id'), p.get('tp_order_id')):
                                        if oid:
                                            try:
                                                exchange.cancel_order(oid, symbol, params={})
                                            except Exception:
                                                pass
                                    positions.pop(symbol, None)
                            elif p.get('side') == 'sell':
                                # short: hit SL if last >= sl_price; hit TP if last <= tp_price
                                if p.get('sl_price') and last_px >= float(p['sl_price']):
                                    logger.info(f"{symbol} local SL hit for short @ {last_px} >= {p['sl_price']} -> closing position")
                                    qty = p.get('qty')
                                    if qty and float(qty) > 0:
                                        try:
                                            exchange.create_market_order(symbol, 'buy', qty, params={'reduceOnly': True, 'tdMode': 'cross', 'posSide': 'short'})
                                        except Exception as e:
                                            logger.error(f"{symbol} local SL close error: {e}")
                                    update_stats_on_close(symbol, p, last_px, "本地止损（空）")
                                    for oid in (p.get('sl_order_id'), p.get('tp_order_id')):
                                        if oid:
                                            try:
                                                exchange.cancel_order(oid, symbol, params={})
                                            except Exception:
                                                pass
                                    positions.pop(symbol, None)
                                elif p.get('tp_price') and last_px <= float(p['tp_price']):
                                    logger.info(f"{symbol} local TP hit for short @ {last_px} <= {p['tp_price']} -> closing position")
                                    qty = p.get('qty')
                                    if qty and float(qty) > 0:
                                        try:
                                            exchange.create_market_order(symbol, 'buy', qty, params={'reduceOnly': True, 'tdMode': 'cross', 'posSide': 'short'})
                                        except Exception as e:
                                            logger.error(f"{symbol} local TP close error: {e}")
                                    update_stats_on_close(symbol, p, last_px, "本地止盈（空）")
                                    for oid in (p.get('sl_order_id'), p.get('tp_order_id')):
                                        if oid:
                                            try:
                                                exchange.cancel_order(oid, symbol, params={})
                                            except Exception:
                                                pass
                                    positions.pop(symbol, None)
                    except Exception as e:
                        logger.warning(f"{symbol} local SL/TP fallback check error: {e}")

                    # try to reconcile sl/tp orders; many ccxt implementations return dicts for created orders
                    p = positions[symbol]
                    # if exchange returned order objects with 'id' store them under sl_order_id/tp_order_id
                    if isinstance(p.get('sl_order'), dict) and not p.get('sl_order_id'):
                        p['sl_order_id'] = p['sl_order'].get('id') or p['sl_order'].get('algoId') or p['sl_order'].get('order_id')
                    if isinstance(p.get('tp_order'), dict) and not p.get('tp_order_id'):
                        p['tp_order_id'] = p['tp_order'].get('id') or p['tp_order'].get('algoId') or p['tp_order'].get('order_id')

                    # Check sl/tp order status if ids present
                    for key in ('sl_order_id','tp_order_id'):
                        oid = p.get(key)
                        if not oid:
                            continue
                        try:
                            oinfo = exchange.fetch_order(oid, symbol, params={})
                            status = oinfo.get('status') or oinfo.get('state') or oinfo.get('ordStatus')
                            if status and str(status).lower() in ('closed','filled','triggered','filled_with_trigger','filled'):
                                # one of protective orders triggered -> close position local and cancel other protective order if exists
                                logger.info(f"{symbol} protective order {key} executed (status={status}). Cleaning up local position.")
                                # 统计已实现盈亏（以当前价近似）
                                try:
                                    tkf = exchange.fetch_ticker(symbol)
                                    close_px = tkf.get('last') if tkf and 'last' in tkf else p.get('entry_price')
                                except Exception:
                                    close_px = p.get('entry_price')
                                update_stats_on_close(symbol, p, close_px, f"保护单触发（{key}）")
                                # cancel counterpart if exists
                                other = 'tp_order_id' if key == 'sl_order_id' else 'sl_order_id'
                                other_id = p.get(other)
                                if other_id:
                                    try:
                                        exchange.cancel_order(other_id, symbol, params={})
                                    except Exception as e:
                                        logger.warning(f"cancel other protective order error: {e}")
                                # remove local record
                                positions.pop(symbol, None)
                                break
                        except Exception as e:
                            # fetch_order may not support conditional ids; ignore or log
                            logger.debug(f"fetch_order for protective id {oid} error: {e}")
                            continue

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}\n{traceback.format_exc()}")

        # End for symbols

        # 循环末：对当前所有已记录持仓做保护单数量收敛，最多保留2个（SL+TP）
        try:
            for _sym, _p in list(positions.items()):
                try:
                    _pside = 'long' if _p.get('side') == 'buy' else 'short'
                    enforce_protection_limit(_sym, _pside, keep=2)
                except Exception:
                    continue
        except Exception:
            pass

        # 每60秒输出一次账户与持仓看板
        try:
            global last_summary_ts
            now_ts = time.time()
            if now_ts - last_summary_ts >= 60:
                # 重新取一次余额用于看板
                try:
                    free_usdt_summary = get_free_balance_usdt()
                except Exception:
                    free_usdt_summary = 0.0
                log_summary(free_usdt_summary)
                last_summary_ts = now_ts
        except Exception as _e:
            logger.debug(f"汇总看板输出异常：{_e}")

        # Sleep until next tick (but keep loop responsive)
        elapsed = time.time() - start_ts
        to_sleep = max(1, MAIN_LOOP_INTERVAL - elapsed)
        time.sleep(to_sleep)

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    try:
        logger.info("Starting K-line Momentum bot (exchange-side conditional orders). SANDBOX=%s" % (SANDBOX,))
        monitor_and_run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Exiting.")
    except Exception as e:
        logger.error(f"Fatal exception: {e}\n{traceback.format_exc()}")
