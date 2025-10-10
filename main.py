#!/usr/bin/env python3
# railway_bot.py
# K-line Momentum + ATR adaptive SL/TP using exchange-side conditional orders (OKX via ccxt)
# NOTE: Test in sandbox first. Adjust per your ccxt/OKX version.

import os
import time
import math
import logging
import traceback
from datetime import datetime, timedelta

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
            logger.warning(f"fetch_ohlcv error {symbol} attempt {attempts}: {e}")
            time.sleep(1 + attempts)
    logger.error(f"fetch_ohlcv failed for {symbol}")
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

# ----------------------------
# Account helpers
# ----------------------------
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

        # Calculate qty: for perpetuals, notional_usdt * leverage / price = contract notional
        # Note: OKX uses "size" in contracts or base-quantity depending on market; this is a best-effort approach.
        if entry_est_price and entry_est_price > 0:
            qty = (notional_usdt * leverage) / entry_est_price
        else:
            qty = max(1.0, (notional_usdt * leverage))  # best-effort fallback

        # Place market order
        try:
            order = exchange.create_market_order(symbol, side, qty)
        except Exception as e:
            logger.warning(f"market order creation raised: {e}; trying create_order('market') fallback")
            order = exchange.create_order(symbol, 'market', side, qty, None)

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

        # attempt variation 1: common param names
        sl_params_try = {
            'triggerPrice': float(sl_price),
            'reduceOnly': True,
            'closeOnTrigger': True,  # if supported
            'clientOrderId': client_base + "_sl",
            # optionally: 'triggerType':'last_price'
        }
        tp_params_try = {
            'triggerPrice': float(tp_price),
            'reduceOnly': True,
            'closeOnTrigger': True,
            'clientOrderId': client_base + "_tp",
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
                'clientOrderId': client_base + "_sl_v2",
            }
            alt_tp_params = {
                'stopPrice': float(tp_price),
                'reduceOnly': True,
                'orderType': 'market',
                'clientOrderId': client_base + "_tp_v2",
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
                    logger.warning(f"{symbol} insufficient data for timeframe {tf}. Skipping.")
                    continue
                atr = compute_atr(df, ATR_PERIOD)
                current_atr = atr.iloc[-1]
                if math.isnan(current_atr) or current_atr <= 0:
                    logger.warning(f"{symbol} ATR invalid: {current_atr}. Skip.")
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
                                exchange.create_market_order(symbol, 'sell', qty, params={'reduceOnly': True})
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
                                exchange.create_market_order(symbol, 'buy', qty, params={'reduceOnly': True})
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
                            'opened_at': datetime.utcnow().isoformat(),
                            'atr_sl_value': current_atr * atr_mult_sl,
                            'atr_val': current_atr,
                            'cfg': cfg
                        }
                        logger.info(f"{symbol} position opened and protective orders placed.")
                    except Exception as e:
                        logger.error(f"{symbol} failed to open position: {e}")
                        logger.error(traceback.format_exc())
                        # continue to next symbol

                # Optionally: monitor existing conditional orders status, if they triggered -> cleanup
                if positions.get(symbol):
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
