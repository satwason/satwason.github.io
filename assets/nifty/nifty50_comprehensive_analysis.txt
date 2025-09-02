# nifty50_comprehensive_analysis_improved.py
"""
Improved Nifty50 comprehensive analysis pipeline (final)
- All artifacts saved to `reports/` (preserves original filenames)
- Backtest with transaction costs, slippage, position sizing
- Optional GARCH volatility modeling (if `arch` installed), fallback to EWMA
- Experiment logging to CSV/Excel and SQLite (atomic xlsx append when possible)
- Data validation (basic ETL checks)
- ML pipeline using sklearn Pipeline and RandomizedSearchCV with TimeSeriesSplit
- Model artifact saving (full pipeline + metadata/model card)
- Trades export (CSV + XLSX), feature importance export (CSV + XLSX + PNG)
- Transaction-cost & slippage sensitivity sweep

Run as: python nifty50_comprehensive_analysis_improved.py
"""

import os
import json
import time
import warnings
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Optional arch import for GARCH modeling (not required)
try:
    from arch import arch_model
    HAS_ARCH = True
except Exception:
    HAS_ARCH = False

# Optional openpyxl for atomic xlsx append
try:
    from openpyxl import load_workbook
    HAS_OPENPYXL = True
except Exception:
    HAS_OPENPYXL = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import joblib

warnings.filterwarnings('ignore')

# --- Configuration ---
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Backtest configuration
DEFAULT_TRANSACTION_COST = 0.0005  # 5 bps per trade
DEFAULT_SLIPPAGE_PCT = 0.0005  # 5 bps slippage
MAX_POSITION_FRACTION = 0.5  # fraction of capital allowed in a single position
MINIMUM_CASH_RESERVE = 0.0  # fraction of capital to keep in cash

# SQL / experiment log path
EXPERIMENT_LOG_CSV = os.path.join(REPORTS_DIR, 'experiments.csv')
EXPERIMENT_LOG_XLSX = os.path.join(REPORTS_DIR, 'experiments.xlsx')
EXPERIMENT_DB = os.path.join(REPORTS_DIR, 'experiments.db')

plt.style.use('seaborn-v0_8')

# -----------------------------
# Helpers for atomic Excel append
# -----------------------------

def append_df_to_excel_xlsx(path, df, sheet_name='Sheet1'):
    """Append a dataframe to an xlsx file in an atomic way using openpyxl if available.
    If openpyxl not available, will overwrite by reading CSV fallback (less efficient).
    """
    if HAS_OPENPYXL:
        from openpyxl import load_workbook
        if not os.path.exists(path):
            df.to_excel(path, index=False)
            return
        book = load_workbook(path)
        with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # find last row and append - simple approach: read existing into df_all and write full
            # to avoid complexity we will read existing sheet to preserve simplicity for now
            try:
                df_all = pd.read_excel(path)
                df_out = pd.concat([df_all, df], ignore_index=True)
                df_out.to_excel(path, index=False)
            except Exception:
                # fallback to simple append by writing a new sheet
                df.to_excel(path, index=False, sheet_name=f'sheet_{int(time.time())}')
    else:
        # Fallback: write via CSV -> XLSX conversion
        try:
            # if file exists, read csv then append
            if os.path.exists(EXPERIMENT_LOG_CSV):
                df.to_csv(EXPERIMENT_LOG_CSV, mode='a', header=False, index=False)
            else:
                df.to_csv(EXPERIMENT_LOG_CSV, index=False)
            df_all = pd.read_csv(EXPERIMENT_LOG_CSV)
            df_all.to_excel(path, index=False)
        except Exception as e:
            print('Failed to append to xlsx atomically:', e)

# -----------------------------
# Data loading & validation
# -----------------------------

def load_nifty_data(file_path):
    """Load Nifty data from Excel and standardize columns."""
    try:
        print(f"Loading data from {file_path}...")
        data = pd.read_excel(file_path)
        print("Columns found:", data.columns.tolist())
        # Identify date column
        date_col = next((c for c in ['Date', 'DATE', 'date'] if c in data.columns), data.columns[0])
        data[date_col] = pd.to_datetime(data[date_col])
        data.set_index(date_col, inplace=True)
        # Normalize column names
        column_mapping = {}
        for col in data.columns:
            lc = col.lower()
            if 'open' in lc: column_mapping[col] = 'Open'
            elif 'high' in lc: column_mapping[col] = 'High'
            elif 'low' in lc: column_mapping[col] = 'Low'
            elif 'close' in lc: column_mapping[col] = 'Close'
            elif 'volume' in lc or 'vol' in lc: column_mapping[col] = 'Volume'
        data.rename(columns=column_mapping, inplace=True)
        # required columns
        for req in ['Open', 'High', 'Low', 'Close']:
            if req not in data.columns:
                raise ValueError(f"Missing required column: {req}")
        print(f"Loaded data: {data.index.min()} to {data.index.max()} ({len(data)} rows)")
        return data.sort_index()
    except Exception as e:
        print("Error loading data:", e)
        return None


def validate_data(df, freq='B'):
    """Run basic validation checks for time-series market data.
    - checks for duplicates, missing values, expected frequency gaps
    - returns dict of checks
    """
    checks = {}
    if df is None or df.empty:
        checks['ok'] = False
        checks['reason'] = 'empty dataframe'
        return checks
    checks['rows'] = len(df)
    checks['start'] = df.index.min()
    checks['end'] = df.index.max()
    # duplicates
    dup_count = df.index.duplicated().sum()
    checks['duplicates'] = int(dup_count)
    # missing values
    checks['na_summary'] = df.isna().sum().to_dict()
    # expected frequency gaps
    try:
        expected = pd.date_range(start=checks['start'], end=checks['end'], freq=freq)
        missing_dates = sorted(set(expected) - set(df.index))
        checks['missing_dates_count'] = len(missing_dates)
        checks['missing_dates_sample'] = missing_dates[:5]
    except Exception:
        checks['missing_dates_count'] = None
    checks['ok'] = (checks['duplicates'] == 0) and (sum(df.isna().sum()) == 0)
    # Save validation report
    out = os.path.join(REPORTS_DIR, 'data_validation.json')
    with open(out, 'w') as f:
        json.dump(checks, f, default=str, indent=2)
    print(f"Data validation saved to {out}")
    return checks

# -----------------------------
# Indicators & analysis
# -----------------------------

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calculate_vwma(df, period=20):
    return (df['Close'] * df['Volume']).rolling(period).sum() / df['Volume'].rolling(period).sum()


def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return sma + std * std_dev, sma - std * std_dev


def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_obv(df):
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()


def calculate_adx(df, period=14):
    df = df.copy()
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum((df['High'] - df['Close'].shift()).abs(), (df['Low'] - df['Close'].shift()).abs()))
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['DMplus'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['DMminus'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    tr_smooth = df['TR'].rolling(window=period).mean()
    dmplus_smooth = df['DMplus'].rolling(window=period).mean()
    dmminus_smooth = df['DMminus'].rolling(window=period).mean()
    di_plus = 100 * dmplus_smooth / tr_smooth
    di_minus = 100 * dmminus_smooth / tr_smooth
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()
    return adx, di_plus, di_minus


def analyze_nifty_data(df, period='all'):
    df = get_period_data(df, period)
    print(f"Analyzing {len(df)} rows: {df.index.min().date()} to {df.index.max().date()}")
    df = df.copy()
    df['Daily_Return'] = df['Close'].pct_change()
    df['EMA_12'] = calculate_ema(df['Close'], 12)
    df['EMA_26'] = calculate_ema(df['Close'], 26)
    df['MACD_Line'], df['MACD_Signal'], df['MACD_Histogram'] = calculate_macd(df['Close'])
    if 'Volume' in df.columns:
        df['VWMA_20'] = calculate_vwma(df, 20)
        df['OBV'] = calculate_obv(df)
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['RSI_14'] = calculate_rsi(df['Close'])
    df['ADX_14'], df['DIplus'], df['DIminus'] = calculate_adx(df)
    df['ATR_14'] = calculate_atr(df)
    df['Peak'] = df['Close'].cummax()
    df['Drawdown'] = (df['Close'] - df['Peak']) / df['Peak']
    return df, df['Daily_Return'].std() * np.sqrt(252)


def get_period_data(df, period):
    end = df.index.max()
    if period == 'all':
        return df.copy()
    mapping = {
        '10y': pd.DateOffset(years=10),
        '5y': pd.DateOffset(years=5),
        '3y': pd.DateOffset(years=3),
        '1y': pd.DateOffset(years=1),
        '6m': pd.DateOffset(months=6),
        '3m': pd.DateOffset(months=3),
        '1m': pd.DateOffset(months=1),
        '2w': pd.DateOffset(weeks=2)
    }
    start = end - mapping.get(period, pd.DateOffset(days=60))
    return df.loc[df.index >= start].copy()

# -----------------------------
# GARCH / volatility modeling
# -----------------------------

def compute_garch_volatility(series, p=1, q=1):
    if not HAS_ARCH:
        print('arch package not installed — using EWMA volatility as fallback')
        ewma_vol = series.ewm(span=20).std()
        return ewma_vol
    try:
        am = arch_model(series.dropna() * 100, vol='Garch', p=p, q=q, dist='normal')
        res = am.fit(disp='off')
        cond_vol = res.conditional_volatility / 100.0
        return cond_vol.reindex(series.index).fillna(method='ffill')
    except Exception as e:
        print('GARCH fit failed, using EWMA fallback:', e)
        return series.ewm(span=20).std()

# -----------------------------
# Improved backtest
# -----------------------------

def advanced_backtest(df, signals, initial_capital=100000, transaction_cost=DEFAULT_TRANSACTION_COST, slippage=DEFAULT_SLIPPAGE_PCT, max_position_fraction=MAX_POSITION_FRACTION):
    df = df.copy()
    signals = signals.reindex(df.index).fillna(0)
    cash = initial_capital
    position = 0.0
    equity = []
    trades = []
    price_series = df['Open'].fillna(df['Close'])

    for i in range(1, len(df)):
        idx = df.index[i]
        sig = signals['combined_signal'].iloc[i-1]
        price = price_series.iloc[i]
        max_position_value = initial_capital * max_position_fraction
        if sig > 2 and position == 0:
            target_value = max_position_value
            size = target_value / (price * (1 + slippage))
            exec_cost = target_value * transaction_cost
            slippage_cost = target_value * slippage
            cash -= (size * price) + exec_cost + slippage_cost
            position = size
            trades.append({'type': 'buy', 'date': idx, 'price': price*(1+slippage), 'size': size, 'cash': cash})
        elif sig < -2 and position > 0:
            proceeds = position * price * (1 - slippage)
            exec_cost = proceeds * transaction_cost
            cash += proceeds - exec_cost
            trades.append({'type': 'sell', 'date': idx, 'price': price*(1-slippage), 'size': position, 'cash': cash})
            position = 0
        current_equity = cash + position * price
        equity.append({'date': idx, 'equity': current_equity, 'cash': cash, 'position': position * price})

    equity_df = pd.DataFrame(equity).set_index('date') if equity else pd.DataFrame()
    final_value = cash + position * df['Close'].iloc[-1]
    total_return = (final_value / initial_capital - 1)
    return equity_df, trades, {'final_value': final_value, 'total_return': total_return}

# -----------------------------
# Signals
# -----------------------------

def generate_signals(df):
    s = pd.DataFrame(index=df.index)
    s['price'] = df['Close']
    s['macd_signal'] = 0
    macd_trend = df['MACD_Line'] > df['MACD_Signal']
    macd_mom = df['MACD_Histogram'] > 0
    s['macd_signal'] = np.where(macd_trend & macd_mom & (df['MACD_Line'] > 0), 2, 0)
    s['macd_signal'] = np.where((~macd_trend) & (~macd_mom) & (df['MACD_Line'] < 0), -2, s['macd_signal'])
    s['macd_signal'] = np.where(macd_trend & (df['MACD_Histogram'].diff() > 0), 1, s['macd_signal'])
    s['macd_signal'] = np.where((~macd_trend) & (df['MACD_Histogram'].diff() < 0), -1, s['macd_signal'])
    ema_trend = df['EMA_12'] > df['EMA_26']
    price_above_ema = df['Close'] > df['EMA_12']
    s['ema_signal'] = np.where(ema_trend & price_above_ema & (df['Close'] > df['EMA_26']), 2, 0)
    s['ema_signal'] = np.where((~ema_trend) & (~price_above_ema) & (df['Close'] < df['EMA_26']), -2, s['ema_signal'])
    s['rsi_signal'] = 0
    s['rsi_signal'] = np.where(df['RSI_14'] < 25, 2, 0)
    s['rsi_signal'] = np.where(df['RSI_14'] > 75, -2, s['rsi_signal'])
    s['rsi_signal'] = np.where((df['RSI_14'] < 40) & (df['RSI_14'].diff() > 0), 1, s['rsi_signal'])
    s['rsi_signal'] = np.where((df['RSI_14'] > 60) & (df['RSI_14'].diff() < 0), -1, s['rsi_signal'])
    s['adx_signal'] = np.where(df['ADX_14'] > 25, 1, 0)
    s['adx_signal'] = np.where(df['ADX_14'] > 40, 2, s['adx_signal'])
    if 'OBV' in df.columns:
        obv_trend = df['OBV'].pct_change(5) > 0
        s['volume_signal'] = np.where(obv_trend, 1, -1)
    else:
        s['volume_signal'] = 0
    weights = {'macd_signal': 1.5, 'ema_signal': 1.5, 'rsi_signal': 1.0, 'adx_signal': 1.0, 'volume_signal': 0.5}
    s['combined_signal'] = 0
    for k, w in weights.items():
        s['combined_signal'] += s[k] * w
    return s

# -----------------------------
# Performance metrics
# -----------------------------

def performance_metrics(equity_df, df):
    if equity_df.empty:
        return {}
    eq = equity_df['equity']
    returns = eq.pct_change().dropna()
    cum_return = eq.iloc[-1] / eq.iloc[0] - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    peak = eq.cummax()
    drawdown = (eq - peak) / peak
    max_dd = drawdown.min()
    return {'cumulative_return': cum_return, 'sharpe': sharpe, 'max_drawdown': max_dd}

# -----------------------------
# Experiment logging (CSV + XLSX + SQLite)
# -----------------------------

def log_experiment(name, params, metrics, path_csv=EXPERIMENT_LOG_CSV, path_xlsx=EXPERIMENT_LOG_XLSX, path_db=EXPERIMENT_DB):
    ts = pd.Timestamp.now()
    record = {'timestamp': ts, 'name': name, 'params': json.dumps(params), 'metrics': json.dumps(metrics)}
    df = pd.DataFrame([record])
    # Append to CSV
    if not os.path.exists(path_csv):
        df.to_csv(path_csv, index=False)
    else:
        df.to_csv(path_csv, mode='a', header=False, index=False)
    # Append to XLSX atomically if possible
    try:
        append_df_to_excel_xlsx(path_xlsx, df)
    except Exception as e:
        print('Failed to append experiment to xlsx atomically:', e)
        try:
            df_all = pd.read_csv(path_csv)
            df_all.to_excel(path_xlsx, index=False)
        except Exception as e2:
            print('Fallback write to xlsx failed:', e2)
    # SQLite
    conn = sqlite3.connect(path_db)
    try:
        df.to_sql('experiments', conn, if_exists='append', index=False)
    except Exception as e:
        print('Failed to write experiment log to sqlite:', e)
    finally:
        conn.close()
    print(f"Logged experiment '{name}' to {path_csv} and {path_db} (xlsx saved to {path_xlsx} if possible)")

# -----------------------------
# Machine learning pipeline
# -----------------------------

def prepare_ml_data(df, prediction_horizon=1):
    df = df.copy()
    df['Target'] = np.where(df['Close'].shift(-prediction_horizon) > df['Close'], 1, 0)
    features = ['Daily_Return', 'RSI_14', 'ADX_14', 'MACD_Line', 'EMA_12', 'EMA_26', 'VWMA_20', 'ATR_14']
    for feat in features:
        if feat in df.columns:
            for lag in [1, 2, 3, 5]:
                df[f'{feat}_Lag{lag}'] = df[feat].shift(lag)
    df['Volatility_5'] = df['Daily_Return'].rolling(5).std()
    df['Volatility_20'] = df['Daily_Return'].rolling(20).std()
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['Momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    df = df.dropna()
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=int), df
    feature_columns = [c for c in df.columns if ('Lag' in c) or c in ['Volatility_5', 'Volatility_20', 'Momentum_5', 'Momentum_20']]
    X = df[feature_columns].copy()
    y = df['Target'].astype(int).copy()
    return X, y, df


def train_ml_model_pipeline(X, y, n_iter=20):
    if X.empty or len(X) < 20:
        print('Not enough rows to train ML model reliably')
        return None, None, None
    tscv = TimeSeriesSplit(n_splits=5)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'))
    ])
    param_dist = {
        'clf__n_estimators': [100, 200, 400],
        'clf__max_depth': [3, 5, 8, None],
        'clf__min_samples_split': [2, 5, 10]
    }
    rsearch = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=n_iter, cv=tscv, scoring='accuracy', random_state=RANDOM_STATE, n_jobs=-1)
    rsearch.fit(X, y)
    best = rsearch.best_estimator_
    # persist
    model_out = os.path.join(REPORTS_DIR, 'nifty_ml_pipeline.pkl')
    joblib.dump({'model': best, 'cv_results': rsearch.cv_results_, 'best_params': rsearch.best_params_}, model_out)
    print(f"Saved ML pipeline to {model_out}")
    return best, rsearch.cv_results_, rsearch.best_params_


def evaluate_and_log_model(model, X, y, name='rf_ts_pipeline'):
    try:
        y_pred = model.predict(X)
    except NotFittedError:
        print('Model not fitted')
        return None
    acc = accuracy_score(y, y_pred)
    rep = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred).tolist()
    metrics = {'accuracy': acc, 'classification_report': rep, 'confusion_matrix': cm}
    # compute economic P&L by converting predictions into signals and running a quick backtest
    signals = pd.DataFrame(index=X.index)
    signals['combined_signal'] = np.where(y_pred == 1, 3, 0)  # treat as buy signal
    equity_df, trades, perf = advanced_backtest(df_global.loc[X.index], signals, initial_capital=100000)
    perf_metrics = performance_metrics(equity_df, df_global)
    metrics.update(perf_metrics)
    # log
    params = {'model': str(model)}
    log_experiment(name, params, metrics)
    # additionally save classification report to xlsx for easy review
    try:
        cr_df = pd.DataFrame(rep).T
        cr_out = os.path.join(REPORTS_DIR, 'classification_report.xlsx')
        cr_df.to_excel(cr_out)
        print('Saved classification report to', cr_out)
    except Exception as e:
        print('Failed to save classification report xlsx:', e)
    return metrics

# -----------------------------
# Feature importance export
# -----------------------------

def save_feature_importance(model, feature_names):
    try:
        clf = model.named_steps['clf']
        importances = clf.feature_importances_
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
        csv_out = os.path.join(REPORTS_DIR, 'feature_importance.csv')
        xlsx_out = os.path.join(REPORTS_DIR, 'feature_importance.xlsx')
        png_out = os.path.join(REPORTS_DIR, 'feature_importance.png')
        fi_df.to_csv(csv_out, index=False)
        fi_df.to_excel(xlsx_out, index=False)
        # plot
        plt.figure(figsize=(10, max(4, len(fi_df)*0.25)))
        top = fi_df.head(30)
        plt.barh(range(len(top)), top['importance'])
        plt.yticks(range(len(top)), top['feature'])
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.title('Feature Importance (Top 30)')
        plt.tight_layout()
        plt.savefig(png_out, dpi=300)
        plt.close()
        print('Saved feature importance to', csv_out, xlsx_out, png_out)
        # also save as JSON model card entry
        return fi_df
    except Exception as e:
        print('Failed to save feature importance:', e)
        return None

# -----------------------------
# Transaction-cost sensitivity sweep
# -----------------------------

def transaction_cost_sweep(dfp, signals, tc_list, slippage_list):
    rows = []
    for tc in tc_list:
        for sl in slippage_list:
            equity_df, trades, perf = advanced_backtest(dfp, signals, transaction_cost=tc, slippage=sl)
            perf_metrics = performance_metrics(equity_df, dfp)
            rows.append({'transaction_cost': tc, 'slippage': sl, 'final_value': perf.get('final_value'), 'total_return': perf.get('total_return'), **perf_metrics})
    res_df = pd.DataFrame(rows)
    csv_out = os.path.join(REPORTS_DIR, 'tc_sensitivity.csv')
    xlsx_out = os.path.join(REPORTS_DIR, 'tc_sensitivity.xlsx')
    res_df.to_csv(csv_out, index=False)
    try:
        res_df.to_excel(xlsx_out, index=False)
    except Exception as e:
        print('Failed to save tc sensitivity xlsx:', e)
    print('Saved transaction-cost sensitivity results to', csv_out, xlsx_out)
    # Log summary experiment
    summary_metrics = {'best_return': float(res_df['total_return'].max()), 'best_params_row': res_df.loc[res_df['total_return'].idxmax()].to_dict()}
    log_experiment('tc_sensitivity_sweep', {'tc_list': tc_list, 'sl_list': slippage_list}, summary_metrics)
    return res_df

# -----------------------------
# Visualization helpers
# -----------------------------

def create_visualizations(df, period):
    fig = plt.figure(figsize=(18, 16))
    ax1 = plt.subplot(6, 1, 1)
    ax1.plot(df.index, df['Close'], label='Close')
    ax1.plot(df.index, df['EMA_12'], label='EMA 12')
    ax1.plot(df.index, df['EMA_26'], label='EMA 26')
    if 'VWMA_20' in df.columns:
        ax1.plot(df.index, df['VWMA_20'], label='VWMA 20')
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.2)
    ax1.set_title(f'Price & Moving Averages ({period})')
    ax1.legend()

    ax2 = plt.subplot(6, 1, 2)
    ax2.plot(df.index, df['MACD_Line'], label='MACD')
    ax2.plot(df.index, df['MACD_Signal'], label='Signal')
    ax2.bar(df.index, df['MACD_Histogram'], label='Hist', alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(6, 1, 3)
    ax3.plot(df.index, df['RSI_14'], label='RSI')
    ax3.axhline(70, linestyle='--')
    ax3.axhline(30, linestyle='--')
    ax3.set_ylim(0, 100)

    ax4 = plt.subplot(6, 1, 4)
    ax4.plot(df.index, df['ADX_14'], label='ADX')
    ax4.plot(df.index, df['DIplus'], label='+DI')
    ax4.plot(df.index, df['DIminus'], label='-DI')
    ax4.legend()

    ax5 = plt.subplot(6, 1, 5)
    if 'OBV' in df.columns:
        ax5.plot(df.index, df['OBV'], label='OBV')
        ax5.legend()

    ax6 = plt.subplot(6, 1, 6)
    ax6.fill_between(df.index, df['Drawdown'], 0, alpha=0.3)
    ax6.plot(df.index, df['Drawdown'], label='Drawdown')
    ax6.legend()

    plt.tight_layout()
    outp = os.path.join(REPORTS_DIR, f'nifty50_advanced_analysis_{period}.png')
    fig.savefig(outp, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved visualization to {outp}")
    return outp

# -----------------------------
# Main execution
# -----------------------------

def main():
    global df_global
    print('='*70)
    print('COMPREHENSIVE NIFTY 50 ANALYSIS - FINAL')
    print('='*70)

    file_path = 'Nifty_50_Historical_Data.xlsx'
    df = load_nifty_data(file_path)
    if df is None:
        return
    df_global = df.copy()

    # validation
    val = validate_data(df)

    periods = ['1y', '6m', '3m', '1m']
    for period in periods:
        print('\n' + '-'*50)
        print(f'ANALYZING PERIOD: {period}')
        print('-'*50)
        dfp, vol = analyze_nifty_data(df, period)
        # compute optional GARCH volatility column
        dfp['GARCH_VOL'] = compute_garch_volatility(dfp['Daily_Return'])
        signals = generate_signals(dfp)
        # Save period dataset
        out_xlsx = os.path.join(REPORTS_DIR, f'nifty_analysis_{period}.xlsx')
        dfp.to_excel(out_xlsx)
        print('Saved dataset to', out_xlsx)
        # visualization
        create_visualizations(dfp, period)
        # Backtest
        equity_df, trades, perf = advanced_backtest(dfp, signals)
        perf_metrics = performance_metrics(equity_df, dfp)
        print('Backtest perf:', perf_metrics)
        # save equity curve to both CSV & XLSX
        if not equity_df.empty:
            equity_file_csv = os.path.join(REPORTS_DIR, f'equity_curve_{period}.csv')
            equity_file_xlsx = os.path.join(REPORTS_DIR, f'equity_curve_{period}.xlsx')
            equity_df.to_csv(equity_file_csv)
            try:
                equity_df.to_excel(equity_file_xlsx)
            except Exception as e:
                print('Failed to save equity curve xlsx:', e)
            print('Saved equity curve to', equity_file_csv, 'and', equity_file_xlsx)
        # save trades if any
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_csv = os.path.join(REPORTS_DIR, f'trades_{period}.csv')
            trades_xlsx = os.path.join(REPORTS_DIR, f'trades_{period}.xlsx')
            trades_df.to_csv(trades_csv, index=False)
            try:
                trades_df.to_excel(trades_xlsx, index=False)
            except Exception as e:
                print('Failed to save trades xlsx:', e)
            print('Saved trades to', trades_csv, 'and', trades_xlsx)
        # transaction-cost sensitivity sweep
        tc_list = [0.0, DEFAULT_TRANSACTION_COST, 0.001, 0.0025]
        sl_list = [0.0, DEFAULT_SLIPPAGE_PCT, 0.001]
        tc_res = transaction_cost_sweep(dfp, signals, tc_list, sl_list)
        # log experiment
        params = {'period': period, 'transaction_cost': DEFAULT_TRANSACTION_COST, 'slippage': DEFAULT_SLIPPAGE_PCT}
        metrics = {'final_value': perf.get('final_value'), 'total_return': perf.get('total_return'), **perf_metrics}
        log_experiment(f'strategy_{period}', params, metrics)

    # SQL database (full 1y snapshot)
    analyzed_1y, _ = analyze_nifty_data(df, '1y')
    db_file = os.path.join(REPORTS_DIR, 'nifty_analysis.db')
    conn = sqlite3.connect(db_file)
    analyzed_1y.to_sql('nifty_daily', conn, if_exists='replace')
    conn.close()
    print('Saved sqlite DB to', db_file)

    # Statistical analysis on 1y
    stats_results = perform_statistical_analysis(analyzed_1y)
    regression_model = time_series_regression(analyzed_1y)
    print_statistical_results(stats_results, regression_model)

    # Machine Learning
    X, y, ml_df = prepare_ml_data(analyzed_1y)
    model, cv_results, best_params = train_ml_model_pipeline(X, y, n_iter=25)
    if model is not None:
        # Save feature importance (needs feature names)
        fi_df = save_feature_importance(model, X.columns.tolist())
        # Save model card / metadata
        model_card = {
            'timestamp': str(pd.Timestamp.now()),
            'model_path': os.path.join(REPORTS_DIR, 'nifty_ml_pipeline.pkl'),
            'features': X.columns.tolist(),
            'best_params': best_params,
        }
        with open(os.path.join(REPORTS_DIR, 'model_card.json'), 'w') as f:
            json.dump(model_card, f, indent=2)
        print('Saved model card to', os.path.join(REPORTS_DIR, 'model_card.json'))
        metrics = evaluate_and_log_model(model, X, y)
        print('ML metrics saved to experiments log')

    # Save full indicators file
    out_all = os.path.join(REPORTS_DIR, 'nifty50_with_indicators.xlsx')
    df_all, _ = analyze_nifty_data(df, 'all')
    df_all.to_excel(out_all)
    print('Saved full indicators file to', out_all)

    # Save environment metadata
    meta = {
        'timestamp': str(pd.Timestamp.now()),
        'rows': int(len(df)),
        'python_version': os.sys.version,
        'has_arch': HAS_ARCH,
        'has_openpyxl': HAS_OPENPYXL
    }
    with open(os.path.join(REPORTS_DIR, 'run_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('Run complete — metadata saved')

# -----------------------------
# Statistical / regression / ML helper functions re-used from original script
# -----------------------------

def perform_statistical_analysis(data):
    results = {}
    returns = data['Daily_Return'].dropna()
    if len(returns) < 10:
        print('Not enough returns for statistical tests')
        return {}
    sw = stats.shapiro(returns)
    jb = stats.jarque_bera(returns)
    results['normality'] = {'shapiro_stat': float(sw[0]), 'shapiro_pvalue': float(sw[1]), 'jarque_bera': float(jb[0]), 'jarque_bera_pvalue': float(jb[1])}
    adf_res = adfuller(returns)
    results['adf'] = {'adf_stat': float(adf_res[0]), 'pvalue': float(adf_res[1]), 'crit_vals': adf_res[4]}
    sq = returns ** 2
    lb = sm.stats.diagnostic.acorr_ljungbox(sq, lags=[5, 10, 20])
    results['ljung_box'] = lb
    corr_data = data[['Daily_Return', 'Volume', 'RSI_14', 'ADX_14']].dropna()
    results['correlation_matrix'] = corr_data.corr().to_dict()
    ttest = stats.ttest_1samp(returns, 0)
    results['mean_return_test'] = {'t_stat': float(ttest.statistic), 'pvalue': float(ttest.pvalue)}
    return results


def time_series_regression(data):
    analysis_data = data[['Daily_Return', 'RSI_14', 'ADX_14', 'MACD_Line']].dropna()
    analysis_data['Return_Lag1'] = analysis_data['Daily_Return'].shift(1)
    analysis_data['RSI_Lag1'] = analysis_data['RSI_14'].shift(1)
    analysis_data = analysis_data.dropna()
    X = analysis_data[['RSI_Lag1', 'ADX_14', 'MACD_Line', 'Return_Lag1']]
    X = sm.add_constant(X)
    y = analysis_data['Daily_Return']
    model = sm.OLS(y, X).fit()
    return model


def print_statistical_results(results, regression_model):
    print('\n' + '='*60)
    print('STATISTICAL ANALYSIS RESULTS')
    print('='*60)
    if not results:
        print('No statistical results to show')
        return
    nr = results.get('normality', {})
    print('Shapiro-Wilk p-value:', nr.get('shapiro_pvalue'))
    print('Jarque-Bera p-value:', nr.get('jarque_bera_pvalue'))
    adf = results.get('adf', {})
    print('ADF statistic:', adf.get('adf_stat'))
    mrt = results.get('mean_return_test', {})
    print('Mean return t-test p-value:', mrt.get('pvalue'))
    print('\nCorrelation matrix (sample):')
    print(pd.DataFrame(results.get('correlation_matrix', {})).round(3).head())
    print('\nRegression summary (head):')
    try:
        print(regression_model.summary())
    except Exception as e:
        print('Could not print regression summary:', e)

# -----------------------------
# Entry point
# -----------------------------

if __name__ == '__main__':
    main()
