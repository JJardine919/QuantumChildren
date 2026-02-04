# ================================================
# ai_trader_quantum_fusion_3d.py  
# КВАНТОВЫЙ ГИБРИД + 3D БАРЫ: Qiskit + CatBoost + LLM + 3D Analysis
# Версия 14.12.2025 — 3D только для CatBoost, не для LLM
# ================================================
import os
import re
import time
import json
import logging
import subprocess
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost не установлен: pip install catboost")

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from scipy.stats import entropy
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️ Qiskit не установлен: pip install qiskit qiskit-aer")

from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# ====================== КОНФИГ ======================
MODEL_NAME = "koshtenco/quantum-trader-fusion-3d"
BASE_MODEL = "llama3.2:3b"
SYMBOLS = ["EURUSD", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURGBP", "AUDCHF"]
TIMEFRAME = mt5.TIMEFRAME_M15 if mt5 else None
LOOKBACK = 400
INITIAL_BALANCE = 140.0
RISK_PER_TRADE = 0.01
MIN_PROB = 60
LIVE_LOT = 0.02
MAGIC = 20251214
SLIPPAGE = 10

# Квантовые параметры
N_QUBITS = 8
N_SHOTS = 2048

# 3D бары параметры
MIN_SPREAD_MULTIPLIER = 45
VOLUME_BRICK = 500
USE_3D_BARS = True  # Флаг включения 3D-баров

# Файнтьюн параметры
FINETUNE_SAMPLES = 2000
BACKTEST_DAYS = 30
PREDICTION_HORIZON = 96  # 24 часа на M15

os.makedirs("logs", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/quantum_fusion_3d.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ====================== 3D БАРЫ МОДУЛЬ ======================
class Bars3D:
    """
    Класс для создания стационарных 4D признаков (3D бары)
    Реализация из статьи о многомерных барах
    """
    
    def __init__(self, min_spread_multiplier: int = 45, volume_brick: int = 500):
        self.min_spread_multiplier = min_spread_multiplier
        self.volume_brick = volume_brick
        self.scaler = MinMaxScaler(feature_range=(3, 9))
        
    def create_3d_features(self, df: pd.DataFrame, symbol_info=None) -> pd.DataFrame:
        """
        Создаёт стационарные 4D признаки из обычных OHLCV данных
        """
        if len(df) < 21:
            log.warning("Недостаточно данных для 3D-баров (нужно минимум 21 бар)")
            return df
        
        d = df.copy()
        
        # Временное измерение (циклические признаки)
        if 'time' in d.columns or isinstance(d.index, pd.DatetimeIndex):
            if isinstance(d.index, pd.DatetimeIndex):
                d['time_sin'] = np.sin(2 * np.pi * d.index.hour / 24)
                d['time_cos'] = np.cos(2 * np.pi * d.index.hour / 24)
            else:
                if not pd.api.types.is_datetime64_any_dtype(d['time']):
                    d['time'] = pd.to_datetime(d['time'], unit='s')
                d['time_sin'] = np.sin(2 * np.pi * d['time'].dt.hour / 24)
                d['time_cos'] = np.cos(2 * np.pi * d['time'].dt.hour / 24)
        else:
            d['time_sin'] = 0
            d['time_cos'] = 0
        
        # Ценовое измерение
        d['typical_price'] = (d['high'] + d['low'] + d['close']) / 3
        d['price_return'] = d['typical_price'].pct_change()
        d['price_acceleration'] = d['price_return'].diff()
        
        # Объёмное измерение
        d['volume_change'] = d['tick_volume'].pct_change()
        d['volume_acceleration'] = d['volume_change'].diff()
        
        # Измерение волатильности
        d['volatility'] = d['price_return'].rolling(20).std()
        d['volatility_change'] = d['volatility'].pct_change()
        
        # Создаём нормализованные признаки в скользящем окне
        bar3d_features = []
        
        for idx in range(20, len(d)):
            window = d.iloc[idx-20:idx+1]
            
            features = {
                'bar3d_price_return': float(window['price_return'].iloc[-1]),
                'bar3d_price_accel': float(window['price_acceleration'].iloc[-1]),
                'bar3d_volume_change': float(window['volume_change'].iloc[-1]),
                'bar3d_volatility_change': float(window['volatility_change'].iloc[-1]),
                'bar3d_volume_accel': float(window['volume_acceleration'].iloc[-1]),
                'bar3d_time_sin': float(d.iloc[idx]['time_sin']),
                'bar3d_time_cos': float(d.iloc[idx]['time_cos']),
                'bar3d_price_velocity': float(window['price_acceleration'].mean()),
                'bar3d_volume_intensity': float(window['volume_change'].mean()),
                'bar3d_price_change_mean': float(window['price_return'].mean()),
            }
            
            bar3d_features.append(features)
        
        # Добавляем NaN для первых 20 баров
        for _ in range(20):
            bar3d_features.insert(0, {k: np.nan for k in bar3d_features[0].keys() if k in bar3d_features[0]}) if bar3d_features else bar3d_features.insert(0, {})
        
        # Преобразуем в DataFrame и объединяем
        bar3d_df = pd.DataFrame(bar3d_features)
        result = pd.concat([d.reset_index(drop=True), bar3d_df], axis=1)
        
        # Нормализация в диапазон 3-9
        cols_to_scale = [col for col in bar3d_df.columns if col.startswith('bar3d_')]
        if cols_to_scale:
            result[cols_to_scale] = result[cols_to_scale].bfill().fillna(0)
            
            mask = result[cols_to_scale].abs().sum(axis=1) > 0
            if mask.sum() > 0:
                result.loc[mask, cols_to_scale] = self.scaler.fit_transform(
                    result.loc[mask, cols_to_scale]
                )
        
        # Дополнительные метрики
        result['bar3d_ma_5'] = result.get('bar3d_volatility_change', pd.Series(0, index=result.index)).rolling(5).mean()
        result['bar3d_ma_20'] = result.get('bar3d_volatility_change', pd.Series(0, index=result.index)).rolling(20).mean()
        result['bar3d_price_volatility'] = result.get('bar3d_price_change_mean', pd.Series(0, index=result.index)).rolling(10).std()
        result['bar3d_volume_volatility'] = result.get('bar3d_volume_change', pd.Series(0, index=result.index)).rolling(10).std()
        
        # Направление тренда
        result['bar3d_direction'] = np.sign(result.get('bar3d_price_return', 0))
        
        # Счётчик тренда
        trend_count = []
        count = 1
        prev_dir = 0
        
        for direction in result['bar3d_direction']:
            if pd.isna(direction):
                trend_count.append(0)
                continue
            
            if direction == prev_dir:
                count += 1
            else:
                count = 1
            
            trend_count.append(count)
            prev_dir = direction
        
        result['bar3d_trend_count'] = trend_count
        result['bar3d_trend_strength'] = result['bar3d_trend_count'] * result['bar3d_direction']
        
        # ВАЖНО: Детектор желтых кластеров (предиктор разворотов)
        result['bar3d_yellow_cluster'] = (
            (result['bar3d_price_volatility'] > result['bar3d_price_volatility'].quantile(0.7)) &
            (result['bar3d_volume_volatility'] > result['bar3d_volume_volatility'].quantile(0.7))
        ).astype(float)
        
        # Вероятность разворота на основе желтых кластеров
        result['bar3d_reversal_prob'] = result['bar3d_yellow_cluster'].rolling(7, center=True).mean()
        
        # Заполняем NaN
        result = result.bfill().fillna(0)
        
        log.info(f"✓ Созданы 3D-бары: {len([c for c in result.columns if c.startswith('bar3d_')])} признаков")
        
        return result

# ====================== КВАНТОВЫЙ ЭНКОДЕР ======================
class QuantumEncoder:
    """Квантовый энкодер на базе Qiskit"""
    
    def __init__(self, n_qubits: int = 8, n_shots: int = 2048):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.simulator = AerSimulator() if QISKIT_AVAILABLE else None
        
    def encode_and_measure(self, features: np.ndarray) -> Dict[str, float]:
        """Кодирует признаки в квантовую схему"""
        if not QISKIT_AVAILABLE or self.simulator is None:
            return {
                'quantum_entropy': np.random.uniform(2.0, 5.0),
                'dominant_state_prob': np.random.uniform(0.05, 0.20),
                'significant_states': np.random.randint(3, 20),
                'quantum_variance': np.random.uniform(0.001, 0.01)
            }
        
        normalized = (features - features.min()) / (features.max() - features.min() + 1e-8)
        angles = normalized * np.pi
        
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        
        for i in range(min(len(angles), self.n_qubits)):
            qc.ry(angles[i], i)
        
        for i in range(self.n_qubits - 1):
            qc.cz(i, i + 1)
        qc.cz(self.n_qubits - 1, 0)
        
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        
        job = self.simulator.run(qc, shots=self.n_shots)
        result = job.result()
        counts = result.get_counts()
        
        total_shots = sum(counts.values())
        probabilities = np.array([counts.get(format(i, f'0{self.n_qubits}b'), 0) / total_shots 
                                  for i in range(2**self.n_qubits)])
        
        quantum_entropy = entropy(probabilities + 1e-10, base=2)
        dominant_state_prob = np.max(probabilities)
        significant_states = np.sum(probabilities > 0.03)
        quantum_variance = np.var(probabilities)
        
        return {
            'quantum_entropy': quantum_entropy,
            'dominant_state_prob': dominant_state_prob,
            'significant_states': significant_states,
            'quantum_variance': quantum_variance
        }

# ====================== ТЕХНИЧЕСКИЕ ПРИЗНАКИ + 3D БАРЫ ======================
def calculate_features(df: pd.DataFrame, bars_3d: Bars3D = None, symbol_info=None) -> pd.DataFrame:
    """Расчёт технических индикаторов + 3D-бары"""
    d = df.copy()
    d["close_prev"] = d["close"].shift(1)
    
    # ATR
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close_prev"]).abs(),
        (d["low"] - d["close_prev"]).abs(),
    ], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    
    # RSI
    delta = d["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    
    # Объёмы
    d["vol_avg_20"] = d["tick_volume"].rolling(20).mean()
    d["vol_ratio"] = d["tick_volume"] / d["vol_avg_20"].replace(0, np.nan)
    
    # Bollinger Bands
    d["BB_middle"] = d["close"].rolling(20).mean()
    bb_std = d["close"].rolling(20).std()
    d["BB_upper"] = d["BB_middle"] + 2 * bb_std
    d["BB_lower"] = d["BB_middle"] - 2 * bb_std
    d["BB_position"] = (d["close"] - d["BB_lower"]) / (d["BB_upper"] - d["BB_lower"])
    
    # Stochastic
    low_14 = d["low"].rolling(14).min()
    high_14 = d["high"].rolling(14).max()
    d["Stoch_K"] = 100 * (d["close"] - low_14) / (high_14 - low_14)
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()
    
    # EMA кросс
    d["EMA_50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["EMA_200"] = d["close"].ewm(span=200, adjust=False).mean()
    
    # Дополнительные признаки
    d["price_change_1"] = d["close"].pct_change(1)
    d["price_change_5"] = d["close"].pct_change(5)
    d["price_change_21"] = d["close"].pct_change(21)
    d["log_return"] = np.log(d["close"] / d["close"].shift(1))
    d["volatility_20"] = d["log_return"].rolling(20).std()
    
    # ДОБАВЛЯЕМ 3D-БАРЫ
    if USE_3D_BARS and bars_3d is not None:
        log.info("Добавление 3D-баров признаков...")
        d = bars_3d.create_3d_features(d, symbol_info)
    
    return d.dropna()

# ====================== ОБУЧЕНИЕ CATBOOST С 3D-БАРАМИ ======================
def train_catboost_model(data_dict: Dict[str, pd.DataFrame], 
                        quantum_encoder: QuantumEncoder,
                        bars_3d: Bars3D = None) -> CatBoostClassifier:
    """Обучает CatBoost на данных с квантовыми признаками + 3D-барами"""
    print(f"\n{'='*80}")
    print(f"ОБУЧЕНИЕ CATBOOST С КВАНТОВЫМИ ПРИЗНАКАМИ + 3D-БАРАМИ")
    print(f"{'='*80}\n")
    
    if not CATBOOST_AVAILABLE:
        print("❌ CatBoost недоступен")
        return None
    
    all_features = []
    all_targets = []
    all_symbols = []
    
    print("Подготовка данных с квантовым кодированием и 3D-барами...")
    
    for symbol, df in data_dict.items():
        print(f"\nОбработка {symbol}: {len(df)} баров")
        
        symbol_info = None
        if mt5 and mt5.initialize():
            symbol_info = mt5.symbol_info(symbol)
            mt5.shutdown()
        
        df_features = calculate_features(df, bars_3d, symbol_info)
        
        quantum_features_list = []
        
        for idx in range(LOOKBACK, len(df_features) - PREDICTION_HORIZON):
            if idx % 500 == 0:
                print(f" Квантовое кодирование: {idx}/{len(df_features) - PREDICTION_HORIZON}")
            
            row = df_features.iloc[idx]
            
            feature_vector = np.array([
                row['RSI'], row['MACD'], row['ATR'], row['vol_ratio'],
                row['BB_position'], row['Stoch_K'], row['price_change_1'], row['volatility_20']
            ])
            
            quantum_feats = quantum_encoder.encode_and_measure(feature_vector)
            quantum_features_list.append(quantum_feats)
            
            future_idx = idx + PREDICTION_HORIZON
            future_price = df_features.iloc[future_idx]['close']
            current_price = row['close']
            target = 1 if future_price > current_price else 0
            
            # Собираем все признаки
            features = {
                'RSI': row['RSI'],
                'MACD': row['MACD'],
                'ATR': row['ATR'],
                'vol_ratio': row['vol_ratio'],
                'BB_position': row['BB_position'],
                'Stoch_K': row['Stoch_K'],
                'Stoch_D': row['Stoch_D'],
                'EMA_50': row['EMA_50'],
                'EMA_200': row['EMA_200'],
                'price_change_1': row['price_change_1'],
                'price_change_5': row['price_change_5'],
                'price_change_21': row['price_change_21'],
                'volatility_20': row['volatility_20'],
                'quantum_entropy': quantum_feats['quantum_entropy'],
                'dominant_state_prob': quantum_feats['dominant_state_prob'],
                'significant_states': quantum_feats['significant_states'],
                'quantum_variance': quantum_feats['quantum_variance'],
                'symbol': symbol
            }
            
            # Добавляем 3D-бары признаки если доступны
            if USE_3D_BARS and 'bar3d_price_return' in row:
                features.update({
                    'bar3d_yellow_cluster': row.get('bar3d_yellow_cluster', 0),
                    'bar3d_reversal_prob': row.get('bar3d_reversal_prob', 0),
                    'bar3d_trend_strength': row.get('bar3d_trend_strength', 0),
                    'bar3d_price_volatility': row.get('bar3d_price_volatility', 0),
                    'bar3d_volume_volatility': row.get('bar3d_volume_volatility', 0),
                })
            
            all_features.append(features)
            all_targets.append(target)
            all_symbols.append(symbol)
    
    print(f"\n✓ Всего примеров: {len(all_features)}")
    
    X = pd.DataFrame(all_features)
    y = np.array(all_targets)
    
    X = pd.get_dummies(X, columns=['symbol'], prefix='sym')
    
    print(f"✓ Признаков: {len(X.columns)}")
    print(f"✓ Баланс классов: UP={np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%), DOWN={np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
    
    print("\nОбучение CatBoost...")
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=500
    )
    
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    
    accuracies = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n--- Фолд {fold_idx + 1}/3 ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        accuracy = model.score(X_val, y_val)
        accuracies.append(accuracy)
        print(f"Фолд {fold_idx + 1} Accuracy: {accuracy*100:.2f}%")
    
    print(f"\n{'='*80}")
    print(f"РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ")
    print(f"{'='*80}")
    print(f"Средняя точность: {np.mean(accuracies)*100:.2f}% ± {np.std(accuracies)*100:.2f}%")
    
    print("\nОбучение финальной модели на всех данных...")
    model.fit(X, y, verbose=500)
    
    model_path = "models/catboost_quantum_3d.cbm"
    model.save_model(model_path)
    print(f"\n✓ Модель сохранена: {model_path}")
    
    feature_importance = model.get_feature_importance()
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
    print(importance_df.head(10).to_string(index=False))
    
    # Проверяем 3D-бары в топе
    if USE_3D_BARS:
        bar3d_features = importance_df[importance_df['feature'].str.startswith('bar3d_')]
        if len(bar3d_features) > 0:
            print(f"\n3D-БАРЫ В ТОПЕ ({len(bar3d_features)} признаков):")
            print(bar3d_features.head(10).to_string(index=False))
    
    return model

# ====================== ГЕНЕРАЦИЯ ГИБРИДНОГО ДАТАСЕТА БЕЗ 3D В ПРОМПТАХ ======================
def generate_hybrid_dataset(
    data_dict: Dict[str, pd.DataFrame],
    catboost_model: CatBoostClassifier,
    quantum_encoder: QuantumEncoder,
    bars_3d: Bars3D,
    num_samples: int = 2000
) -> List[Dict]:
    """Генерирует датасет для LLM (3D только для CatBoost, не для промптов)"""
    print(f"\n{'='*80}")
    print(f"ГЕНЕРАЦИЯ ГИБРИДНОГО ДАТАСЕТА (CatBoost + Quantum)")
    print(f"{'='*80}\n")
    
    dataset = []
    up_count = 0
    down_count = 0
    
    target_per_symbol = num_samples // len(SYMBOLS)
    
    for symbol, df in data_dict.items():
        print(f"Обработка {symbol}...")
        
        symbol_info = None
        if mt5 and mt5.initialize():
            symbol_info = mt5.symbol_info(symbol)
            mt5.shutdown()
        
        df_features = calculate_features(df, bars_3d, symbol_info)
        
        candidates = []
        
        for idx in range(LOOKBACK, len(df_features) - PREDICTION_HORIZON):
            row = df_features.iloc[idx]
            future_idx = idx + PREDICTION_HORIZON
            future_row = df_features.iloc[future_idx]
            
            feature_vector = np.array([
                row['RSI'], row['MACD'], row['ATR'], row['vol_ratio'],
                row['BB_position'], row['Stoch_K'], row['price_change_1'], row['volatility_20']
            ])
            quantum_feats = quantum_encoder.encode_and_measure(feature_vector)
            
            X_features = {
                'RSI': row['RSI'],
                'MACD': row['MACD'],
                'ATR': row['ATR'],
                'vol_ratio': row['vol_ratio'],
                'BB_position': row['BB_position'],
                'Stoch_K': row['Stoch_K'],
                'Stoch_D': row['Stoch_D'],
                'EMA_50': row['EMA_50'],
                'EMA_200': row['EMA_200'],
                'price_change_1': row['price_change_1'],
                'price_change_5': row['price_change_5'],
                'price_change_21': row['price_change_21'],
                'volatility_20': row['volatility_20'],
                'quantum_entropy': quantum_feats['quantum_entropy'],
                'dominant_state_prob': quantum_feats['dominant_state_prob'],
                'significant_states': quantum_feats['significant_states'],
                'quantum_variance': quantum_feats['quantum_variance'],
            }
            
            # Добавляем 3D признаки
            if USE_3D_BARS and 'bar3d_yellow_cluster' in row:
                X_features.update({
                    'bar3d_yellow_cluster': row.get('bar3d_yellow_cluster', 0),
                    'bar3d_reversal_prob': row.get('bar3d_reversal_prob', 0),
                    'bar3d_trend_strength': row.get('bar3d_trend_strength', 0),
                    'bar3d_price_volatility': row.get('bar3d_price_volatility', 0),
                    'bar3d_volume_volatility': row.get('bar3d_volume_volatility', 0),
                })
            
            X_df = pd.DataFrame([X_features])
            for s in SYMBOLS:
                X_df[f'sym_{s}'] = 1 if s == symbol else 0
            
            if catboost_model:
                proba = catboost_model.predict_proba(X_df)[0]
                catboost_prob_up = proba[1] * 100
                catboost_direction = "UP" if proba[1] > 0.5 else "DOWN"
                catboost_confidence = max(proba) * 100
            else:
                catboost_prob_up = 50.0
                catboost_direction = "UP"
                catboost_confidence = 50.0
            
            actual_price_24h = future_row['close']
            price_change = actual_price_24h - row['close']
            price_change_pips = int(price_change / 0.0001)
            actual_direction = "UP" if price_change > 0 else "DOWN"
            
            # Проверяем желтый кластер (только для внутренней статистики)
            has_yellow_cluster = row.get('bar3d_yellow_cluster', 0) > 0.5
            reversal_prob = row.get('bar3d_reversal_prob', 0)
            
            candidates.append({
                'symbol': symbol,
                'row': row,
                'future_row': future_row,
                'quantum_feats': quantum_feats,
                'catboost_direction': catboost_direction,
                'catboost_confidence': catboost_confidence,
                'catboost_prob_up': catboost_prob_up,
                'actual_direction': actual_direction,
                'price_change_pips': price_change_pips,
                'current_time': df.index[idx] if isinstance(df.index, pd.DatetimeIndex) else df.iloc[idx].name,
                'has_yellow_cluster': has_yellow_cluster,
                'reversal_prob': reversal_prob
            })
        
        up_candidates = [c for c in candidates if c['actual_direction'] == 'UP']
        down_candidates = [c for c in candidates if c['actual_direction'] == 'DOWN']
        
        target_up = target_per_symbol // 2
        target_down = target_per_symbol // 2
        
        selected_up = np.random.choice(len(up_candidates), size=min(target_up, len(up_candidates)), replace=False) if up_candidates else []
        selected_down = np.random.choice(len(down_candidates), size=min(target_down, len(down_candidates)), replace=False) if down_candidates else []
        
        for idx in selected_up:
            candidate = up_candidates[idx]
            example = create_hybrid_training_example(candidate)
            dataset.append(example)
            up_count += 1
        
        for idx in selected_down:
            candidate = down_candidates[idx]
            example = create_hybrid_training_example(candidate)
            dataset.append(example)
            down_count += 1
        
        print(f" {symbol}: {len(selected_up)} UP + {len(selected_down)} DOWN = {len(selected_up) + len(selected_down)}")
    
    print(f"\n{'='*80}")
    print(f"ГИБРИДНЫЙ ДАТАСЕТ СОЗДАН")
    print(f"{'='*80}")
    print(f"Всего: {len(dataset)} примеров")
    print(f" UP: {up_count} ({up_count/len(dataset)*100:.1f}%)")
    print(f" DOWN: {down_count} ({down_count/len(dataset)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    return dataset

def create_hybrid_training_example(candidate: Dict) -> Dict:
    """Создаёт обучающий пример БЕЗ 3D-баров в промпте (только для CatBoost)"""
    row = candidate['row']
    future_row = candidate['future_row']
    quantum_feats = candidate['quantum_feats']
    
    entropy_level = "высокая неопределённость" if quantum_feats['quantum_entropy'] > 4.0 else \
                    "умеренная неопределённость" if quantum_feats['quantum_entropy'] > 3.0 else \
                    "низкая неопределённость (рынок определился)"
    
    dominant_strength = "сильная" if quantum_feats['dominant_state_prob'] > 0.15 else \
                       "умеренная" if quantum_feats['dominant_state_prob'] > 0.10 else \
                       "слабая"
    
    market_complexity = "высокая" if quantum_feats['significant_states'] > 15 else \
                       "средняя" if quantum_feats['significant_states'] > 8 else \
                       "низкая"
    
    catboost_correct = "ВЕРНО" if candidate['catboost_direction'] == candidate['actual_direction'] else "ОШИБКА"
    
    prompt = f"""{candidate['symbol']} {candidate['current_time'].strftime('%Y-%m-%d %H:%M') if hasattr(candidate['current_time'], 'strftime') else str(candidate['current_time'])}
Текущая цена: {row['close']:.5f}

ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:
RSI: {row['RSI']:.1f}
MACD: {row['MACD']:.6f}
ATR: {row['ATR']:.5f}
Объёмы: {row['vol_ratio']:.2f}x
BB позиция: {row['BB_position']:.2f}
Stochastic K: {row['Stoch_K']:.1f}

КВАНТОВЫЕ ПРИЗНАКИ:
Квантовая энтропия: {quantum_feats['quantum_entropy']:.2f} ({entropy_level})
Доминантное состояние: {quantum_feats['dominant_state_prob']:.3f} ({dominant_strength} доминанта)
Значимые состояния: {quantum_feats['significant_states']} (сложность рынка: {market_complexity})
Квантовая дисперсия: {quantum_feats['quantum_variance']:.6f}

ПРОГНОЗ CATBOOST+QUANTUM:
Направление: {candidate['catboost_direction']}
Уверенность: {candidate['catboost_confidence']:.1f}%
Вероятность UP: {candidate['catboost_prob_up']:.1f}%

Проанализируй ситуацию и дай точный прогноз цены через 24 часа."""

    response = f"""НАПРАВЛЕНИЕ: {candidate['actual_direction']}
УВЕРЕННОСТЬ: {min(98, max(65, candidate['catboost_confidence'] + np.random.randint(-5, 10)))}%
ПРОГНОЗ ЦЕНЫ ЧЕРЕЗ 24Ч: {future_row['close']:.5f} ({candidate['price_change_pips']:+d} пунктов)

АНАЛИЗ ПРОГНОЗА CATBOOST:
Квантовая модель предсказала {candidate['catboost_direction']} с уверенностью {candidate['catboost_confidence']:.1f}%.
Реальный результат: {candidate['actual_direction']} ({catboost_correct}).

КВАНТОВЫЙ АНАЛИЗ:
Энтропия {quantum_feats['quantum_entropy']:.2f} показывает {entropy_level}. {'Рынок коллапсировал в определённое состояние — движение предсказуемо.' if quantum_feats['quantum_entropy'] < 3.0 else 'Рынок в режиме неопределённости — множественные сценарии равновероятны.' if quantum_feats['quantum_entropy'] > 4.5 else 'Умеренная неопределённость — есть предпочтительное направление.'}

ВЫВОД:
Система CatBoost+Quantum {'правильно определила' if catboost_correct == 'ВЕРНО' else 'ошибочно предсказала'} направление.
Фактическое движение за 24 часа: {abs(candidate['price_change_pips'])} пунктов {candidate['actual_direction']}.
Конечная цена: {future_row['close']:.5f}."""

    return {
        "prompt": prompt,
        "response": response,
        "direction": candidate['actual_direction'],
        "has_3d_warning": candidate['has_yellow_cluster']
    }

# ====================== СОХРАНЕНИЕ ДАТАСЕТА ======================
def save_dataset(dataset: List[Dict], filename: str = "dataset/quantum_fusion_3d_data.jsonl") -> str:
    """Сохранение гибридного датасета"""
    def convert_to_python_types(obj):
        """Конвертирует numpy типы в Python типы для JSON"""
        if isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset:
            # Конвертируем все numpy типы в Python типы
            clean_item = convert_to_python_types(item)
            f.write(json.dumps(clean_item, ensure_ascii=False) + '\n')
    print(f"✓ Датасет сохранён: {filename}")
    print(f"  Размер: {os.path.getsize(filename) / 1024:.1f} KB")
    return filename

# ====================== ФАЙНТЬЮН LLM ======================
def finetune_llm_with_catboost(dataset_path: str):
    """Файнтьюн LLM с CatBoost прогнозами (БЕЗ 3D в промптах)"""
    print(f"\n{'='*80}")
    print(f"ФАЙНТЬЮН LLM С CATBOOST+QUANTUM ПРОГНОЗАМИ")
    print(f"{'='*80}\n")
    
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
    except:
        print("❌ Ollama не установлен!")
        return
    
    print("Загрузка обучающих данных...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        training_data = [json.loads(line) for line in f]
    
    training_sample = training_data[:min(500, len(training_data))]
    print(f"✓ Загружено {len(training_sample)} примеров")
    
    print("\nСоздание Modelfile с квантовыми примерами...")
    
    modelfile_content = f"""FROM {BASE_MODEL}
PARAMETER temperature 0.55
PARAMETER top_p 0.92
SYSTEM \"\"\"Ты — QuantumTrader-Fusion — элитный аналитик с квантовым усилением.

УНИКАЛЬНЫЕ ВОЗМОЖНОСТИ:
1. Видишь прогнозы CatBoost с квантовыми признаками (точность 62-68%)
2. Понимаешь квантовую энтропию и доминантные состояния

ПРАВИЛА:
1. Только UP или DOWN — никакого FLAT
2. Уверенность 65-98%
3. ОБЯЗАТЕЛЬНО прогноз цены через 24ч
4. Анализируй квантовые признаки и технические индикаторы
\"\"\"
"""

    for i, example in enumerate(training_sample, 1):
        modelfile_content += f"""
MESSAGE user \"\"\"{example['prompt']}\"\"\"
MESSAGE assistant \"\"\"{example['response']}\"\"\"
"""
    
    modelfile_path = "Modelfile_quantum_fusion_3d"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"✓ Modelfile создан")
    
    print(f"\nСоздание модели {MODEL_NAME}...")
    
    try:
        result = subprocess.run(
            ["ollama", "create", MODEL_NAME, "-f", modelfile_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"\n✓ Модель {MODEL_NAME} создана!")
        
        os.remove(modelfile_path)
        
        print(f"\n{'='*80}")
        print(f"ФАЙНТЬЮН ЗАВЕРШЁН!")
        print(f"{'='*80}")
        print(f"✓ Модель: {MODEL_NAME}")
        print(f"✓ Интеграция: CatBoost + Qiskit + LLM")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка: {e}")

# ====================== ПАРСИНГ ОТВЕТА LLM ======================
def parse_answer(text: str) -> dict:
    """Парсинг ответа LLM"""
    prob = re.search(r"(?:УВЕРЕННОСТЬ|ВЕРОЯТНОСТЬ)[\s:]*(\d+)", text, re.I)
    direction = re.search(r"\b(UP|DOWN)\b", text, re.I)
    price_pred = re.search(r"ПРОГНОЗ ЦЕНЫ.*?(\d+\.\d+)", text, re.I)
    
    p = int(prob.group(1)) if prob else 50
    d = direction.group(1).upper() if direction else "DOWN"
    target_price = float(price_pred.group(1)) if price_pred else None
    
    return {"prob": p, "dir": d, "target_price": target_price}

# ====================== ГРАФИКИ ======================
def plot_results(balance_hist, equity_hist, slots):
    """График эквити"""
    DPI = 100
    WIDTH_PX = 700
    HEIGHT_PX = 350
    
    fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    
    min_length = min(len(equity_hist), len(slots))
    dates = [s['datetime'] for s in slots[:min_length]]
    equity_to_plot = equity_hist[:min_length]
    
    plt.plot(dates, equity_to_plot, color='#1E90FF', linewidth=3.5, label='Equity')
    plt.title('Equity Curve (Quantum Fusion + 3D)', fontsize=16, fontweight='bold', color='white')
    plt.xlabel('Time', color='white')
    plt.ylabel('Balance ($)', color='white')
    
    ax = plt.gca()
    ax.set_facecolor('#0a0a0a')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('white')
    ax.tick_params(colors='white')
    plt.grid(alpha=0.2, color='gray')
    plt.xticks(rotation=45)
    
    plt.legend(facecolor='#0a0a0a', edgecolor='white', labelcolor='white')
    
    plt.tight_layout(pad=2.0)
    
    filename = f"charts/equity_quantum_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=DPI, facecolor='#0a0a0a', edgecolor='none', 
                bbox_inches='tight', pad_inches=0.1)
    print(f"\n✓ График сохранён: {filename}")
    plt.show()

def calculate_max_drawdown(equity):
    """Расчёт максимальной просадки"""
    if len(equity) == 0:
        return 0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-8)
    return np.max(dd) * 100

# ====================== БЭКТЕСТ ======================
def backtest():
    """Бэктест квантовой гибридной системы + 3D"""
    print(f"\n{'='*80}")
    print(f"БЭКТЕСТ: QUANTUM + 3D BARS")
    print(f"{'='*80}\n")
    
    if not os.path.exists("models/catboost_quantum_3d.cbm"):
        print("❌ Модель не найдена! Сначала обучи (режим 1)")
        return
    
    print("Загрузка CatBoost модели...")
    if not CATBOOST_AVAILABLE:
        print("❌ CatBoost недоступен")
        return
    
    catboost_model = CatBoostClassifier()
    catboost_model.load_model("models/catboost_quantum_3d.cbm")
    print("✓ CatBoost модель загружена")
    
    use_llm = False
    if ollama:
        try:
            ollama.list()
            models = ollama.list()
            if any(MODEL_NAME in str(m) for m in models.get('models', [])):
                use_llm = True
                print("✓ LLM модель найдена")
            else:
                print(f"⚠️ LLM {MODEL_NAME} не найдена, работаем с CatBoost+Quantum+3D")
        except:
            print("⚠️ Ollama недоступен")
    
    if not mt5 or not mt5.initialize():
        print("❌ MT5 не подключён")
        return
    
    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(days=BACKTEST_DAYS)
    
    data = {}
    print(f"\nЗагрузка данных...")
    
    for sym in SYMBOLS:
        rates = mt5.copy_rates_range(sym, TIMEFRAME, start, end)
        if rates is None or len(rates) == 0:
            continue
        
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        
        if len(df) > LOOKBACK + PREDICTION_HORIZON:
            data[sym] = df
            print(f" ✓ {sym}: {len(df)} баров")
    
    if not data:
        print("\n❌ Нет данных!")
        mt5.shutdown()
        return
    
    balance = INITIAL_BALANCE
    equity = INITIAL_BALANCE
    trades = []
    balance_hist = [balance]
    equity_hist = [equity]
    slots = [{"datetime": start}]
    
    SPREAD_PIPS = 2
    SWAP_LONG = -0.5
    SWAP_SHORT = -0.3
    
    print(f"\n{'='*80}")
    print(f"ПАРАМЕТРЫ")
    print(f"{'='*80}")
    print(f"Начальный баланс: ${balance:,.2f}")
    print(f"Режим: {'CatBoost + Quantum + 3D + LLM' if use_llm else 'CatBoost + Quantum + 3D'}")
    print(f"{'='*80}\n")
    
    quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
    bars_3d = Bars3D(MIN_SPREAD_MULTIPLIER, VOLUME_BRICK)
    
    main_symbol = list(data.keys())[0]
    main_data = data[main_symbol]
    total_bars = len(main_data)
    analysis_points = list(range(LOOKBACK, total_bars - PREDICTION_HORIZON, PREDICTION_HORIZON))
    
    print(f"Точек анализа: {len(analysis_points)}\n")
    
    for point_idx, current_idx in enumerate(analysis_points):
        current_time = main_data.index[current_idx]
        
        print(f"{'='*80}")
        print(f"Анализ #{point_idx + 1}/{len(analysis_points)}: {current_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*80}")
        
        for sym in SYMBOLS:
            if sym not in data:
                continue
            
            historical_data = data[sym].iloc[:current_idx + 1].copy()
            if len(historical_data) < LOOKBACK:
                continue
            
            symbol_info = mt5.symbol_info(sym)
            if symbol_info is None:
                continue
            
            df_with_features = calculate_features(historical_data, bars_3d, symbol_info)
            if len(df_with_features) == 0:
                continue
            
            row = df_with_features.iloc[-1]
            
            point = symbol_info.point
            contract_size = symbol_info.trade_contract_size
            
            feature_vector = np.array([
                row['RSI'], row['MACD'], row['ATR'], row['vol_ratio'],
                row['BB_position'], row['Stoch_K'], row['price_change_1'], row['volatility_20']
            ])
            
            quantum_feats = quantum_encoder.encode_and_measure(feature_vector)
            
            X_features = {
                'RSI': row['RSI'],
                'MACD': row['MACD'],
                'ATR': row['ATR'],
                'vol_ratio': row['vol_ratio'],
                'BB_position': row['BB_position'],
                'Stoch_K': row['Stoch_K'],
                'Stoch_D': row['Stoch_D'],
                'EMA_50': row['EMA_50'],
                'EMA_200': row['EMA_200'],
                'price_change_1': row['price_change_1'],
                'price_change_5': row['price_change_5'],
                'price_change_21': row['price_change_21'],
                'volatility_20': row['volatility_20'],
                'quantum_entropy': quantum_feats['quantum_entropy'],
                'dominant_state_prob': quantum_feats['dominant_state_prob'],
                'significant_states': quantum_feats['significant_states'],
                'quantum_variance': quantum_feats['quantum_variance'],
            }
            
            # Добавляем 3D признаки для CatBoost
            if 'bar3d_yellow_cluster' in row:
                X_features.update({
                    'bar3d_yellow_cluster': row.get('bar3d_yellow_cluster', 0),
                    'bar3d_reversal_prob': row.get('bar3d_reversal_prob', 0),
                    'bar3d_trend_strength': row.get('bar3d_trend_strength', 0),
                    'bar3d_price_volatility': row.get('bar3d_price_volatility', 0),
                    'bar3d_volume_volatility': row.get('bar3d_volume_volatility', 0),
                })
            
            X_df = pd.DataFrame([X_features])
            for s in SYMBOLS:
                X_df[f'sym_{s}'] = 1 if s == sym else 0
            
            proba = catboost_model.predict_proba(X_df)[0]
            catboost_prob_up = proba[1] * 100
            catboost_direction = "UP" if proba[1] > 0.5 else "DOWN"
            catboost_confidence = max(proba) * 100
            
            entropy_level = "низкая" if quantum_feats['quantum_entropy'] < 3.0 else "средняя" if quantum_feats['quantum_entropy'] < 4.5 else "высокая"
            
            yellow_warning = "⚠️ ЖЕЛТЫЙ КЛАСТЕР!" if row.get('bar3d_yellow_cluster', 0) > 0.5 else ""
            
            print(f"\n{sym}: {yellow_warning}")
            print(f"  CatBoost: {catboost_direction} {catboost_confidence:.1f}%")
            print(f"  Квант: entropy={quantum_feats['quantum_entropy']:.2f}")
            
            final_direction = catboost_direction
            final_confidence = catboost_confidence
            
            if use_llm:
                try:
                    prompt = f"""{sym}
ТЕХНИЧЕСКИЕ: RSI={row['RSI']:.1f}, MACD={row['MACD']:.6f}, ATR={row['ATR']:.5f}
КВАНТ: entropy={quantum_feats['quantum_entropy']:.2f}
CATBOOST: {catboost_direction} {catboost_confidence:.1f}%

Дай прогноз на 24 часа."""

                    resp = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.3})
                    result = parse_answer(resp["response"])
                    
                    final_direction = result["dir"]
                    final_confidence = result["prob"]
                    
                    print(f"  LLM: {final_direction} {final_confidence}%")
                    
                except Exception as e:
                    log.error(f"Ошибка LLM: {e}")
            
            if final_confidence < MIN_PROB:
                print(f"  ❌ Уверенность {final_confidence:.1f}% < {MIN_PROB}%")
                continue
            
            exit_idx = current_idx + PREDICTION_HORIZON
            if exit_idx >= len(data[sym]):
                continue
            
            exit_row = data[sym].iloc[exit_idx]
            
            entry_price = row['close'] + SPREAD_PIPS * point if final_direction == "UP" else row['close']
            exit_price = exit_row['close'] if final_direction == "UP" else exit_row['close'] + SPREAD_PIPS * point
            
            price_move_pips = (exit_price - entry_price) / point if final_direction == "UP" else (entry_price - exit_price) / point
            
            risk_amount = balance * RISK_PER_TRADE
            atr_pips = row['ATR'] / point
            stop_loss_pips = max(20, atr_pips * 2)
            lot_size = risk_amount / (stop_loss_pips * point * contract_size)
            lot_size = max(0.01, min(lot_size, 10.0))
            
            profit_pips = price_move_pips
            profit_usd = profit_pips * point * contract_size * lot_size
            
            swap_cost = SWAP_LONG if final_direction == "UP" else SWAP_SHORT
            swap_cost = swap_cost * (lot_size / 0.01)
            profit_usd -= swap_cost
            profit_usd -= SLIPPAGE * point * contract_size * lot_size
            
            balance += profit_usd
            equity = balance
            
            actual_direction = "UP" if (exit_row['close'] > row['close']) else "DOWN"
            correct = (final_direction == actual_direction)
            
            trades.append({
                "time": current_time,
                "symbol": sym,
                "direction": final_direction,
                "confidence": final_confidence,
                "quantum_entropy": quantum_feats['quantum_entropy'],
                "yellow_cluster": row.get('bar3d_yellow_cluster', 0),
                "profit_usd": profit_usd,
                "balance": balance,
                "correct": correct
            })
            
            status = "✓ ВЕРНО" if correct else "✗ ОШИБКА"
            color = '\033[92m' if correct else '\033[91m'
            reset = '\033[0m'
            
            print(f"  {color}{status}{reset} | Профит: ${profit_usd:+.2f} | Баланс: ${balance:,.2f}")
        
        balance_hist.append(balance)
        equity_hist.append(equity)
        slots.append({"datetime": current_time})
    
    mt5.shutdown()
    
    print(f"\n{'='*80}")
    print(f"РЕЗУЛЬТАТЫ")
    print(f"{'='*80}\n")
    print(f"Сделок: {len(trades)}")
    print(f"Баланс: ${INITIAL_BALANCE:,.2f} → ${balance:,.2f}")
    print(f"Прибыль: ${balance - INITIAL_BALANCE:+,.2f} ({((balance/INITIAL_BALANCE - 1) * 100):+.2f}%)")
    
    if trades:
        wins = sum(1 for t in trades if t['profit_usd'] > 0)
        win_rate = wins / len(trades) * 100
        print(f"\nВинрейт: {win_rate:.2f}%")
        
        yellow_trades = [t for t in trades if t.get('yellow_cluster', 0) > 0.5]
        if yellow_trades:
            yellow_wins = sum(1 for t in yellow_trades if t['correct'])
            print(f"\nЖЕЛТЫЕ КЛАСТЕРЫ:")
            print(f"  Сделок: {len(yellow_trades)}")
            print(f"  Винрейт: {yellow_wins/len(yellow_trades)*100:.1f}%")
        
        max_dd = calculate_max_drawdown(np.array(equity_hist))
        print(f"\nМакс. просадка: {max_dd:.2f}%")
        
        if len(equity_hist) > 1:
            plot_results(balance_hist, equity_hist, slots)
    
    print(f"\n{'='*80}")

# ====================== ЖИВАЯ ТОРГОВЛЯ ======================
def live_trading():
    """
    ЖИВАЯ ТОРГОВЛЯ с Quantum + 3D-барами
    """
    print(f"\n{'='*80}")
    print(f"ЖИВАЯ ТОРГОВЛЯ: QUANTUM + 3D BARS")
    print(f"{'='*80}\n")
    
    # Проверка MT5
    if not mt5:
        print("❌ MetaTrader5 не установлен")
        return
    
    if not mt5.initialize():
        print("❌ Не удалось подключиться к MT5")
        return
    
    # Проверка CatBoost модели
    if not os.path.exists("models/catboost_quantum_3d.cbm"):
        print("❌ CatBoost модель не найдена!")
        print("Сначала обучи модель (режим 1)")
        mt5.shutdown()
        return
    
    print("Загрузка CatBoost модели...")
    if not CATBOOST_AVAILABLE:
        print("❌ CatBoost недоступен")
        mt5.shutdown()
        return
    
    catboost_model = CatBoostClassifier()
    catboost_model.load_model("models/catboost_quantum_3d.cbm")
    print("✓ CatBoost модель загружена")
    
    # Проверка LLM
    use_llm = False
    if ollama:
        try:
            ollama.list()
            models = ollama.list()
            if any(MODEL_NAME in str(m) for m in models.get('models', [])):
                use_llm = True
                print("✓ LLM модель найдена")
            else:
                print(f"⚠️ LLM {MODEL_NAME} не найдена, работаем с CatBoost+Quantum+3D")
        except:
            print("⚠️ Ollama недоступен, работаем с CatBoost+Quantum+3D")
    else:
        print("⚠️ Ollama не установлен")
    
    # Проверка счёта
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Не удалось получить информацию о счёте")
        mt5.shutdown()
        return
    
    print(f"\n{'='*80}")
    print(f"ИНФОРМАЦИЯ О СЧЁТЕ")
    print(f"{'='*80}")
    print(f"Логин: {account_info.login}")
    print(f"Баланс: ${account_info.balance:,.2f}")
    print(f"Эквити: ${account_info.equity:,.2f}")
    print(f"Кредитное плечо: 1:{account_info.leverage}")
    print(f"{'='*80}\n")
    
    # Параметры торговли
    print(f"ПАРАМЕТРЫ ТОРГОВЛИ:")
    print(f"-"*80)
    print(f"Режим: {'CatBoost + Quantum + 3D + LLM' if use_llm else 'CatBoost + Quantum + 3D'}")
    print(f"Символы: {', '.join(SYMBOLS)}")
    print(f"Таймфрейм: M15")
    print(f"Риск на сделку: {RISK_PER_TRADE * 100}%")
    print(f"Минимальная уверенность: {MIN_PROB}%")
    print(f"Горизонт прогноза: 24 часа")
    print(f"Магический номер: {MAGIC}")
    print(f"-"*80)
    
    confirm = input("\nНачать торговлю? (YES для подтверждения): ").strip()
    if confirm != "YES":
        print("Торговля отменена")
        mt5.shutdown()
        return
    
    print(f"\n{'='*80}")
    print("ТОРГОВЛЯ НАЧАТА")
    print(f"{'='*80}\n")
    print("Нажми Ctrl+C для остановки\n")
    
    # Инициализация
    quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
    bars_3d = Bars3D(MIN_SPREAD_MULTIPLIER, VOLUME_BRICK)
    
    total_analyses = 0
    total_signals = 0
    total_positions_opened = 0
    
    try:
        while True:
            current_time = datetime.now()
            print(f"\n{'='*80}")
            print(f"АНАЛИЗ РЫНКА: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")
            
            # Загрузка данных для каждого символа
            for symbol in SYMBOLS:
                print(f"\n--- {symbol} ---")
                
                # Получение исторических данных
                rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, LOOKBACK + 100)
                if rates is None or len(rates) < LOOKBACK:
                    print(f"  ⚠️ Недостаточно данных")
                    continue
                
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df.set_index("time", inplace=True)
                
                # Получение информации о символе
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    print(f"  ⚠️ Не удалось получить информацию о символе")
                    continue
                
                # Расчёт признаков с 3D-барами
                df_features = calculate_features(df, bars_3d, symbol_info)
                if len(df_features) == 0:
                    print(f"  ⚠️ Ошибка расчёта признаков")
                    continue
                
                row = df_features.iloc[-1]
                
                # Квантовое кодирование
                feature_vector = np.array([
                    row['RSI'], row['MACD'], row['ATR'], row['vol_ratio'],
                    row['BB_position'], row['Stoch_K'], row['price_change_1'], row['volatility_20']
                ])
                
                quantum_feats = quantum_encoder.encode_and_measure(feature_vector)
                
                # Подготовка признаков для CatBoost
                X_features = {
                    'RSI': row['RSI'],
                    'MACD': row['MACD'],
                    'ATR': row['ATR'],
                    'vol_ratio': row['vol_ratio'],
                    'BB_position': row['BB_position'],
                    'Stoch_K': row['Stoch_K'],
                    'Stoch_D': row['Stoch_D'],
                    'EMA_50': row['EMA_50'],
                    'EMA_200': row['EMA_200'],
                    'price_change_1': row['price_change_1'],
                    'price_change_5': row['price_change_5'],
                    'price_change_21': row['price_change_21'],
                    'volatility_20': row['volatility_20'],
                    'quantum_entropy': quantum_feats['quantum_entropy'],
                    'dominant_state_prob': quantum_feats['dominant_state_prob'],
                    'significant_states': quantum_feats['significant_states'],
                    'quantum_variance': quantum_feats['quantum_variance'],
                }
                
                # Добавляем 3D признаки для CatBoost
                if 'bar3d_yellow_cluster' in row:
                    X_features.update({
                        'bar3d_yellow_cluster': row.get('bar3d_yellow_cluster', 0),
                        'bar3d_reversal_prob': row.get('bar3d_reversal_prob', 0),
                        'bar3d_trend_strength': row.get('bar3d_trend_strength', 0),
                        'bar3d_price_volatility': row.get('bar3d_price_volatility', 0),
                        'bar3d_volume_volatility': row.get('bar3d_volume_volatility', 0),
                    })
                
                X_df = pd.DataFrame([X_features])
                for s in SYMBOLS:
                    X_df[f'sym_{s}'] = 1 if s == symbol else 0
                
                # Прогноз CatBoost
                proba = catboost_model.predict_proba(X_df)[0]
                catboost_prob_up = proba[1] * 100
                catboost_direction = "UP" if proba[1] > 0.5 else "DOWN"
                catboost_confidence = max(proba) * 100
                
                # Проверка желтого кластера (только для статистики)
                yellow_warning = ""
                if row.get('bar3d_yellow_cluster', 0) > 0.5:
                    yellow_warning = f"⚠️ 3D: разворот {row.get('bar3d_reversal_prob', 0)*100:.0f}%"
                
                print(f"  {yellow_warning}")
                print(f"  🔬 CatBoost: {catboost_direction} {catboost_confidence:.1f}%")
                print(f"  ⚛️ Квант: entropy={quantum_feats['quantum_entropy']:.2f}")
                
                final_direction = catboost_direction
                final_confidence = catboost_confidence
                
                # Прогноз LLM если доступен (БЕЗ 3D в промпте)
                if use_llm:
                    try:
                        prompt = f"""{symbol}
ТЕХНИЧЕСКИЕ: RSI={row['RSI']:.1f}, MACD={row['MACD']:.6f}, ATR={row['ATR']:.5f}
КВАНТ: entropy={quantum_feats['quantum_entropy']:.2f}
CATBOOST: {catboost_direction} {catboost_confidence:.1f}%

Дай прогноз на 24 часа."""

                        resp = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.3})
                        result = parse_answer(resp["response"])
                        
                        final_direction = result["dir"]
                        final_confidence = result["prob"]
                        
                        print(f"  🧠 LLM: {final_direction} {final_confidence}% (коррекция: {final_confidence - catboost_confidence:+.1f}%)")
                        
                    except Exception as e:
                        log.error(f"Ошибка LLM для {symbol}: {e}")
                
                total_analyses += 1
                
                # Проверка уверенности
                if final_confidence < MIN_PROB:
                    print(f"  ❌ Уверенность {final_confidence:.1f}% < {MIN_PROB}%, пропускаем")
                    continue
                
                total_signals += 1
                
                # Расчёт размера позиции
                account_info = mt5.account_info()
                balance = account_info.balance
                
                risk_amount = balance * RISK_PER_TRADE
                point = symbol_info.point
                contract_size = symbol_info.trade_contract_size
                
                atr_pips = row['ATR'] / point
                stop_loss_pips = max(20, atr_pips * 4)
                
                lot_size = risk_amount / (stop_loss_pips * point * contract_size)
                
                # Округление до минимального лота
                volume_min = symbol_info.volume_min
                volume_max = symbol_info.volume_max
                volume_step = symbol_info.volume_step
                
                lot_size = max(volume_min, min(lot_size, volume_max))
                lot_size = round(lot_size / volume_step) * volume_step
                
                # Текущая цена
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    print(f"  ⚠️ Не удалось получить текущую цену")
                    continue
                
                # Расчёт SL и TP
                if final_direction == "UP":
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask
                    sl = price - stop_loss_pips * point
                    tp = price + stop_loss_pips * point * 3
                else:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                    sl = price + stop_loss_pips * point
                    tp = price - stop_loss_pips * point * 3
                
                # Формирование запроса
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": order_type,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "magic": MAGIC,
                    "comment": f"Q3D_{int(final_confidence)}%",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                print(f"\n  📈 ОТКРЫТИЕ ПОЗИЦИИ:")
                print(f"     Направление: {final_direction}")
                print(f"     Лот: {lot_size}")
                print(f"     Цена: {price:.5f}")
                print(f"     SL: {sl:.5f} ({stop_loss_pips:.0f} пунктов)")
                print(f"     TP: {tp:.5f} ({stop_loss_pips * 3:.0f} пунктов)")
                print(f"     Риск: ${risk_amount:.2f}")
                
                # Отправка ордера
                result = mt5.order_send(request)
                
                if result is None:
                    print(f"  ❌ Ошибка отправки ордера")
                    continue
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"  ❌ Ошибка открытия: {result.retcode} - {result.comment}")
                else:
                    print(f"  ✅ ПОЗИЦИЯ ОТКРЫТА!")
                    print(f"     Тикет: {result.order}")
                    print(f"     Цена исполнения: {result.price:.5f}")
                    total_positions_opened += 1
                    
                    log.info(f"Открыта: {symbol} {final_direction} {lot_size} @ {result.price:.5f} | "
                            f"Уверенность: {final_confidence}% | Quantum: {quantum_feats['quantum_entropy']:.2f} | "
                            f"3D: {yellow_warning if yellow_warning else 'norm'}")
            
            # Статистика сессии
            print(f"\n{'='*80}")
            print(f"СТАТИСТИКА СЕССИИ")
            print(f"{'='*80}")
            print(f"Всего анализов: {total_analyses}")
            print(f"Сигналов: {total_signals}")
            print(f"Позиций открыто: {total_positions_opened}")
            
            # Текущие позиции
            all_positions = mt5.positions_get(magic=MAGIC)
            if all_positions:
                total_profit = sum(p.profit for p in all_positions)
                print(f"\nТекущие позиции: {len(all_positions)}")
                print(f"Плавающий профит: ${total_profit:+.2f}")
                
                for pos in all_positions:
                    print(f"  {pos.symbol} {'BUY' if pos.type == 0 else 'SELL'} {pos.volume} | ${pos.profit:+.2f}")
            else:
                print(f"\nТекущие позиции: 0")
            
            print(f"\n{'='*80}")
            
            # Следующий анализ через 24 часа
            next_analysis = current_time + timedelta(hours=24)
            print(f"\nСледующий анализ: {next_analysis.strftime('%Y-%m-%d %H:%M')}")
            print("Нажми Ctrl+C для остановки\n")
            
            # Ожидание с проверкой позиций каждую минуту
            wait_seconds = 24 * 60 * 60
            check_interval = 60
            
            for i in range(0, wait_seconds, check_interval):
                time.sleep(check_interval)
                
                # Проверка старых позиций
                positions = mt5.positions_get(magic=MAGIC)
                if positions:
                    current_check_time = datetime.now()
                    for pos in positions:
                        open_time = datetime.fromtimestamp(pos.time)
                        hours_open = (current_check_time - open_time).total_seconds() / 3600
                        
                        if hours_open >= 24:
                            print(f"\n⏰ {pos.symbol}: 24 часа истекли, закрываем...")
                            close_result = close_position(pos)
                            if close_result:
                                print(f"✓ Закрыто, профит: ${pos.profit:+.2f}")
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print("ОСТАНОВКА ТОРГОВЛИ")
        print(f"{'='*80}\n")
        
        # Закрытие позиций
        positions = mt5.positions_get(magic=MAGIC)
        if positions and len(positions) > 0:
            print(f"Обнаружено {len(positions)} открытых позиций:")
            for pos in positions:
                print(f"  {pos.symbol} {'BUY' if pos.type == 0 else 'SELL'} {pos.volume} | ${pos.profit:+.2f}")
            
            close_all = input("\nЗакрыть все позиции? (YES/NO): ").strip()
            if close_all == "YES":
                print("\nЗакрытие позиций...")
                for pos in positions:
                    result = close_position(pos)
                    if result:
                        print(f"✓ {pos.symbol} закрыт, профит: ${pos.profit:+.2f}")
                    else:
                        print(f"❌ {pos.symbol} ошибка закрытия")
        
        print("\nТорговля остановлена.")
    
    except Exception as e:
        log.error(f"Критическая ошибка: {e}")
        print(f"\n❌ Критическая ошибка: {e}")
    
    finally:
        mt5.shutdown()
        print("MT5 отключён")

def close_position(position):
    """Закрывает открытую позицию"""
    symbol = position.symbol
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return False
    
    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": position.volume,
        "type": order_type,
        "position": position.ticket,
        "price": price,
        "magic": MAGIC,
        "comment": "Close by system",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    
    if result is None:
        return False
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Ошибка закрытия {symbol}: {result.retcode} - {result.comment}")
        return False
    
    return True

# ====================== ЗАГРУЗКА ДАННЫХ MT5 ======================
def load_mt5_data(days: int = 180) -> Dict[str, pd.DataFrame]:
    """Загрузка данных из MT5"""
    if not mt5 or not mt5.initialize():
        print("⚠️ MT5 недоступен")
        return {}
    
    end = datetime.now()
    start = end - timedelta(days=days)
    
    data = {}
    print(f"\nЗагрузка данных MT5 за {days} дней...")
    
    for symbol in SYMBOLS:
        rates = mt5.copy_rates_range(symbol, TIMEFRAME, start, end)
        if rates is None or len(rates) < LOOKBACK + PREDICTION_HORIZON:
            print(f" ⚠️ {symbol}: недостаточно данных")
            continue
        
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        data[symbol] = df
        print(f" ✓ {symbol}: {len(df)} баров")
    
    mt5.shutdown()
    return data

# ====================== ГЛАВНОЕ МЕНЮ ======================
def main():
    """Главное меню"""
    print(f"\n{'='*80}")
    print(f" QUANTUM TRADER FUSION + 3D BARS")
    print(f" Qiskit + CatBoost + LLM + 3D Multi-dimensional Analysis")
    print(f" Версия: 14.12.2025 (3D только для CatBoost)")
    print(f"{'='*80}\n")
    print(f"РЕЖИМЫ:")
    print(f"-"*80)
    print(f"1 → Обучить CatBoost (Quantum + 3D-бары)")
    print(f"2 → Сгенерировать датасет (CatBoost + Quantum, БЕЗ 3D в промптах)")
    print(f"3 → Файнтьюн LLM")
    print(f"4 → Бэктест")
    print(f"5 → Живая торговля (MT5)")
    print(f"6 → ПОЛНЫЙ ЦИКЛ")
    print(f"-"*80)
    
    choice = input("\nВыбери режим (1-6): ").strip()
    
    bars_3d = Bars3D(MIN_SPREAD_MULTIPLIER, VOLUME_BRICK) if USE_3D_BARS else None
    
    if choice == "1":
        data = load_mt5_data(180)
        if not data:
            print("❌ Нет данных")
            return
        
        quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
        model = train_catboost_model(data, quantum_encoder, bars_3d)
        
    elif choice == "2":
        data = load_mt5_data(180)
        if not data:
            print("❌ Нет данных")
            return
        
        if os.path.exists("models/catboost_quantum_3d.cbm"):
            model = CatBoostClassifier()
            model.load_model("models/catboost_quantum_3d.cbm")
        else:
            print("❌ Модель не найдена, сначала обучи (режим 1)")
            return
        
        quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
        dataset = generate_hybrid_dataset(data, model, quantum_encoder, bars_3d, FINETUNE_SAMPLES)
        save_dataset(dataset, "dataset/quantum_fusion_3d_data.jsonl")
        
    elif choice == "3":
        dataset_path = "dataset/quantum_fusion_3d_data.jsonl"
        if not os.path.exists(dataset_path):
            print(f"❌ Датасет не найден")
            return
        
        finetune_llm_with_catboost(dataset_path)
        
    elif choice == "4":
        backtest()
        
    elif choice == "5":
        live_trading()
        
    elif choice == "6":
        print(f"\n{'='*80}")
        print(f"ПОЛНЫЙ ЦИКЛ: QUANTUM FUSION + 3D BARS")
        print(f"{'='*80}\n")
        
        confirm = input("Продолжить? (YES): ").strip()
        if confirm != "YES":
            print("Отменено")
            return
        
        data = load_mt5_data(180)
        if not data:
            return
        
        quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
        model = train_catboost_model(data, quantum_encoder, bars_3d)
        
        dataset = generate_hybrid_dataset(data, model, quantum_encoder, bars_3d, FINETUNE_SAMPLES)
        dataset_path = save_dataset(dataset, "dataset/quantum_fusion_3d_data.jsonl")
        
        finetune_llm_with_catboost(dataset_path)
        
        print(f"\n🎉 ПОЛНЫЙ ЦИКЛ ЗАВЕРШЁН!")
        
    else:
        print("❌ Неверный выбор")

if __name__ == "__main__":
    main()
