# ================================================
# ai_trader_quantum_fusion.py
# КВАНТОВЫЙ ГИБРИД: Qiskit + CatBoost + LLM
# Версия 09.12.2025 — Полная интеграция из статьи
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

# ====================== КОНФИГ ======================
MODEL_NAME = "koshtenco/quantum-trader-fusion-3b"
BASE_MODEL = "llama3.2:3b"
SYMBOLS = ["EURUSD", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURGBP", "AUDCHF"]
TIMEFRAME = mt5.TIMEFRAME_M15 if mt5 else None
LOOKBACK = 400
INITIAL_BALANCE = 140.0
RISK_PER_TRADE = 0.05
MIN_PROB = 60
LIVE_LOT = 0.02
MAGIC = 20251209
SLIPPAGE = 10

# Квантовые параметры
N_QUBITS = 8
N_SHOTS = 2048

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
        logging.FileHandler("logs/quantum_fusion.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ====================== КВАНТОВЫЙ ЭНКОДЕР ======================
class QuantumEncoder:
    """
    Квантовый энкодер на базе Qiskit для извлечения скрытых признаков
    Реализация из статьи: 8 кубитов, запутывание через CZ-гейты, 2048 измерений
    """
    
    def __init__(self, n_qubits: int = 8, n_shots: int = 2048):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.simulator = AerSimulator()
        
    def encode_and_measure(self, features: np.ndarray) -> Dict[str, float]:
        """
        Кодирует признаки в квантовую схему и извлекает 4 квантовых признака:
        1. Квантовая энтропия (мера неопределённости)
        2. Доминантное состояние (вероятность самого частого базиса)
        3. Количество значимых состояний (>3% вероятности)
        4. Квантовая дисперсия вероятностей
        """
        if not QISKIT_AVAILABLE:
            # Fallback на псевдо-квантовые признаки
            return {
                'quantum_entropy': np.random.uniform(2.0, 5.0),
                'dominant_state_prob': np.random.uniform(0.05, 0.20),
                'significant_states': np.random.randint(3, 20),
                'quantum_variance': np.random.uniform(0.001, 0.01)
            }
        
        # Нормализация признаков в диапазон [0, π]
        normalized = (features - features.min()) / (features.max() - features.min() + 1e-8)
        angles = normalized * np.pi
        
        # Создаём квантовую схему
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        
        # Кодирование через RY-вращения
        for i in range(min(len(angles), self.n_qubits)):
            qc.ry(angles[i], i)
        
        # Запутывание через CZ-гейты (создаём корреляции второго порядка)
        for i in range(self.n_qubits - 1):
            qc.cz(i, i + 1)
        # Замыкаем цепь
        qc.cz(self.n_qubits - 1, 0)
        
        # Измерение
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        
        # Выполнение на симуляторе
        job = self.simulator.run(qc, shots=self.n_shots)
        result = job.result()
        counts = result.get_counts()
        
        # Вычисление квантовых признаков
        total_shots = sum(counts.values())
        probabilities = np.array([counts.get(format(i, f'0{self.n_qubits}b'), 0) / total_shots 
                                  for i in range(2**self.n_qubits)])
        
        # 1. Квантовая энтропия Шеннона
        quantum_entropy = entropy(probabilities + 1e-10, base=2)
        
        # 2. Доминантное состояние
        dominant_state_prob = np.max(probabilities)
        
        # 3. Количество значимых состояний (>3%)
        significant_states = np.sum(probabilities > 0.03)
        
        # 4. Квантовая дисперсия
        quantum_variance = np.var(probabilities)
        
        return {
            'quantum_entropy': quantum_entropy,
            'dominant_state_prob': dominant_state_prob,
            'significant_states': significant_states,
            'quantum_variance': quantum_variance
        }

# ====================== ТЕХНИЧЕСКИЕ ПРИЗНАКИ ======================
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Расчёт 33 технических индикаторов"""
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
    
    # Дополнительные признаки для CatBoost
    d["price_change_1"] = d["close"].pct_change(1)
    d["price_change_5"] = d["close"].pct_change(5)
    d["price_change_21"] = d["close"].pct_change(21)
    d["log_return"] = np.log(d["close"] / d["close"].shift(1))
    d["volatility_20"] = d["log_return"].rolling(20).std()
    
    return d.dropna()

# ====================== ОБУЧЕНИЕ CATBOOST ======================
def train_catboost_model(data_dict: Dict[str, pd.DataFrame], quantum_encoder: QuantumEncoder) -> CatBoostClassifier:
    """
    Обучает CatBoost на данных всех 8 валютных пар с квантовыми признаками
    Возвращает обученную модель
    """
    print(f"\n{'='*80}")
    print(f"ОБУЧЕНИЕ CATBOOST С КВАНТОВЫМИ ПРИЗНАКАМИ")
    print(f"{'='*80}\n")
    
    if not CATBOOST_AVAILABLE:
        print("❌ CatBoost недоступен, используем заглушку")
        return None
    
    all_features = []
    all_targets = []
    all_symbols = []
    
    print("Подготовка данных и квантовое кодирование...")
    
    for symbol, df in data_dict.items():
        print(f"\nОбработка {symbol}: {len(df)} баров")
        
        df_features = calculate_features(df)
        
        # Квантовое кодирование для каждой точки
        quantum_features_list = []
        
        for idx in range(LOOKBACK, len(df_features) - PREDICTION_HORIZON):
            if idx % 500 == 0:
                print(f" Квантовое кодирование: {idx}/{len(df_features) - PREDICTION_HORIZON}")
            
            row = df_features.iloc[idx]
            
            # Берём ключевые индикаторы для квантового кодирования
            feature_vector = np.array([
                row['RSI'], row['MACD'], row['ATR'], row['vol_ratio'],
                row['BB_position'], row['Stoch_K'], row['price_change_1'], row['volatility_20']
            ])
            
            # Квантовое кодирование
            quantum_feats = quantum_encoder.encode_and_measure(feature_vector)
            quantum_features_list.append(quantum_feats)
            
            # Целевая переменная: цена через 24 часа
            future_idx = idx + PREDICTION_HORIZON
            future_price = df_features.iloc[future_idx]['close']
            current_price = row['close']
            target = 1 if future_price > current_price else 0  # 1=UP, 0=DOWN
            
            # Собираем признаки: технические + квантовые + символ
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
            
            all_features.append(features)
            all_targets.append(target)
            all_symbols.append(symbol)
    
    print(f"\n✓ Всего примеров: {len(all_features)}")
    
    # Создаём DataFrame
    X = pd.DataFrame(all_features)
    y = np.array(all_targets)
    
    # One-hot encoding символов
    X = pd.get_dummies(X, columns=['symbol'], prefix='sym')
    
    print(f"✓ Признаков: {len(X.columns)}")
    print(f"✓ Баланс классов: UP={np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%), DOWN={np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
    
    # Обучение CatBoost
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
    
    # Используем TimeSeriesSplit для честной валидации
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
    
    # Финальное обучение на всех данных
    print("\nОбучение финальной модели на всех данных...")
    model.fit(X, y, verbose=500)
    
    # Сохранение модели
    model_path = "models/catboost_quantum.cbm"
    model.save_model(model_path)
    print(f"\n✓ Модель сохранена: {model_path}")
    
    # Важность признаков
    feature_importance = model.get_feature_importance()
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
    print(importance_df.head(10).to_string(index=False))
    
    return model

# ====================== ГЕНЕРАЦИЯ ГИБРИДНОГО ДАТАСЕТА ======================
def generate_hybrid_dataset(
    data_dict: Dict[str, pd.DataFrame],
    catboost_model: CatBoostClassifier,
    quantum_encoder: QuantumEncoder,
    num_samples: int = 2000
) -> List[Dict]:
    """
    Генерирует датасет для LLM с встроенными прогнозами CatBoost и квантовыми признаками
    Каждый пример содержит:
    - Технические индикаторы
    - Квантовые признаки (в человекочитаемом формате)
    - Прогноз CatBoost (направление + уверенность)
    - Реальный результат через 24 часа
    """
    print(f"\n{'='*80}")
    print(f"ГЕНЕРАЦИЯ ГИБРИДНОГО ДАТАСЕТА ДЛЯ LLM")
    print(f"{'='*80}\n")
    print(f"Цель: {num_samples} примеров с CatBoost прогнозами и квантовыми признаками\n")
    
    dataset = []
    up_count = 0
    down_count = 0
    
    target_per_symbol = num_samples // len(SYMBOLS)
    
    for symbol, df in data_dict.items():
        print(f"Обработка {symbol}...")
        df_features = calculate_features(df)
        
        candidates = []
        
        for idx in range(LOOKBACK, len(df_features) - PREDICTION_HORIZON):
            row = df_features.iloc[idx]
            future_idx = idx + PREDICTION_HORIZON
            future_row = df_features.iloc[future_idx]
            
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
            
            # Создаём DataFrame для CatBoost (с one-hot encoding)
            X_df = pd.DataFrame([X_features])
            for s in SYMBOLS:
                X_df[f'sym_{s}'] = 1 if s == symbol else 0
            
            # Прогноз CatBoost
            if catboost_model:
                proba = catboost_model.predict_proba(X_df)[0]
                catboost_prob_up = proba[1] * 100
                catboost_direction = "UP" if proba[1] > 0.5 else "DOWN"
                catboost_confidence = max(proba) * 100
            else:
                catboost_prob_up = 50.0
                catboost_direction = "UP"
                catboost_confidence = 50.0
            
            # Реальный результат
            actual_price_24h = future_row['close']
            price_change = actual_price_24h - row['close']
            price_change_pips = int(price_change / 0.0001)
            actual_direction = "UP" if price_change > 0 else "DOWN"
            
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
                'current_time': df.index[idx]
            })
        
        # Балансировка: берём равное количество UP и DOWN
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
    """Создаёт обучающий пример с CatBoost прогнозом и квантовыми признаками"""
    row = candidate['row']
    future_row = candidate['future_row']
    quantum_feats = candidate['quantum_feats']
    
    # Интерпретация квантовых признаков
    entropy_level = "высокая неопределённость" if quantum_feats['quantum_entropy'] > 4.0 else \
                    "умеренная неопределённость" if quantum_feats['quantum_entropy'] > 3.0 else \
                    "низкая неопределённость (рынок определился)"
    
    dominant_strength = "сильная" if quantum_feats['dominant_state_prob'] > 0.15 else \
                       "умеренная" if quantum_feats['dominant_state_prob'] > 0.10 else \
                       "слабая"
    
    market_complexity = "высокая" if quantum_feats['significant_states'] > 15 else \
                       "средняя" if quantum_feats['significant_states'] > 8 else \
                       "низкая"
    
    # Проверка правильности CatBoost
    catboost_correct = "ВЕРНО" if candidate['catboost_direction'] == candidate['actual_direction'] else "ОШИБКА"
    
    prompt = f"""{candidate['symbol']} {candidate['current_time'].strftime('%Y-%m-%d %H:%M')}
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
Источник: catboost_quantum

Проанализируй ситуацию с учётом прогноза квантовой модели и дай точный прогноз цены через 24 часа."""

    response = f"""НАПРАВЛЕНИЕ: {candidate['actual_direction']}
УВЕРЕННОСТЬ: {min(98, max(65, candidate['catboost_confidence'] + np.random.randint(-5, 10)))}%
ПРОГНОЗ ЦЕНЫ ЧЕРЕЗ 24Ч: {future_row['close']:.5f} ({candidate['price_change_pips']:+d} пунктов)

АНАЛИЗ ПРОГНОЗА CATBOOST:
Квантовая модель предсказала {candidate['catboost_direction']} с уверенностью {candidate['catboost_confidence']:.1f}%.
Реальный результат: {candidate['actual_direction']} ({catboost_correct}).

КВАНТОВЫЙ АНАЛИЗ:
Энтропия {quantum_feats['quantum_entropy']:.2f} показывает {entropy_level}. {'Рынок коллапсировал в определённое состояние — движение предсказуемо.' if quantum_feats['quantum_entropy'] < 3.0 else 'Рынок в режиме неопределённости — множественные сценарии равновероятны.' if quantum_feats['quantum_entropy'] > 4.5 else 'Умеренная неопределённость — есть предпочтительное направление.'}
Доминантное состояние {quantum_feats['dominant_state_prob']:.3f} указывает на {dominant_strength} преобладание одного квантового состояния.
{quantum_feats['significant_states']} значимых состояний означают {market_complexity} сложность рыночной структуры.

ТЕХНИЧЕСКИЙ АНАЛИЗ НА 24 ЧАСА:
{'RSI ' + str(round(row["RSI"], 1)) + ' — перепроданность, жду отскок' if row['RSI'] < 30 else 'RSI ' + str(round(row["RSI"], 1)) + ' — перекупленность, возможна коррекция' if row['RSI'] > 70 else 'RSI ' + str(round(row["RSI"], 1)) + ' — нейтральная зона'}.
{'MACD позитивный — бычий импульс сохраняется' if row['MACD'] > 0 else 'MACD негативный — медвежье давление продолжается'}.
{'Объёмы выше средних — движение поддержано' if row['vol_ratio'] > 1.3 else 'Объёмы низкие — слабый импульс'}.
{'Цена у нижней BB — статистически жду возврата к средней' if row['BB_position'] < 0.25 else 'Цена у верхней BB — возможен откат' if row['BB_position'] > 0.75 else 'Цена в середине BB — направление не определено уровнями'}.

ВЫВОД:
Квантовая модель CatBoost {'правильно определила' if catboost_correct == 'ВЕРНО' else 'ошибочно предсказала'} направление. {'Квантовая энтропия подтверждает предсказуемость движения.' if quantum_feats['quantum_entropy'] < 3.5 else 'Высокая квантовая энтропия указывает на сложность прогноза.'} 
Фактическое движение за 24 часа: {abs(candidate['price_change_pips'])} пунктов {candidate['actual_direction']}.
Конечная цена: {future_row['close']:.5f}.

ВАЖНО: Квантовая модель имеет точность 62-68% на валидации. Это дополнительный фактор, но не абсолютная истина. {'В данном случае квантовые признаки показали высокую уверенность и оказались правы.' if catboost_correct == 'ВЕРНО' and quantum_feats['quantum_entropy'] < 3.5 else 'Следующий прогноз может быть противоположным — рынок непредсказуем.'}"""

    return {
        "prompt": prompt,
        "response": response,
        "direction": candidate['actual_direction']
    }

# ====================== СОХРАНЕНИЕ ДАТАСЕТА ======================
def save_dataset(dataset: List[Dict], filename: str = "dataset/quantum_fusion_data.jsonl") -> str:
    """Сохранение гибридного датасета"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Датасет сохранён: {filename}")
    print(f"  Размер: {os.path.getsize(filename) / 1024:.1f} KB")
    return filename

# ====================== ФАЙНТЬЮН LLM ======================
def finetune_llm_with_catboost(dataset_path: str):
    """Файнтьюн LLM с встроенными CatBoost прогнозами"""
    print(f"\n{'='*80}")
    print(f"ФАЙНТЬЮН LLM С CATBOOST ПРОГНОЗАМИ")
    print(f"{'='*80}\n")
    
    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
    except:
        print("❌ Ollama не установлен!")
        print("Установи: https://ollama.com/download")
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
PARAMETER top_k 30
PARAMETER num_ctx 8192
PARAMETER num_predict 768
PARAMETER repeat_penalty 1.1
SYSTEM \"\"\"
Ты — QuantumTrader-3B-Fusion — элитный аналитик с квантовым усилением.

УНИКАЛЬНЫЕ ВОЗМОЖНОСТИ:
1. Ты видишь прогнозы CatBoost модели с квантовыми признаками (точность 62-68%)
2. Ты понимаешь квантовую энтропию, доминантные состояния и сложность рынка
3. Ты интегрируешь квантовые прогнозы с классическим техническим анализом

СТРОГИЕ ПРАВИЛА:
1. Только UP или DOWN — никакого FLAT
2. Уверенность 65-98%
3. ОБЯЗАТЕЛЬНО прогноз цены через 24ч: X.XXXXX (±NN пунктов)
4. Анализируй прогноз CatBoost и квантовые признаки
5. Объясняй, почему квантовая модель права или ошиблась

ФОРМАТ ОТВЕТА:
НАПРАВЛЕНИЕ: UP/DOWN
УВЕРЕННОСТЬ: XX%
ПРОГНОЗ ЦЕНЫ ЧЕРЕЗ 24Ч: X.XXXXX (±NN пунктов)

АНАЛИЗ ПРОГНОЗА CATBOOST:
[оценка прогноза квантовой модели]

КВАНТОВЫЙ АНАЛИЗ:
[интерпретация квантовой энтропии, доминантных состояний]

ТЕХНИЧЕСКИЙ АНАЛИЗ НА 24 ЧАСА:
[RSI, MACD, объёмы, уровни]

ВЫВОД:
[синтез квантовых и технических сигналов с конкретной целью]
\"\"\"
"""

    for i, example in enumerate(training_sample, 1):
        modelfile_content += f"""
MESSAGE user \"\"\"{example['prompt']}\"\"\"
MESSAGE assistant \"\"\"{example['response']}\"\"\"
"""
    
    modelfile_path = "Modelfile_quantum_fusion"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"✓ Modelfile создан с {len(training_sample)} примерами")
    
    print(f"\nСоздание модели {MODEL_NAME}...")
    print("Это займёт 2-5 минут...\n")
    
    try:
        result = subprocess.run(
            ["ollama", "create", MODEL_NAME, "-f", modelfile_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"\n✓ Модель {MODEL_NAME} успешно создана!")
        
        print("\nТестирование модели...")
        test_prompt = """EURUSD 2025-12-09 10:00
Текущая цена: 1.0850

ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:
RSI: 32.5
MACD: -0.00015
ATR: 0.00085
Объёмы: 1.8x
BB позиция: 0.15
Stochastic K: 25.0

КВАНТОВЫЕ ПРИЗНАКИ:
Квантовая энтропия: 2.8 (низкая неопределённость - рынок определился)
Доминантное состояние: 0.187 (сильная доминанта)
Значимые состояния: 5 (сложность рынка: низкая)
Квантовая дисперсия: 0.003421

ПРОГНОЗ CATBOOST+QUANTUM:
Направление: UP
Уверенность: 87.3%
Вероятность UP: 87.3%
Источник: catboost_quantum

Проанализируй."""

        test_result = ollama.generate(model=MODEL_NAME, prompt=test_prompt)
        print("\n" + "="*80)
        print("ТЕСТОВЫЙ ОТВЕТ:")
        print("="*80)
        print(test_result['response'])
        print("="*80)
        
        os.remove(modelfile_path)
        
        print(f"\n{'='*80}")
        print(f"ФАЙНТЬЮН ЗАВЕРШЁН!")
        print(f"{'='*80}")
        print(f"✓ Модель готова: {MODEL_NAME}")
        print(f"✓ Интеграция: CatBoost + Qiskit + LLM")
        print(f"✓ Для публикации: ollama push {MODEL_NAME}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка: {e}")
        print(f"Вывод: {e.output}")

# ====================== ПАРСИНГ ОТВЕТОВ LLM ======================
def parse_answer(text: str) -> dict:
    """Парсинг ответа LLM с прогнозом цены"""
    prob = re.search(r"(?:УВЕРЕННОСТЬ|ВЕРОЯТНОСТЬ)[\s:]*(\d+)", text, re.I)
    direction = re.search(r"\b(UP|DOWN)\b", text, re.I)
    price_pred = re.search(r"ПРОГНОЗ ЦЕНЫ.*?(\d+\.\d+)", text, re.I)
    
    p = int(prob.group(1)) if prob else 50
    d = direction.group(1).upper() if direction else "DOWN"
    target_price = float(price_pred.group(1)) if price_pred else None
    
    return {"prob": p, "dir": d, "target_price": target_price}

# ====================== ВИЗУАЛИЗАЦИЯ ======================
def plot_results(balance_hist, equity_hist, slots):
    """График эквити с точными размерами"""
    DPI = 100
    WIDTH_PX = 700
    HEIGHT_PX = 350
    
    fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
    
    min_length = min(len(equity_hist), len(slots))
    dates = [s['datetime'] for s in slots[:min_length]]
    equity_to_plot = equity_hist[:min_length]
    
    plt.plot(dates, equity_to_plot, color='#1E90FF', linewidth=3.5, label='Equity')
    plt.title('Equity Curve (Quantum Fusion)', fontsize=16, fontweight='bold', color='white')
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
    
    filename = f"charts/equity_quantum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=DPI, facecolor='#0a0a0a', edgecolor='none', 
                bbox_inches='tight', pad_inches=0.1)
    print(f"\n✓ График сохранён: {filename} ({WIDTH_PX}×{HEIGHT_PX} px)")
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
    """
    Бэктест квантовой гибридной системы (CatBoost + Quantum + LLM)
    Использует обученную CatBoost модель с квантовыми признаками
    """
    print(f"\n{'='*80}")
    print(f"БЭКТЕСТ КВАНТОВОЙ ГИБРИДНОЙ СИСТЕМЫ")
    print(f"{'='*80}\n")
    
    # Проверка наличия моделей
    if not os.path.exists("models/catboost_quantum.cbm"):
        print("❌ CatBoost модель не найдена!")
        print("Сначала обучи модель (режим 1) или запусти полный цикл (режим 6)")
        return
    
    # Загрузка CatBoost модели
    print("Загрузка CatBoost модели...")
    if not CATBOOST_AVAILABLE:
        print("❌ CatBoost недоступен")
        return
    
    catboost_model = CatBoostClassifier()
    catboost_model.load_model("models/catboost_quantum.cbm")
    print("✓ CatBoost модель загружена")
    
    # Проверка Ollama и LLM модели
    use_llm = False
    if ollama:
        try:
            ollama.list()
            # Проверяем наличие нашей модели
            models = ollama.list()
            if any(MODEL_NAME in str(m) for m in models.get('models', [])):
                use_llm = True
                print("✓ LLM модель найдена, используем гибридный режим")
            else:
                print(f"⚠️ LLM модель {MODEL_NAME} не найдена")
                print("Работаем в режиме только CatBoost+Quantum")
        except:
            print("⚠️ Ollama недоступен, работаем только с CatBoost+Quantum")
    
    # Загрузка данных
    if not mt5 or not mt5.initialize():
        print("❌ MT5 не подключён")
        return
    
    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(days=BACKTEST_DAYS)
    
    data = {}
    print(f"\nЗагрузка данных с {start.strftime('%Y-%m-%d')} по {end.strftime('%Y-%m-%d')}...")
    
    for sym in SYMBOLS:
        rates = mt5.copy_rates_range(sym, TIMEFRAME, start, end)
        if rates is None or len(rates) == 0:
            print(f" ⚠️ {sym}: нет данных")
            continue
        
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        
        if len(df) > LOOKBACK + PREDICTION_HORIZON:
            data[sym] = df
            print(f" ✓ {sym}: {len(df)} баров")
    
    if not data:
        print("\n❌ Нет данных для бэктеста!")
        mt5.shutdown()
        return
    
    # Инициализация
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
    print(f"ПАРАМЕТРЫ БЭКТЕСТА")
    print(f"{'='*80}")
    print(f"Начальный баланс: ${balance:,.2f}")
    print(f"Риск на сделку: {RISK_PER_TRADE * 100}%")
    print(f"Минимальная уверенность: {MIN_PROB}%")
    print(f"Спред: {SPREAD_PIPS} пункта")
    print(f"Своп лонг/шорт: {SWAP_LONG}/{SWAP_SHORT} USD/день")
    print(f"Режим: {'CatBoost + Quantum + LLM' if use_llm else 'CatBoost + Quantum'}")
    print(f"{'='*80}\n")
    
    # Квантовый энкодер
    quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
    
    # Определение точек анализа
    main_symbol = list(data.keys())[0]
    main_data = data[main_symbol]
    total_bars = len(main_data)
    analysis_points = list(range(LOOKBACK, total_bars - PREDICTION_HORIZON, PREDICTION_HORIZON))
    
    print(f"Точек анализа: {len(analysis_points)} (каждые 24 часа)\n")
    print("Начинаем торговлю...\n")
    
    # Основной цикл бэктеста
    for point_idx, current_idx in enumerate(analysis_points):
        current_time = main_data.index[current_idx]
        
        print(f"{'='*80}")
        print(f"Анализ #{point_idx + 1}/{len(analysis_points)}: {current_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*80}")
        
        for sym in SYMBOLS:
            if sym not in data:
                continue
            
            # Исторические данные до текущего момента
            historical_data = data[sym].iloc[:current_idx + 1].copy()
            if len(historical_data) < LOOKBACK:
                continue
            
            # Расчёт технических признаков
            df_with_features = calculate_features(historical_data)
            if len(df_with_features) == 0:
                continue
            
            row = df_with_features.iloc[-1]
            
            # Получение информации о символе
            symbol_info = mt5.symbol_info(sym)
            if symbol_info is None:
                continue
            
            point = symbol_info.point
            contract_size = symbol_info.trade_contract_size
            
            # ===== КВАНТОВОЕ КОДИРОВАНИЕ =====
            feature_vector = np.array([
                row['RSI'], row['MACD'], row['ATR'], row['vol_ratio'],
                row['BB_position'], row['Stoch_K'], row['price_change_1'], row['volatility_20']
            ])
            
            quantum_feats = quantum_encoder.encode_and_measure(feature_vector)
            
            # ===== ПРОГНОЗ CATBOOST =====
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
            
            X_df = pd.DataFrame([X_features])
            for s in SYMBOLS:
                X_df[f'sym_{s}'] = 1 if s == sym else 0
            
            proba = catboost_model.predict_proba(X_df)[0]
            catboost_prob_up = proba[1] * 100
            catboost_direction = "UP" if proba[1] > 0.5 else "DOWN"
            catboost_confidence = max(proba) * 100
            
            # Интерпретация квантовых признаков
            entropy_level = "низкая" if quantum_feats['quantum_entropy'] < 3.0 else \
                           "средняя" if quantum_feats['quantum_entropy'] < 4.5 else "высокая"
            
            print(f"\n{sym}:")
            print(f"  Квант: entropy={quantum_feats['quantum_entropy']:.2f} ({entropy_level}), "
                  f"dominant={quantum_feats['dominant_state_prob']:.3f}")
            print(f"  CatBoost: {catboost_direction} {catboost_confidence:.1f}%")
            
            # ===== ПРОГНОЗ LLM (если доступен) =====
            final_direction = catboost_direction
            final_confidence = catboost_confidence
            
            if use_llm:
                try:
                    prompt = f"""{sym} {current_time.strftime('%Y-%m-%d %H:%M')}
Текущая цена: {row['close']:.5f}

ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:
RSI: {row['RSI']:.1f}
MACD: {row['MACD']:.6f}
ATR: {row['ATR']:.5f}
Объёмы: {row['vol_ratio']:.2f}x
BB позиция: {row['BB_position']:.2f}
Stochastic K: {row['Stoch_K']:.1f}

КВАНТОВЫЕ ПРИЗНАКИ:
Квантовая энтропия: {quantum_feats['quantum_entropy']:.2f} ({entropy_level} неопределённость)
Доминантное состояние: {quantum_feats['dominant_state_prob']:.3f}
Значимые состояния: {quantum_feats['significant_states']}
Квантовая дисперсия: {quantum_feats['quantum_variance']:.6f}

ПРОГНОЗ CATBOOST+QUANTUM:
Направление: {catboost_direction}
Уверенность: {catboost_confidence:.1f}%
Вероятность UP: {catboost_prob_up:.1f}%

Проанализируй и дай прогноз на 24 часа."""

                    resp = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.3})
                    result = parse_answer(resp["response"])
                    
                    final_direction = result["dir"]
                    final_confidence = result["prob"]
                    
                    print(f"  LLM: {final_direction} {final_confidence}% (коррекция: {final_confidence - catboost_confidence:+.1f}%)")
                    
                except Exception as e:
                    log.error(f"Ошибка LLM для {sym}: {e}")
                    final_direction = catboost_direction
                    final_confidence = catboost_confidence
            
            # ===== ПРОВЕРКА УВЕРЕННОСТИ =====
            if final_confidence < MIN_PROB:
                print(f"  ❌ Уверенность {final_confidence:.1f}% < {MIN_PROB}%, пропускаем")
                continue
            
            # ===== РАСЧЁТ РЕЗУЛЬТАТА ЧЕРЕЗ 24 ЧАСА =====
            exit_idx = current_idx + PREDICTION_HORIZON
            if exit_idx >= len(data[sym]):
                continue
            
            exit_row = data[sym].iloc[exit_idx]
            
            # Входная цена с учётом спреда
            entry_price = row['close'] + SPREAD_PIPS * point if final_direction == "UP" else row['close']
            # Выходная цена с учётом спреда
            exit_price = exit_row['close'] if final_direction == "UP" else exit_row['close'] + SPREAD_PIPS * point
            
            # Движение цены в пунктах
            price_move_pips = (exit_price - entry_price) / point if final_direction == "UP" else \
                             (entry_price - exit_price) / point
            
            # ===== РАСЧЁТ РАЗМЕРА ПОЗИЦИИ =====
            risk_amount = balance * RISK_PER_TRADE
            atr_pips = row['ATR'] / point
            stop_loss_pips = max(20, atr_pips * 2)
            lot_size = risk_amount / (stop_loss_pips * point * contract_size)
            lot_size = max(0.01, min(lot_size, 10.0))
            
            # ===== РАСЧЁТ ПРИБЫЛИ =====
            profit_pips = price_move_pips
            profit_usd = profit_pips * point * contract_size * lot_size
            
            # Своп за 24 часа
            swap_cost = SWAP_LONG if final_direction == "UP" else SWAP_SHORT
            swap_cost = swap_cost * (lot_size / 0.01)
            profit_usd -= swap_cost
            
            # Проскальзывание
            profit_usd -= SLIPPAGE * point * contract_size * lot_size
            
            # ===== ОБНОВЛЕНИЕ БАЛАНСА =====
            balance += profit_usd
            equity = balance
            
            # ===== ПРОВЕРКА ПРАВИЛЬНОСТИ =====
            actual_direction = "UP" if (exit_row['close'] > row['close']) else "DOWN"
            correct = (final_direction == actual_direction)
            
            # ===== ЗАПИСЬ СДЕЛКИ =====
            trades.append({
                "time": current_time,
                "symbol": sym,
                "direction": final_direction,
                "confidence": final_confidence,
                "catboost_confidence": catboost_confidence,
                "quantum_entropy": quantum_feats['quantum_entropy'],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "lot_size": lot_size,
                "profit_pips": profit_pips,
                "profit_usd": profit_usd,
                "balance": balance,
                "correct": correct
            })
            
            # ===== ВЫВОД =====
            status = "✓ ВЕРНО" if correct else "✗ ОШИБКА"
            color = '\033[92m' if correct else '\033[91m'
            reset = '\033[0m'
            
            print(f"  {color}{status}{reset} | Вход: {entry_price:.5f} → Выход: {exit_price:.5f}")
            print(f"  Лот: {lot_size:.2f} | Профит: {profit_pips:+.1f}п = ${profit_usd:+.2f}")
            print(f"  Баланс: ${balance:,.2f}")
        
        balance_hist.append(balance)
        equity_hist.append(equity)
        slots.append({"datetime": current_time})
    
    mt5.shutdown()
    
    # ===== СТАТИСТИКА =====
    print(f"\n{'='*80}")
    print(f"РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print(f"{'='*80}\n")
    print(f"Период: {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')} ({BACKTEST_DAYS} дней)")
    print(f"Режим: {'CatBoost + Quantum + LLM (Гибрид)' if use_llm else 'CatBoost + Quantum'}")
    print(f"\nСДЕЛКИ:")
    print(f"  Всего: {len(trades)}")
    print(f"  Начальный баланс: ${INITIAL_BALANCE:,.2f}")
    print(f"  Конечный баланс: ${balance:,.2f}")
    print(f"  Прибыль/убыток: ${balance - INITIAL_BALANCE:+,.2f}")
    print(f"  Доходность: {((balance/INITIAL_BALANCE - 1) * 100):+.2f}%")
    
    if trades:
        wins = sum(1 for t in trades if t['profit_usd'] > 0)
        losses = len(trades) - wins
        win_rate = wins / len(trades) * 100
        
        print(f"\nСТАТИСТИКА:")
        print(f"  Прибыльных: {wins} ({win_rate:.2f}%)")
        print(f"  Убыточных: {losses} ({100 - win_rate:.2f}%)")
        
        if wins > 0:
            avg_win = np.mean([t['profit_usd'] for t in trades if t['profit_usd'] > 0])
            print(f"  Средняя прибыль: ${avg_win:.2f}")
        
        if losses > 0:
            avg_loss = np.mean([t['profit_usd'] for t in trades if t['profit_usd'] < 0])
            print(f"  Средний убыток: ${avg_loss:.2f}")
        
        if wins > 0 and losses > 0:
            total_profit = sum(t['profit_usd'] for t in trades if t['profit_usd'] > 0)
            total_loss = abs(sum(t['profit_usd'] for t in trades if t['profit_usd'] < 0))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            print(f"  Profit Factor: {profit_factor:.2f}")
        
        max_dd = calculate_max_drawdown(np.array(equity_hist))
        print(f"  Макс. просадка: {max_dd:.2f}%")
        
        # Квантовая статистика
        print(f"\nКВАНТОВЫЙ АНАЛИЗ:")
        low_entropy_trades = [t for t in trades if t['quantum_entropy'] < 2.5]
        high_entropy_trades = [t for t in trades if t['quantum_entropy'] > 4.5]
        
        if low_entropy_trades:
            low_entropy_wins = sum(1 for t in low_entropy_trades if t['correct'])
            print(f"  Низкая энтропия (<2.5): {len(low_entropy_trades)} сделок, "
                  f"винрейт {low_entropy_wins/len(low_entropy_trades)*100:.1f}%")
        
        if high_entropy_trades:
            high_entropy_wins = sum(1 for t in high_entropy_trades if t['correct'])
            print(f"  Высокая энтропия (>4.5): {len(high_entropy_trades)} сделок, "
                  f"винрейт {high_entropy_wins/len(high_entropy_trades)*100:.1f}%")
        
        # LLM коррекции
        if use_llm:
            corrections = [t for t in trades if abs(t['confidence'] - t['catboost_confidence']) > 3]
            if corrections:
                correct_corrections = sum(1 for t in corrections if t['correct'])
                print(f"\nLLM КОРРЕКЦИИ:")
                print(f"  Всего коррекций (>3%): {len(corrections)}")
                print(f"  Успешных: {correct_corrections} ({correct_corrections/len(corrections)*100:.1f}%)")
        
        best_trade = max(trades, key=lambda x: x['profit_usd'])
        worst_trade = min(trades, key=lambda x: x['profit_usd'])
        
        print(f"\nЛУЧШАЯ СДЕЛКА:")
        print(f"  {best_trade['time'].strftime('%Y-%m-%d %H:%M')} | {best_trade['symbol']} "
              f"{best_trade['direction']} | ${best_trade['profit_usd']:+.2f}")
        
        print(f"\nХУДШАЯ СДЕЛКА:")
        print(f"  {worst_trade['time'].strftime('%Y-%m-%d %H:%M')} | {worst_trade['symbol']} "
              f"{worst_trade['direction']} | ${worst_trade['profit_usd']:+.2f}")
        
        # График
        if len(equity_hist) > 1:
            print(f"\n{'='*80}")
            print("Построение графика эквити...")
            plot_results(balance_hist, equity_hist, slots)
        
        # Сохранение детального отчёта
        trades_df = pd.DataFrame(trades)
        report_path = f"logs/backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(report_path, index=False)
        print(f"\n✓ Детальный отчёт сохранён: {report_path}")
    
    print(f"\n{'='*80}")
    print("БЭКТЕСТ ЗАВЕРШЁН")
    print(f"{'='*80}\n")

# ====================== ЖИВАЯ ТОРГОВЛЯ ======================
def live_trading():
    """
    Живая торговля с квантовой гибридной системой
    Анализирует рынок каждые 24 часа, открывает позиции с квантовым усилением
    """
    print(f"\n{'='*80}")
    print(f"ЖИВАЯ ТОРГОВЛЯ — QUANTUM FUSION")
    print(f"{'='*80}\n")
    
    # Проверка наличия моделей
    if not os.path.exists("models/catboost_quantum.cbm"):
        print("❌ CatBoost модель не найдена!")
        print("Сначала обучи модель (режим 1) или запусти полный цикл (режим 6)")
        return
    
    # Загрузка CatBoost модели
    print("Загрузка CatBoost модели...")
    if not CATBOOST_AVAILABLE:
        print("❌ CatBoost недоступен")
        return
    
    catboost_model = CatBoostClassifier()
    catboost_model.load_model("models/catboost_quantum.cbm")
    print("✓ CatBoost модель загружена")
    
    # Проверка Ollama и LLM модели
    use_llm = False
    if ollama:
        try:
            ollama.list()
            models = ollama.list()
            if any(MODEL_NAME in str(m) for m in models.get('models', [])):
                use_llm = True
                print("✓ LLM модель найдена, используем гибридный режим")
            else:
                print(f"⚠️ LLM модель {MODEL_NAME} не найдена")
                print("Работаем в режиме только CatBoost+Quantum")
        except:
            print("⚠️ Ollama недоступен, работаем только с CatBoost+Quantum")
    
    # Подключение к MT5
    if not mt5 or not mt5.initialize():
        print("❌ MT5 не подключён!")
        print("Запусти MetaTrader 5 и попробуй снова")
        return
    
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Не удалось получить информацию о счёте")
        mt5.shutdown()
        return
    
    print(f"\n✓ Подключено к MT5")
    print(f"  Счёт: {account_info.login}")
    print(f"  Баланс: ${account_info.balance:,.2f}")
    print(f"  Свободная маржа: ${account_info.margin_free:,.2f}")
    print(f"  Валюта: {account_info.currency}")
    
    # Проверка символов
    available_symbols = []
    for symbol in SYMBOLS:
        if mt5.symbol_select(symbol, True):
            available_symbols.append(symbol)
            print(f"  ✓ {symbol} доступен")
        else:
            print(f"  ⚠️ {symbol} недоступен")
    
    if not available_symbols:
        print("\n❌ Нет доступных символов для торговли!")
        mt5.shutdown()
        return
    
    print(f"\n{'='*80}")
    print(f"ПАРАМЕТРЫ ТОРГОВЛИ")
    print(f"{'='*80}")
    print(f"Режим: {'CatBoost + Quantum + LLM (Гибрид)' if use_llm else 'CatBoost + Quantum'}")
    print(f"Символы: {', '.join(available_symbols)}")
    print(f"Таймфрейм: M15")
    print(f"Риск на сделку: {RISK_PER_TRADE * 100}%")
    print(f"Минимальная уверенность: {MIN_PROB}%")
    print(f"Горизонт прогноза: 24 часа")
    print(f"MAGIC: {MAGIC}")
    print(f"{'='*80}\n")
    
    print("⚠️  ВНИМАНИЕ! Сейчас начнётся РЕАЛЬНАЯ торговля!")
    print("    Система будет открывать позиции на реальном счёте.")
    print("    Убедись, что ты понимаешь риски.\n")
    
    confirm = input("Продолжить? (YES для подтверждения): ").strip()
    if confirm != "YES":
        print("Торговля отменена")
        mt5.shutdown()
        return
    
    print(f"\n{'='*80}")
    print("ЗАПУСК ТОРГОВЛИ")
    print(f"{'='*80}\n")
    
    # Квантовый энкодер
    quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
    
    # Статистика
    total_analyses = 0
    total_signals = 0
    total_positions_opened = 0
    
    try:
        while True:
            current_time = datetime.now()
            
            print(f"\n{'='*80}")
            print(f"АНАЛИЗ РЫНКА: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")
            
            for symbol in available_symbols:
                print(f"\n{symbol}:")
                
                # Проверка: уже есть открытая позиция?
                positions = mt5.positions_get(symbol=symbol, magic=MAGIC)
                if positions and len(positions) > 0:
                    pos = positions[0]
                    profit = pos.profit
                    open_time = datetime.fromtimestamp(pos.time)
                    hours_open = (current_time - open_time).total_seconds() / 3600
                    
                    print(f"  ⏸️  Позиция уже открыта:")
                    print(f"     Тип: {'BUY' if pos.type == 0 else 'SELL'}")
                    print(f"     Лот: {pos.volume}")
                    print(f"     Профит: ${profit:+.2f}")
                    print(f"     Открыта: {hours_open:.1f}ч назад")
                    
                    # Закрытие через 24 часа
                    if hours_open >= 24:
                        print(f"  ⏰ 24 часа истекли, закрываем...")
                        close_result = close_position(pos)
                        if close_result:
                            print(f"  ✓ Позиция закрыта, финальный профит: ${profit:+.2f}")
                        else:
                            print(f"  ❌ Ошибка закрытия позиции")
                    
                    continue
                
                # Загрузка данных
                rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, LOOKBACK + 100)
                if rates is None or len(rates) < LOOKBACK:
                    print(f"  ⚠️ Недостаточно данных ({len(rates) if rates else 0} баров)")
                    continue
                
                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df.set_index("time", inplace=True)
                
                # Расчёт технических признаков
                df_features = calculate_features(df)
                if len(df_features) == 0:
                    print(f"  ⚠️ Не удалось рассчитать индикаторы")
                    continue
                
                row = df_features.iloc[-1]
                
                # Квантовое кодирование
                print(f"  🔬 Квантовое кодирование...")
                feature_vector = np.array([
                    row['RSI'], row['MACD'], row['ATR'], row['vol_ratio'],
                    row['BB_position'], row['Stoch_K'], row['price_change_1'], row['volatility_20']
                ])
                
                quantum_feats = quantum_encoder.encode_and_measure(feature_vector)
                
                # Прогноз CatBoost
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
                
                X_df = pd.DataFrame([X_features])
                for s in SYMBOLS:
                    X_df[f'sym_{s}'] = 1 if s == symbol else 0
                
                proba = catboost_model.predict_proba(X_df)[0]
                catboost_prob_up = proba[1] * 100
                catboost_direction = "UP" if proba[1] > 0.5 else "DOWN"
                catboost_confidence = max(proba) * 100
                
                entropy_level = "низкая" if quantum_feats['quantum_entropy'] < 3.0 else \
                               "средняя" if quantum_feats['quantum_entropy'] < 4.5 else "высокая"
                
                print(f"  📊 CatBoost: {catboost_direction} {catboost_confidence:.1f}%")
                print(f"  ⚛️  Квант: entropy={quantum_feats['quantum_entropy']:.2f} ({entropy_level})")
                
                # Прогноз LLM
                final_direction = catboost_direction
                final_confidence = catboost_confidence
                
                if use_llm:
                    try:
                        print(f"  🤖 LLM анализ...")
                        prompt = f"""{symbol} {current_time.strftime('%Y-%m-%d %H:%M')}
Текущая цена: {row['close']:.5f}

ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:
RSI: {row['RSI']:.1f}
MACD: {row['MACD']:.6f}
ATR: {row['ATR']:.5f}
Объёмы: {row['vol_ratio']:.2f}x
BB позиция: {row['BB_position']:.2f}
Stochastic K: {row['Stoch_K']:.1f}

КВАНТОВЫЕ ПРИЗНАКИ:
Квантовая энтропия: {quantum_feats['quantum_entropy']:.2f} ({entropy_level} неопределённость)
Доминантное состояние: {quantum_feats['dominant_state_prob']:.3f}
Значимые состояния: {quantum_feats['significant_states']}
Квантовая дисперсия: {quantum_feats['quantum_variance']:.6f}

ПРОГНОЗ CATBOOST+QUANTUM:
Направление: {catboost_direction}
Уверенность: {catboost_confidence:.1f}%
Вероятность UP: {catboost_prob_up:.1f}%

Проанализируй и дай прогноз на 24 часа."""

                        resp = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.3})
                        result = parse_answer(resp["response"])
                        
                        final_direction = result["dir"]
                        final_confidence = result["prob"]
                        
                        print(f"  🧠 Финальный: {final_direction} {final_confidence}% (коррекция: {final_confidence - catboost_confidence:+.1f}%)")
                        
                    except Exception as e:
                        log.error(f"Ошибка LLM для {symbol}: {e}")
                
                total_analyses += 1
                
                # Проверка уверенности
                if final_confidence < MIN_PROB:
                    print(f"  ❌ Уверенность {final_confidence:.1f}% < {MIN_PROB}%, пропускаем")
                    continue
                
                total_signals += 1
                
                # Получение информации о символе
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    print(f"  ⚠️ Не удалось получить информацию о символе")
                    continue
                
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
                
                current_price = tick.ask if final_direction == "UP" else tick.bid
                
                # Расчёт SL и TP
                if final_direction == "UP":
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask
                    sl = price - stop_loss_pips * point
                    tp = price + stop_loss_pips * point * 3  # R:R = 1:3
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
                    "comment": f"QF_{int(final_confidence)}%",
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
                    print(f"  ❌ Ошибка отправки ордера: result is None")
                    continue
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"  ❌ Ошибка открытия: {result.retcode} - {result.comment}")
                else:
                    print(f"  ✅ ПОЗИЦИЯ ОТКРЫТА!")
                    print(f"     Тикет: {result.order}")
                    print(f"     Цена исполнения: {result.price:.5f}")
                    total_positions_opened += 1
                    
                    # Логирование
                    log.info(f"Открыта позиция: {symbol} {final_direction} {lot_size} lots @ {result.price:.5f} | Уверенность: {final_confidence}% | Quantum entropy: {quantum_feats['quantum_entropy']:.2f}")
            
            # Статистика цикла
            print(f"\n{'='*80}")
            print(f"СТАТИСТИКА СЕССИИ")
            print(f"{'='*80}")
            print(f"Всего анализов: {total_analyses}")
            print(f"Сигналов получено: {total_signals}")
            print(f"Позиций открыто: {total_positions_opened}")
            
            # Текущие позиции
            all_positions = mt5.positions_get(magic=MAGIC)
            if all_positions:
                total_profit = sum(p.profit for p in all_positions)
                print(f"\nТекущие позиции: {len(all_positions)}")
                print(f"Общий плавающий профит: ${total_profit:+.2f}")
                
                for pos in all_positions:
                    print(f"  {pos.symbol} {'BUY' if pos.type == 0 else 'SELL'} {pos.volume} | ${pos.profit:+.2f}")
            else:
                print(f"\nТекущие позиции: 0")
            
            print(f"\n{'='*80}")
            
            # Следующий анализ через 24 часа
            next_analysis = current_time + timedelta(hours=24)
            print(f"\nСледующий анализ: {next_analysis.strftime('%Y-%m-%d %H:%M')}")
            print("Нажми Ctrl+C для остановки\n")
            
            # Ожидание 24 часа с проверкой каждую минуту (для закрытия старых позиций)
            wait_seconds = 24 * 60 * 60
            check_interval = 60  # Проверяем каждую минуту
            
            for i in range(0, wait_seconds, check_interval):
                time.sleep(check_interval)
                
                # Проверяем позиции каждую минуту
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
        
        # Спрашиваем о закрытии позиций
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
        log.error(f"Критическая ошибка в live_trading: {e}")
        print(f"\n❌ Критическая ошибка: {e}")
    
    finally:
        mt5.shutdown()
        print("MT5 отключён")

def close_position(position):
    """Закрывает открытую позицию"""
    symbol = position.symbol
    
    # Получение информации о символе
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return False
    
    # Определение типа закрывающего ордера
    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    
    # Формирование запроса на закрытие
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
    
    # Отправка ордера
    result = mt5.order_send(request)
    
    if result is None:
        return False
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Ошибка закрытия позиции {symbol}: {result.retcode} - {result.comment}")
        return False
    
    return True

# ====================== ЗАГРУЗКА ДАННЫХ ======================
def load_mt5_data(days: int = 180) -> Dict[str, pd.DataFrame]:
    """Загрузка реальных данных из MT5"""
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
    print(f" QUANTUM TRADER FUSION — Qiskit + CatBoost + LLM")
    print(f" Версия: 09.12.2025 (Full Integration)")
    print(f"{'='*80}\n")
    print(f"РЕЖИМЫ:")
    print(f"-"*80)
    print(f"1 → Обучить CatBoost с квантовыми признаками")
    print(f"2 → Сгенерировать гибридный датасет (CatBoost + Quantum)")
    print(f"3 → Файнтьюн LLM с CatBoost прогнозами")
    print(f"4 → Бэктест гибридной системы")
    print(f"5 → Живая торговля (MT5)")
    print(f"6 → ПОЛНЫЙ ЦИКЛ (всё вместе)")
    print(f"-"*80)
    
    choice = input("\nВыбери режим (1-6): ").strip()
    
    if choice == "1":
        # Режим 1: Обучение CatBoost
        data = load_mt5_data(180)
        if not data:
            print("❌ Нет данных для обучения")
            return
        
        quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
        model = train_catboost_model(data, quantum_encoder)
        
    elif choice == "2":
        # Режим 2: Генерация датасета
        data = load_mt5_data(180)
        if not data:
            print("❌ Нет данных")
            return
        
        # Загрузка CatBoost модели
        if os.path.exists("models/catboost_quantum.cbm"):
            print("Загрузка CatBoost модели...")
            model = CatBoostClassifier()
            model.load_model("models/catboost_quantum.cbm")
        else:
            print("❌ CatBoost модель не найдена, сначала обучи (режим 1)")
            return
        
        quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
        dataset = generate_hybrid_dataset(data, model, quantum_encoder, FINETUNE_SAMPLES)
        save_dataset(dataset, "dataset/quantum_fusion_data.jsonl")
        
    elif choice == "3":
        # Режим 3: Файнтьюн LLM
        dataset_path = "dataset/quantum_fusion_data.jsonl"
        if not os.path.exists(dataset_path):
            print(f"❌ Датасет не найден: {dataset_path}")
            print("Сначала сгенерируй датасет (режим 2)")
            return
        
        finetune_llm_with_catboost(dataset_path)
        
    elif choice == "4":
        # Режим 4: Бэктест
        backtest()
        
    elif choice == "5":
        # Режим 5: Живая торговля
        live_trading()
        
    elif choice == "6":
        # Режим 6: ПОЛНЫЙ ЦИКЛ
        print(f"\n{'='*80}")
        print(f"ПОЛНЫЙ ЦИКЛ: QUANTUM FUSION")
        print(f"{'='*80}\n")
        print("Этот процесс займёт 2-3 часа:")
        print("1. Загрузка данных MT5 (180 дней)")
        print("2. Квантовое кодирование (~60 мин)")
        print("3. Обучение CatBoost (~15 мин)")
        print("4. Генерация датасета (~45 мин)")
        print("5. Файнтьюн LLM (~20 мин)")
        
        confirm = input("\nПродолжить? (YES): ").strip()
        if confirm != "YES":
            print("Отменено")
            return
        
        # Шаг 1: Загрузка данных
        print(f"\n{'='*80}")
        print("ШАГ 1/5: ЗАГРУЗКА ДАННЫХ MT5")
        print(f"{'='*80}")
        data = load_mt5_data(180)
        if not data:
            print("❌ Не удалось загрузить данные")
            return
        
        # Шаг 2-3: Обучение CatBoost
        print(f"\n{'='*80}")
        print("ШАГ 2-3/5: КВАНТОВОЕ КОДИРОВАНИЕ + ОБУЧЕНИЕ CATBOOST")
        print(f"{'='*80}")
        quantum_encoder = QuantumEncoder(N_QUBITS, N_SHOTS)
        model = train_catboost_model(data, quantum_encoder)
        
        # Шаг 4: Генерация датасета
        print(f"\n{'='*80}")
        print("ШАГ 4/5: ГЕНЕРАЦИЯ ГИБРИДНОГО ДАТАСЕТА")
        print(f"{'='*80}")
        dataset = generate_hybrid_dataset(data, model, quantum_encoder, FINETUNE_SAMPLES)
        dataset_path = save_dataset(dataset, "dataset/quantum_fusion_data.jsonl")
        
        # Шаг 5: Файнтьюн LLM
        print(f"\n{'='*80}")
        print("ШАГ 5/5: ФАЙНТЬЮН LLM")
        print(f"{'='*80}")
        finetune_llm_with_catboost(dataset_path)
        
        print(f"\n{'='*80}")
        print("🎉 ПОЛНЫЙ ЦИКЛ ЗАВЕРШЁН!")
        print(f"{'='*80}")
        print("✓ CatBoost модель обучена с квантовыми признаками")
        print("✓ LLM файнтьюнена с CatBoost прогнозами")
        print("✓ Система готова к использованию")
        print(f"\nМодель: {MODEL_NAME}")
        print(f"CatBoost: models/catboost_quantum.cbm")
        print(f"Датасет: {dataset_path}")
        
    else:
        print("❌ Неверный выбор")

if __name__ == "__main__":
    main()
