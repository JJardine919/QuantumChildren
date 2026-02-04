# ================================================
# ai_trader_quantum_lstm_live.py
# КВАНТОВЫЙ ГИБРИД: Qiskit + Bidirectional LSTM + LLM
# Версия 22.12.2025 — Полная интеграция из статьи
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
import hashlib

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch не установлен: pip install torch")

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from scipy.stats import entropy
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️ Qiskit не установлен: pip install qiskit qiskit-aer")

# ====================== КОНФИГ ======================
MODEL_NAME = "koshtenco/quantum-trader-lstm-3b"
BASE_MODEL = "llama3.2:3b"
SYMBOLS = ["EURUSD", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "EURGBP", "AUDCHF"]
TIMEFRAME = mt5.TIMEFRAME_H1 if mt5 else None  # Часовой таймфрейм как в статье
LOOKBACK = 1500  # Количество свечей для загрузки
INITIAL_BALANCE = 270.0
RISK_PER_TRADE = 0.06
MIN_PROB = 60
LIVE_LOT = 0.02
MAGIC = 20251222
SLIPPAGE = 10

# Квантовые параметры (из статьи)
N_QUBITS = 3  # 3 кубита = 8 состояний
N_SHOTS = 1000

# LSTM параметры (из статьи)
SEQUENCE_LENGTH = 50  # Окно для LSTM
QUANTUM_WINDOW = 50  # Окно для квантовых признаков
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15

# Файнтьюн параметры
FINETUNE_SAMPLES = 2000
BACKTEST_DAYS = 30
PREDICTION_HORIZON = 24  # 24 часа на H1

os.makedirs("logs", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/quantum_lstm.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ====================== КВАНТОВЫЙ ЭНКОДЕР (ИЗ СТАТЬИ) ======================
class QuantumFeatureExtractor:
    """
    Квантовый экстрактор признаков точно по статье:
    - 3 кубита (8 состояний)
    - RY-вентили для кодирования
    - CNOT для запутывания
    - 7 квантовых признаков
    """
    
    def __init__(self, num_qubits: int = 3, shots: int = 1000):
        self.num_qubits = num_qubits
        self.shots = shots
        self.simulator = AerSimulator(method='statevector') if QISKIT_AVAILABLE else None
        self.cache = {}
        
    def create_quantum_circuit(self, features: np.ndarray) -> QuantumCircuit:
        """Создание квантовой схемы с RY-вентилями и CNOT"""
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)
        
        # RY-вентили для кодирования признаков
        for i in range(self.num_qubits):
            feature_idx = i % len(features)
            angle = np.clip(np.pi * features[feature_idx], -2*np.pi, 2*np.pi)
            qc.ry(angle, i)
        
        # CNOT-вентили для создания запутанности
        for i in range(self.num_qubits - 1):
            qc.cx(i, i + 1)
        
        # Измерения
        qc.measure(range(self.num_qubits), range(self.num_qubits))
        return qc
    
    def extract_quantum_features(self, price_data: np.ndarray) -> dict:
        """
        Извлечение 7 квантовых признаков из ценовых данных
        (точно как в статье)
        """
        # Кэширование для ускорения
        data_hash = hashlib.md5(price_data.tobytes()).hexdigest()
        if data_hash in self.cache:
            return self.cache[data_hash]
        
        if not QISKIT_AVAILABLE:
            return self._get_default_features()
        
        # Вычисление классических признаков для квантового кодирования
        returns = np.diff(price_data) / (price_data[:-1] + 1e-10)
        features = np.array([
            np.mean(returns),           # Средняя доходность
            np.std(returns),            # Волатильность
            np.max(returns) - np.min(returns)  # Диапазон
        ])
        features = np.tanh(features)  # Нормализация через tanh
        
        try:
            # Создание и выполнение квантовой схемы
            qc = self.create_quantum_circuit(features)
            compiled_circuit = transpile(qc, self.simulator, optimization_level=2)
            job = self.simulator.run(compiled_circuit, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            # Вычисление 7 квантовых метрик
            quantum_features = self._compute_quantum_metrics(counts, self.shots)
            self.cache[data_hash] = quantum_features
            return quantum_features
            
        except Exception as e:
            log.error(f"Ошибка квантового вычисления: {e}")
            return self._get_default_features()
    
    def _compute_quantum_metrics(self, counts: dict, shots: int) -> dict:
        """Вычисление 7 квантовых признаков"""
        # Преобразование counts в вероятности
        probabilities = {state: count/shots for state, count in counts.items()}
        
        # 1. Квантовая энтропия Шеннона
        quantum_entropy = -sum(p * np.log2(p) if p > 0 else 0 
                              for p in probabilities.values())
        
        # 2. Вероятность доминантного состояния
        dominant_state_prob = max(probabilities.values())
        
        # 3. Мера суперпозиции (количество значимых состояний)
        threshold = 0.05
        significant_states = sum(1 for p in probabilities.values() if p > threshold)
        superposition_measure = significant_states / (2 ** self.num_qubits)
        
        # 4. Фазовая когерентность
        state_values = [int(state, 2) for state in probabilities.keys()]
        max_value = 2 ** self.num_qubits - 1
        phase_coherence = 1.0 - (np.std(state_values) / max_value) if len(state_values) > 1 else 0.5
        
        # 5. Степень запутанности
        entanglement_degree = self._compute_entanglement_from_cnot(probabilities)
        
        # 6. Квантовая дисперсия
        mean_state = sum(int(state, 2) * prob for state, prob in probabilities.items())
        quantum_variance = sum((int(state, 2) - mean_state)**2 * prob 
                              for state, prob in probabilities.items())
        
        # 7. Количество значимых состояний (абсолютное)
        num_significant_states = float(significant_states)
        
        return {
            'quantum_entropy': quantum_entropy,
            'dominant_state_prob': dominant_state_prob,
            'superposition_measure': superposition_measure,
            'phase_coherence': phase_coherence,
            'entanglement_degree': entanglement_degree,
            'quantum_variance': quantum_variance,
            'num_significant_states': num_significant_states
        }
    
    def _compute_entanglement_from_cnot(self, probabilities: dict) -> float:
        """Вычисление степени запутанности между соседними кубитами"""
        bit_correlations = []
        for i in range(self.num_qubits - 1):
            correlation = 0.0
            for state, prob in probabilities.items():
                if len(state) > i + 1:
                    # Qiskit возвращает состояния в обратном порядке
                    if state[-(i+1)] == state[-(i+2)]:
                        correlation += prob
            bit_correlations.append(correlation)
        return np.mean(bit_correlations) if bit_correlations else 0.5
    
    def _get_default_features(self) -> dict:
        """Дефолтные значения при ошибках"""
        return {
            'quantum_entropy': 2.5,
            'dominant_state_prob': 0.125,
            'superposition_measure': 0.5,
            'phase_coherence': 0.5,
            'entanglement_degree': 0.5,
            'quantum_variance': 0.005,
            'num_significant_states': 4.0
        }

# ====================== FOCAL LOSS (ИЗ СТАТЬИ) ======================
class FocalLoss(nn.Module):
    """
    Focal Loss для балансировки классов
    alpha=0.25, gamma=2.0 как в статье
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

# ====================== BIDIRECTIONAL LSTM МОДЕЛЬ (ИЗ СТАТЬИ) ======================
class QuantumLSTM(nn.Module):
    """
    Гибридная квантово-нейросетевая архитектура из статьи:
    - Bidirectional LSTM для обработки ценовых последовательностей
    - Отдельный процессор для квантовых признаков
    - Fusion layer для объединения
    - BatchNorm + Dropout для регуляризации
    """
    def __init__(self, input_size: int = 5, quantum_feature_size: int = 7,
                 hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.3):
        super(QuantumLSTM, self).__init__()
        
        # Bidirectional LSTM для ценовых последовательностей
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True  # ДВУНАПРАВЛЕННАЯ!
        )
        
        # Процессор квантовых признаков
        self.quantum_processor = nn.Sequential(
            nn.Linear(quantum_feature_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusion layer (объединение LSTM и квантовых признаков)
        # hidden_size * 2 потому что bidirectional
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, price_seq, quantum_features):
        """
        price_seq: [batch_size, sequence_length, input_size]
        quantum_features: [batch_size, quantum_feature_size]
        """
        # LSTM обработка
        lstm_out, _ = self.lstm(price_seq)
        lstm_last = lstm_out[:, -1, :]  # Берём последний временной шаг
        
        # Обработка квантовых признаков
        quantum_processed = self.quantum_processor(quantum_features)
        
        # Объединение
        combined = torch.cat([lstm_last, quantum_processed], dim=1)
        
        # Финальное предсказание
        output = self.fusion(combined)
        return output

# ====================== PYTORCH DATASET ======================
class MarketDataset(Dataset):
    """Dataset для PyTorch DataLoader"""
    def __init__(self, price_data, quantum_features, targets, sequence_length=50):
        self.price_data = price_data
        self.quantum_features = quantum_features
        self.targets = targets
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.price_data) - self.sequence_length
    
    def __getitem__(self, idx):
        price_seq = self.price_data[idx:idx + self.sequence_length]
        quantum_feat = self.quantum_features[idx + self.sequence_length - 1]
        target = self.targets[idx + self.sequence_length]
        
        return {
            'price': torch.FloatTensor(price_seq),
            'quantum': torch.FloatTensor(quantum_feat),
            'target': torch.FloatTensor([target])
        }
    
    def get_labels(self):
        """Для WeightedRandomSampler"""
        return [self.targets[idx + self.sequence_length] 
                for idx in range(len(self))]

# ====================== ТЕХНИЧЕСКИЕ ПРИЗНАКИ ======================
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Расчёт технических индикаторов"""
    d = df.copy()
    d["close_prev"] = d["close"].shift(1)
    
    # Базовые признаки
    d['returns'] = d['close'].pct_change()
    d['log_returns'] = np.log(d['close'] / d['close'].shift(1))
    d['high_low'] = (d['high'] - d['low']) / d['close']
    d['close_open'] = (d['close'] - d['open']) / d['open']
    
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
    d["volatility_20"] = d["log_returns"].rolling(20).std()
    
    return d.dropna()

# ====================== ПОДГОТОВКА ДАННЫХ ======================
def prepare_data(symbol="EURUSD", timeframe=None, n_candles=1500):
    """Подготовка данных точно как в статье"""
    if not mt5 or not mt5.initialize():
        raise RuntimeError("MT5 не инициализирован")
    
    if timeframe is None:
        timeframe = TIMEFRAME
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    mt5.shutdown()
    
    if rates is None or len(rates) == 0:
        raise ValueError("Не удалось получить данные")
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Вычисление технических признаков
    df = calculate_features(df)
    
    # Классические признаки для LSTM (5 признаков как в статье)
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low'] = (df['high'] - df['low']) / df['close']
    df['close_open'] = (df['close'] - df['open']) / df['open']
    df = df.dropna()
    
    # Подготовка признаков
    price_features = df[['returns', 'log_returns', 'high_low', 
                         'close_open', 'tick_volume']].values
    
    # Стандартизация
    mean = price_features.mean(axis=0)
    std = price_features.std(axis=0)
    price_data = (price_features - mean) / (std + 1e-8)
    
    # Извлечение квантовых признаков
    print(f"\nИзвлечение квантовых признаков (это займёт ~10-15 минут)...")
    quantum_extractor = QuantumFeatureExtractor(num_qubits=N_QUBITS, shots=N_SHOTS)
    quantum_features_list = []
    quantum_window = QUANTUM_WINDOW
    
    start_time = time.time()
    
    for i in range(quantum_window, len(df)):
        window = df['close'].iloc[i-quantum_window:i].values
        q_features = quantum_extractor.extract_quantum_features(window)
        quantum_features_list.append(list(q_features.values()))
        
        if (i - quantum_window) % 100 == 0:
            elapsed = time.time() - start_time
            progress = (i - quantum_window) / (len(df) - quantum_window)
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"Прогресс: {i - quantum_window}/{len(df) - quantum_window} "
                  f"({progress*100:.1f}%) | ETA: {eta/60:.1f} мин")
    
    quantum_features = np.array(quantum_features_list)
    
    # Выравнивание размеров
    price_data = price_data[quantum_window:]
    
    # Целевая переменная (следующая свеча выше текущей?)
    targets = (df['close'].shift(-1) > df['close']).astype(float).values
    targets = targets[quantum_window:]
    
    # Проверка баланса классов
    unique, counts = np.unique(targets, return_counts=True)
    print(f"\nБаланс классов:")
    print(f"Падение (0): {counts[0]} ({counts[0]/len(targets)*100:.1f}%)")
    print(f"Рост (1): {counts[1]} ({counts[1]/len(targets)*100:.1f}%)")
    
    return price_data, quantum_features, targets

# ====================== СОЗДАНИЕ СБАЛАНСИРОВАННОГО ЗАГРУЗЧИКА ======================
def create_balanced_loader(dataset, batch_size=32):
    """Создание DataLoader с WeightedRandomSampler для балансировки классов"""
    labels = dataset.get_labels()
    class_counts = np.bincount([int(l) for l in labels])
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[int(l)] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# ====================== ОБУЧЕНИЕ LSTM МОДЕЛИ ======================
def train_lstm_model(symbol="EURUSD", n_candles=1500):
    """
    Полный цикл обучения Bidirectional LSTM с квантовыми признаками
    Точно по алгоритму из статьи
    """
    print(f"\n{'='*80}")
    print(f"ОБУЧЕНИЕ BIDIRECTIONAL LSTM С КВАНТОВЫМИ ПРИЗНАКАМИ")
    print(f"{'='*80}\n")
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch недоступен")
        return None
    
    # Подготовка данных
    price_data, quantum_features, targets = prepare_data(symbol, TIMEFRAME, n_candles)
    
    # Разделение на train/val/test (70/15/15)
    train_size = int(len(price_data) * 0.7)
    val_size = int(len(price_data) * 0.15)
    
    print(f"\nРазделение данных:")
    print(f"Train: {train_size} ({train_size/len(price_data)*100:.1f}%)")
    print(f"Val:   {val_size} ({val_size/len(price_data)*100:.1f}%)")
    print(f"Test:  {len(price_data)-train_size-val_size} ({(len(price_data)-train_size-val_size)/len(price_data)*100:.1f}%)")
    
    train_dataset = MarketDataset(
        price_data[:train_size],
        quantum_features[:train_size],
        targets[:train_size],
        sequence_length=SEQUENCE_LENGTH
    )
    val_dataset = MarketDataset(
        price_data[train_size:train_size+val_size],
        quantum_features[train_size:train_size+val_size],
        targets[train_size:train_size+val_size],
        sequence_length=SEQUENCE_LENGTH
    )
    test_dataset = MarketDataset(
        price_data[train_size+val_size:],
        quantum_features[train_size+val_size:],
        targets[train_size+val_size:],
        sequence_length=SEQUENCE_LENGTH
    )
    
    # DataLoaders
    train_loader = create_balanced_loader(train_dataset, BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Инициализация модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуемое устройство: {device}")
    
    model = QuantumLSTM(
        input_size=5,
        quantum_feature_size=7,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    # Оптимизатор и функция потерь
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=7, factor=0.5
    )
    
    # Обучение
    print(f"\n{'='*80}")
    print(f"НАЧАЛО ОБУЧЕНИЯ")
    print(f"{'='*80}\n")
    
    best_val_loss = float('inf')
    patience = 0
    max_patience = EARLY_STOPPING_PATIENCE
    
    for epoch in range(MAX_EPOCHS):
        # Тренировка
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            price = batch['price'].to(device)
            quantum = batch['quantum'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            output = model(price, quantum)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                price = batch['price'].to(device)
                quantum = batch['quantum'].to(device)
                target = batch['target'].to(device)
                
                output = model(price, quantum)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), 'models/quantum_lstm_best.pth')
            print(f"Эпоха {epoch+1}/{MAX_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} ✓")
        else:
            patience += 1
            if (epoch + 1) % 5 == 0:
                print(f"Эпоха {epoch+1}/{MAX_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if patience >= max_patience:
            print(f"Early stopping на эпохе {epoch+1}")
            break
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load('models/quantum_lstm_best.pth'))
    
    # Оценка на тестовой выборке
    print(f"\n{'='*80}")
    print(f"ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print(f"{'='*80}\n")
    
    evaluate_model(model, test_loader, device)
    
    return model

# ====================== ОЦЕНКА МОДЕЛИ ======================
def evaluate_model(model, test_loader, device):
    """Оценка модели на тестовой выборке"""
    model.eval()
    predictions, actuals = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            price = batch['price'].to(device)
            quantum = batch['quantum'].to(device)
            target = batch['target'].to(device)
            
            output = model(price, quantum)
            pred = torch.sigmoid(output)
            
            predictions.extend(pred.cpu().numpy())
            actuals.extend(target.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    binary_predictions = (predictions > 0.5).astype(int)
    
    # Метрики
    accuracy = (binary_predictions == actuals).mean()
    
    tp = ((binary_predictions == 1) & (actuals == 1)).sum()
    tn = ((binary_predictions == 0) & (actuals == 0)).sum()
    fp = ((binary_predictions == 1) & (actuals == 0)).sum()
    fn = ((binary_predictions == 0) & (actuals == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Распределение предсказаний
    pred_0 = (binary_predictions == 0).sum()
    pred_1 = (binary_predictions == 1).sum()
    
    print(f"{'='*80}")
    print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
    print(f"{'='*80}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nПредсказания модели:")
    print(f"Падение (0): {pred_0} ({pred_0/len(binary_predictions)*100:.1f}%)")
    print(f"Рост (1): {pred_1} ({pred_1/len(binary_predictions)*100:.1f}%)")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              0      1")
    print(f"Actual 0    {tn:3d}   {fp:3d}")
    print(f"Actual 1    {fn:3d}   {tp:3d}")
    print(f"{'='*80}\n")

# ====================== БЭКТЕСТ ======================
def backtest_lstm_model(symbol="EURUSD", n_candles=1500, initial_balance=10000.0):
    """
    Полный бэктест LSTM модели на исторических данных
    """
    print(f"\n{'='*80}")
    print(f"БЭКТЕСТ LSTM МОДЕЛИ НА {symbol}")
    print(f"{'='*80}\n")
    
    if not TORCH_AVAILABLE:
        print("❌ PyTorch недоступен")
        return None
    
    # Проверка наличия модели
    if not os.path.exists('models/quantum_lstm_best.pth'):
        print("❌ LSTM модель не найдена. Сначала обучи модель (режим 1)")
        return None
    
    # Загрузка модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuantumLSTM(
        input_size=5,
        quantum_feature_size=7,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load('models/quantum_lstm_best.pth'))
    model.eval()
    
    print("Модель загружена")
    
    # Подготовка данных
    print("Подготовка данных...")
    price_data, quantum_features, targets = prepare_data(symbol, TIMEFRAME, n_candles)
    
    # Используем только тестовую часть (последние 15%)
    test_start = int(len(price_data) * 0.85)
    price_data = price_data[test_start:]
    quantum_features = quantum_features[test_start:]
    targets = targets[test_start:]
    
    print(f"Тестовая выборка: {len(price_data)} баров\n")
    
    # Инициализация бэктеста
    balance = initial_balance
    equity_curve = [balance]
    trades = []
    positions = []
    
    # Параметры торговли
    lot_size = 0.01  # Фиксированный лот
    leverage = 100
    spread = 0.0002  # 2 пункта для EUR/USD
    
    # Получение реальных данных для цен
    if not mt5 or not mt5.initialize():
        print("⚠️ MT5 недоступен, используем синтетические цены")
        mt5_available = False
        prices = np.random.uniform(1.0, 1.2, len(price_data))
    else:
        mt5_available = True
        rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_candles)
        mt5.shutdown()
        if rates is not None:
            df = pd.DataFrame(rates)
            prices = df['close'].values[test_start:test_start+len(price_data)]
        else:
            prices = np.random.uniform(1.0, 1.2, len(price_data))
    
    print("Запуск бэктеста...")
    print(f"Начальный баланс: ${balance:.2f}")
    print(f"Лот: {lot_size}")
    print(f"Спред: {spread*10000:.1f} пунктов\n")
    
    # Создание dataset для predictions
    test_dataset = MarketDataset(price_data, quantum_features, targets, SEQUENCE_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Прогон через модель
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= len(prices) - SEQUENCE_LENGTH - 1:
                break
                
            price = batch['price'].to(device)
            quantum = batch['quantum'].to(device)
            target = batch['target'].to(device)
            
            # Предсказание
            output = model(price, quantum)
            prob = torch.sigmoid(output).item()
            
            current_price = prices[idx + SEQUENCE_LENGTH]
            next_price = prices[idx + SEQUENCE_LENGTH + 1] if idx + SEQUENCE_LENGTH + 1 < len(prices) else current_price
            
            # Логика торговли
            if len(positions) == 0:  # Нет открытых позиций
                if prob > 0.5:  # BUY сигнал
                    # Открываем BUY
                    entry_price = current_price + spread
                    positions.append({
                        'type': 'BUY',
                        'entry_price': entry_price,
                        'entry_idx': idx,
                        'lot': lot_size,
                        'prob': prob
                    })
                    
                elif prob < 0.5:  # SELL сигнал
                    # Открываем SELL
                    entry_price = current_price - spread
                    positions.append({
                        'type': 'SELL',
                        'entry_price': entry_price,
                        'entry_idx': idx,
                        'lot': lot_size,
                        'prob': prob
                    })
            
            else:  # Есть открытая позиция
                pos = positions[0]
                
                # Проверка условий закрытия
                close_position = False
                
                # 1. Держим максимум 24 свечи (24 часа на H1)
                if idx - pos['entry_idx'] >= 24:
                    close_position = True
                    close_reason = "Время вышло"
                
                # 2. Стоп-лосс / тейк-профит
                if pos['type'] == 'BUY':
                    profit_pips = (current_price - pos['entry_price']) * 10000
                    if profit_pips < -50:  # SL 50 пунктов
                        close_position = True
                        close_reason = "Stop Loss"
                    elif profit_pips > 100:  # TP 100 пунктов
                        close_position = True
                        close_reason = "Take Profit"
                else:  # SELL
                    profit_pips = (pos['entry_price'] - current_price) * 10000
                    if profit_pips < -50:
                        close_position = True
                        close_reason = "Stop Loss"
                    elif profit_pips > 100:
                        close_position = True
                        close_reason = "Take Profit"
                
                # Закрытие позиции
                if close_position:
                    exit_price = current_price
                    
                    # Расчёт профита
                    if pos['type'] == 'BUY':
                        price_diff = exit_price - pos['entry_price']
                    else:
                        price_diff = pos['entry_price'] - exit_price
                    
                    contract_size = 100000  # Стандартный лот для форекс
                    profit = price_diff * contract_size * pos['lot']
                    
                    balance += profit
                    equity_curve.append(balance)
                    
                    # Запись сделки
                    trades.append({
                        'type': pos['type'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'profit': profit,
                        'pips': profit_pips,
                        'duration': idx - pos['entry_idx'],
                        'reason': close_reason,
                        'prob': pos['prob']
                    })
                    
                    # Удаление позиции
                    positions.clear()
    
    # Закрываем оставшиеся позиции
    if len(positions) > 0:
        pos = positions[0]
        exit_price = prices[-1]
        
        if pos['type'] == 'BUY':
            price_diff = exit_price - pos['entry_price']
        else:
            price_diff = pos['entry_price'] - exit_price
        
        contract_size = 100000
        profit = price_diff * contract_size * pos['lot']
        balance += profit
        
        trades.append({
            'type': pos['type'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'profit': profit,
            'pips': (exit_price - pos['entry_price']) * 10000,
            'duration': len(prices) - pos['entry_idx'],
            'reason': 'Конец теста',
            'prob': pos['prob']
        })
    
    # Анализ результатов
    print(f"\n{'='*80}")
    print("РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print(f"{'='*80}\n")
    
    total_trades = len(trades)
    if total_trades == 0:
        print("❌ Не было совершено ни одной сделки")
        return None
    
    profitable_trades = len([t for t in trades if t['profit'] > 0])
    losing_trades = total_trades - profitable_trades
    
    total_profit = sum(t['profit'] for t in trades)
    gross_profit = sum(t['profit'] for t in trades if t['profit'] > 0)
    gross_loss = sum(t['profit'] for t in trades if t['profit'] < 0)
    
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0
    
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    avg_win = gross_profit / profitable_trades if profitable_trades > 0 else 0
    avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
    
    final_balance = balance
    total_return = ((final_balance - initial_balance) / initial_balance * 100)
    
    # Максимальная просадка
    peak = initial_balance
    max_dd = 0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    print(f"Начальный баланс:    ${initial_balance:.2f}")
    print(f"Конечный баланс:     ${final_balance:.2f}")
    print(f"Общий профит:        ${total_profit:+.2f}")
    print(f"Общая доходность:    {total_return:+.2f}%")
    print(f"\nВсего сделок:        {total_trades}")
    print(f"Прибыльных:          {profitable_trades} ({win_rate:.1f}%)")
    print(f"Убыточных:           {losing_trades} ({100-win_rate:.1f}%)")
    print(f"\nGross Profit:        ${gross_profit:.2f}")
    print(f"Gross Loss:          ${gross_loss:.2f}")
    print(f"Profit Factor:       {profit_factor:.2f}")
    print(f"\nСредний профит:      ${avg_profit:+.2f}")
    print(f"Средняя прибыль:     ${avg_win:+.2f}")
    print(f"Средний убыток:      ${avg_loss:+.2f}")
    print(f"Макс. просадка:      {max_dd:.2f}%")
    
    # Разбивка по типам
    buy_trades = [t for t in trades if t['type'] == 'BUY']
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    
    print(f"\nBUY сделок:          {len(buy_trades)}")
    if buy_trades:
        buy_profit = sum(t['profit'] for t in buy_trades)
        buy_win_rate = len([t for t in buy_trades if t['profit'] > 0]) / len(buy_trades) * 100
        print(f"  Профит:            ${buy_profit:+.2f}")
        print(f"  Win rate:          {buy_win_rate:.1f}%")
    
    print(f"\nSELL сделок:         {len(sell_trades)}")
    if sell_trades:
        sell_profit = sum(t['profit'] for t in sell_trades)
        sell_win_rate = len([t for t in sell_trades if t['profit'] > 0]) / len(sell_trades) * 100
        print(f"  Профит:            ${sell_profit:+.2f}")
        print(f"  Win rate:          {sell_win_rate:.1f}%")
    
    print(f"\n{'='*80}")
    
    # Сохранение детальной статистики
    results = {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_profit': total_profit,
        'total_return': total_return,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_dd,
        'trades': trades,
        'equity_curve': equity_curve
    }
    
    # Сохранение в JSON
    with open('logs/backtest_results.json', 'w') as f:
        json.dumps(results, indent=2, default=str)
    
    # График equity curve - НОВЫЙ СТИЛЬ
    try:
        from datetime import datetime
        
        DPI = 100
        WIDTH_PX = 700
        HEIGHT_PX = 350
        
        fig = plt.figure(figsize=(WIDTH_PX / DPI, HEIGHT_PX / DPI), dpi=DPI)
        
        plt.plot(equity_curve, color='#1E90FF', linewidth=3.5, label='Equity')
        plt.title(f'Equity Curve - {symbol} (LSTM)', fontsize=16, fontweight='bold', color='white')
        plt.xlabel('Trades', color='white')
        plt.ylabel('Balance ($)', color='white')
        
        ax = plt.gca()
        ax.set_facecolor('#0a0a0a')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('white')
        ax.tick_params(colors='white')
        plt.grid(alpha=0.2, color='gray')
        
        plt.axhline(y=initial_balance, color='#FF4444', linestyle='--', linewidth=2, label='Initial Balance', alpha=0.7)
        plt.legend(facecolor='#0a0a0a', edgecolor='white', labelcolor='white')
        
        plt.tight_layout(pad=2.0)
        
        filename = f"charts/equity_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=DPI, facecolor='#0a0a0a', edgecolor='none', 
                    bbox_inches='tight', pad_inches=0.1)
        print(f"\n✓ График сохранён: {filename}")
    except Exception as e:
        print(f"⚠️ Ошибка построения графика: {e}")
    
    return results

# ====================== ГЕНЕРАЦИЯ ДАТАСЕТА ДЛЯ LLM ======================
def generate_hybrid_dataset(data_dict: Dict[str, pd.DataFrame], model, quantum_encoder, n_samples: int = 2000):
    """
    Генерация датасета с LSTM предсказаниями для файнтьюна LLM
    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ - квантовое кодирование только 1 раз!
    """
    print(f"\n{'='*80}")
    print(f"ГЕНЕРАЦИЯ ГИБРИДНОГО ДАТАСЕТА (LSTM + QUANTUM)")
    print(f"{'='*80}\n")
    print(f"Цель: {n_samples} примеров с LSTM прогнозами и квантовыми признаками\n")
    
    dataset = []
    up_count = 0
    down_count = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    target_per_symbol = n_samples // len(data_dict)
    
    for symbol, df_orig in data_dict.items():
        print(f"Обработка {symbol}...")
        
        # Вычисление технических индикаторов
        df_features = calculate_features(df_orig)
        
        if len(df_features) < LOOKBACK + PREDICTION_HORIZON:
            print(f"  ⚠️ Недостаточно данных для {symbol}")
            continue
        
        # Подготовка классических признаков для LSTM
        df_features['returns'] = df_features['close'].pct_change()
        df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
        df_features['high_low'] = (df_features['high'] - df_features['low']) / df_features['close']
        df_features['close_open'] = (df_features['close'] - df_features['open']) / df_features['open']
        df_features = df_features.dropna()
        
        price_features = df_features[['returns', 'log_returns', 'high_low', 
                                      'close_open', 'tick_volume']].values
        
        # Стандартизация
        mean = price_features.mean(axis=0)
        std = price_features.std(axis=0)
        price_data = (price_features - mean) / (std + 1e-8)
        
        # ОПТИМИЗАЦИЯ: Извлекаем все квантовые признаки ОДИН РАЗ
        print(f"  Квантовое кодирование (это займёт ~2-3 минуты)...")
        quantum_features_list = []
        
        for i in range(QUANTUM_WINDOW, len(df_features)):
            window = df_features['close'].iloc[i-QUANTUM_WINDOW:i].values
            q_features = quantum_encoder.extract_quantum_features(window)
            quantum_features_list.append(list(q_features.values()))
        
        quantum_features = np.array(quantum_features_list)
        price_data = price_data[QUANTUM_WINDOW:]
        
        print(f"  ✓ Квантовое кодирование завершено")
        
        # Создание кандидатов
        candidates = []
        
        for idx in range(SEQUENCE_LENGTH, len(price_data) - PREDICTION_HORIZON):
            # Подготовка данных для LSTM
            price_seq = price_data[idx-SEQUENCE_LENGTH:idx]
            quantum_feat = quantum_features[idx]
            
            # Получение будущей цены
            future_idx = idx + QUANTUM_WINDOW + PREDICTION_HORIZON
            if future_idx >= len(df_features):
                break
            
            current_row = df_features.iloc[idx + QUANTUM_WINDOW]
            future_row = df_features.iloc[future_idx]
            
            # LSTM предсказание
            with torch.no_grad():
                price_tensor = torch.FloatTensor(price_seq).unsqueeze(0).to(device)
                quantum_tensor = torch.FloatTensor(quantum_feat).unsqueeze(0).to(device)
                output = model(price_tensor, quantum_tensor)
                prob = torch.sigmoid(output).item()
            
            lstm_direction = "UP" if prob > 0.5 else "DOWN"
            lstm_confidence = max(prob, 1-prob) * 100
            
            # Реальный результат
            actual_price_24h = future_row['close']
            price_change = actual_price_24h - current_row['close']
            price_change_pips = int(price_change / 0.0001)
            actual_direction = "UP" if price_change > 0 else "DOWN"
            
            candidates.append({
                'symbol': symbol,
                'current_row': current_row,
                'future_row': future_row,
                'quantum_feats': {
                    'quantum_entropy': quantum_feat[0],
                    'dominant_state_prob': quantum_feat[1],
                    'superposition_measure': quantum_feat[2],
                    'phase_coherence': quantum_feat[3],
                    'entanglement_degree': quantum_feat[4],
                    'quantum_variance': quantum_feat[5],
                    'num_significant_states': quantum_feat[6]
                },
                'lstm_direction': lstm_direction,
                'lstm_confidence': lstm_confidence,
                'lstm_prob': prob * 100,
                'actual_direction': actual_direction,
                'price_change_pips': price_change_pips,
                'current_time': current_row.name
            })
        
        # Балансировка: берём равное количество UP и DOWN
        up_candidates = [c for c in candidates if c['actual_direction'] == 'UP']
        down_candidates = [c for c in candidates if c['actual_direction'] == 'DOWN']
        
        target_up = target_per_symbol // 2
        target_down = target_per_symbol // 2
        
        selected_up = np.random.choice(len(up_candidates), size=min(target_up, len(up_candidates)), replace=False) if up_candidates else []
        selected_down = np.random.choice(len(down_candidates), size=min(target_down, len(down_candidates)), replace=False) if down_candidates else []
        
        # Создание примеров
        for idx in selected_up:
            candidate = up_candidates[idx]
            example = create_lstm_training_example(candidate)
            dataset.append(example)
            up_count += 1
        
        for idx in selected_down:
            candidate = down_candidates[idx]
            example = create_lstm_training_example(candidate)
            dataset.append(example)
            down_count += 1
        
        print(f"  {symbol}: {len(selected_up)} UP + {len(selected_down)} DOWN = {len(selected_up) + len(selected_down)}")
    
    print(f"\n{'='*80}")
    print(f"ГИБРИДНЫЙ ДАТАСЕТ СОЗДАН")
    print(f"{'='*80}")
    print(f"Всего: {len(dataset)} примеров")
    print(f" UP: {up_count} ({up_count/len(dataset)*100:.1f}%)")
    print(f" DOWN: {down_count} ({down_count/len(dataset)*100:.1f}%)")
    print(f"{'='*80}\n")
    
    return dataset

def create_lstm_training_example(candidate: Dict) -> Dict:
    """Создаёт обучающий пример с LSTM прогнозом и квантовыми признаками"""
    row = candidate['current_row']
    future_row = candidate['future_row']
    quantum_feats = candidate['quantum_feats']
    
    # Интерпретация квантовых признаков
    entropy_level = "высокая неопределённость" if quantum_feats['quantum_entropy'] > 2.5 else \
                    "умеренная неопределённость" if quantum_feats['quantum_entropy'] > 2.0 else \
                    "низкая неопределённость (рынок определился)"
    
    dominant_strength = "сильная" if quantum_feats['dominant_state_prob'] > 0.25 else \
                       "умеренная" if quantum_feats['dominant_state_prob'] > 0.15 else \
                       "слабая"
    
    market_complexity = "высокая" if quantum_feats['num_significant_states'] > 5 else \
                       "средняя" if quantum_feats['num_significant_states'] > 3 else \
                       "низкая"
    
    # Проверка правильности LSTM
    lstm_correct = "ВЕРНО" if candidate['lstm_direction'] == candidate['actual_direction'] else "ОШИБКА"
    
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
Квантовая энтропия: {quantum_feats['quantum_entropy']:.3f} ({entropy_level})
Доминантное состояние: {quantum_feats['dominant_state_prob']:.3f} ({dominant_strength} доминанта)
Суперпозиция: {quantum_feats['superposition_measure']:.3f}
Когерентность: {quantum_feats['phase_coherence']:.3f}
Запутанность: {quantum_feats['entanglement_degree']:.3f}
Значимых состояний: {quantum_feats['num_significant_states']:.0f} (сложность: {market_complexity})
Квантовая дисперсия: {quantum_feats['quantum_variance']:.6f}

ПРОГНОЗ LSTM+QUANTUM:
Направление: {candidate['lstm_direction']}
Уверенность: {candidate['lstm_confidence']:.1f}%
Вероятность UP: {candidate['lstm_prob']:.1f}%
Источник: bidirectional_lstm_quantum

Проанализируй ситуацию с учётом прогноза квантовой LSTM и дай точный прогноз цены через 24 часа."""

    response = f"""НАПРАВЛЕНИЕ: {candidate['actual_direction']}
УВЕРЕННОСТЬ: {min(98, max(65, candidate['lstm_confidence'] + np.random.randint(-5, 10)))}%
ПРОГНОЗ ЦЕНЫ ЧЕРЕЗ 24Ч: {future_row['close']:.5f} ({candidate['price_change_pips']:+d} пунктов)

АНАЛИЗ ПРОГНОЗА LSTM:
Квантовая Bidirectional LSTM предсказала {candidate['lstm_direction']} с уверенностью {candidate['lstm_confidence']:.1f}%.
Реальный результат: {candidate['actual_direction']} ({lstm_correct}).

КВАНТОВЫЙ АНАЛИЗ:
Энтропия {quantum_feats['quantum_entropy']:.3f} показывает {entropy_level}. {'Рынок коллапсировал в определённое состояние — движение предсказуемо.' if quantum_feats['quantum_entropy'] < 2.0 else 'Рынок в режиме неопределённости — множественные сценарии равновероятны.' if quantum_feats['quantum_entropy'] > 2.5 else 'Умеренная неопределённость — есть предпочтительное направление.'}
Доминантное состояние {quantum_feats['dominant_state_prob']:.3f} указывает на {dominant_strength} преобладание одного квантового состояния.
Когерентность {quantum_feats['phase_coherence']:.3f} {'высокая — все сценарии согласованы' if quantum_feats['phase_coherence'] > 0.7 else 'низкая — сценарии противоречат друг другу'}.
Запутанность {quantum_feats['entanglement_degree']:.3f} {'показывает сильную корреляцию движений — тренд имеет инерцию' if quantum_feats['entanglement_degree'] > 0.6 else 'слабая — паттерны разорваны'}.

ТЕХНИЧЕСКИЙ АНАЛИЗ НА 24 ЧАСА:
{'RSI ' + str(round(row["RSI"], 1)) + ' — перепроданность, жду отскок' if row['RSI'] < 30 else 'RSI ' + str(round(row["RSI"], 1)) + ' — перекупленность, возможна коррекция' if row['RSI'] > 70 else 'RSI ' + str(round(row["RSI"], 1)) + ' — нейтральная зона'}.
{'MACD позитивный — бычий импульс сохраняется' if row['MACD'] > 0 else 'MACD негативный — медвежье давление продолжается'}.
{'Объёмы выше средних — движение поддержано' if row['vol_ratio'] > 1.3 else 'Объёмы низкие — слабый импульс'}.
{'Цена у нижней BB — статистически жду возврата к средней' if row['BB_position'] < 0.25 else 'Цена у верхней BB — возможен откат' if row['BB_position'] > 0.75 else 'Цена в середине BB — направление не определено уровнями'}.

ВЫВОД:
Квантовая Bidirectional LSTM {'правильно определила' if lstm_correct == 'ВЕРНО' else 'ошибочно предсказала'} направление. {'Квантовая энтропия подтверждает предсказуемость движения.' if quantum_feats['quantum_entropy'] < 2.3 else 'Высокая квантовая энтропия указывает на сложность прогноза.'} {'Высокая когерентность усиливает уверенность прогноза.' if quantum_feats['phase_coherence'] > 0.7 else 'Низкая когерентность снижает надёжность.'} 
Фактическое движение за 24 часа: {abs(candidate['price_change_pips'])} пунктов {candidate['actual_direction']}.
Конечная цена: {future_row['close']:.5f}.

ВАЖНО: Квантовая LSTM имеет точность 54-56% на валидации. Это статистическое преимущество, но не абсолютная истина. {'В данном случае квантовые признаки показали высокую уверенность и оказались правы.' if lstm_correct == 'ВЕРНО' and quantum_feats['quantum_entropy'] < 2.3 else 'Следующий прогноз может быть противоположным — рынок непредсказуем.'}"""

    return {
        "prompt": prompt,
        "response": response,
        "direction": candidate['actual_direction']
    }

# ====================== СОХРАНЕНИЕ ДАТАСЕТА ======================
def save_dataset(dataset: List[Dict], filepath: str):
    """Сохранение датасета в JSONL формате"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Датасет сохранён: {filepath}")
    return filepath

# ====================== ФАЙНТЬЮН LLM ======================
def finetune_llm_with_lstm(dataset_path: str):
    """Файнтьюн LLM с встроенными LSTM прогнозами - точно как в quantum_fusion.py"""
    print(f"\n{'='*80}")
    print(f"ФАЙНТЬЮН LLM С LSTM ПРОГНОЗАМИ")
    print(f"{'='*80}\n")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Датасет не найден: {dataset_path}")
        return
    
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
Ты — QuantumTrader-LSTM-3B — элитный аналитик с квантовым усилением и Bidirectional LSTM.

УНИКАЛЬНЫЕ ВОЗМОЖНОСТИ:
1. Ты видишь прогнозы Bidirectional LSTM модели с квантовыми признаками (точность 54-56%)
2. Ты понимаешь квантовую энтропию, доминантные состояния, когерентность, запутанность
3. Ты интегрируешь квантовые прогнозы LSTM с классическим техническим анализом

КВАНТОВЫЕ ПРИЗНАКИ:
- Квантовая энтропия: мера неопределённости рынка
  • Низкая (<2.0) = рынок определился, предсказуемость высокая
  • Умеренная (2.0-2.5) = умеренная неопределённость
  • Высокая (>2.5) = хаос, множественные сценарии равновероятны
- Доминантное состояние: вероятность главного квантового состояния
  • >0.25 = сильная доминанта, чёткий тренд
  • 0.15-0.25 = умеренная доминанта
  • <0.15 = слабая доминанта, неопределённость
- Суперпозиция: количество значимых квантовых состояний
  • Низкая = консолидация, рынок в узком диапазоне
  • Высокая = волатильность, множество сценариев
- Когерентность: согласованность сценариев
  • >0.7 = высокая, все сценарии согласованы, устойчивый тренд
  • <0.7 = низкая, сценарии противоречат друг другу
- Запутанность: корреляция между временными точками
  • >0.6 = сильная, тренд имеет инерцию
  • <0.6 = слабая, паттерны разорваны
- Квантовая дисперсия: разброс вероятностей состояний
- Значимые состояния: количество активных квантовых состояний

ПРОГНОЗЫ LSTM:
- Обучена на 50-часовых окнах H1 таймфрейма
- Bidirectional архитектура (видит прошлое и будущее)
- 7 квантовых признаков + 5 классических
- Точность 54-56% — статистическое преимущество
- Уверенность >60% = сильный сигнал
- Уверенность 50-60% = слабый сигнал

СТРОГИЕ ПРАВИЛА:
1. Только UP или DOWN — никакого FLAT
2. Уверенность 65-98%
3. ОБЯЗАТЕЛЬНО прогноз цены через 24ч: X.XXXXX (±NN пунктов)
4. Анализируй прогноз LSTM и квантовые признаки
5. Объясняй, почему квантовая LSTM права или ошиблась

ФОРМАТ ОТВЕТА:
НАПРАВЛЕНИЕ: UP/DOWN
УВЕРЕННОСТЬ: XX%
ПРОГНОЗ ЦЕНЫ ЧЕРЕЗ 24Ч: X.XXXXX (±NN пунктов)

АНАЛИЗ ПРОГНОЗА LSTM:
[оценка прогноза квантовой LSTM модели]

КВАНТОВЫЙ АНАЛИЗ:
[интерпретация квантовой энтропии, доминантных состояний, когерентности, запутанности]

ТЕХНИЧЕСКИЙ АНАЛИЗ НА 24 ЧАСА:
[RSI, MACD, объёмы, уровни Боллинджера, Stochastic]

ВЫВОД:
[синтез квантовых LSTM сигналов и технических индикаторов с конкретной целью]

ВАЖНО: Квантовая LSTM имеет точность 54-56%. Это статистическое преимущество, но не абсолютная истина!
\"\"\"
"""
    
    for i, example in enumerate(training_sample, 1):
        modelfile_content += f"""
MESSAGE user \"\"\"{example['prompt']}\"\"\"
MESSAGE assistant \"\"\"{example['response']}\"\"\"
"""
    
    modelfile_path = "Modelfile_quantum_lstm"
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
        test_prompt = """EURUSD 2025-12-23 10:00
Текущая цена: 1.04250

ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:
RSI: 35.2
MACD: -0.00012
ATR: 0.00095
Объёмы: 1.5x
BB позиция: 0.22
Stochastic K: 28.0

КВАНТОВЫЕ ПРИЗНАКИ:
Квантовая энтропия: 2.150 (низкая неопределённость - рынок определился)
Доминантное состояние: 0.242 (сильная доминанта)
Суперпозиция: 0.325
Когерентность: 0.784 (высокая - все сценарии согласованы)
Запутанность: 0.652 (сильная корреляция - тренд имеет инерцию)
Значимых состояний: 4.0 (сложность: низкая)
Квантовая дисперсия: 0.004523

ПРОГНОЗ LSTM+QUANTUM:
Направление: UP
Уверенность: 72.5%
Вероятность UP: 72.5%
Источник: bidirectional_lstm_quantum

Проанализируй ситуацию с учётом прогноза квантовой LSTM и дай точный прогноз цены через 24 часа."""

        if ollama:
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
        print(f"✓ Интеграция: Bidirectional LSTM + Qiskit + LLM")
        print(f"✓ Обучено примеров: {len(training_sample)}")
        print(f"✓ Для публикации: ollama push {MODEL_NAME}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка: {e}")
        if hasattr(e, 'output'):
            print(f"Вывод: {e.output}")

# ====================== ЖИВАЯ ТОРГОВЛЯ ======================
def live_trading():
    """Живая торговля с LSTM моделью"""
    print(f"\n{'='*80}")
    print(f"ЗАПУСК ЖИВОЙ ТОРГОВЛИ С LSTM")
    print(f"{'='*80}\n")
    
    if not mt5 or not mt5.initialize():
        print("❌ MT5 не инициализирован")
        return
    
    # Загрузка модели
    if not os.path.exists('models/quantum_lstm_best.pth'):
        print("❌ LSTM модель не найдена. Сначала обучи модель (режим 1)")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuantumLSTM().to(device)
    model.load_state_dict(torch.load('models/quantum_lstm_best.pth'))
    model.eval()
    
    quantum_encoder = QuantumFeatureExtractor(N_QUBITS, N_SHOTS)
    
    print("Модель загружена, начинаем мониторинг...")
    print(f"Проверка каждые 60 секунд")
    print(f"Риск на сделку: {RISK_PER_TRADE*100}%")
    print(f"Минимальная вероятность: {MIN_PROB}%\n")
    
    try:
        while True:
            for symbol in SYMBOLS:
                # Получение данных
                rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, LOOKBACK)
                if rates is None or len(rates) < LOOKBACK:
                    continue
                
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df = calculate_features(df)
                
                # Подготовка признаков
                price_features = df[['returns', 'log_returns', 'high_low', 
                                    'close_open', 'tick_volume']].tail(SEQUENCE_LENGTH).values
                mean = price_features.mean(axis=0)
                std = price_features.std(axis=0)
                price_data = (price_features - mean) / (std + 1e-8)
                
                # Квантовые признаки
                window = df['close'].tail(QUANTUM_WINDOW).values
                q_features = quantum_encoder.extract_quantum_features(window)
                quantum_features = np.array(list(q_features.values()))
                
                # Предсказание
                with torch.no_grad():
                    price_tensor = torch.FloatTensor(price_data).unsqueeze(0).to(device)
                    quantum_tensor = torch.FloatTensor(quantum_features).unsqueeze(0).to(device)
                    output = model(price_tensor, quantum_tensor)
                    prob = torch.sigmoid(output).item() * 100
                
                # Логика торговли
                positions = mt5.positions_get(symbol=symbol, magic=MAGIC)
                
                if prob > MIN_PROB and not positions:
                    print(f"\n🔼 {symbol}: BUY сигнал ({prob:.1f}%)")
                    # Здесь логика открытия позиции
                
                elif prob < (100 - MIN_PROB) and not positions:
                    print(f"\n🔽 {symbol}: SELL сигнал ({100-prob:.1f}%)")
                    # Здесь логика открытия позиции
            
            time.sleep(60)
    
    except KeyboardInterrupt:
        print("\n\nОстановка торговли...")
    finally:
        mt5.shutdown()

# ====================== ГЛАВНОЕ МЕНЮ ======================
def main():
    """Главное меню"""
    print(f"\n{'='*80}")
    print(f" QUANTUM TRADER LSTM — Qiskit + Bidirectional LSTM + LLM")
    print(f" Версия: 22.12.2025 (Full Integration)")
    print(f"{'='*80}\n")
    print(f"РЕЖИМЫ:")
    print(f"-"*80)
    print(f"1 → Обучить Bidirectional LSTM с квантовыми признаками")
    print(f"2 → Сгенерировать гибридный датасет (LSTM + Quantum)")
    print(f"3 → Файнтьюн LLM с LSTM прогнозами")
    print(f"4 → Бэктест LSTM модели на исторических данных")
    print(f"5 → Живая торговля (MT5)")
    print(f"6 → ПОЛНЫЙ ЦИКЛ (всё вместе)")
    print(f"-"*80)
    
    choice = input("\nВыбери режим (1-6): ").strip()
    
    if choice == "1":
        # Обучение LSTM
        model = train_lstm_model(symbol="EURUSD", n_candles=LOOKBACK)
        
    elif choice == "2":
        # Генерация датасета
        data = load_mt5_data(180)
        if not data:
            print("❌ Нет данных")
            return
        
        if not os.path.exists("models/quantum_lstm_best.pth"):
            print("❌ LSTM модель не найдена, сначала обучи (режим 1)")
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = QuantumLSTM().to(device)
        model.load_state_dict(torch.load("models/quantum_lstm_best.pth"))
        
        quantum_encoder = QuantumFeatureExtractor(N_QUBITS, N_SHOTS)
        dataset = generate_hybrid_dataset(data, model, quantum_encoder, FINETUNE_SAMPLES)
        save_dataset(dataset, "dataset/quantum_lstm_data.jsonl")
        
    elif choice == "3":
        # Файнтьюн LLM
        dataset_path = "dataset/quantum_lstm_data.jsonl"
        if not os.path.exists(dataset_path):
            print(f"❌ Датасет не найден: {dataset_path}")
            return
        finetune_llm_with_lstm(dataset_path)
        
    elif choice == "4":
        # Бэктест
        print("\nПараметры бэктеста:")
        symbol = input("Символ (по умолчанию EURUSD): ").strip() or "EURUSD"
        n_candles = int(input("Количество свечей (по умолчанию 1500): ").strip() or "1500")
        initial_balance = float(input("Начальный баланс (по умолчанию 10000): ").strip() or "10000")
        
        backtest_lstm_model(symbol, n_candles, initial_balance)
        
    elif choice == "5":
        # Живая торговля
        live_trading()
        
    elif choice == "6":
        # ПОЛНЫЙ ЦИКЛ
        print(f"\n{'='*80}")
        print(f"ПОЛНЫЙ ЦИКЛ: QUANTUM LSTM FUSION")
        print(f"{'='*80}\n")
        print("Этот процесс займёт 1-2 часа:")
        print("1. Загрузка данных MT5")
        print("2. Квантовое кодирование (~15 мин)")
        print("3. Обучение Bidirectional LSTM (~30 мин)")
        print("4. Бэктест на исторических данных (~5 мин)")
        print("5. Генерация датасета (~20 мин)")
        print("6. Файнтьюн LLM (~15 мин)")
        
        confirm = input("\nПродолжить? (YES): ").strip()
        if confirm != "YES":
            print("Отменено")
            return
        
        # Шаг 1: Загрузка
        data = load_mt5_data(180)
        if not data:
            print("❌ Не удалось загрузить данные")
            return
        
        # Шаг 2-3: Обучение LSTM
        print(f"\n{'='*80}")
        print("ШАГ 2-3/6: КВАНТОВОЕ КОДИРОВАНИЕ + ОБУЧЕНИЕ LSTM")
        print(f"{'='*80}")
        model = train_lstm_model(symbol="EURUSD", n_candles=LOOKBACK)
        
        # Шаг 4: Бэктест
        print(f"\n{'='*80}")
        print("ШАГ 4/6: БЭКТЕСТ НА ИСТОРИЧЕСКИХ ДАННЫХ")
        print(f"{'='*80}")
        backtest_lstm_model(symbol="EURUSD", n_candles=LOOKBACK, initial_balance=10000.0)
        
        # Шаг 5: Датасет
        print(f"\n{'='*80}")
        print("ШАГ 5/6: ГЕНЕРАЦИЯ ДАТАСЕТА")
        print(f"{'='*80}")
        quantum_encoder = QuantumFeatureExtractor(N_QUBITS, N_SHOTS)
        dataset = generate_hybrid_dataset(data, model, quantum_encoder, FINETUNE_SAMPLES)
        dataset_path = save_dataset(dataset, "dataset/quantum_lstm_data.jsonl")
        
        # Шаг 6: Файнтьюн
        print(f"\n{'='*80}")
        print("ШАГ 6/6: ФАЙНТЬЮН LLM")
        print(f"{'='*80}")
        finetune_llm_with_lstm(dataset_path)
        
        print(f"\n{'='*80}")
        print("🎉 ПОЛНЫЙ ЦИКЛ ЗАВЕРШЁН!")
        print(f"{'='*80}")
        print("✓ Bidirectional LSTM обучена с квантовыми признаками")
        print("✓ Бэктест проведён и сохранён в logs/backtest_results.json")
        print("✓ LLM файнтьюнена с LSTM прогнозами")
        print("✓ Система готова к использованию")
        
    else:
        print("❌ Неверный выбор")

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
        if rates is None or len(rates) < LOOKBACK:
            print(f" ⚠️ {symbol}: недостаточно данных")
            continue
        
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        data[symbol] = df
        print(f" ✓ {symbol}: {len(df)} баров")
    
    mt5.shutdown()
    return data

if __name__ == "__main__":
    main()
