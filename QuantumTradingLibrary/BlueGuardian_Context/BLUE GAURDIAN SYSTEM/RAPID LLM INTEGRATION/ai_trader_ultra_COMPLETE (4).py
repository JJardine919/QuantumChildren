

# ================================================
# ai_trader_ultra_with_finetune.py
# Версия с РЕАЛЬНЫМИ ДАННЫМИ MT5 — 21.11.2025
# 5 режимов: 1-Push, 2-Finetune, 3-Backtest, 4-Live, 5-Dataset
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

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    import ollama
except ImportError:
    ollama = None

# ====================== КОНФИГ ======================
MODEL_NAME = "koshtenco/shtencoaitrader-3b-ultra-analyst-v3"
BASE_MODEL = "llama3.2:3b"
SYMBOLS = ["EURUSD", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
TIMEFRAME = mt5.TIMEFRAME_M15 if mt5 else None
LOOKBACK = 400
INITIAL_BALANCE = 10000.0
RISK_PER_TRADE = 0.005
MIN_PROB = 70
LIVE_LOT = 2.00
MAGIC = 20251121
SLIPPAGE = 10

# Файнтьун параметры
FINETUNE_SAMPLES = 10000
FINETUNE_EPOCHS = 3
BACKTEST_DAYS = 30 
# Прогноз на 24 часа (96 баров по 15 минут)
PREDICTION_HORIZON = 96
balance_ratio: float = 1.0

os.makedirs("logs", exist_ok=True)
os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("charts", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/ai_trader_ultra.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ====================== ПРИЗНАКИ ======================
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Расчёт технических индикаторов"""
    d = df.copy()
    d["close_prev"] = d["close"].shift(1)
    # ATR
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - d["close_prev"]).abs(),
            (d["low"] - d["close_prev"]).abs(),
        ],
        axis=1,
    ).max(axis=1)
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
    return d.dropna()


# ====================== ГЕНЕРАЦИЯ ДАТАСЕТА ИЗ MT5 ======================
def generate_real_dataset_from_mt5(num_samples: int = 1000) -> list:
    """
    Генерация СБАЛАНСИРОВАННОГО датасета на основе реальных данных MT5

    Args:
        num_samples: Общее количество примеров
        balance_ratio: Целевое соотношение UP/DOWN (1.0 = идеальный баланс 50/50)

    Returns:
        Список сбалансированных примеров
    """
    print(f"\n{'='*80}")
    print(f"ГЕНЕРАЦИЯ СБАЛАНСИРОВАННОГО ДАТАСЕТА ИЗ MT5")
    print(f"{'='*80}\n")
    print(f"Цель: {num_samples} примеров с балансом UP/DOWN = {balance_ratio}:1")

    if not mt5 or not mt5.initialize():
        print("MT5 не подключен! Используй синтетический датасет.")
        return generate_balanced_synthetic_dataset(num_samples, balance_ratio)

    # Счётчики для балансировки
    up_count = 0
    down_count = 0
    target_up = int(num_samples * balance_ratio / (1 + balance_ratio))
    target_down = num_samples - target_up

    print(f"Целевое распределение:")
    print(f" UP: {target_up} примеров ({target_up/num_samples*100:.1f}%)")
    print(f" DOWN: {target_down} примеров ({target_down/num_samples*100:.1f}%)\n")

    dataset = []

    # Загружаем данные за последние 6 месяцев
    end = datetime.now()
    start = end - timedelta(days=180)

    for symbol in SYMBOLS:
        print(f"Загрузка {symbol}...")
        rates = mt5.copy_rates_range(symbol, TIMEFRAME, start, end)

        if rates is None or len(rates) < LOOKBACK + PREDICTION_HORIZON:
            print(f"Недостаточно данных для {symbol}")
            continue

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = calculate_features(df)

        # Собираем ВСЕ возможные точки для анализа
        all_candidates = []

        for idx in range(LOOKBACK, len(df) - PREDICTION_HORIZON):
            row = df.iloc[idx]
            future_idx = idx + PREDICTION_HORIZON
            future_row = df.iloc[future_idx]

            actual_price_24h = future_row['close']
            price_change = actual_price_24h - row['close']

            direction = "UP" if price_change > 0 else "DOWN"

            all_candidates.append({
                'idx': idx,
                'direction': direction,
                'price_change': abs(price_change),
                'symbol': symbol,
                'row': row,
                'future_row': future_row
            })

        print(f" Найдено {len(all_candidates)} возможных точек")

        # Разделяем по направлениям
        up_candidates = [c for c in all_candidates if c['direction'] == 'UP']
        down_candidates = [c for c in all_candidates if c['direction'] == 'DOWN']

        print(f" UP: {len(up_candidates)} | DOWN: {len(down_candidates)}")

        # Сэмплируем с учётом баланса
        symbol_target = num_samples // len(SYMBOLS)
        symbol_up_target = int(symbol_target * balance_ratio / (1 + balance_ratio))
        symbol_down_target = symbol_target - symbol_up_target

        selected_up = np.random.choice(
            len(up_candidates),
            size=min(symbol_up_target, len(up_candidates)),
            replace=False
        ) if len(up_candidates) > 0 else []

        selected_down = np.random.choice(
            len(down_candidates),
            size=min(symbol_down_target, len(down_candidates)),
            replace=False
        ) if len(down_candidates) > 0 else []

        # Создаём примеры
        for idx in selected_up:
            candidate = up_candidates[idx]
            example = create_training_example(
                candidate['symbol'],
                candidate['row'],
                candidate['future_row'],
                df.index[candidate['idx']]
            )
            dataset.append(example)
            up_count += 1

        for idx in selected_down:
            candidate = down_candidates[idx]
            example = create_training_example(
                candidate['symbol'],
                candidate['row'],
                candidate['future_row'],
                df.index[candidate['idx']]
            )
            dataset.append(example)
            down_count += 1

        print(f"{symbol}: создано {len(selected_up)} UP + {len(selected_down)} DOWN = {len(selected_up) + len(selected_down)} примеров")

    mt5.shutdown()

    # Финальная статистика
    print(f"\n{'='*80}")
    print(f"ДАТАСЕТ СОЗДАН")
    print(f"{'='*80}")
    print(f"Всего примеров: {len(dataset)}")
    print(f" UP: {up_count} ({up_count/len(dataset)*100:.1f}%)")
    print(f" DOWN: {down_count} ({down_count/len(dataset)*100:.1f}%)")

    actual_ratio = max(up_count, down_count) / min(up_count, down_count) if min(up_count, down_count) > 0 else 0
    print(f" Фактическое соотношение: {actual_ratio:.2f}:1")

    if actual_ratio <= 1.2:
        print(f" ОТЛИЧНО! Датасет сбалансирован")
    elif actual_ratio <= 1.5:
        print(f" ПРИЕМЛЕМО. Лёгкий дисбаланс")
    else:
        print(f" ПРОБЛЕМА! Требуется балансировка")

    print(f"{'='*80}\n")

    if actual_ratio > 1.3:
        print(f"Применяю балансировку через oversampling...")
        dataset = balance_dataset_oversampling(dataset, up_count, down_count)

    return dataset


def create_training_example(symbol: str, row: pd.Series, future_row: pd.Series, current_time: datetime) -> dict:
    """Создание обучающего примера из данных"""
    actual_price_24h = future_row['close']
    price_change = actual_price_24h - row['close']
    price_change_pips = int(price_change / 0.0001)
    direction = "UP" if price_change > 0 else "DOWN"

    bullish_signals = 0
    bearish_signals = 0
    analysis_parts = []

    # RSI анализ
    if row['RSI'] < 30:
        bullish_signals += 2
        analysis_parts.append(f"RSI {row['RSI']:.1f} — сильная перепроданность, через 24ч произошёл отскок на {abs(price_change_pips)} пунктов")
    elif row['RSI'] > 70:
        bearish_signals += 2
        analysis_parts.append(f"RSI {row['RSI']:.1f} — перекупленность, за сутки случилась коррекция на {abs(price_change_pips)} пунктов")
    else:
        if row['RSI'] < 50:
            bullish_signals += 1
        else:
            bearish_signals += 1
        analysis_parts.append(f"RSI {row['RSI']:.1f} — нейтральная зона, движение {abs(price_change_pips)} пунктов за 24ч")

    # MACD анализ
    if row['MACD'] > 0:
        bullish_signals += 2
        analysis_parts.append("MACD позитивный — бычий импульс подтвердился в течение суток")
    else:
        bearish_signals += 2
        analysis_parts.append("MACD негативный — медвежье давление сохранилось 24 часа")

    # ATR анализ
    if row['ATR'] > row['ATR'] * 1.3:
        analysis_parts.append(f"ATR {row['ATR']:.5f} — высокая волатильность обеспечила движение {abs(price_change_pips)} пунктов")
    else:
        analysis_parts.append(f"ATR {row['ATR']:.5f} — умеренная волатильность, движение {abs(price_change_pips)} пунктов")

    # Объёмы
    if row['vol_ratio'] > 1.5:
        if direction == "UP":
            bullish_signals += 1
        else:
            bearish_signals += 1
        analysis_parts.append("Объёмы выше средних на 50%+ — импульс продолжился в течение суток")

    # BB позиция
    if row['BB_position'] < 0.2:
        bullish_signals += 1
        analysis_parts.append("Цена у нижней границы Боллинджера — через 24ч произошёл возврат к средней")
    elif row['BB_position'] > 0.8:
        bearish_signals += 1
        analysis_parts.append("Цена у верхней границы Боллинджера — за сутки случился откат")

    # Stochastic
    if row['Stoch_K'] < 20:
        bullish_signals += 1
        analysis_parts.append(f"Stochastic {row['Stoch_K']:.1f} — перепроданность дала разворот вверх")
    elif row['Stoch_K'] > 80:
        bearish_signals += 1
        analysis_parts.append(f"Stochastic {row['Stoch_K']:.1f} — перекупленность привела к коррекции")

    # Уверенность
    if direction == "UP":
        confidence = min(98, max(65, 60 + bullish_signals * 8))
    else:
        confidence = min(98, max(65, 60 + bearish_signals * 8))

    analysis = "\n- ".join(analysis_parts)

    prompt = f"""{symbol} {current_time.strftime('%Y-%m-%d %H:%M')}
Текущая цена: {row['close']:.5f}
RSI: {row['RSI']:.1f}
MACD: {row['MACD']:.6f}
ATR: {row['ATR']:.5f}
Объёмы: {row['vol_ratio']:.2f}x
BB позиция: {row['BB_position']:.2f}
Stochastic K: {row['Stoch_K']:.1f}
Проанализируй ситуацию объективно и дай точный прогноз цены через 24 часа.
ВАЖНО: Не склоняйся к одному направлению - анализируй реальные данные без предвзятости."""

    response = f"""НАПРАВЛЕНИЕ: {direction}
УВЕРЕННОСТЬ: {confidence}%
ПРОГНОЗ ЦЕНЫ ЧЕРЕЗ 24Ч: {actual_price_24h:.5f} ({price_change_pips:+d} пунктов)
ОБЪЕКТИВНЫЙ АНАЛИЗ НА 24 ЧАСА (РЕАЛЬНЫЙ РЕЗУЛЬТАТ):
- {analysis}
ВЫВОД: Анализ {abs(bullish_signals + bearish_signals)} индикаторов показывает {'бычий' if direction == 'UP' else 'медвежий'} сценарий. Фактическое движение за 24 часа составило {abs(price_change_pips)} пунктов {direction}. Конечная цена: {actual_price_24h:.5f}.
ВАЖНО: Этот прогноз основан исключительно на технических индикаторах, без предвзятости к направлению. В следующий раз рыночная ситуация может быть противоположной."""

    return {
        "prompt": prompt,
        "response": response,
        "direction": direction
    }


def balance_dataset_oversampling(dataset: list, up_count: int, down_count: int) -> list:
    """Балансировка датасета через oversampling (дублирование меньшинства)"""
    if up_count == down_count:
        return dataset

    minority_class = 'UP' if up_count < down_count else 'DOWN'
    majority_count = max(up_count, down_count)
    minority_count = min(up_count, down_count)

    print(f" Класс меньшинства: {minority_class}")
    print(f" Нужно добавить: {majority_count - minority_count} примеров\n")

    up_examples = [ex for ex in dataset if ex.get('direction') == 'UP']
    down_examples = [ex for ex in dataset if ex.get('direction') == 'DOWN']

    minority_examples = up_examples if minority_class == 'UP' else down_examples

    while len(minority_examples) < majority_count:
        original = np.random.choice(minority_examples)
        variation = original.copy()
        minority_examples.append(variation)

    if minority_class == 'UP':
        balanced = minority_examples[:majority_count] + down_examples
    else:
        balanced = up_examples + minority_examples[:majority_count]

    np.random.shuffle(balanced)

    print(f"Балансировка завершена")
    print(f" Итоговое распределение: UP={len([ex for ex in balanced if ex.get('direction') == 'UP'])}, "
          f"DOWN={len([ex for ex in balanced if ex.get('direction') == 'DOWN'])}\n")

    return balanced


def generate_balanced_synthetic_dataset(num_samples: int = 1000, balance_ratio: float = 1.0) -> list:
    """Генерация сбалансированного синтетического датасета (заглушка, если MT5 недоступен)"""
    print("Генерируем синтетический датасет как fallback...")
    return generate_synthetic_dataset(num_samples)


def generate_synthetic_dataset(num_samples: int = 1000) -> list:
    """Генерация СБАЛАНСИРОВАННОГО синтетического датасета"""
    print(f"\nГенерация {num_samples} СБАЛАНСИРОВАННЫХ синтетических примеров...")
    print(f"Целевой баланс UP/DOWN: {balance_ratio}:1\n")

    dataset = []
    symbols = ["EURUSD", "GBPUSD", "USDCHF", "USDCAD"]

    target_up = int(num_samples * balance_ratio / (1 + balance_ratio))
    target_down = num_samples - target_up

    up_count = 0
    down_count = 0

    print(f"Целевое распределение:")
    print(f" UP: {target_up} примеров ({target_up/num_samples*100:.1f}%)")
    print(f" DOWN: {target_down} примеров ({target_down/num_samples*100:.1f}%)\n")

    attempts = 0
    max_attempts = num_samples * 3

    while len(dataset) < num_samples and attempts < max_attempts:
        attempts += 1

        symbol = np.random.choice(symbols)
        price = np.random.uniform(1.0500, 1.2000) if "EUR" in symbol else np.random.uniform(1.2000, 1.4000)
        rsi = np.random.uniform(20, 80)
        macd = np.random.uniform(-0.001, 0.001)
        atr = np.random.uniform(0.0005, 0.0030)
        vol_ratio = np.random.uniform(0.5, 2.0)
        bb_pos = np.random.uniform(0, 1)
        stoch_k = np.random.uniform(20, 80)

        bullish_signals = 0
        bearish_signals = 0

        if rsi < 30:
            bullish_signals += 2
        elif rsi > 70:
            bearish_signals += 2
        elif 40 < rsi < 50:
            bullish_signals += 1
        elif 50 < rsi < 60:
            bearish_signals += 1

        if macd > 0:
            bullish_signals += 2
        else:
            bearish_signals += 2

        if bb_pos < 0.2:
            bullish_signals += 1
        elif bb_pos > 0.8:
            bearish_signals += 1

        if vol_ratio > 1.5:
            bullish_signals += 1 if bullish_signals > bearish_signals else bearish_signals + 1

        if stoch_k < 20:
            bullish_signals += 1
        elif stoch_k > 80:
            bearish_signals += 1

        direction = "UP" if bullish_signals > bearish_signals else "DOWN"

        if direction == "UP" and up_count >= target_up:
            continue
        if direction == "DOWN" and down_count >= target_down:
            continue

        confidence = min(98, max(65, 60 + abs(bullish_signals - bearish_signals) * 8))

        signal_strength = abs(bullish_signals - bearish_signals)
        base_move = signal_strength * 15 + np.random.randint(10, 40)
        price_24h_move = base_move if direction == "UP" else -base_move
        price_24h = price + (price_24h_move * 0.0001)

        analysis_parts = []
        if rsi < 30:
            analysis_parts.append(f"RSI {rsi:.1f} — сильная перепроданность, через 24ч жду отскок на {abs(price_24h_move)} пунктов")
        elif rsi > 70:
            analysis_parts.append(f"RSI {rsi:.1f} — перекупленность, за сутки возможна коррекция до {abs(price_24h_move)} пунктов")
        else:
            analysis_parts.append(f"RSI {rsi:.1f} — нейтральная зона, прогноз движения {abs(price_24h_move)} пунктов за 24ч")

        if macd > 0.0005:
            analysis_parts.append("MACD сильно позитивный — бычий импульс продолжится в течение суток")
        elif macd < -0.0005:
            analysis_parts.append("MACD негативный — медвежье давление сохранится 24 часа")
        else:
            analysis_parts.append("MACD около нуля — слабый тренд, но направление определено")

        if atr > 0.002:
            analysis_parts.append(f"ATR {atr:.5f} — высокая волатильность, за сутки возможен размах {int(atr/0.0001 * 1.5)} пунктов")
        else:
            analysis_parts.append(f"ATR {atr:.5f} — умеренная волатильность, движение {abs(price_24h_move)} пунктов реалистично")

        if vol_ratio > 1.5:
            analysis_parts.append("Объёмы выше средних на 50%+ — импульс сохранится на ближайшие 24 часа")
        elif vol_ratio < 0.7:
            analysis_parts.append("Объёмы низкие — движение будет медленным, но направление верное")

        if bb_pos < 0.2:
            analysis_parts.append("Цена у нижней границы Боллинджера — через сутки ожидаю возврат к середине канала")
        elif bb_pos > 0.8:
            analysis_parts.append("Цена у верхней границы Боллинджера — за 24ч возможен откат к середине")

        if stoch_k < 20:
            analysis_parts.append(f"Stochastic {stoch_k:.1f} — экстремальная перепроданность, разворот вверх в течение суток")
        elif stoch_k > 80:
            analysis_parts.append(f"Stochastic {stoch_k:.1f} — экстремальная перекупленность, коррекция вниз за 24ч")

        analysis = "\n- ".join(analysis_parts)

        prompt = f"""{symbol} {datetime.now().strftime('%Y-%m-%d %H:%M')}
Текущая цена: {price:.5f}
RSI: {rsi:.1f}
MACD: {macd:.6f}
ATR: {atr:.5f}
Объёмы: {vol_ratio:.2f}x
BB позиция: {bb_pos:.2f}
Stochastic K: {stoch_k:.1f}
Проанализируй ситуацию объективно и дай точный прогноз цены через 24 часа.
ВАЖНО: Давай прогноз на основе данных, без предвзятости к направлению."""

        response = f"""НАПРАВЛЕНИЕ: {direction}
УВЕРЕННОСТЬ: {confidence}%
ПРОГНОЗ ЦЕНЫ ЧЕРЕЗ 24Ч: {price_24h:.5f} ({price_24h_move:+d} пунктов)
ОБЪЕКТИВНЫЙ АНАЛИЗ НА 24 ЧАСА:
- {analysis}
ВЫВОД: Технический анализ по {abs(bullish_signals + bearish_signals)} индикаторам указывает на {'бычий' if direction == 'UP' else 'медвежий'} сценарий. За ближайшие 24 часа жду движение {abs(price_24h_move)} пунктов {direction} к цели {price_24h:.5f}.
НАПОМИНАНИЕ: Рынок непредсказуем. Этот анализ основан на текущих технических данных, но ситуация может измениться. Следующий сигнал может быть противоположным."""

        dataset.append({
            "prompt": prompt,
            "response": response,
            "direction": direction
        })

        if direction == "UP":
            up_count += 1
        else:
            down_count += 1

        if len(dataset) % 100 == 0:
            current_ratio = max(up_count, down_count) / max(1, min(up_count, down_count))
            print(f"Создано {len(dataset)}/{num_samples} | UP: {up_count} | DOWN: {down_count} | Ratio: {current_ratio:.2f}:1")

    print(f"\nСинтетический датасет готов: {len(dataset)} примеров")
    print(f" UP: {up_count} ({up_count/len(dataset)*100:.1f}%)")
    print(f" DOWN: {down_count} ({down_count/len(dataset)*100:.1f}%)")

    actual_ratio = max(up_count, down_count) / min(up_count, down_count)
    print(f" Фактический баланс: {actual_ratio:.2f}:1")
    print(" ОТЛИЧНО! Датасет сбалансирован\n" if actual_ratio <= 1.2 else " Небольшой дисбаланс, но приемлемо\n")

    return dataset


def save_dataset(dataset: list, filename: str = "dataset/finetune_data.jsonl"):
    """Сохранение датасета в JSONL формате"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Датасет сохранён: {filename}")
    return filename


# ====================== ФАЙНТЬЮН ЧЕРЕЗ OLLAMA ======================
def finetune_with_ollama(dataset_path: str):
    """Файнтьюн модели через Ollama"""
    print("\nЗАПУСК ФАЙНТЬЮНА ЧЕРЕЗ OLLAMA\n")
    print("=" * 80)

    try:
        subprocess.run(["ollama", "--version"], check=True, capture_output=True)
    except:
        print("Ollama не установлен!")
        print("Установи: https://ollama.com/download")
        return

    print("Создание Modelfile с обучающими данными...")

    with open(dataset_path, 'r', encoding='utf-8') as f:
        training_data = [json.loads(line) for line in f]

    training_sample = training_data[:min(100, len(training_data))]

    modelfile_content = f"""FROM {BASE_MODEL}
PARAMETER temperature 0.55
PARAMETER top_p 0.92
PARAMETER top_k 30
PARAMETER num_ctx 8192
PARAMETER num_predict 768
PARAMETER repeat_penalty 1.1
SYSTEM \"\"\"
Ты — ShtencoAiTrader-3B-Ultra-Analyst v3 — элитный аналитик валютного рынка с прогнозами на 24 часа.
СТРОГИЕ ПРАВИЛА:
1. Только UP или DOWN — никакого FLAT, боковика, неуверенности
2. Уверенность всегда 65-98%
3. ОБЯЗАТЕЛЬНО давай прогноз цены через 24 часа в формате: X.XXXXX (±NN пунктов)
4. Детальный анализ каждого индикатора с учётом суточного таймфрейма
5. Конкретные рекомендации с целевой ценой
ФОРМАТ ОТВЕТА (СТРОГО):
НАПРАВЛЕНИЕ: UP/DOWN
УВЕРЕННОСТЬ: XX%
ПРОГНОЗ ЦЕНЫ ЧЕРЕЗ 24Ч: X.XXXXX (±NN пунктов)
ПОЛНЫЙ АНАЛИЗ НА 24 ЧАСА:
- RSI: [детальный анализ с прогнозом на сутки]
- MACD: [детальный анализ с прогнозом на сутки]
- ATR: [детальный анализ с прогнозом на сутки]
- Объёмы: [детальный анализ с прогнозом на сутки]
- Bollinger Bands: [детальный анализ с прогнозом на сутки]
- Stochastic: [детальный анализ с прогнозом на сутки]
ИТОГ: [конкретная рекомендация с целевой ценой через 24 часа и обоснованием]
\"\"\"
"""

    for i, example in enumerate(training_sample[:500], 1):
        modelfile_content += f"""
MESSAGE user \"\"\"{example['prompt']}\"\"\"
MESSAGE assistant \"\"\"{example['response']}\"\"\"
"""

    modelfile_path = "Modelfile_finetune"
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)

    print(f"Modelfile создан с {training_sample} примерами")

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
        print(f"\nМодель {MODEL_NAME} успешно создана!")

        print("\nТестирование модели...")
        test_prompt = """EURUSD 2025-11-21 10:00
Текущая цена: 1.0850
RSI: 32.5
MACD: -0.00015
ATR: 0.00085
Объёмы: 1.8x
BB позиция: 0.15
Stochastic K: 25.0
Проанализируй и дай точный прогноз цены через 24 часа."""
        test_result = ollama.generate(model=MODEL_NAME, prompt=test_prompt)
        print("\n" + "=" * 80)
        print("ТЕСТОВЫЙ ОТВЕТ:")
        print("=" * 80)
        print(test_result['response'])
        print("=" * 80)

        os.remove(modelfile_path)

        print(f"\nФАЙНТЬЮН ЗАВЕРШЁН!")
        print(f"Модель готова к использованию: {MODEL_NAME}")
        print(f"\nДля публикации в реестр Ollama:")
        print(f" ollama push {MODEL_NAME}")

    except subprocess.CalledProcessError as e:
        print(f"Ошибка при создании модели: {e}")
        print(f"Вывод: {e.output}")


# ====================== ПАРСИНГ ======================
def parse_answer(text: str) -> dict:
    """Парсинг ответа модели с прогнозом цены"""
    prob = re.search(r"(?:УВЕРЕННОСТЬ|ВЕРОЯТНОСТЬ)[\s:]*(\d+)", text, re.I)
    direction = re.search(r"\b(UP|DOWN)\b", text, re.I)
    price_pred = re.search(r"ПРОГНОЗ ЦЕНЫ.*?(\d+\.\d+)", text, re.I)

    p = int(prob.group(1)) if prob else 50
    d = direction.group(1).upper() if direction else "DOWN"
    target_price = float(price_pred.group(1)) if price_pred else None

    return {"prob": p, "dir": d, "target_price": target_price}


# ====================== ВИЗУАЛИЗАЦИЯ ======================
def plot_results(balance_hist, equity_hist, slots):
    """Построение графика эквити"""
    plt.figure(figsize=(16, 6))

    min_length = min(len(equity_hist), len(slots))
    dates = [s['datetime'] for s in slots[:min_length]]
    equity_to_plot = equity_hist[:min_length]

    plt.plot(dates, equity_to_plot, color='#1E90FF', linewidth=3.5, label='Эквити')

    plt.title('Эквити', fontsize=16, fontweight='bold', color='white')
    plt.xlabel('Время', color='white')
    plt.ylabel('Баланс ($)', color='white')

    ax = plt.gca()
    ax.set_facecolor('#0a0a0a')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('white')
    ax.tick_params(colors='white')
    plt.grid(alpha=0.2, color='gray')

    plt.xticks(rotation=45)
    plt.tight_layout()

    filename = f"charts/equity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=300, facecolor='#0a0a0a')
    print(f"\nГрафик сохранён: {filename}")
    plt.show()


def calculate_max_drawdown(equity):
    """Расчёт максимальной просадки"""
    if len(equity) == 0:
        return 0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-8)
    return np.max(dd) * 100


# ====================== 1. PUSH ======================
def mode_push():
    """Пуш базовой модели в Ollama"""
    print("\n" + "=" * 80)
    print("1. ПУШ БАЗОВОЙ МОДЕЛИ В OLLAMA")
    print("=" * 80)

    content = f"""FROM {BASE_MODEL}
PARAMETER temperature 0.55
PARAMETER top_p 0.92
PARAMETER top_k 30
PARAMETER num_ctx 8192
PARAMETER num_predict 768
SYSTEM \"\"\"
Ты — ShtencoAiTrader-3B-Ultra-Analyst v3 — лучший в мире аналитик валютного рынка.
Ты всегда даешь четкое направление: UP или DOWN. Слова FLAT, боковик, не уверен — полностью запрещены.
Ты ОБЯЗАТЕЛЬНО даёшь прогноз цены через 24 часа в формате: X.XXXXX (±NN пунктов)
Ты подробно разбираешь каждый индикатор (RSI, MACD, объемы, ATR, уровни, свечи и т.д.).
Формат ответа строго такой:
НАПРАВЛЕНИЕ: UP
УВЕРЕННОСТЬ: 87%
ПРОГНОЗ ЦЕНЫ ЧЕРЕЗ 24Ч: 1.08750 (+45 пунктов)
ПОЛНЫЙ АНАЛИЗ НА 24 ЧАСА:
- RSI: ...
- MACD: ...
- Объемы: ...
- ATR и волатильность: ...
- Уровни поддержки/сопротивления: ...
- Свечной паттерн: ...
ИТОГ: мощный бычий импульс с подтверждением по всем индикаторам, через 24ч цель 1.08750
Уверенность всегда 65–98%. Никаких сомнений.
\"\"\"
"""

    with open("Modelfile", "w", encoding="utf-8") as f:
        f.write(content)
    print("Modelfile создан")
    print("Скачивание базовой модели...")
    subprocess.run(["ollama", "pull", BASE_MODEL], check=True)

    print("Создание модели...")
    subprocess.run(["ollama", "create", MODEL_NAME, "-f", "Modelfile"], check=True)

    print("Пушим в реестр Ollama (5-20 минут)...")
    subprocess.run(["ollama", "push", MODEL_NAME], check=True)

    os.remove("Modelfile")
    print(f"\nГОТОВО! Модель доступна: https://ollama.com/{MODEL_NAME}")


# ====================== 2. ФАЙНТЬЮН ======================
def mode_finetune():
    """Режим файнтьюна"""
    print("\n" + "=" * 80)
    print("2. ФАЙНТЬЮН МОДЕЛИ (ПРОГНОЗ НА 24 ЧАСА)")
    print("=" * 80)
    print("\nВыбери вариант:")
    print("A. Обучение на РЕАЛЬНЫХ данных MT5 (10000 примеров)")
    print("B. Обучение на синтетических данных (10000 примеров)")
    print("C. Использовать существующий датасет")
    print("D. Только сгенерировать датасет")

    choice = input("\nВыбор (A/B/C/D): ").strip().upper()

    if choice == "A":
        dataset = generate_real_dataset_from_mt5(FINETUNE_SAMPLES)
        dataset_path = save_dataset(dataset, "dataset/finetune_real_mt5.jsonl")
        finetune_with_ollama(dataset_path)

    elif choice == "B":
        dataset = generate_synthetic_dataset(FINETUNE_SAMPLES)
        dataset_path = save_dataset(dataset)
        finetune_with_ollama(dataset_path)

    elif choice == "C":
        dataset_path = input("Путь к датасету (JSONL): ").strip()
        if os.path.exists(dataset_path):
            finetune_with_ollama(dataset_path)
        else:
            print(f"Файл {dataset_path} не найден")

    elif choice == "D":
        print("\nВыбери тип датасета:")
        print("1. Реальные данные MT5")
        print("2. Синтетические данные")

        dtype = input("Выбор (1/2): ").strip()
        num_samples = int(input("Количество примеров (по умолчанию 10000): ").strip() or "1000")

        if dtype == "1":
            dataset = generate_real_dataset_from_mt5(num_samples)
            save_dataset(dataset, "dataset/finetune_real_mt5.jsonl")
        else:
            dataset = generate_synthetic_dataset(num_samples)
            save_dataset(dataset)

        print("\nДатасет готов для файнтьюна!")

    else:
        print("Неверный выбор")


# ====================== 3. БЭКТЕСТ ======================
def backtest():
    print("\n" + "=" * 80)
    print("3. ИСПРАВЛЕННЫЙ БЭКТЕСТ (БЕЗ УТЕЧКИ ДАННЫХ)")
    print("=" * 80)

    if not mt5 or not mt5.initialize():
        print("MT5 не подключен → запускаем на синтетических данных")
        mock = True
    else:
        mock = False
        print("MT5 подключен, загружаем реальные данные...")
    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(days=BACKTEST_DAYS)

    data = {}
    print(f"\nЗагрузка данных с {start.strftime('%Y-%m-%d')} по {end.strftime('%Y-%m-%d')}...")

    for sym in SYMBOLS:
        if not mock:
            rates = mt5.copy_rates_range(sym, TIMEFRAME, start, end)
            if rates is None or len(rates) == 0:
                print(f"Нет данных для {sym}")
                continue
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
        else:
            dates = pd.date_range(start, end, freq="15min")
            close = 1.0800 + np.cumsum(np.random.randn(len(dates)) * 0.0002)
            df = pd.DataFrame({
                "time": dates,
                "open": close + np.random.randn(len(dates)) * 0.0001,
                "high": close + abs(np.random.randn(len(dates))) * 0.0003,
                "low": close - abs(np.random.randn(len(dates))) * 0.0003,
                "close": close,
                "tick_volume": np.random.randint(1000, 10000, len(dates)),
            })
        df.set_index("time", inplace=True)

        if len(df) > LOOKBACK + PREDICTION_HORIZON:
            data[sym] = df
            print(f"{sym}: {len(df)} баров загружено")

    if not data:
        print("\nНет данных для бэктеста!")
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

    print(f"\nСимуляция торговли...")
    print(f"Начальный баланс: ${balance:,.2f}")
    print(f"Риск на сделку: {RISK_PER_TRADE * 100}%")
    print(f"Спред: {SPREAD_PIPS} пункта")
    print(f"Своп лонг/шорт: {SWAP_LONG}/{SWAP_SHORT} USD/день\n")

    use_ai = False
    if ollama:
        try:
            ollama.list()
            use_ai = True
            print("Ollama подключен\n")
        except:
            print("Ollama недоступен, используем простую логику\n")

    main_symbol = list(data.keys())[0]
    main_data = data[main_symbol]
    total_bars = len(main_data)
    analysis_points = list(range(LOOKBACK, total_bars - PREDICTION_HORIZON, PREDICTION_HORIZON))

    print(f"Точек анализа: {len(analysis_points)}\n")

    for point_idx, current_idx in enumerate(analysis_points):
        current_time = main_data.index[current_idx]

        for sym in SYMBOLS:
            if sym not in data:
                continue

            historical_data = data[sym].iloc[:current_idx + 1].copy()
            if len(historical_data) < LOOKBACK:
                continue

            df_with_features = calculate_features(historical_data)
            if len(df_with_features) == 0:
                continue

            row = df_with_features.iloc[-1]

            if not mock:
                symbol_info = mt5.symbol_info(sym)
                if symbol_info is None:
                    continue
                point = symbol_info.point
                contract_size = symbol_info.trade_contract_size
            else:
                point = 0.0001
                contract_size = 100000

            prompt = f"""{sym} {current_time.strftime('%Y-%m-%d %H:%M')}
Текущая цена: {row['close']:.5f}
RSI: {row['RSI']:.1f}
MACD: {row['MACD']:.6f}
ATR: {row['ATR']:.5f}
Объёмы: {row['vol_ratio']:.2f}x
BB позиция: {row['BB_position']:.2f}
Stochastic K: {row['Stoch_K']:.1f}
Проанализируй и дай точный прогноз цены через 24 часа."""

            try:
                if use_ai:
                    resp = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.3})
                    result = parse_answer(resp["response"])
                else:
                    rsi_signal = 1 if row['RSI'] < 50 else -1
                    macd_signal = 1 if row['MACD'] > 0 else -1
                    combined = rsi_signal + macd_signal
                    result = {
                        "prob": min(95, max(65, 70 + abs(combined) * 10)),
                        "dir": "UP" if combined > 0 else "DOWN",
                        "target_price": None
                    }

                if result["prob"] < MIN_PROB:
                    continue

                entry_price = row['close'] + SPREAD_PIPS * point if result["dir"] == "UP" else row['close']
                exit_idx = current_idx + PREDICTION_HORIZON
                if exit_idx >= len(data[sym]):
                    continue
                exit_row = data[sym].iloc[exit_idx]
                exit_price = exit_row['close'] if result["dir"] == "UP" else exit_row['close'] + SPREAD_PIPS * point

                price_move_pips = (exit_price - entry_price) / point if result["dir"] == "UP" else (entry_price - exit_price) / point

                risk_amount = balance * RISK_PER_TRADE
                atr_pips = row['ATR'] / point
                stop_loss_pips = max(20, atr_pips * 2)
                lot_size = risk_amount / (stop_loss_pips * point * contract_size)
                lot_size = max(0.01, min(lot_size, 10.0))

                profit_pips = price_move_pips
                profit_usd = profit_pips * point * contract_size * lot_size
                swap_cost = SWAP_LONG if result["dir"] == "UP" else SWAP_SHORT
                swap_cost = swap_cost * (lot_size / 0.01)
                profit_usd -= swap_cost
                profit_usd -= SLIPPAGE * point * contract_size * lot_size

                balance += profit_usd
                equity = balance

                actual_direction = "UP" if (exit_row['close'] > row['close']) else "DOWN"
                correct = (result["dir"] == actual_direction)

                trades.append({
                    "time": current_time,
                    "symbol": sym,
                    "direction": result["dir"],
                    "prob": result["prob"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "lot_size": lot_size,
                    "profit_pips": profit_pips,
                    "profit_usd": profit_usd,
                    "balance": balance,
                    "correct": correct
                })

                status = "ПРАВИЛЬНО" if correct else "ОШИБКА"
                print(f"{status} {current_time.strftime('%m-%d %H:%M')} | {sym} {result['dir']} {result['prob']}% | "
                      f"Лот {lot_size:.2f} | {entry_price:.5f} → {exit_price:.5f} | "
                      f"{profit_pips:+.1f}p | ${profit_usd:+.2f} | Баланс: ${balance:,.2f}")

            except Exception as e:
                log.error(f"Ошибка при анализе {sym}: {e}")

        balance_hist.append(balance)
        equity_hist.append(equity)
        slots.append({"datetime": current_time})

    print(f"\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ БЭКТЕСТА (ИСПРАВЛЕННАЯ ВЕРСИЯ)")
    print("=" * 80)
    print(f"Всего сделок: {len(trades)}")
    print(f"Начальный баланс: ${INITIAL_BALANCE:,.2f}")
    print(f"Конечный баланс: ${balance:,.2f}")
    print(f"Прибыль/убыток: ${balance - INITIAL_BALANCE:+,.2f} ({((balance/INITIAL_BALANCE - 1) * 100):+.2f}%)")

    if trades:
        wins = sum(1 for t in trades if t['profit_usd'] > 0)
        losses = len(trades) - wins
        win_rate = wins / len(trades) * 100

        print(f"\nСТАТИСТИКА:")
        print(f"Прибыльных: {wins} ({win_rate:.1f}%)")
        print(f"Убыточных: {losses} ({100 - win_rate:.1f}%)")

        if wins > 0:
            avg_win = np.mean([t['profit_usd'] for t in trades if t['profit_usd'] > 0])
            print(f"Средняя прибыль: ${avg_win:.2f}")

        if losses > 0:
            avg_loss = np.mean([t['profit_usd'] for t in trades if t['profit_usd'] < 0])
            print(f"Средний убыток: ${avg_loss:.2f}")

        if wins > 0 and losses > 0:
            profit_factor = abs(sum(t['profit_usd'] for t in trades if t['profit_usd'] > 0)) / abs(sum(t['profit_usd'] for t in trades if t['profit_usd'] < 0))
            print(f"Профит-фактор: {profit_factor:.2f}")

        max_dd = calculate_max_drawdown(np.array(equity_hist))
        print(f"Макс. просадка: {max_dd:.2f}%")

        best_trade = max(trades, key=lambda x: x['profit_usd'])
        worst_trade = min(trades, key=lambda x: x['profit_usd'])


        if len(equity_hist) > 1:
            plot_results(balance_hist, equity_hist, slots)

    if not mock:
        mt5.shutdown()


# ====================== 4. ЛАЙВ ======================
def live():
    """ИСПРАВЛЕННАЯ ЖИВАЯ ТОРГОВЛЯ"""
    print("\n" + "=" * 80)
    print("4. ИСПРАВЛЕННАЯ ЖИВАЯ ТОРГОВЛЯ")
    print("=" * 80)

    if not mt5 or not mt5.initialize():
        print("MT5 не найден → режим недоступен")
        return
    account_info = mt5.account_info()
    if account_info is None:
        print("Не удалось получить информацию о счёте")
        return

    print(f"Подключен к счёту: {account_info.login}")
    print(f"Баланс: ${account_info.balance:.2f}")
    print(f"Эквити: ${account_info.equity:.2f}")
    print(f"Свободная маржа: ${account_info.margin_free:.2f}")

    print("\nВНИМАНИЕ! Сейчас начнётся РЕАЛЬНАЯ торговля!")
    print(" - Позиции будут открываться автоматически")
    print(" - Анализ каждые 24 часа")
    print(" - Закрытие позиций через 24 часа")

    confirm = input("\nПродолжить? (YES для подтверждения): ").strip()
    if confirm != "YES":
        print("Торговля отменена")
        return

    print("\nЗапуск живой торговли...")
    print("Ctrl+C для остановки\n")

    open_positions = {}
    last_analysis_time = None

    while True:
        try:
            now = datetime.now()
            positions = mt5.positions_get()

            # Закрытие позиций по истечении 24 часов
            if positions:
                for pos in positions:
                    if pos.magic == MAGIC:
                        open_time = datetime.fromtimestamp(pos.time)
                        if (now - open_time).total_seconds() >= 86400:
                            request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": pos.symbol,
                                "volume": pos.volume,
                                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                "position": pos.ticket,
                                "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                                "deviation": SLIPPAGE,
                                "magic": MAGIC,
                                "comment": "24h close",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_IOC,
                            }
                            result = mt5.order_send(request)
                            if result.retcode == mt5.TRADE_RETCODE_DONE:
                                print(f"Закрыта {pos.symbol} через 24ч | Тикет: {pos.ticket} | Профит: ${pos.profit:+.2f}")
                                if pos.ticket in open_positions:
                                    del open_positions[pos.ticket]

            # Новый анализ каждые 24 часа
            if last_analysis_time is None or (now - last_analysis_time).total_seconds() >= 86400:
                last_analysis_time = now
                print(f"\n{'='*80}")
                print(f"АНАЛИЗ РЫНКА: {now.strftime('%Y-%m-%d %H:%M')}")
                print(f"{'='*80}\n")

                for sym in SYMBOLS:
                    has_position = any(p.symbol == sym and p.magic == MAGIC for p in (positions or []))
                    if has_position:
                        print(f"{sym}: уже есть открытая позиция, пропускаем")
                        continue

                    rates = mt5.copy_rates_from_pos(sym, TIMEFRAME, 0, LOOKBACK)
                    if rates is None or len(rates) == 0:
                        continue

                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s")
                    df.set_index("time", inplace=True)
                    df = calculate_features(df)
                    if len(df) == 0:
                        continue

                    row = df.iloc[-1]
                    symbol_info = mt5.symbol_info(sym)
                    if symbol_info is None or not symbol_info.visible:
                        continue

                    prompt = f"""{sym} {now.strftime('%Y-%m-%d %H:%M')}
Текущая цена: {row['close']:.5f}
RSI: {row['RSI']:.1f}
MACD: {row['MACD']:.6f}
ATR: {row['ATR']:.5f}
Объёмы: {row['vol_ratio']:.2f}x
BB позиция: {row['BB_position']:.2f}
Stochastic K: {row['Stoch_K']:.1f}
Проанализируй и дай точный прогноз цены через 24 часа."""

                    resp = ollama.generate(model=MODEL_NAME, prompt=prompt, options={"temperature": 0.3})
                    result = parse_answer(resp["response"])

                    print(f"{sym}: {result['dir']} ({result['prob']}%)")
                    if result.get('target_price'):
                        print(f" Текущая: {row['close']:.5f} → Цель 24ч: {result['target_price']:.5f}")

                    if result["prob"] < MIN_PROB:
                        print(f" Уверенность {result['prob']}% < {MIN_PROB}%, пропускаем\n")
                        continue

                    order_type = mt5.ORDER_TYPE_BUY if result["dir"] == "UP" else mt5.ORDER_TYPE_SELL
                    tick = mt5.symbol_info_tick(sym)
                    if tick is None:
                        continue
                    price = tick.ask if result["dir"] == "UP" else tick.bid

                    risk_amount = mt5.account_info().balance * RISK_PER_TRADE
                    point = symbol_info.point
                    atr_pips = row['ATR'] / point
                    stop_loss_pips = max(20, atr_pips * 2)
                    lot_size = risk_amount / (stop_loss_pips * point * symbol_info.trade_contract_size)
                    lot_step = symbol_info.volume_step
                    lot_size = round(lot_size / lot_step) * lot_step
                    lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))

                    sl = price - stop_loss_pips * point if result["dir"] == "UP" else price + stop_loss_pips * point
                    tp = price + stop_loss_pips * 3 * point if result["dir"] == "UP" else price - stop_loss_pips * 3 * point

                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": sym,
                        "volume": lot_size,
                        "type": order_type,
                        "price": price,
                        "sl": sl,
                        "tp": tp,
                        "deviation": SLIPPAGE,
                        "magic": MAGIC,
                        "comment": f"AI_{result['prob']}%",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    result_order = mt5.order_send(request)
                    if result_order.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f" Позиция открыта! Тикет: {result_order.order}, Лот: {lot_size}, Цена: {result_order.price:.5f}\n")
                        open_positions[result_order.order] = {"symbol": sym, "open_time": now, "direction": result["dir"], "lot": lot_size}
                    else:
                        print(f" Ошибка открытия: {result_order.comment}\n")

                print(f"{'='*80}")
                print(f"Открыто позиций: {len(open_positions)}")
                print(f"Следующий анализ: {(now + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')}")
                print(f"{'='*80}\n")

            time.sleep(60)

        except KeyboardInterrupt:
            print("\nОстановка торговли...")
            positions = mt5.positions_get(magic=MAGIC)
            if positions:
                for pos in positions:
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": pos.symbol,
                        "volume": pos.volume,
                        "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                        "position": pos.ticket,
                        "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                        "deviation": SLIPPAGE,
                        "magic": MAGIC,
                        "comment": "manual close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"{pos.symbol} закрыт, профит: ${pos.profit:+.2f}")
            print("Торговля остановлена")
            break
        except Exception as e:
            log.error(f"Критическая ошибка: {e}")
            time.sleep(60)

    mt5.shutdown()


# ====================== 5. ГЕНЕРАЦИЯ ДАТАСЕТА ======================
def mode_dataset():
    """Режим генерации датасета"""
    print("\n" + "=" * 80)
    print("5. ГЕНЕРАЦИЯ ДАТАСЕТА (ПРОГНОЗ НА 24 ЧАСА)")
    print("=" * 80)

    print("\nВыбери тип датасета:")
    print("1. Реальные данные из MT5")
    print("2. Синтетические данные")

    dtype = input("\nВыбор (1/2): ").strip()
    num_samples = input("Количество примеров (по умолчанию 1000): ").strip()
    num_samples = int(num_samples) if num_samples else 1000

    if dtype == "1":
        dataset = generate_real_dataset_from_mt5(num_samples)
        dataset_path = save_dataset(dataset, "dataset/finetune_real_m5.jsonl")
    else:
        dataset = generate_synthetic_dataset(num_samples)
        dataset_path = save_dataset(dataset)

    print(f"\nДатасет сохранён: {dataset_path}")
    print(f"Примеров: {len(dataset)}")
    print(f"Размер: {os.path.getsize(dataset_path) / 1024:.1f} KB")

    print("\n" + "=" * 80)
    print("ПРИМЕР ИЗ ДАТАСЕТА:")
    print("=" * 80)
    example = dataset[0]
    print("\nПРОМПТ:")
    print(example['prompt'])
    print("\nОТВЕТ:")
    print(example['response'])


# ====================== МЕНЮ ======================
def main():
    """Главное меню"""
    print("\n" + "=" * 80)
    print(" SHTENCO AI TRADER ULTRA 3B — РЕАЛЬНЫЕ ДАННЫЕ MT5")
    print(" Версия: 21.11.2025 (Real MT5 Data + 24h prediction)")
    print("=" * 80)
    print("\nРЕЖИМЫ РАБОТЫ:")
    print("-" * 80)
    print("1 → Пуш базовой модели в Ollama")
    print("2 → ФАЙНТЬЮН (реальные данные MT5 или синтетика)")
    print("3 → Бэктест на исторических данных (24h прогнозы)")
    print("4 → Живая торговля (MT5, анализ каждые 24 часа)")
    print("5 → Генерация обучающего датасета (MT5 или синтетика)")
    print("-" * 80)

    choice = input("\nВыбери режим (1-5): ").strip()
    if choice == "1":
        mode_push()
    elif choice == "2":
        mode_finetune()
    elif choice == "3":
        backtest()
    elif choice == "4":
        live()
    elif choice == "5":
        mode_dataset()
    else:
        print("Неверный выбор")


if __name__ == "__main__":
    main()
