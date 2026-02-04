import matplotlib

matplotlib.use("Agg")  # Set the backend to 'Agg' before importing pyplot
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from pathlib import Path

# Constants
V_rest = -65.0  # Resting potential (mV)
Cm = 1.0  # Membrane capacitance (μF/cm²)
g_Na = 120.0  # Maximum Na+ channel conductance (mS/cm²)
g_K = 36.0  # Maximum K+ channel conductance (mS/cm²)
g_L = 0.3  # Leak conductance (mS/cm²)
E_Na = 50.0  # Na+ equilibrium potential (mV)
E_K = -77.0  # K+ equilibrium potential (mV)
E_L = -54.4  # Leak equilibrium potential (mV)

# Plasma parameters
plasma_strength = 1.0  # Plasma influence strength
plasma_decay = 20.0  # Plasma influence decay time

# STDP parameters
A_plus = 0.1  # Enhancement coefficient for positive Δt
A_minus = 0.1  # Weakening coefficient for negative Δt
tau_plus = 20.0  # Decay time for positive Δt
tau_minus = 20.0  # Decay time for negative Δt


# Market features calculation class
class MarketFeatures:
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.is_fitted = False

    def calculate_features(self, ohlc_data: pd.DataFrame) -> np.ndarray:
        if len(ohlc_data) < self.window_size:
            return np.zeros(100)

        features = []
        close = ohlc_data["close"]
        volume = ohlc_data["tick_volume"]

        # Moving averages (4)
        features.append(self._calculate_sma(close, window=10))
        features.append(self._calculate_sma(close, window=20))
        features.append(self._calculate_ema(close, window=10))
        features.append(self._calculate_ema(close, window=20))

        # RSI (1)
        features.append(self._calculate_rsi(close, window=14))

        # MACD (2)
        macd, signal = self._calculate_macd(close)
        features.append(macd)
        features.append(signal)

        # Bollinger Bands (2)
        upper, lower = self._calculate_bollinger_bands(close, window=20)
        features.append(upper)
        features.append(lower)

        # ATR (1)
        features.append(self._calculate_atr(ohlc_data, window=14))

        # Momentum (1)
        features.append(self._calculate_momentum(close, window=10))

        # Volume (2)
        features.append(self._calculate_sma(volume, window=10))
        features.append(self._calculate_sma(volume, window=20))

        # Time features (3)
        features.append(ohlc_data.index[-1].dayofweek / 7.0)
        features.append(ohlc_data.index[-1].hour / 24.0)
        features.append(ohlc_data.index[-1].month / 12.0)

        # Pad to 100 features
        while len(features) < 100:
            features.append(0.0)

        return np.array(features[:100])

    def _calculate_sma(self, series: pd.Series, window: int) -> float:
        if len(series) < window: return 0.0
        return series.rolling(window=window).mean().iloc[-1]

    def _calculate_ema(self, series: pd.Series, window: int) -> float:
        if len(series) < window: return 0.0
        return series.ewm(span=window, adjust=False).mean().iloc[-1]

    def _calculate_rsi(self, series: pd.Series, window: int) -> float:
        if len(series) < window + 1: return 50.0
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        if loss.iloc[-1] == 0: return 100.0
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def _calculate_macd(self, series: pd.Series) -> Tuple[float, float]:
        if len(series) < 26: return 0.0, 0.0
        ema_12 = series.ewm(span=12, adjust=False).mean()
        ema_26 = series.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.iloc[-1], signal.iloc[-1]

    def _calculate_bollinger_bands(self, series: pd.Series, window: int) -> Tuple[float, float]:
        if len(series) < window: return 0.0, 0.0
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return sma.iloc[-1] + (2 * std.iloc[-1]), sma.iloc[-1] - (2 * std.iloc[-1])

    def _calculate_atr(self, ohlc_data: pd.DataFrame, window: int) -> float:
        if len(ohlc_data) < window + 1: return 0.0
        tr = np.maximum(ohlc_data["high"] - ohlc_data["low"], 
                        np.maximum(abs(ohlc_data["high"] - ohlc_data["close"].shift()), 
                                   abs(ohlc_data["low"] - ohlc_data["close"].shift())))
        return tr.rolling(window=window).mean().iloc[-1]

    def _calculate_momentum(self, series: pd.Series, window: int) -> float:
        if len(series) < window: return 0.0
        return series.iloc[-1] - series.iloc[-window]


class HodgkinHuxleyNeuron:
    def __init__(self):
        self.V = V_rest
        self.m = 0.05
        self.h = 0.6
        self.n = 0.32
        self.last_spike_time = -100.0
        self.spike_count = 0

    def update(self, I_ext, dt=0.1):
        # Safety wrapper for exp
        def safe_exp(x):
            return np.exp(np.clip(x, -50, 50))

        # α and β functions
        v_shifted = self.V + 40
        if abs(v_shifted) < 1e-6:
            am = 1.0
        else:
            am = 0.1 * v_shifted / (1 - safe_exp(-v_shifted / 10))
            
        bm = 4.0 * safe_exp(-(self.V + 65) / 18)
        ah = 0.07 * safe_exp(-(self.V + 65) / 20)
        bh = 1.0 / (1 + safe_exp(-(self.V + 35) / 10))
        
        v_shifted_n = self.V + 55
        if abs(v_shifted_n) < 1e-6:
            an = 0.1
        else:
            an = 0.01 * v_shifted_n / (1 - safe_exp(-v_shifted_n / 10))
            
        bn = 0.125 * safe_exp(-(self.V + 65) / 80)

        # Update gates
        self.m += dt * (am * (1 - self.m) - bm * self.m)
        self.h += dt * (ah * (1 - self.h) - bh * self.h)
        self.n += dt * (an * (1 - self.n) - bn * self.n)

        # Calculate currents
        I_Na = g_Na * self.m**3 * self.h * (self.V - E_Na)
        I_K = g_K * self.n**4 * (self.V - E_K)
        I_L = g_L * (self.V - E_L)

        # Update potential
        dV = (I_ext - I_Na - I_K - I_L) / Cm
        self.V += dV * dt

        # Check for spike
        spiked = False
        if self.V > 30:
            self.V = V_rest
            spiked = True
            self.spike_count += 1
        
        # Stability: clip potential
        self.V = np.clip(self.V, -100, 50)
        return spiked


class BioTradingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BioTradingModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.Tanh()
        
        # Bio-inspired components
        self.hidden_size = hidden_size
        self.neurons = [HodgkinHuxleyNeuron() for _ in range(hidden_size)]
        self.spike_history = []

    def forward(self, x):
        # Standard pass
        h1 = self.activation(self.fc1(x))
        h1 = self.dropout(h1)
        
        # Integrate HH neurons: use h1 as input current I_ext
        # In a real-time pass, we simulate one step
        current_spikes = []
        h1_np = h1.detach().cpu().numpy().flatten()
        for i, neuron in enumerate(self.neurons):
            # Scale input current to meaningful HH range
            spiked = neuron.update(I_ext=h1_np[i] * 20.0, dt=0.5)
            current_spikes.append(1.0 if spiked else 0.0)
        
        self.spike_history.append(current_spikes)
        if len(self.spike_history) > 100: self.spike_history.pop(0)

        # Pass h1 through second layer
        h2 = self.activation(self.fc2(h1))
        h2 = self.dropout(h2)
        out = self.fc3(h2)
        return out

    def apply_stdp(self, pre_features):
        if len(self.spike_history) < 2: return
        
        # Simplified STDP update for fc1 weights
        post_spikes = self.spike_history[-1]
        pre_vals = pre_features.detach().cpu().numpy().flatten()
        
        with torch.no_grad():
            for i in range(self.hidden_size):
                if post_spikes[i] > 0:
                    # Potentiation: if neuron spiked, strengthen connections to active inputs
                    # Simplified: use pre_vals as a proxy for "recent" pre-synaptic activity
                    adjustment = torch.from_numpy(pre_vals * A_plus).to(self.fc1.weight.device)
                    self.fc1.weight[i] += adjustment
                else:
                    # Depression: slightly weaken connections if no spike
                    self.fc1.weight[i] *= (1.0 - A_minus * 0.01)


class EnhancedPlasmaBrainTrader:
    def __init__(self, input_size, hidden_size, output_size):
        self.model = BioTradingModel(input_size, hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        self.price_history = []
        self.predictions = []

    def predict(self, price: float, features_np: np.ndarray, train=True):
        self.price_history.append(price)
        
        # Normalize features
        if not self.is_scaler_fitted:
            # First time, we can't really normalize, but let's fit it
            self.scaler.fit(features_np.reshape(1, -1))
            self.is_scaler_fitted = True
        
        norm_features = self.scaler.transform(features_np.reshape(1, -1))
        features_tensor = torch.tensor(norm_features, dtype=torch.float32)

        # Model prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features_tensor)
        self.predictions.append(prediction.item())

        # Model training
        if train and len(self.price_history) > 1:
            self.model.train()
            self.optimizer.zero_grad()
            
            # Target is the current price (we are predicting the next price)
            # Actually, the loop in main passes current price and OHLC up to now.
            # So the model predicts 'next' price. 
            # We can only train when we HAVE the next price.
            # So we train on the PREVIOUS prediction vs CURRENT price.
            if len(self.predictions) > 1:
                prev_features = self.last_features
                target = torch.tensor([[price]], dtype=torch.float32)
                
                # Re-run forward to get gradients
                pred_to_train = self.model(prev_features)
                loss = self.criterion(pred_to_train, target)
                loss.backward()
                self.optimizer.step()
                
                # Apply STDP
                self.model.apply_stdp(prev_features)

        self.last_features = features_tensor
        return prediction.item()

    def get_stats(self):
        if len(self.predictions) < 2:
            return {"correlation": 0, "mse": 0, "total_trades": 0}
        
        actuals = np.array(self.price_history[1:])
        preds = np.array(self.predictions[:-1])
        
        if len(actuals) < 2: return {"correlation": 0, "mse": 0, "total_trades": 0}
        
        correlation = np.corrcoef(actuals, preds)[0, 1]
        if np.isnan(correlation): correlation = 0
        mse = np.mean((actuals - preds) ** 2)
        return {
            "correlation": correlation,
            "mse": mse,
            "total_trades": len(actuals),
        }


# Main code
if __name__ == "__main__":
    # MT5 initialization
    if not mt5.initialize():
        print("Error: MT5 initialization failed")
        quit()

    # Data loading
    symbol = "EURUSD"
    timeframe = mt5.TIMEFRAME_D1
    start_date = datetime.now() - timedelta(days=365 * 8)
    end_date = datetime.now()
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    prices = df["close"].values

    # Split data into training and testing sets
    train_size = int(len(prices) * 0.8)
    train_prices, test_prices = prices[:train_size], prices[train_size:]

    # Iterative improvement
    best_stats = None
    best_config = None
    best_trader = None

    print("\nStarting training process...")
    for iteration in range(20):
        print(f"\nIteration {iteration + 1}/20")
        input_size = 100
        hidden_size = 64
        output_size = 1
        trader = EnhancedPlasmaBrainTrader(input_size, hidden_size, output_size)
        mf = MarketFeatures()
        predictions = []

        print("Processing training data...")
        for i in range(20, len(train_prices)):
            price = train_prices[i]
            ohlc_data = df.iloc[:i]
            features = mf.calculate_features(ohlc_data)
            pred = trader.predict(price, features)
            predictions.append(pred)

            if i % 200 == 0:
                print(f"Processed {i}/{len(train_prices)} samples")

        stats = trader.get_stats()
        total_spikes = sum(n.spike_count for n in trader.model.neurons)
        if best_stats is None or stats["correlation"] > best_stats["correlation"]:
            best_stats = stats
            best_config = (input_size, hidden_size, output_size)
            best_trader = trader # Keep best trader for testing
        print(
            f"Current correlation: {stats['correlation']:.3f}, MSE: {stats['mse']:.6f}, Total Spikes: {total_spikes}"
        )

    print("\nStarting testing process...")
    test_predictions = []
    mf_test = MarketFeatures()
    trader = best_trader # Use the best one found
    
    # We should continue from where training ended for the trader's state
    # But for simplicity, let's just run on test data
    for i in range(train_size, len(prices)):
        price = prices[i]
        ohlc_data = df.iloc[:i]
        features = mf_test.calculate_features(ohlc_data)
        # In test mode, we might not want to train, but bio-systems often keep learning
        pred = trader.predict(price, features, train=True) 
        test_predictions.append(pred)

        if (i - train_size) % 100 == 0:
            print(f"Processed {i - train_size}/{len(prices) - train_size} test samples")

    # Stats for test
    actuals = prices[train_size+1:]
    preds = np.array(test_predictions[:-1])
    test_correlation = np.corrcoef(actuals, preds)[0, 1] if len(actuals) > 1 else 0
    test_mse = np.mean((actuals - preds) ** 2) if len(actuals) > 1 else 0

    # Create charts directory
    Path("./charts").mkdir(exist_ok=True)

    # Training process visualization
    plt.figure(figsize=(8, 4))
    actual_train = train_prices[20:]
    plt.plot(range(len(predictions)), predictions, label="Prediction", alpha=0.7)
    plt.plot(range(len(actual_train)), actual_train, label="Actual", alpha=0.7)
    plt.title("Training Process: Prediction vs Reality")
    plt.legend()
    plt.grid(True)
    plt.gcf().set_size_inches(6, 4)
    plt.savefig("./charts/training_process.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Test results visualization
    plt.figure(figsize=(8, 4))
    actual_test = prices[train_size:]
    plt.plot(
        range(len(test_predictions)),
        test_predictions,
        label="Prediction",
        color="blue",
        alpha=0.7,
    )
    plt.plot(
        range(len(actual_test)), actual_test, label="Actual", color="red", alpha=0.7
    )
    plt.title("Test Results: Prediction vs Reality")
    plt.legend()
    plt.grid(True)
    plt.gcf().set_size_inches(6, 4)
    plt.savefig("./charts/test_results.png", dpi=100, bbox_inches="tight")
    plt.close()

    # Training error visualization
    plt.figure(figsize=(8, 4))
    errors = np.abs(np.array(predictions) - actual_train)
    plt.plot(range(len(errors)), errors, label="Error", color="red")
    plt.title("Training Error Dynamics")
    plt.xlabel("Iteration")
    plt.ylabel("Absolute Error")
    plt.grid(True)
    plt.gcf().set_size_inches(6, 4)
    plt.savefig("./charts/training_error.png", dpi=100, bbox_inches="tight")
    plt.close()

    print("\nBest configuration:")
    print(f"Architecture: {best_config}")
    print(f"Test data correlation: {test_correlation:.3f}, MSE: {test_mse:.6f}")
    print("\nCharts saved in ./charts/ directory")

    # Save the best model
    if best_trader:
        torch.save(best_trader.model.state_dict(), "best_bio_model.pth")
        import joblib
        joblib.dump(best_trader.scaler, "best_scaler.gz")
        print("Model and scaler saved to disk.")

    # Close MT5
    mt5.shutdown()
