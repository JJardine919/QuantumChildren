import tkinter as tk
from tkinter import ttk
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path

# Import classes from the main file
from BioTraderLearn import MarketFeatures, EnhancedPlasmaBrainTrader


class PlasmaBrainPredictor(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PlasmaBrain Predictor")
        self.geometry("1000x800")

        # Default parameters
        self.symbol = "EURUSD"
        self.timeframe = mt5.TIMEFRAME_H1
        self.historic_bars = 20
        self.forecast_bars = 5

        # Initialize MT5
        if not mt5.initialize():
            print("Error: MT5 initialization failed")
            quit()

        # Get the list of available symbols
        self.symbols = mt5.symbols_get()
        self.symbol_names = [sym.name for sym in self.symbols]

        # Dictionary of timeframes
        self.timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

        self.setup_gui()
        self.setup_model()

    def setup_gui(self):
        # Upper control panel
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Select symbol
        ttk.Label(control_frame, text="Symbol:").pack(side=tk.LEFT, padx=5)
        self.symbol_var = tk.StringVar(value=self.symbol)
        symbol_combo = ttk.Combobox(
            control_frame,
            textvariable=self.symbol_var,
            values=self.symbol_names,
            width=10,
        )
        symbol_combo.pack(side=tk.LEFT, padx=5)

        # Select timeframe
        ttk.Label(control_frame, text="Timeframe:").pack(side=tk.LEFT, padx=5)
        self.timeframe_var = tk.StringVar(value="H1")
        timeframe_combo = ttk.Combobox(
            control_frame,
            textvariable=self.timeframe_var,
            values=list(self.timeframes.keys()),
            width=5,
        )
        timeframe_combo.pack(side=tk.LEFT, padx=5)

        # Number of historical bars
        ttk.Label(control_frame, text="History bars:").pack(side=tk.LEFT, padx=5)
        self.historic_var = tk.StringVar(value="20")
        historic_spin = ttk.Spinbox(
            control_frame, from_=20, to=100, textvariable=self.historic_var, width=5
        )
        historic_spin.pack(side=tk.LEFT, padx=5)

        # Number of forecast bars
        ttk.Label(control_frame, text="Forecast bars:").pack(side=tk.LEFT, padx=5)
        self.forecast_var = tk.StringVar(value="5")
        forecast_spin = ttk.Spinbox(
            control_frame, from_=5, to=20, textvariable=self.forecast_var, width=5
        )
        forecast_spin.pack(side=tk.LEFT, padx=5)

        # Update button
        ttk.Button(control_frame, text="Update", command=self.update_forecast).pack(
            side=tk.LEFT, padx=20
        )

        # Chart
        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def setup_model(self):
        # Initialize model with parameters from the main file
        input_size = 100
        hidden_size = 64
        output_size = 1
        self.trader = EnhancedPlasmaBrainTrader(input_size, hidden_size, output_size)
        
        # Try to load saved model
        model_path = Path("best_bio_model.pth")
        scaler_path = Path("best_scaler.gz")
        
        if model_path.exists():
            import torch
            try:
                self.trader.model.load_state_dict(torch.load(model_path))
                print(f"Loaded saved model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                
        if scaler_path.exists():
            import joblib
            try:
                self.trader.scaler = joblib.load(scaler_path)
                self.trader.is_scaler_fitted = True
                print(f"Loaded saved scaler from {scaler_path}")
            except Exception as e:
                print(f"Error loading scaler: {e}")

    def update_forecast(self):
        symbol = self.symbol_var.get()
        timeframe = self.timeframes[self.timeframe_var.get()]
        historic_bars = int(self.historic_var.get())
        forecast_bars = int(self.forecast_var.get())

        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        if rates is None or len(rates) == 0:
            print(f"Error: No data for {symbol}")
            return
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)

        # Last historic_bars bars
        last_prices = df["close"].values[-historic_bars:]
        last_data = df.iloc[-historic_bars:]

        # Forecast
        predictions = []
        current_data = last_data.copy()
        current_price = last_prices[-1]
        mf = MarketFeatures()

        for _ in range(forecast_bars):
            features = mf.calculate_features(current_data)
            pred = self.trader.predict(current_price, features, train=False)
            predictions.append(pred)
            # Update data for the next forecast
            current_price = pred
            # Rough estimate for next time point
            delta = (current_data.index[-1] - current_data.index[-2]) if len(current_data) > 1 else timedelta(hours=1)
            next_time = current_data.index[-1] + delta
            
            # Add a new row for the forecast
            new_row = current_data.iloc[-1].copy()
            new_row["close"] = pred
            current_data.loc[next_time] = new_row

        # Visualization
        self.ax.clear()
        x_hist = range(historic_bars)
        x_pred = range(historic_bars - 1, historic_bars + forecast_bars)

        self.ax.plot(x_hist, last_prices, label="Historical", color="blue", linewidth=2)
        self.ax.plot(
            x_pred,
            [last_prices[-1]] + predictions,
            label="Forecast",
            color="red",
            linestyle="--",
            linewidth=2,
        )

        # Add price values to the chart
        for i, price in enumerate(last_prices):
            self.ax.text(i, price, f"{price:.5f}", fontsize=8)
        for i, price in enumerate(predictions):
            self.ax.text(
                historic_bars + i, price, f"{price:.5f}", fontsize=8, color="red"
            )

        self.ax.set_title(f"{symbol} Forecast ({self.timeframe_var.get()})")
        self.ax.grid(True)
        self.ax.legend()

        self.canvas.draw()


if __name__ == "__main__":
    app = PlasmaBrainPredictor()
    app.mainloop()
