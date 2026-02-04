import numpy as np
from mnemonic import Mnemonic
import hashlib
import MetaTrader5 as mt5
import matplotlib
matplotlib.use('Agg')  # Set Agg backend
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime, timedelta

class PriceAnalysis:
   def __init__(self):
       self.mnemo = Mnemonic("english")
       # Initialize MT5 connection
       if not mt5.initialize():
           raise Exception("MetaTrader5 initialization failed")
   
   def get_price_data(self, symbol, period='1y', interval='1d'):
       # Convert period string to number of days
       if period == '1y':
           days = 365
       elif period == '6m':
           days = 180
       elif period == '1m':
           days = 30
       else:
           days = 365  # default to 1 year
           
       # Convert interval string to MT5 timeframe
       if interval == '1d':
           timeframe = mt5.TIMEFRAME_D1
       elif interval == '1h':
           timeframe = mt5.TIMEFRAME_H1
       elif interval == '4h':
           timeframe = mt5.TIMEFRAME_H4
       else:
           timeframe = mt5.TIMEFRAME_D1  # default to daily
           
       # Calculate start and end dates
       end_date = datetime.now()
       start_date = end_date - timedelta(days=days)
       
       # Get historical data from MT5
       rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
       
       if rates is None or len(rates) == 0:
           raise ValueError(f"No data available for {symbol}")
           
       # Extract close prices
       prices = np.array([rate[4] for rate in rates])  # index 4 is close price
       return prices
   
   def encode_price_movement(self, prices):
       # Normalize prices to range 0-255
       normalized = ((prices - np.min(prices)) / (np.max(prices) - np.min(prices)) * 255).astype(np.uint8)
       # Convert to bytes
       data = bytes(normalized)
       # Create entropy
       entropy = hashlib.sha256(data).digest()
       # Get mnemonic phrase
       mnemonic = self.mnemo.to_mnemonic(entropy)
       return mnemonic
   
   def analyze_words_frequency(self, prices):
       if len(prices) < 2:
           raise ValueError("Insufficient data for analysis.")
       
       price_diff = np.diff(prices)  # Price changes
       bullish_mnemonics = []
       bearish_mnemonics = []
       
       for i in range(len(price_diff)):
           window = prices[max(0, i-4):i+1]  # Sliding window (5 values)
           if len(window) < 5:
               continue
           mnemonic = self.encode_price_movement(window)
           if price_diff[i] > 0:
               bullish_mnemonics.extend(mnemonic.split())
           else:
               bearish_mnemonics.extend(mnemonic.split())
       
       # Count word frequencies
       bullish_freq = Counter(bullish_mnemonics)
       bearish_freq = Counter(bearish_mnemonics)
       return bullish_freq, bearish_freq
   
   def plot_frequencies(self, bullish_freq, bearish_freq, top_n=500):
       # Select top N words
       bullish_top = dict(bullish_freq.most_common(top_n))
       bearish_top = dict(bearish_freq.most_common(top_n))
       
       # Calculate figure size with fixed width of 750 pixels (at 100 DPI)
       width_inches = 7.5  # 750px / 100dpi = 7.5 inches
       height_inches = width_inches / 2  # maintain 2:1 aspect ratio
       
       # Create plots
       fig, ax = plt.subplots(1, 2, figsize=(width_inches, height_inches), dpi=100)
       
       # Plot bullish frequencies without x-axis labels
       ax[0].bar(range(len(bullish_top)), bullish_top.values(), color='green')
       ax[0].set_title(f"Top {top_n} Words for Bullish Movements")
       ax[0].set_ylabel("Frequency")
       ax[0].set_xticks([])  # Remove x-axis ticks
       
       # Plot bearish frequencies without x-axis labels
       ax[1].bar(range(len(bearish_top)), bearish_top.values(), color='red')
       ax[1].set_title(f"Top {top_n} Words for Bearish Movements")
       ax[1].set_ylabel("Frequency")
       ax[1].set_xticks([])  # Remove x-axis ticks
       
       plt.tight_layout()
       # Save to file
       plt.savefig('frequency_analysis.png', dpi=100, bbox_inches='tight')
       plt.close()
       
   def __del__(self):
       mt5.shutdown()

# Data analysis
analyzer = PriceAnalysis()
try:
   # Load data
   prices = analyzer.get_price_data('USDJPY', period='1y', interval='1d')
   
   # Analyze word frequencies
   bullish_freq, bearish_freq = analyzer.analyze_words_frequency(prices)
   
   # Output results
   print("Top 500 words for bullish movements:")
   print(bullish_freq.most_common(500))
   print("\nTop 500 words for bearish movements:")
   print(bearish_freq.most_common(500))
   
   # Create plots and save to file
   analyzer.plot_frequencies(bullish_freq, bearish_freq, top_n=500)
except ValueError as e:
   print(f"Error: {e}")
except Exception as e:
   print(f"Unexpected error: {e}")
finally:
   mt5.shutdown()
