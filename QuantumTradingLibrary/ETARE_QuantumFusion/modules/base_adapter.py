from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    """
    Base class for all system adapters in the Strike Boss Fusion Engine.
    Ensures a standardized output for the Fusion Engine.
    """
    
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_signal(self, symbol, timeframe, lookback):
        """
        Processes market data and returns a standardized signal.
        
        Returns:
            dict: {
                "name": self.name,
                "signal": float (-1.0 to 1.0),
                "confidence": float (0.0 to 1.0),
                "metadata": dict (system-specific data)
            }
        """
        pass
