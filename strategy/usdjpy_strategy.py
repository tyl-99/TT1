import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import time

class USDJPYSTRATEGY:
    def __init__(self, target_pair="USD/JPY"):
        self.target_pair = target_pair
        self.pip_size = 0.01  # USD/JPY pip size is 0.01 (different from EUR/USD)
        
        # BEST FROM WIN RATE TUNER: 43.9% win rate, 82 trades
        self.fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]  # Best from win rate tuner
        self.atr_period = 8  # Best from win rate tuner
        self.atr_threshold = 0.2  # Best from win rate tuner
        self.swing_pips = 10  # Best from win rate tuner
        self.risk_reward_ratio = 3.0  # Updated to 1:3 RR
        self.tolerance = 0.0008  # Best from win rate tuner
        
        # Time filters
        self.trading_hours = {'start': time(0, 0), 'end': time(23, 59)}
        self.avoid_weekends = False
        
        # Risk management
        self.min_rr_ratio = 1.5
        self.max_rr_ratio = 15.0
        
        # Risk per trade for lot size calculation
        self.risk_per_trade = 50.0  # $50 USD risk per trade
        
        self.filter_stats = {'total_candles': 0, 'signals_generated': 0}

    def _check_time_filter(self, timestamp) -> bool:
        return True

    def _find_swing_points(self, df, window=3):
        highs, lows = [], []
        for i in range(window, len(df) - window):
            if all(df['high'].iloc[i] >= df['high'].iloc[i-window:i]) and all(df['high'].iloc[i] >= df['high'].iloc[i+1:i+window+1]):
                highs.append(i)
            if all(df['low'].iloc[i] <= df['low'].iloc[i-window:i]) and all(df['low'].iloc[i] <= df['low'].iloc[i+1:i+window+1]):
                lows.append(i)
        return highs, lows

    def _calculate_fibonacci_levels(self, swing_high, swing_low, direction):
        swing_range = swing_high - swing_low
        fib_levels = {}
        if direction == 'BUY':
            for level in self.fib_levels:
                fib_levels[f'ret_{level}'] = swing_low + (swing_range * level)
        else:
            for level in self.fib_levels:
                fib_levels[f'ret_{level}'] = swing_high - (swing_range * level)
        return fib_levels

    def _calculate_indicators(self, df):
        df = df.copy()
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.atr_period).mean()
        df['atr_pips'] = df['atr'] / self.pip_size
        return df

    def _identify_fibonacci_setup(self, df, current_price):
        lookback = 20
        recent_data = df.tail(lookback)
        highs, lows = self._find_swing_points(recent_data, window=2)
        
        if len(highs) < 1 or len(lows) < 1:
            return None
            
        recent_high_idx = max(highs)
        recent_low_idx = max(lows)
        
        if recent_high_idx > recent_low_idx:
            swing_high = recent_data['high'].iloc[recent_high_idx]
            swing_low = recent_data['low'].iloc[recent_low_idx]
            direction = 'BUY'
        else:
            swing_high = recent_data['high'].iloc[recent_high_idx]
            swing_low = recent_data['low'].iloc[recent_low_idx]
            direction = 'SELL'
            
        swing_range_pips = abs(swing_high - swing_low) / self.pip_size
        if swing_range_pips < self.swing_pips:
            return None
            
        fib_levels = self._calculate_fibonacci_levels(swing_high, swing_low, direction)
        tolerance = self.tolerance
        
        nearest_level = None
        nearest_distance = float('inf')
        
        for level_name, level_price in fib_levels.items():
            distance = abs(current_price - level_price)
            if distance < tolerance and distance < nearest_distance:
                nearest_level = level_name
                nearest_distance = distance
                    
        if not nearest_level:
            return None
            
        entry_price = fib_levels[nearest_level]
        
        if direction == 'BUY':
            stop_loss = swing_low - tolerance
            take_profit = entry_price + (self.risk_reward_ratio * (entry_price - stop_loss))
        else:
            stop_loss = swing_high + tolerance
            take_profit = entry_price - (self.risk_reward_ratio * (stop_loss - entry_price))
            
        sl_distance = abs(entry_price - stop_loss) / self.pip_size
        tp_distance = abs(take_profit - entry_price) / self.pip_size
        rr_ratio = tp_distance / sl_distance
        
        if rr_ratio < self.min_rr_ratio or rr_ratio > self.max_rr_ratio:
            return None
            
        return {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'rr_ratio': rr_ratio,
            'sl_pips': sl_distance,
            'tp_pips': tp_distance
        }

    def calculate_lot_size(self, entry_price: float, stop_loss: float, direction: str) -> float:
        """Calculate lot size for $50 risk per trade based on stop loss distance"""
        # Pip values for different pairs
        pip_values = {
            'USD/JPY': 10.0,  # $10 per pip for 1 lot
            'EUR/JPY': 7.0,   # $7 per pip for 1 lot  
            'GBP/JPY': 7.0,   # $7 per pip for 1 lot
            'EUR/USD': 10.0,  # $10 per pip for 1 lot
            'GBP/USD': 10.0,  # $10 per pip for 1 lot
            'default': 10.0
        }
        
        pip_value = pip_values.get(self.target_pair, pip_values['default'])
        
        # Calculate pip size
        pip_size = 0.01 if "JPY" in self.target_pair else 0.0001
        
        # Calculate risk in pips
        if direction == "BUY":
            risk_pips = abs(entry_price - stop_loss) / pip_size
        else:  # SELL
            risk_pips = abs(stop_loss - entry_price) / pip_size
        
        # Calculate lot size: Risk / (Risk Pips * Pip Value)
        lot_size = self.risk_per_trade / (risk_pips * pip_value)
        
        # Round to 2 decimal places (standard lot size format)
        lot_size = round(lot_size, 2)
        
        # Ensure minimum lot size of 0.01
        lot_size = max(lot_size, 0.01)
        
        return lot_size

    def analyze_trade_signal(self, df: pd.DataFrame, pair: str) -> Dict[str, Any]:
        if len(df) < 30:
            return {'action': 'HOLD', 'reason': 'Insufficient data'}
            
        df_with_indicators = self._calculate_indicators(df)
        current = df_with_indicators.iloc[-1]
        self.filter_stats['total_candles'] += 1
        
        # Volatility filter
        if current['atr_pips'] < self.atr_threshold:
            return {'action': 'HOLD', 'reason': 'Low volatility'}
            
        fib_setup = self._identify_fibonacci_setup(df_with_indicators, current['close'])
        if not fib_setup:
            return {'action': 'HOLD', 'reason': 'No valid Fibonacci setup'}
            
        self.filter_stats['signals_generated'] += 1
        
        # Calculate lot size based on stop loss
        lot_size = self.calculate_lot_size(
            fib_setup['entry_price'],
            fib_setup['stop_loss'],
            fib_setup['direction']
        )
        
        return {
            'decision': fib_setup['direction'],
            'direction': fib_setup['direction'],
            'action': fib_setup['direction'],
            'entry_price': fib_setup['entry_price'],
            'stop_loss': fib_setup['stop_loss'],
            'take_profit': fib_setup['take_profit'],
            'lot_size': lot_size,  # Strategy calculates lot size
            'sl_pips': fib_setup['sl_pips'],
            'tp_pips': fib_setup['tp_pips'],
            'risk_reward': f"1:{fib_setup['rr_ratio']:.1f}",
            'reason': f'Fibonacci retracement',
            'confidence': 'high',
            'setup_type': 'fibonacci_retracement'
        }

    def print_filter_stats(self):
        total = self.filter_stats['total_candles']
        if total == 0:
            return
        print(f"\n📊 USD/JPY FIBONACCI STRATEGY - FILTER STATS:")
        print(f"Total candles analyzed: {total}")
        print(f"Signals generated: {self.filter_stats['signals_generated']} ({self.filter_stats['signals_generated']/total*100:.1f}%)") 