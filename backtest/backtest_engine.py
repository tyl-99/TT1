import pandas as pd
import numpy as np
import datetime
import os
import logging
import json
import pickle
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trade_models import Trade
import warnings

# 🎯 DYNAMIC STRATEGY IMPORTS - ONE LINE PER IMPORT
from strategy.eurusd_strategy import EURUSDSupplyDemandStrategy
from strategy.gbpusd_strategy import GBPUSDDemandStrategy
from strategy.usdjpy_strategy import USDJPYStrategy
from strategy.eurgbp_strategy import EURGBPSupplyDemandStrategy
from strategy.gbpjpy_strategy import GBPJPYStrategy
from strategy.eurjpy_strategy import EURJPYSupplyDemandStrategy
# from strategy.audusd_strategy import AUDUSDStrategy
# from strategy.usdcad_strategy import USDCADStrategy
# from strategy.nzdusd_strategy import NZDUSDStrategy
# from strategy.usdchf_strategy import USDCHFStrategy

# Ignore FutureWarnings from pandas
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, target_pair="EUR/USD", initial_balance=1000, strategy=None):
        # 🎯 DYNAMIC PAIR CONFIGURATION
        self.target_pair = target_pair
        self.pair_code = target_pair.replace("/", "_").replace("-", "_")
        
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.lowest_balance = initial_balance
        
        self.trades = []
        self.open_trades = []
        
        # 🎯 DYNAMIC STRATEGY INITIALIZATION
        self.strategy = strategy if strategy is not None else self._initialize_strategy()
        
        # 🕯️ COMPREHENSIVE CANDLE DATA STORAGE
        self.trade_candle_data = {}  # Will store all candle data per trade
        
        # EXCEL & PICKLE LOGGING SETUP
        self.excel_log_path = None
        self.candle_data_path = None
        self.setup_logging()
        
        logger.info(f"💰 Backtest Engine: ${initial_balance} initial balance")
        logger.info(f"🎯 TARGET PAIR: {self.target_pair}")
        logger.info(f"🧠 STRATEGY: {self.strategy.__class__.__name__ if self.strategy else 'No Strategy Available'}")
        logger.info(f"🕯️ COMPREHENSIVE CANDLE DATA: Entry to Exit tracking")
        logger.info("📊 NO TRADE LIMITS - Complete dataset analysis")
    
    def _initialize_strategy(self):
        """
        🎯 DYNAMIC STRATEGY INITIALIZATION BASED ON CURRENCY PAIR
        Initialize the appropriate strategy class for the target pair
        """
        # Create strategy mapping dictionary
        strategy_mapping = {
            "EUR/USD": EURUSDSupplyDemandStrategy,
            "GBP/USD": GBPUSDDemandStrategy,
            "USD/JPY": USDJPYStrategy,
            "EUR/GBP": EURGBPSupplyDemandStrategy,
            "GBP/JPY": GBPJPYStrategy,
            "EUR/JPY": EURJPYSupplyDemandStrategy,
            # "AUD/USD": AUDUSDStrategy,
            # "USD/CAD": USDCADStrategy,
            # "NZD/USD": NZDUSDStrategy,
            # "USD/CHF": USDCHFStrategy,
        }
        
        # Get strategy class for target pair
        strategy_class = strategy_mapping.get(self.target_pair)
        
        if strategy_class is None:
            logger.warning(f"⚠️ No strategy available for {self.target_pair}")
            logger.warning(f"📝 Create strategy/{''.join(self.target_pair.lower().split('/'))}_strategy.py")
            logger.warning(f"📝 Class name should be: {''.join(self.target_pair.split('/'))}Strategy")
            logger.warning(f"📝 Then uncomment: # from strategy.{''.join(self.target_pair.lower().split('/'))}_strategy import {''.join(self.target_pair.split('/'))}Strategy")
            return None
        
        try:
            # Initialize strategy instance
            strategy_instance = strategy_class()
            logger.info(f"✅ {strategy_class.__name__} initialized for {self.target_pair}")
            return strategy_instance
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize strategy for {self.target_pair}: {e}")
            return None
    
    def get_strategy_file_info(self):
        """
        📝 GET STRATEGY FILE CREATION INFO FOR MISSING PAIRS
        Provides guidance on creating new strategy files
        """
        pair_clean = ''.join(self.target_pair.lower().split('/'))
        class_name = ''.join(self.target_pair.split('/'))
        
        return {
            'file_path': f"strategy/{pair_clean}_strategy.py",
            'class_name': f"{class_name}Strategy",
            'import_statement': f"from strategy.{pair_clean}_strategy import {class_name}Strategy",
            'uncomment_line': f"# from strategy.{pair_clean}_strategy import {class_name}Strategy",
            'example_template': f'''
# strategy/{pair_clean}_strategy.py
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class {class_name}Strategy:
    def __init__(self):
        self.check_count = 0
        self.filtered_count = 0
        self.passed_count = 0
    
    def analyze_trade_signal(self, df: pd.DataFrame, pair: str):
        """
        {self.target_pair} specific trading strategy
        Implement your {self.target_pair} logic here
        """
        self.check_count += 1
        
        try:
            # Add your {self.target_pair} strategy logic here
            # Return format should be:
            # {{
            #     "decision": "BUY" | "SELL" | "NO TRADE",
            #     "entry_price": float,
            #     "stop_loss": float,
            #     "take_profit": float,
            #     "volume": float,
            #     "reason": str
            # }}
            
            # Placeholder - replace with actual strategy
            return {{"decision": "NO TRADE", "reason": "Strategy not implemented"}}
            
        except Exception as e:
            logger.error(f"❌ {class_name}Strategy error: {{e}}")
            return {{"decision": "NO TRADE", "reason": f"Error: {{str(e)}}"}}
    
    def get_statistics(self):
        return {{
            'total_checks': self.check_count,
            'filtered_out': self.filtered_count,
            'passed_to_strategy': self.passed_count
        }}
    
    def print_final_stats(self):
        stats = self.get_statistics()
        print(f"\\n🎯 {class_name.upper()} STRATEGY STATS:")
        print(f"   Total Checks: {{stats['total_checks']:,}}")
        print(f"   Signals Generated: {{stats['passed_to_strategy']:,}}")
'''
        }
    
    def setup_logging(self):
        """Setup logging for dynamic pair with candle data storage"""
        os.makedirs('backtest_logs', exist_ok=True)
        os.makedirs('candle_data', exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Trade logging file
        self.excel_log_path = f"backtest_logs/{self.pair_code}_FULL_trades_{timestamp}.xlsx"
        
        # 🕯️ CANDLE DATA STORAGE (Pickle for efficiency)
        self.candle_data_path = f"candle_data/{self.pair_code}_candle_data_{timestamp}.pkl"
        
        logger.info(f"📊 {self.target_pair} Trade logging: {self.excel_log_path}")
        logger.info(f"🕯️ {self.target_pair} Candle data: {self.candle_data_path}")
    
    def get_pip_size(self, pair):
        """Get pip size for any currency pair"""
        if "JPY" in pair:
            return 0.01
        else:
            return 0.0001
    
    def get_trading_costs(self, pair):
        """Get trading costs for different pairs"""
        costs = {
            "EUR/USD": {"spread": 1.2, "slippage": 0.8},
            "GBP/USD": {"spread": 1.5, "slippage": 1.0},
            "USD/JPY": {"spread": 1.0, "slippage": 0.8},
            "AUD/USD": {"spread": 1.8, "slippage": 1.2},
            "USD/CAD": {"spread": 1.5, "slippage": 1.0},
            "EUR/GBP": {"spread": 2.0, "slippage": 1.5},
            "GBP/JPY": {"spread": 2.5, "slippage": 2.0},
            "EUR/JPY": {"spread": 2.0, "slippage": 1.5},
            "NZD/USD": {"spread": 2.0, "slippage": 1.5},
            "USD/CHF": {"spread": 1.8, "slippage": 1.2},
        }
        
        # Default for any other pair
        default_costs = {"spread": 2.0, "slippage": 1.5}
        return costs.get(pair, default_costs)
    
    def collect_comprehensive_candle_data(self, trade, full_df, entry_index, exit_index):
        """
        Collect comprehensive candle data from entry-50 candles until exit
        🕯️ COMPLETE PRICE ACTION HISTORY
        """
        try:
            trade_id = len(self.trades)
            
            # Calculate range: 50 candles before entry until exit
            start_index = max(0, entry_index - 50)
            end_index = min(len(full_df) - 1, exit_index)
            
            # Extract all candles in range
            candle_range = full_df.iloc[start_index:end_index + 1].copy()
            candle_range.reset_index(drop=True, inplace=True)
            
            # Add relative positioning
            entry_relative_index = entry_index - start_index
            exit_relative_index = exit_index - start_index
            
            # Mark special candles
            candle_range['candle_type'] = 'normal'
            candle_range.loc[:entry_relative_index - 1, 'candle_type'] = 'pre_entry'
            candle_range.loc[entry_relative_index, 'candle_type'] = 'entry'
            candle_range.loc[entry_relative_index + 1:exit_relative_index - 1, 'candle_type'] = 'in_trade'
            candle_range.loc[exit_relative_index, 'candle_type'] = 'exit'
            
            # Calculate price movements relative to entry
            entry_price = trade.entry_price
            pip_size = self.get_pip_size(trade.pair)
            
            candle_range['pips_from_entry'] = ((candle_range['close'] - entry_price) / pip_size).round(1)
            candle_range['high_pips_from_entry'] = ((candle_range['high'] - entry_price) / pip_size).round(1)
            candle_range['low_pips_from_entry'] = ((candle_range['low'] - entry_price) / pip_size).round(1)
            
            # Mark SL/TP levels hit
            if trade.direction == "BUY":
                candle_range['sl_hit'] = candle_range['low'] <= trade.stop_loss
                candle_range['tp_hit'] = candle_range['high'] >= trade.take_profit
            else:
                candle_range['sl_hit'] = candle_range['high'] >= trade.stop_loss
                candle_range['tp_hit'] = candle_range['low'] <= trade.take_profit
            
            # Calculate candle patterns
            candle_range['body_size'] = abs(candle_range['close'] - candle_range['open'])
            candle_range['upper_wick'] = candle_range['high'] - candle_range[['open', 'close']].max(axis=1)
            candle_range['lower_wick'] = candle_range[['open', 'close']].min(axis=1) - candle_range['low']
            candle_range['total_range'] = candle_range['high'] - candle_range['low']
            candle_range['body_pct'] = (candle_range['body_size'] / candle_range['total_range'] * 100).round(1)
            
            # Bullish/Bearish classification
            candle_range['bullish'] = candle_range['close'] > candle_range['open']
            candle_range['bearish'] = candle_range['close'] < candle_range['open']
            candle_range['doji'] = abs(candle_range['close'] - candle_range['open']) < (candle_range['total_range'] * 0.1)
            
            # Store comprehensive trade candle data
            trade_candle_info = {
                'trade_id': trade_id,
                'pair': trade.pair,
                'direction': trade.direction,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'exit_reason': trade.exit_reason,
                'pips_gained': trade.pips_gained,
                'usd_pnl': trade.usd_pnl,
                'duration_hours': trade.duration_hours,
                'total_candles': len(candle_range),
                'pre_entry_candles': entry_relative_index,
                'in_trade_candles': exit_relative_index - entry_relative_index,
                'entry_index_in_data': entry_index,
                'exit_index_in_data': exit_index,
                'candle_data': candle_range.to_dict('records')  # All candle data as list of dicts
            }
            
            # Store in trade_candle_data dictionary
            self.trade_candle_data[trade_id] = trade_candle_info
            
            logger.info(f"🕯️ Collected {len(candle_range)} candles for Trade #{trade_id} ({trade.direction} {trade.pair})")
            
            return trade_candle_info
            
        except Exception as e:
            logger.error(f"❌ Error collecting candle data for trade: {e}")
            return None
    
    def save_candle_data_to_files(self):
        """
        Save candle data to multiple formats:
        1. Pickle for Python processing
        2. Excel for manual analysis
        3. JSON for other applications
        """
        try:
            if not self.trade_candle_data:
                logger.warning("⚠️ No candle data to save")
                return
            
            # 1. Save as Pickle (most efficient for Python)
            with open(self.candle_data_path, 'wb') as f:
                pickle.dump(self.trade_candle_data, f)
            logger.info(f"🕯️ Candle data saved to pickle: {self.candle_data_path}")
            
            # 2. Save as Excel for manual analysis
            excel_candle_path = self.candle_data_path.replace('.pkl', '.xlsx')
            
            with pd.ExcelWriter(excel_candle_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for trade_id, trade_info in self.trade_candle_data.items():
                    summary_data.append({
                        'trade_id': trade_id,
                        'pair': trade_info['pair'],
                        'direction': trade_info['direction'],
                        'entry_time': trade_info['entry_time'],
                        'exit_time': trade_info['exit_time'],
                        'pips_gained': trade_info['pips_gained'],
                        'usd_pnl': trade_info['usd_pnl'],
                        'exit_reason': trade_info['exit_reason'],
                        'total_candles': trade_info['total_candles'],
                        'duration_hours': trade_info['duration_hours']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Trade_Summary', index=False)
                
                # Individual trade sheets (first 20 trades to avoid Excel limits)
                for trade_id, trade_info in list(self.trade_candle_data.items())[:20]:
                    candle_df = pd.DataFrame(trade_info['candle_data'])
                    sheet_name = f"Trade_{trade_id}_{trade_info['direction']}"[:31]  # Excel sheet name limit
                    candle_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"🕯️ Candle data saved to Excel: {excel_candle_path}")
            
            # 3. Save as JSON for other applications
            json_candle_path = self.candle_data_path.replace('.pkl', '.json')
            
            # Convert datetime objects to strings for JSON serialization
            json_data = {}
            for trade_id, trade_info in self.trade_candle_data.items():
                json_trade_info = trade_info.copy()
                json_trade_info['entry_time'] = trade_info['entry_time'].isoformat() if trade_info['entry_time'] else None
                json_trade_info['exit_time'] = trade_info['exit_time'].isoformat() if trade_info['exit_time'] else None
                
                # Convert candle timestamps
                for candle in json_trade_info['candle_data']:
                    if 'timestamp' in candle and pd.notna(candle['timestamp']):
                        candle['timestamp'] = pd.to_datetime(candle['timestamp']).isoformat()
                
                json_data[str(trade_id)] = json_trade_info
            
            with open(json_candle_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            logger.info(f"🕯️ Candle data saved to JSON: {json_candle_path}")
            
            # 4. Create a quick access method file
            method_file_path = self.candle_data_path.replace('.pkl', '_access_methods.py')
            method_code = f'''
# {self.target_pair} Candle Data Access Methods
# Generated on {datetime.datetime.now()}

import pickle
import pandas as pd
import json

def load_candle_data():
    """Load all candle data from pickle file"""
    with open('{self.candle_data_path}', 'rb') as f:
        return pickle.load(f)

def get_trade_candles(trade_id):
    """Get candle data for specific trade"""
    data = load_candle_data()
    return data.get(trade_id, None)

def get_trade_as_dataframe(trade_id):
    """Get trade candle data as pandas DataFrame"""
    trade_data = get_trade_candles(trade_id)
    if trade_data:
        return pd.DataFrame(trade_data['candle_data'])
    return None

def list_all_trades():
    """List all available trades"""
    data = load_candle_data()
    trades = []
    for trade_id, trade_info in data.items():
        trades.append({{
            'trade_id': trade_id,
            'pair': trade_info['pair'],
            'direction': trade_info['direction'],
            'pips_gained': trade_info['pips_gained'],
            'exit_reason': trade_info['exit_reason']
        }})
    return trades

def get_winning_trades():
    """Get only winning trades"""
    data = load_candle_data()
    return {{k: v for k, v in data.items() if v['usd_pnl'] > 0}}

def get_losing_trades():
    """Get only losing trades"""
    data = load_candle_data()
    return {{k: v for k, v in data.items() if v['usd_pnl'] <= 0}}

# Example usage:
# data = load_candle_data()
# trade_1_df = get_trade_as_dataframe(1)
# all_trades = list_all_trades()
'''
            
            with open(method_file_path, 'w') as f:
                f.write(method_code)
            
            logger.info(f"🕯️ Access methods saved: {method_file_path}")
            
            # Print summary
            print(f"\n🕯️ CANDLE DATA SAVED:")
            print(f"   📊 Total trades with candle data: {len(self.trade_candle_data):,}")
            print(f"   🐍 Pickle file (Python): {self.candle_data_path}")
            print(f"   📊 Excel file (Analysis): {excel_candle_path}")
            print(f"   🌐 JSON file (Universal): {json_candle_path}")
            print(f"   🔧 Access methods: {method_file_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving candle data: {e}")
    
    def load_excel_data(self, file_path='backtest_data/forex_data1.xlsx'):
        """Load data for target pair from Excel file - DYNAMIC"""
        if not os.path.exists(file_path):
            logger.error(f"❌ Data file not found: {file_path}")
            return {}
        
        logger.info(f"📊 Loading {self.target_pair} data from {file_path}")
        data = {}
        
        try:
            with pd.ExcelFile(file_path) as excel_file:
                sheet_names = excel_file.sheet_names
                target_sheet = None
                
                # Try different possible sheet names for target pair
                pair_variations = [
                    self.target_pair,
                    self.target_pair.replace("/", "_"),
                    self.target_pair.replace("/", ""),
                    self.target_pair.replace("-", "_"),
                    self.target_pair.replace("-", ""),
                    self.target_pair.lower(),
                    self.target_pair.lower().replace("/", "_"),
                    self.target_pair.lower().replace("/", ""),
                    self.target_pair.upper(),
                    self.target_pair.upper().replace("/", "_"),
                    self.target_pair.upper().replace("/", "")
                ]
                
                for variation in pair_variations:
                    if variation in sheet_names:
                        target_sheet = variation
                        break
                
                if not target_sheet:
                    logger.error(f"❌ {self.target_pair} sheet not found. Available sheets: {sheet_names}")
                    return {}
                
                # Read target pair sheet
                df = pd.read_excel(excel_file, sheet_name=target_sheet)
                
                # Ensure timestamp column is datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Sort by timestamp (ascending for backtest)
                df.sort_values('timestamp', inplace=True, ascending=True)
                df.reset_index(drop=True, inplace=True)
                
                data[self.target_pair] = df
                logger.info(f"✅ Loaded {self.target_pair}: {len(df):,} bars")
                logger.info(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading {self.target_pair} data: {e}")
            return {}
    
    def log_trade_to_excel(self, trade):
        """Save trade result to Excel with basic info"""
        if not self.excel_log_path:
            return
        
        try:
            pip_size = self.get_pip_size(trade.pair)
            
            # Basic trade data
            trade_data = {
                'trade_id': len(self.trades),
                'entry_time': trade.entry_time.strftime('%Y-%m-%d %H:%M:%S') if trade.entry_time else '',
                'exit_time': trade.exit_time.strftime('%Y-%m-%d %H:%M:%S') if trade.exit_time else '',
                'pair': trade.pair,
                'direction': trade.direction,
                'entry_price': round(trade.entry_price, 5),
                'exit_price': round(trade.exit_price, 5) if trade.exit_price else 0,
                'stop_loss': round(trade.stop_loss, 5),
                'take_profit': round(trade.take_profit, 5),
                'volume': trade.volume,
                'pips_gained': round(trade.pips_gained, 1),
                'usd_pnl': round(trade.usd_pnl, 2),
                'exit_reason': trade.exit_reason or '',
                'duration_hours': round(trade.duration_hours, 1),
                'balance_after': round(self.current_balance, 2),
                'win_loss': "WIN" if trade.usd_pnl > 0 else "LOSS",
                'candle_data_available': 'YES'  # All trades have candle data
            }
            
            # Read existing data or create new
            try:
                existing_df = pd.read_excel(self.excel_log_path, sheet_name='Trades')
            except:
                existing_df = pd.DataFrame(columns=list(trade_data.keys()))
            
            # Append new trade
            new_trade_df = pd.DataFrame([trade_data])
            updated_df = pd.concat([existing_df, new_trade_df], ignore_index=True)
            
            # Save back to Excel
            updated_df.to_excel(self.excel_log_path, index=False, sheet_name='Trades')
            
            win_loss = trade_data['win_loss']
            trade_count = len(self.trades)
            logger.info(f"📊 Trade #{trade_count}: {win_loss} ${trade.usd_pnl:+.2f} | Balance: ${self.current_balance:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log trade to Excel: {e}")
    
    def open_trade(self, signal, timestamp, pair, market_price, current_index, full_df):
        """Open a new trade and store the data index for candle collection"""
        if signal.get("decision", "NO TRADE").upper() == "NO TRADE":
            return
        
        # Validate signal
        required = ["entry_price", "stop_loss", "take_profit", "volume"]
        if not all(field in signal for field in required):
            logger.warning(f"⚠️ Invalid signal for {pair}: missing required fields")
            return
        
        if signal["volume"] <= 0:
            logger.warning(f"⚠️ Invalid volume for {pair}: {signal['volume']}")
            return
        
        # Apply trading costs
        costs = self.get_trading_costs(pair)
        pip_size = self.get_pip_size(pair)
        total_cost = (costs["spread"] + costs["slippage"]) * pip_size
        
        if signal["decision"].upper() == "BUY":
            actual_entry = market_price + total_cost
        else:
            actual_entry = market_price - total_cost
        
        # Create trade
        trade = Trade(
            timestamp, pair, signal["decision"].upper(), actual_entry,
            signal["stop_loss"], signal["take_profit"], 
            signal["volume"], signal.get("reason", "Strategy signal")
        )
        
        # 🕯️ STORE ENTRY INDEX FOR CANDLE DATA COLLECTION
        trade.entry_index = current_index
        trade.full_df_reference = full_df  # Store reference to full dataset
        
        self.open_trades.append(trade)
        logger.info(f"📈 OPENED {trade.direction} {pair} @ {trade.entry_price:.5f}")
    
    def check_trade_exits(self, timestamp, bar_data, pair, current_index):
        """Check if any open trades should be closed and collect candle data"""
        trades_to_close = []
        
        for trade in self.open_trades:
            if trade.pair != pair:
                continue
            
            high = bar_data['high']
            low = bar_data['low']
            open_price = bar_data['open']
            
            exit_price = None
            exit_reason = None
            
            if trade.direction == "BUY":
                sl_hit = low <= trade.stop_loss
                tp_hit = high >= trade.take_profit
                
                if sl_hit and tp_hit:
                    # Both hit - determine order by distance to open
                    distance_to_sl = abs(open_price - trade.stop_loss)
                    distance_to_tp = abs(open_price - trade.take_profit)
                    
                    if distance_to_sl <= distance_to_tp:
                        exit_price = trade.stop_loss
                        exit_reason = "Stop Loss"
                    else:
                        exit_price = trade.take_profit
                        exit_reason = "Take Profit"
                elif sl_hit:
                    exit_price = trade.stop_loss
                    exit_reason = "Stop Loss"
                elif tp_hit:
                    exit_price = trade.take_profit
                    exit_reason = "Take Profit"
                
            else:  # SELL
                sl_hit = high >= trade.stop_loss
                tp_hit = low <= trade.take_profit
                
                if sl_hit and tp_hit:
                    distance_to_sl = abs(open_price - trade.stop_loss)
                    distance_to_tp = abs(open_price - trade.take_profit)
                    
                    if distance_to_sl <= distance_to_tp:
                        exit_price = trade.stop_loss
                        exit_reason = "Stop Loss"
                    else:
                        exit_price = trade.take_profit
                        exit_reason = "Take Profit"
                elif sl_hit:
                    exit_price = trade.stop_loss
                    exit_reason = "Stop Loss"
                elif tp_hit:
                    exit_price = trade.take_profit
                    exit_reason = "Take Profit"
        
            if exit_price:
                # 🕯️ COLLECT COMPREHENSIVE CANDLE DATA
                self.close_trade_with_candle_data(trade, timestamp, exit_price, exit_reason, current_index)
                trades_to_close.append(trade)
        
        # Remove closed trades
        for trade in trades_to_close:
            if trade in self.open_trades:
                self.open_trades.remove(trade)
    
    def close_trade_with_candle_data(self, trade, exit_time, exit_price, exit_reason, exit_index):
        """Close trade and collect comprehensive candle data"""
        pnl = trade.close_trade(exit_time, exit_price, exit_reason)
        
        # Update balance and tracking
        self.current_balance += pnl
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        if self.current_balance < self.lowest_balance:
            self.lowest_balance = self.current_balance
        
        # 🕯️ COLLECT COMPREHENSIVE CANDLE DATA
        self.collect_comprehensive_candle_data(
            trade, 
            trade.full_df_reference, 
            trade.entry_index, 
            exit_index
        )
        
        # Add to completed trades
        self.trades.append(trade)
        
        # Log to Excel
        self.log_trade_to_excel(trade)
        
        emoji = "✅ WIN" if pnl > 0 else "❌ LOSS"
        logger.info(f"{emoji} CLOSED {trade.direction} {trade.pair} | {exit_reason} | {trade.pips_gained:+.1f} pips | ${pnl:+.2f}")
    
    def run_backtest(self):
        """Run the complete backtest on target pair data with candle collection"""
        logger.info(f"🚀 Starting COMPLETE {self.target_pair} backtest with candle data collection...")
        
        # 🎯 CHECK IF STRATEGY IS AVAILABLE
        if self.strategy is None:
            strategy_info = self.get_strategy_file_info()
            logger.error(f"❌ No strategy available for {self.target_pair}")
            logger.error(f"📝 Create file: {strategy_info['file_path']}")
            logger.error(f"📝 Class name: {strategy_info['class_name']}")
            logger.error(f"📝 Then uncomment: {strategy_info['uncomment_line']}")
            print(f"\n🚫 BACKTEST ABORTED - NO STRATEGY AVAILABLE")
            print(f"📝 Create strategy file for {self.target_pair}:")
            print(f"   File: {strategy_info['file_path']}")
            print(f"   Class: {strategy_info['class_name']}")
            print(f"   Then uncomment: {strategy_info['uncomment_line']}")
            return
        
        # Load target pair data
        data = self.load_excel_data()
        if not data or self.target_pair not in data:
            logger.error(f"❌ No {self.target_pair} data loaded. Aborting backtest.")
            return
        
        pair_df = data[self.target_pair]
        if pair_df.empty:
            logger.error(f"❌ {self.target_pair} data is empty. Aborting backtest.")
            return
        
        logger.info(f"📊 {self.target_pair} data loaded: {len(pair_df):,} bars")
        
        # Backtest parameters
        signal_check_interval = 1  # Check every bar
        min_bars_for_strategy = 250
        
        if len(pair_df) < min_bars_for_strategy:
            logger.error(f"❌ Insufficient data. Need {min_bars_for_strategy} bars, got {len(pair_df)}")
            return
        
        logger.info(f"📊 Processing {len(pair_df) - min_bars_for_strategy + 1:,} bars for signals...")
        
        # Main backtest loop
        for i in range(min_bars_for_strategy - 1, len(pair_df)):
            current_bar = pair_df.iloc[i]
            timestamp = current_bar['timestamp']
            
            # Check exits first (pass current index for candle data)
            self.check_trade_exits(timestamp, current_bar, self.target_pair, i)
            
            # Check for new signals
            bars_since_start = i - (min_bars_for_strategy - 1)
            if bars_since_start % signal_check_interval == 0:
                # Only one trade at a time
                has_open_trade = any(t.pair == self.target_pair for t in self.open_trades)
                
                if not has_open_trade:
                    try:
                        # Prepare strategy data
                        start_idx = max(0, i - min_bars_for_strategy + 1)
                        strategy_data = pair_df.iloc[start_idx:i + 1].copy()
                        strategy_data.reset_index(drop=True, inplace=True)
                        
                        if len(strategy_data) >= min_bars_for_strategy:
                            # 🎯 GET STRATEGY SIGNAL FROM PAIR-SPECIFIC STRATEGY
                            signal = self.strategy.analyze_trade_signal(strategy_data, self.target_pair)
                            
                            # Open trade if valid (pass current index and full df)
                            if signal.get("decision", "NO TRADE").upper() != "NO TRADE":
                                self.open_trade(signal, timestamp, self.target_pair, current_bar['close'], i, pair_df)
                                
                    except Exception as e:
                        logger.error(f"❌ Strategy error at {timestamp}: {e}")
            
            # Progress every 5000 bars
            if (i - min_bars_for_strategy + 1) % 5000 == 0:
                progress = ((i - min_bars_for_strategy + 1) / (len(pair_df) - min_bars_for_strategy)) * 100
                logger.info(f"📊 Progress: {progress:.1f}% | Balance: ${self.current_balance:.2f} | Trades: {len(self.trades):,}")
        
        # Close remaining trades
        if self.open_trades:
            logger.info("🔒 Closing remaining trades...")
            final_bar = pair_df.iloc[-1]
            final_index = len(pair_df) - 1
            for trade in self.open_trades[:]:
                self.close_trade_with_candle_data(trade, final_bar['timestamp'], final_bar['close'], "End of backtest", final_index)
        
        # Save all candle data
        self.save_candle_data_to_files()
        
        # Print final statistics
        if self.strategy:
            self.strategy.print_final_stats()
        self.print_final_results()
        
        logger.info(f"🎯 COMPLETE {self.target_pair} backtest with candle data finished!")
    
    def print_final_results(self):
        """Print final backtest results"""
        total_trades = len(self.trades)
        if total_trades == 0:
            print(f"\n🚫 No trades executed for {self.target_pair}")
            return
        
        winning_trades = [t for t in self.trades if t.usd_pnl > 0]
        win_rate = len(winning_trades) / total_trades * 100
        total_pnl = sum(t.usd_pnl for t in self.trades)
        
        print(f"\n" + "="*80)
        print(f"🎯 {self.target_pair} BACKTEST RESULTS")
        print(f"="*80)
        print(f"🧠 Strategy: {self.strategy.__class__.__name__ if self.strategy else 'No Strategy'}")
        print(f"📊 Total Trades: {total_trades:,}")
        print(f"🎯 Win Rate: {win_rate:.2f}%")
        print(f"💰 Total P&L: ${total_pnl:+,.2f}")
        print(f"📈 Final Balance: ${self.current_balance:,.2f}")
        print(f"📊 Return: {((self.current_balance - self.initial_balance) / self.initial_balance * 100):+.2f}%")
        print(f"🕯️ Candle Data: {len(self.trade_candle_data):,} trades with full candle history")
        print(f"📋 Trade Log: {self.excel_log_path}")
        print(f"🕯️ Candle Data: {self.candle_data_path}")
        print(f"="*80)

if __name__ == "__main__":
    try:
        # 🎯 CHANGE THIS TO ANY CURRENCY PAIR
        TARGET_PAIR = "EUR/USD"  # 🔧 MODIFY THIS FOR DIFFERENT PAIRS
        # TARGET_PAIR = "GBP/USD"  # Will auto-use GBPUSDStrategy when uncommented
        # TARGET_PAIR = "USD/JPY"  # Will auto-use USDJPYStrategy when uncommented
        
        # Initialize backtest engine for target pair
        engine = BacktestEngine(
            target_pair=TARGET_PAIR,
            initial_balance=1000
        )
        
        # Run the complete backtest with candle data collection
        engine.run_backtest()
        
    except KeyboardInterrupt:
        logger.info(f"🛑 {TARGET_PAIR} backtest interrupted by user")
    except Exception as e:
        logger.error(f"❌ {TARGET_PAIR} backtest error: {e}", exc_info=True)