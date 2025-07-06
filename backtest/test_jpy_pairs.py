#!/usr/bin/env python3
"""
🎯 JPY PAIRS STRATEGY TESTER
Tests the optimized supply/demand strategy on JPY pairs to collect initial data.

Testing:
- USD/JPY
- GBP/JPY  
- EUR/JPY

Using the existing supply/demand strategy as baseline.
"""

import pandas as pd
import datetime
import logging
import os
from backtest_engine import BacktestEngine
from strategy.eurusd_strategy import EURUSDStrategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_jpy_pair_strategy(pair="USD/JPY"):
    """
    Test the supply/demand strategy on a JPY pair
    """
    print(f"\n🎯 TESTING SUPPLY/DEMAND STRATEGY - {pair}")
    print("=" * 60)
    
    # Initialize strategy (adapted for JPY pair)
    strategy = EURUSDStrategy(target_pair=pair)
    
    # Adjust pip size for JPY pairs
    if "JPY" in pair:
        strategy.pip_size = 0.01  # JPY pairs have different pip size
        print(f"📊 Adjusted pip size for JPY pair: {strategy.pip_size}")
    
    # Initialize backtest engine
    engine = BacktestEngine(pair, initial_balance=1000, strategy=strategy)
    
    # Load data
    print(f"📊 Loading {pair} data...")
    data = engine.load_excel_data()
    
    if not data or pair not in data:
        print(f"❌ No data available for {pair}")
        return None
    
    pair_df = data[pair].copy()
    print(f"✅ Loaded {len(pair_df):,} candles")
    print(f"📅 Date range: {pair_df['timestamp'].min()} to {pair_df['timestamp'].max()}")
    
    # Find all zones first
    all_zones = strategy.find_all_zones(pair_df)
    print(f"🎯 Found {len(all_zones)} supply/demand zones")
    
    if not all_zones:
        print("❌ No zones found!")
        return {
            'pair': pair,
            'zones_found': 0,
            'trades_executed': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'candles_analyzed': len(pair_df)
        }
    
    # Display zone breakdown
    supply_zones = [z for z in all_zones if z['type'] == 'supply']
    demand_zones = [z for z in all_zones if z['type'] == 'demand']
    print(f"   📈 Supply zones: {len(supply_zones)}")
    print(f"   📉 Demand zones: {len(demand_zones)}")
    
    # Simulate trading
    active_trade = None
    all_trades = []
    trade_number = 1
    zones_tested = 0
    
    for i in range(1, len(pair_df)):
        current_bar = pair_df.iloc[i]
        
        # Exit check for active trade
        if active_trade is not None:
            exit_price = None
            exit_type = None
            
            if active_trade['decision'] == "BUY":
                if current_bar['high'] >= active_trade['take_profit']:
                    exit_price = active_trade['take_profit']
                    exit_type = "TP"
                elif current_bar['low'] <= active_trade['stop_loss']:
                    exit_price = active_trade['stop_loss']
                    exit_type = "SL"
            else:  # SELL
                if current_bar['low'] <= active_trade['take_profit']:
                    exit_price = active_trade['take_profit']
                    exit_type = "TP"
                elif current_bar['high'] >= active_trade['stop_loss']:
                    exit_price = active_trade['stop_loss']
                    exit_type = "SL"
            
            if exit_price is not None:
                # Calculate PnL
                pnl_mult = 1 if active_trade['decision'] == "BUY" else -1
                
                # For JPY pairs, use different position sizing
                if "JPY" in pair:
                    # JPY pairs: 1 pip = 0.01, standard lot = 1000 units per pip
                    raw_pnl = (exit_price - active_trade['entry_price']) * pnl_mult * 1000
                else:
                    # Major pairs: 1 pip = 0.0001, standard lot = 10 units per pip  
                    raw_pnl = (exit_price - active_trade['entry_price']) * pnl_mult * 100000
                
                pnl = raw_pnl - 1.0  # Subtract spread cost
                
                trade_result = {
                    'trade_number': trade_number,
                    'entry_date': active_trade['entry_date'],
                    'exit_date': current_bar['timestamp'],
                    'direction': active_trade['decision'],
                    'zone_type': active_trade.get('zone_type', 'unknown'),
                    'entry_price': active_trade['entry_price'],
                    'exit_price': exit_price,
                    'stop_loss': active_trade['stop_loss'],
                    'take_profit': active_trade['take_profit'],
                    'pnl': pnl,
                    'exit_type': exit_type,
                    'is_win': exit_type == "TP"
                }
                all_trades.append(trade_result)
                trade_number += 1
                active_trade = None
                continue
        
        # Entry check - look for zone entries (only if no active trade)
        if active_trade is None:
            for zone in all_zones:
                if zone['is_fresh'] and zone['created_at_index'] < i:
                    # Check both high and low of current candle for zone entry
                    for price_point in [current_bar['high'], current_bar['low']]:
                        signal = strategy.check_entry_signal(price_point, zone)
                        if signal:
                            active_trade = signal.copy()
                            active_trade['entry_date'] = current_bar['timestamp']
                            active_trade['zone_type'] = zone['type']
                            zone['is_fresh'] = False  # Mark zone as tested
                            zones_tested += 1
                            break  # Exit price_point loop
                    if active_trade:
                        break  # Exit zone loop if we found a trade
    
    # Calculate results
    if all_trades:
        wins = [t for t in all_trades if t['is_win']]
        losses = [t for t in all_trades if not t['is_win']]
        win_rate = (len(wins) / len(all_trades)) * 100
        total_pnl = sum(t['pnl'] for t in all_trades)
        
        # Breakdown by zone type
        supply_trades = [t for t in all_trades if t['zone_type'] == 'supply']
        demand_trades = [t for t in all_trades if t['zone_type'] == 'demand']
        
        supply_wins = len([t for t in supply_trades if t['is_win']])
        demand_wins = len([t for t in demand_trades if t['is_win']])
        
        print(f"\n📊 TRADING RESULTS:")
        print(f"   ✅ Total trades: {len(all_trades)}")
        print(f"   🎯 Win rate: {win_rate:.1f}% ({len(wins)} wins, {len(losses)} losses)")
        print(f"   💰 Total PnL: ${total_pnl:.2f}")
        print(f"   🎯 Zones tested: {zones_tested}/{len(all_zones)}")
        
        print(f"\n📈 ZONE BREAKDOWN:")
        print(f"   📊 Supply zone trades: {len(supply_trades)} ({supply_wins} wins, {supply_wins/len(supply_trades)*100:.1f}% win rate)" if supply_trades else "   📊 Supply zone trades: 0")
        print(f"   📊 Demand zone trades: {len(demand_trades)} ({demand_wins} wins, {demand_wins/len(demand_trades)*100:.1f}% win rate)" if demand_trades else "   📊 Demand zone trades: 0")
        
        if len(wins) > 0 and len(losses) > 0:
            avg_win = sum(t['pnl'] for t in wins) / len(wins)
            avg_loss = sum(t['pnl'] for t in losses) / len(losses)
            profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses))
            print(f"   📊 Average win: ${avg_win:.2f}")
            print(f"   📊 Average loss: ${avg_loss:.2f}")
            print(f"   📊 Profit factor: {profit_factor:.2f}")
        
        return {
            'pair': pair,
            'candles_analyzed': len(pair_df),
            'zones_found': len(all_zones),
            'supply_zones': len(supply_zones),
            'demand_zones': len(demand_zones),
            'zones_tested': zones_tested,
            'trades_executed': len(all_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'supply_trades': len(supply_trades),
            'demand_trades': len(demand_trades),
            'supply_win_rate': (supply_wins/len(supply_trades)*100) if supply_trades else 0,
            'demand_win_rate': (demand_wins/len(demand_trades)*100) if demand_trades else 0,
            'profit_factor': profit_factor if len(wins) > 0 and len(losses) > 0 else 0,
            'all_trades': all_trades
        }
    else:
        print(f"\n❌ NO TRADES EXECUTED")
        print(f"   🎯 Zones found: {len(all_zones)}")
        print(f"   🎯 Zones tested: {zones_tested}")
        return {
            'pair': pair,
            'zones_found': len(all_zones),
            'trades_executed': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'candles_analyzed': len(pair_df)
        }

def main():
    """
    Test all three JPY pairs
    """
    print("🚀 JPY PAIRS SUPPLY/DEMAND STRATEGY TESTING")
    print("=" * 60)
    print("🎯 Testing optimized supply/demand strategy on:")
    print("   - USD/JPY")
    print("   - GBP/JPY") 
    print("   - EUR/JPY")
    print()
    
    jpy_pairs = ["USD/JPY", "GBP/JPY", "EUR/JPY"]
    results = []
    
    for pair in jpy_pairs:
        try:
            result = test_jpy_pair_strategy(pair)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Error testing {pair}: {e}")
            continue
    
    # Summary of all results
    if results:
        print("\n" + "=" * 60)
        print("📊 OVERALL JPY PAIRS SUMMARY")
        print("=" * 60)
        
        total_trades = sum(r['trades_executed'] for r in results)
        total_zones = sum(r['zones_found'] for r in results)
        total_pnl = sum(r['total_pnl'] for r in results)
        
        print(f"📈 COMBINED RESULTS:")
        print(f"   💼 Pairs tested: {len(results)}")
        print(f"   🎯 Total zones found: {total_zones}")
        print(f"   ⚡ Total trades: {total_trades}")
        print(f"   💰 Combined PnL: ${total_pnl:.2f}")
        
        if total_trades > 0:
            total_wins = sum(r['wins'] for r in results)
            overall_win_rate = (total_wins / total_trades) * 100
            print(f"   🎯 Overall win rate: {overall_win_rate:.1f}%")
        
        print(f"\n📊 INDIVIDUAL PAIR PERFORMANCE:")
        for result in results:
            if result['trades_executed'] > 0:
                print(f"   {result['pair']}: {result['trades_executed']} trades, "
                      f"{result['win_rate']:.1f}% win rate, "
                      f"${result['total_pnl']:.2f} PnL")
            else:
                print(f"   {result['pair']}: No trades executed ({result['zones_found']} zones found)")
        
        print(f"\n🎯 STRATEGY INSIGHTS:")
        if any(r['trades_executed'] > 0 for r in results):
            best_pair = max([r for r in results if r['trades_executed'] > 0], key=lambda x: x['win_rate'])
            worst_pair = min([r for r in results if r['trades_executed'] > 0], key=lambda x: x['win_rate'])
            print(f"   🏆 Best performing: {best_pair['pair']} ({best_pair['win_rate']:.1f}% win rate)")
            print(f"   📉 Needs improvement: {worst_pair['pair']} ({worst_pair['win_rate']:.1f}% win rate)")
        
        # Zone analysis
        avg_zones_per_pair = total_zones / len(results)
        print(f"   🎯 Average zones per pair: {avg_zones_per_pair:.1f}")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   🔧 Ready for autotuning on JPY pairs")
        print(f"   📊 Strategy shows potential for optimization")
        print(f"   🎯 Consider JPY-specific parameter adjustments")
        
    else:
        print("❌ No successful tests completed")

if __name__ == "__main__":
    main() 