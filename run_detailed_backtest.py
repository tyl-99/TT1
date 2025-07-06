#!/usr/bin/env python3
"""
Detailed Strategy Testing with Excel Export
Tests each strategy once and exports all trade details to Excel files
"""
import sys
import os
import pandas as pd
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest.backtest_engine import BacktestEngine

class DetailedStrategyTester:
    def __init__(self):
        self.pairs = [
            "EUR/USD",
            "GBP/USD", 
            "USD/JPY",
            "EUR/GBP",
            "GBP/JPY",
            "EUR/JPY"
        ]
        self.results = {}
        self.all_trades = {}
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.max_risk_per_trade = 50.0  # Maximum risk per trade in USD
        
        # Create results folder if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
        
    def calculate_position_size(self, entry_price, stop_loss, pair):
        """Calculate position size based on maximum risk per trade"""
        pip_size = 0.01 if "JPY" in pair else 0.0001
        pip_value = 10  # $10 per pip for standard lot
        
        # Calculate risk in pips
        risk_pips = abs(entry_price - stop_loss) / pip_size
        
        # Calculate position size to limit risk to $50
        if risk_pips > 0:
            max_lots = self.max_risk_per_trade / (risk_pips * pip_value)
            return min(max_lots, 1.0)  # Cap at 1 standard lot
        return 0.1  # Minimum position size
        
    def test_strategy_detailed(self, pair):
        """Test a single strategy for a pair with detailed trade logging"""
        print(f"\n{'='*50}")
        print(f"🎯 TESTING {pair} STRATEGY WITH DETAILED LOGGING")
        print(f"{'='*50}")
        
        try:
            # Create backtest engine for this pair
            engine = BacktestEngine(target_pair=pair, initial_balance=1000)
            
            if engine.strategy is None:
                print(f"❌ No strategy available for {pair}")
                return None, []
                
            # Load data
            print(f"📊 Loading data for {pair}...")
            data = engine.load_excel_data()
            
            if not data or pair not in data:
                print(f"❌ No data available for {pair}")
                return None, []
                
            pair_df = data[pair].copy()
            print(f"📈 Data loaded: {len(pair_df)} candles")
            
            # Find all zones using the strategy
            print(f"🔍 Finding supply/demand zones...")
            all_zones = engine.strategy.find_all_zones(pair_df)
            print(f"📍 Found {len(all_zones)} zones")
            
            if not all_zones:
                print(f"❌ No zones found for {pair}")
                return None, []
            
            # Simulate trading with detailed logging
            print(f"🚀 Simulating trades with detailed logging...")
            active_trade = None
            all_trades = []
            trade_id = 1
            
            for i in range(1, len(pair_df)):
                current_bar = pair_df.iloc[i]
                current_time = str(pair_df.iloc[i].name) if hasattr(pair_df.iloc[i], 'name') else str(i)
                
                # Check for trade exit
                if active_trade is not None:
                    exit_price = None
                    exit_reason = None
                    
                    if active_trade['decision'] == "BUY":
                        if current_bar['high'] >= active_trade['take_profit']: 
                            exit_price = active_trade['take_profit']
                            exit_reason = "Take Profit Hit"
                        elif current_bar['low'] <= active_trade['stop_loss']: 
                            exit_price = active_trade['stop_loss']
                            exit_reason = "Stop Loss Hit"
                    else:  # SELL
                        if current_bar['low'] <= active_trade['take_profit']: 
                            exit_price = active_trade['take_profit']
                            exit_reason = "Take Profit Hit"
                        elif current_bar['high'] >= active_trade['stop_loss']: 
                            exit_price = active_trade['stop_loss']
                            exit_reason = "Stop Loss Hit"
                    
                    if exit_price is not None:
                        # Calculate risk and reward in pips first
                        pip_size = 0.01 if "JPY" in pair else 0.0001
                        pip_value = 10  # $10 per pip for standard lot
                        
                        # Calculate position size and actual P&L
                        position_size = active_trade['position_size']
                        
                        # Calculate P&L properly based on pip movement and position size
                        pnl_mult = 1 if active_trade['decision'] == "BUY" else -1
                        price_diff = (exit_price - active_trade['entry_price']) * pnl_mult
                        pip_movement = price_diff / pip_size
                        pnl = pip_movement * pip_value * position_size - 1.0  # $1 commission
                        
                        # Calculate R:R using direct price distances (simpler)
                        risk_distance = abs(active_trade['entry_price'] - active_trade['stop_loss'])
                        reward_distance = abs(active_trade['take_profit'] - active_trade['entry_price'])
                        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
                        
                        # Still calculate pips for reporting and P&L calculations
                        if active_trade['decision'] == "BUY":
                            risk_pips = (active_trade['entry_price'] - active_trade['stop_loss']) / pip_size
                            reward_pips = (active_trade['take_profit'] - active_trade['entry_price']) / pip_size
                        else:
                            risk_pips = (active_trade['stop_loss'] - active_trade['entry_price']) / pip_size
                            reward_pips = (active_trade['entry_price'] - active_trade['take_profit']) / pip_size
                        
                        # Calculate actual risk and potential reward in dollars
                        risk_amount_usd = risk_pips * pip_value * position_size
                        potential_reward_usd = reward_pips * pip_value * position_size
                        
                        trade_detail = {
                            'Trade_ID': trade_id,
                            'Pair': pair,
                            'Entry_Time': active_trade['entry_time'],
                            'Exit_Time': current_time,
                            'Direction': active_trade['decision'],
                            'Entry_Price': active_trade['entry_price'],
                            'Exit_Price': exit_price,
                            'Stop_Loss': active_trade['stop_loss'],
                            'Take_Profit': active_trade['take_profit'],
                            'Position_Size': position_size,
                            'Risk_Amount_USD': risk_amount_usd,
                            'Potential_Loss_USD': risk_amount_usd,  # Same as risk amount
                            'Potential_Reward_USD': potential_reward_usd,
                            'Actual_PnL_USD': pnl,
                            'Risk_Pips': risk_pips,
                            'Reward_Pips': reward_pips,
                            'RR_Ratio': rr_ratio,
                            'Exit_Reason': exit_reason,
                            'Zone_Type': active_trade.get('zone_type', 'Unknown'),
                            'Zone_High': active_trade.get('zone_price_high', 0),
                            'Zone_Low': active_trade.get('zone_price_low', 0),
                            'Result': 'Win' if pnl > 0 else 'Loss'
                        }
                        
                        all_trades.append(trade_detail)
                        trade_id += 1
                        active_trade = None
                        
                    continue

                # Check for trade entry
                if active_trade is None:
                    for zone in all_zones:
                        if zone['is_fresh'] and zone['created_at_index'] < i:
                            # Check both high and low of current candle
                            for price_point in [current_bar['high'], current_bar['low']]:
                                signal = engine.strategy.check_entry_signal(price_point, zone)
                                if signal and signal.get('decision') in ['BUY', 'SELL']:
                                    # Calculate position size based on risk management
                                    position_size = self.calculate_position_size(
                                        signal['entry_price'], 
                                        signal['stop_loss'], 
                                        pair
                                    )
                                    
                                    active_trade = signal.copy()
                                    active_trade['entry_time'] = current_time
                                    active_trade['zone_type'] = zone['type']
                                    active_trade['zone_price_high'] = zone['price_high']
                                    active_trade['zone_price_low'] = zone['price_low']
                                    active_trade['position_size'] = position_size
                                    zone['is_fresh'] = False  # Mark zone as used
                                    break
                            if active_trade:
                                break
            
            # Calculate summary results
            if not all_trades:
                print(f"❌ No trades executed for {pair}")
                return None, []
                
            total_trades = len(all_trades)
            wins = sum(1 for t in all_trades if t['Actual_PnL_USD'] > 0)
            losses = total_trades - wins
            win_rate = (wins / total_trades) * 100
            total_pnl = sum(t['Actual_PnL_USD'] for t in all_trades)
            
            avg_win = sum(t['Actual_PnL_USD'] for t in all_trades if t['Actual_PnL_USD'] > 0) / wins if wins > 0 else 0
            avg_loss = sum(t['Actual_PnL_USD'] for t in all_trades if t['Actual_PnL_USD'] < 0) / losses if losses > 0 else 0
            profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else float('inf')
            
            max_win = max(t['Actual_PnL_USD'] for t in all_trades) if all_trades else 0
            max_loss = min(t['Actual_PnL_USD'] for t in all_trades) if all_trades else 0
            
            avg_rr = sum(t['RR_Ratio'] for t in all_trades) / len(all_trades) if all_trades else 0
            avg_risk = sum(t['Risk_Amount_USD'] for t in all_trades) / len(all_trades) if all_trades else 0
            max_risk = max(t['Risk_Amount_USD'] for t in all_trades) if all_trades else 0
            
            result = {
                'pair': pair,
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'max_win': max_win,
                'max_loss': max_loss,
                'profit_factor': profit_factor,
                'zones_found': len(all_zones),
                'strategy_class': engine.strategy.__class__.__name__,
                'avg_rr_ratio': avg_rr,
                'avg_risk_usd': avg_risk,
                'max_risk_usd': max_risk
            }
            
            # Print results
            print(f"\n✅ {pair} DETAILED RESULTS:")
            print(f"   💰 Total P&L: ${total_pnl:.2f}")
            print(f"   📊 Trades: {total_trades} ({wins}W / {losses}L)")
            print(f"   📈 Win Rate: {win_rate:.2f}%")
            print(f"   💚 Avg Win: ${avg_win:.2f} | Max Win: ${max_win:.2f}")
            print(f"   💔 Avg Loss: ${avg_loss:.2f} | Max Loss: ${max_loss:.2f}")
            print(f"   🔥 Profit Factor: {profit_factor:.2f}")
            print(f"   ⚖️ Avg R:R Ratio: {avg_rr:.2f}")
            print(f"   💸 Avg Risk: ${avg_risk:.2f} | Max Risk: ${max_risk:.2f}")
            print(f"   📍 Zones Found: {len(all_zones)}")
            print(f"   💾 Trades logged for Excel export")
            
            return result, all_trades
            
        except Exception as e:
            print(f"❌ Error testing {pair}: {str(e)}")
            return None, []

    def export_to_excel(self, pair, trades, result):
        """Export detailed trades to Excel for a specific pair"""
        if not trades:
            print(f"⚠️ No trades to export for {pair}")
            return
            
        filename = f"results/DETAILED_TRADES_{pair.replace('/', '_')}_{self.timestamp}.xlsx"
        
        try:
            # Convert trades to DataFrame
            df = pd.DataFrame(trades)
            
            # Create Excel writer
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Write trades data
                df.to_excel(writer, sheet_name='All_Trades', index=False)
                
                # Create summary sheet
                summary_data = {
                    'Metric': [
                        'Currency Pair',
                        'Strategy Used',
                        'Total Trades',
                        'Winning Trades',
                        'Losing Trades', 
                        'Win Rate (%)',
                        'Total P&L ($)',
                        'Average Win ($)',
                        'Average Loss ($)',
                        'Maximum Win ($)',
                        'Maximum Loss ($)',
                        'Profit Factor',
                        'Average R:R Ratio',
                        'Average Risk per Trade ($)',
                        'Maximum Risk per Trade ($)',
                        'Zones Found'
                    ],
                    'Value': [
                        result['pair'],
                        result['strategy_class'],
                        result['total_trades'],
                        result['wins'],
                        result['losses'],
                        f"{result['win_rate']:.2f}%",
                        f"${result['total_pnl']:.2f}",
                        f"${result['avg_win']:.2f}",
                        f"${result['avg_loss']:.2f}",
                        f"${result['max_win']:.2f}",
                        f"${result['max_loss']:.2f}",
                        f"{result['profit_factor']:.2f}",
                        f"{result['avg_rr_ratio']:.2f}",
                        f"${result['avg_risk_usd']:.2f}",
                        f"${result['max_risk_usd']:.2f}",
                        result['zones_found']
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
            print(f"✅ Exported {len(trades)} trades to: {filename}")
            
        except Exception as e:
            print(f"❌ Error exporting {pair} to Excel: {str(e)}")
    
    def create_master_summary(self, all_results):
        """Create master summary Excel file"""
        if not all_results:
            print("❌ No results to create master summary")
            return
            
        filename = f"results/MASTER_STRATEGY_SUMMARY_{self.timestamp}.xlsx"
        
        try:
            # Prepare summary data
            summary_data = []
            for result in all_results:
                summary_data.append({
                    'Currency_Pair': result['pair'],
                    'Strategy_Used': result['strategy_class'],
                    'Total_Trades': result['total_trades'],
                    'Wins': result['wins'],
                    'Losses': result['losses'],
                    'Win_Rate_%': f"{result['win_rate']:.2f}%",
                    'Total_PnL_$': f"${result['total_pnl']:.2f}",
                    'Avg_Win_$': f"${result['avg_win']:.2f}",
                    'Avg_Loss_$': f"${result['avg_loss']:.2f}",
                    'Max_Win_$': f"${result['max_win']:.2f}",
                    'Max_Loss_$': f"${result['max_loss']:.2f}",
                    'Profit_Factor': f"{result['profit_factor']:.2f}",
                    'Avg_RR_Ratio': f"{result['avg_rr_ratio']:.2f}",
                    'Avg_Risk_$': f"${result['avg_risk_usd']:.2f}",
                    'Max_Risk_$': f"${result['max_risk_usd']:.2f}",
                    'Zones_Found': result['zones_found']
                })
            
            df = pd.DataFrame(summary_data)
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='All_Pairs_Summary', index=False)
                
                # Calculate overall totals
                total_trades = sum(r['total_trades'] for r in all_results)
                total_wins = sum(r['wins'] for r in all_results)
                total_pnl = sum(r['total_pnl'] for r in all_results)
                overall_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
                avg_risk_all = sum(r['avg_risk_usd'] for r in all_results) / len(all_results) if all_results else 0
                max_risk_all = max(r['max_risk_usd'] for r in all_results) if all_results else 0
                
                # Create overall summary
                overall_data = {
                    'Portfolio_Metric': [
                        'Total Currency Pairs Tested',
                        'Successful Pairs (with trades)',
                        'Total Trades Across All Pairs',
                        'Total Winning Trades',
                        'Total Losing Trades',
                        'Overall Win Rate (%)',
                        'Total Portfolio P&L ($)',
                        'Average P&L per Pair ($)',
                        'Average Risk per Trade ($)',
                        'Maximum Risk per Trade ($)',
                        'Best Performing Pair',
                        'Worst Performing Pair'
                    ],
                    'Value': [
                        len(self.pairs),
                        len(all_results),
                        total_trades,
                        total_wins,
                        total_trades - total_wins,
                        f"{overall_win_rate:.2f}%",
                        f"${total_pnl:.2f}",
                        f"${total_pnl/len(all_results):.2f}" if all_results else "$0.00",
                        f"${avg_risk_all:.2f}",
                        f"${max_risk_all:.2f}",
                        max(all_results, key=lambda x: x['total_pnl'])['pair'] if all_results else "N/A",
                        min(all_results, key=lambda x: x['total_pnl'])['pair'] if all_results else "N/A"
                    ]
                }
                
                overall_df = pd.DataFrame(overall_data)
                overall_df.to_excel(writer, sheet_name='Portfolio_Overview', index=False)
                
            print(f"✅ Master summary exported to: {filename}")
            
        except Exception as e:
            print(f"❌ Error creating master summary: {str(e)}")
    
    def run_all_detailed_tests(self):
        """Run detailed tests for all pairs and export to Excel"""
        print("🚀 RUNNING DETAILED STRATEGY TESTS FOR ALL PAIRS")
        print("📊 Exporting all trade data to Excel files")
        print(f"💰 Maximum risk per trade: ${self.max_risk_per_trade}")
        print("=" * 70)
        
        all_results = []
        
        for pair in self.pairs:
            result, trades = self.test_strategy_detailed(pair)
            if result and trades:
                all_results.append(result)
                self.results[pair] = result
                self.all_trades[pair] = trades
                
                # Export individual pair to Excel
                self.export_to_excel(pair, trades, result)
            else:
                print(f"⚠️ Skipping Excel export for {pair} (no trades)")
        
        # Create master summary
        if all_results:
            self.create_master_summary(all_results)
        
        # Print final summary
        self.print_final_summary(all_results)
    
    def print_final_summary(self, all_results):
        """Print comprehensive final summary"""
        print(f"\n{'='*70}")
        print("📊 COMPREHENSIVE STRATEGY TESTING COMPLETE")
        print(f"{'='*70}")
        
        if not all_results:
            print("❌ No successful tests completed")
            return
        
        print(f"\n📈 PERFORMANCE RANKING:")
        print("-" * 50)
        
        # Sort by P&L
        sorted_results = sorted(all_results, key=lambda x: x['total_pnl'], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            print(f"{i}. {result['pair']:<8} | ${result['total_pnl']:>8.2f} | {result['win_rate']:>5.1f}% | {result['total_trades']:>3} trades")
        
        # Overall statistics
        total_pnl = sum(r['total_pnl'] for r in all_results)
        total_trades = sum(r['total_trades'] for r in all_results)
        total_wins = sum(r['wins'] for r in all_results)
        overall_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
        avg_risk = sum(r['avg_risk_usd'] for r in all_results) / len(all_results) if all_results else 0
        max_risk = max(r['max_risk_usd'] for r in all_results) if all_results else 0
        
        print(f"\n🎯 PORTFOLIO SUMMARY:")
        print(f"   💰 Total P&L: ${total_pnl:,.2f}")
        print(f"   📊 Total Trades: {total_trades:,}")
        print(f"   📈 Overall Win Rate: {overall_win_rate:.2f}%")
        print(f"   💸 Average Risk per Trade: ${avg_risk:.2f}")
        print(f"   🚨 Maximum Risk per Trade: ${max_risk:.2f}")
        print(f"   🎲 Active Pairs: {len(all_results)}/{len(self.pairs)}")
        print(f"   📁 Excel Files Created: {len(all_results) + 1}")
        
        print(f"\n✅ All data exported to Excel files with timestamp: {self.timestamp}")
        print(f"📂 Files created in 'results' folder:")
        for result in all_results:
            print(f"   • DETAILED_TRADES_{result['pair'].replace('/', '_')}_{self.timestamp}.xlsx")
        print(f"   • MASTER_STRATEGY_SUMMARY_{self.timestamp}.xlsx")

def main():
    tester = DetailedStrategyTester()
    tester.run_all_detailed_tests()

if __name__ == "__main__":
    main() 