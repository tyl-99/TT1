#!/usr/bin/env python3
"""
Firebase integration for Trading Bot
Handles both Firestore (database) and Storage (files)
WITH PROPER POSITION ID TRACKING
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from firebase_admin import firestore, storage
from firebase_config import initialize_firebase
from dotenv import load_dotenv

# Initialize Firebase
firebase_app = initialize_firebase()
logger = logging.getLogger(__name__)

class FirebaseTrader:
    def __init__(self):
        """Initialize Firebase Trader with enhanced trade tracking"""
        self.db = firestore.client()
        self.bucket = storage.bucket()
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create temp directory for Excel files
        os.makedirs('temp', exist_ok=True)
        
        logger.info("🔥 Firebase Trader initialized with enhanced structure")
        logger.info(f"📊 Session ID: {self.session_id}")
        
        # Track pending trades by position_id for lifecycle management
        self.pending_trades = {}  # {position_id: trade_doc_id}
        
    # ===============================
    # TRADE LIFECYCLE MANAGEMENT
    # ===============================
    
    def save_pending_trade(self, pending_order_data, order_id=None):
        """Save trade when order is first sent (before position creation)"""
        try:
            trade_data = {
                # Basic trade info
                'symbol': pending_order_data.get('symbol'),
                'decision': pending_order_data.get('decision'),
                'status': 'order_sent',
                'timestamp': datetime.now(),
                
                # Order details
                'order_id': order_id,
                'position_id': None,  # Will be updated when position is created
                
                # Volume info
                'volume_lots': pending_order_data.get('volume', 0) / 100000,
                'volume_units': pending_order_data.get('volume', 0),
                
                # Price levels from strategy
                'entry_price': pending_order_data.get('entry_price'),
                'stop_loss': pending_order_data.get('stop_loss'),
                'take_profit': pending_order_data.get('take_profit'),
                
                # Risk management
                'risk_reward_ratio': float(pending_order_data.get('risk_reward_ratio', '0').replace(':', '')),
                'potential_loss_usd': float(pending_order_data.get('potential_loss_usd', '$0').replace('$', '')),
                'potential_win_usd': float(pending_order_data.get('potential_win_usd', '$0').replace('$', '')),
                
                # Strategy info
                'trade_reason': pending_order_data.get('reason', ''),
                'winrate': pending_order_data.get('winrate', ''),
                'volume_calculation': pending_order_data.get('volume_calculation', ''),
                'loss_calculation': pending_order_data.get('loss_calculation', ''),
                'win_calculation': pending_order_data.get('win_calculation', ''),
                
                # Session info
                'session_date': datetime.now().strftime('%Y-%m-%d'),
                'created_at': datetime.now()
            }
            
            # Add to Firestore
            doc_ref = self.db.collection('trades').add(trade_data)
            trade_doc_id = doc_ref[1].id
            
            print(f"✅ Pending trade saved: {trade_doc_id}")
            
            # If we have order_id, track it for position updates
            if order_id:
                self.pending_trades[f"order_{order_id}"] = trade_doc_id
            
            return trade_doc_id
            
        except Exception as e:
            print(f"❌ Error saving pending trade: {e}")
            return None
    
    def update_trade_with_position(self, position_id, order_id=None, actual_entry_price=None):
        """Update trade when position is created with actual position_id"""
        try:
            # Find the trade document
            trade_doc_id = None
            
            if order_id and f"order_{order_id}" in self.pending_trades:
                trade_doc_id = self.pending_trades[f"order_{order_id}"]
                # Move to position tracking
                self.pending_trades[position_id] = trade_doc_id
                del self.pending_trades[f"order_{order_id}"]
            else:
                print(f"⚠️ Could not find pending trade for order_id: {order_id}")
                return None
            
            # Update with position info
            update_data = {
                'position_id': position_id,
                'status': 'position_open',
                'position_created_at': datetime.now()
            }
            
            if actual_entry_price:
                update_data['actual_entry_price'] = actual_entry_price
            
            self.db.collection('trades').document(trade_doc_id).update(update_data)
            print(f"✅ Trade updated with position_id: {position_id}")
            
            return trade_doc_id
            
        except Exception as e:
            print(f"❌ Error updating trade with position: {e}")
            return None
    
    def update_trade_sl_tp_set(self, position_id):
        """Update trade when SL/TP are successfully set"""
        try:
            if position_id in self.pending_trades:
                trade_doc_id = self.pending_trades[position_id]
                
                update_data = {
                    'status': 'active',
                    'sl_tp_set_at': datetime.now()
                }
                
                self.db.collection('trades').document(trade_doc_id).update(update_data)
                print(f"✅ Trade SL/TP confirmed for position: {position_id}")
                
        except Exception as e:
            print(f"❌ Error updating SL/TP status: {e}")
    
    def complete_trade(self, position_id, final_data):
        """Complete trade when position is closed with final results"""
        try:
            if position_id in self.pending_trades:
                trade_doc_id = self.pending_trades[position_id]
                
                # Calculate final metrics
                entry_price = final_data.get('entry_price', 0)
                exit_price = final_data.get('exit_price', 0)
                volume_lots = final_data.get('volume_lots', 0)
                
                # Determine if win/loss
                decision = final_data.get('decision', 'BUY')
                if decision == 'BUY':
                    is_winner = exit_price > entry_price
                else:
                    is_winner = exit_price < entry_price
                
                update_data = {
                    'status': 'completed',
                    'exit_price': exit_price,
                    'actual_pnl_usd': final_data.get('pnl', 0),
                    'is_winner': is_winner,
                    'closed_at': datetime.now(),
                    'execution_time_ms': final_data.get('execution_time_ms', 0),
                    'slippage_pips': final_data.get('slippage_pips', 0)
                }
                
                self.db.collection('trades').document(trade_doc_id).update(update_data)
                print(f"✅ Trade completed: {trade_doc_id} | P&L: ${final_data.get('pnl', 0)}")
                
                # Remove from pending tracking
                del self.pending_trades[position_id]
                
                # Update daily summary
                self.update_daily_summary(final_data.get('pnl', 0), is_winner)
                
                return trade_doc_id
            else:
                print(f"⚠️ No pending trade found for position_id: {position_id}")
                return None
                
        except Exception as e:
            print(f"❌ Error completing trade: {e}")
            return None
    
    def handle_trade_error(self, position_id=None, order_id=None, error_info=None):
        """Handle trade errors (failed orders, bad stops, etc.)"""
        try:
            trade_doc_id = None
            
            # Find the trade
            if position_id and position_id in self.pending_trades:
                trade_doc_id = self.pending_trades[position_id]
                del self.pending_trades[position_id]
            elif order_id and f"order_{order_id}" in self.pending_trades:
                trade_doc_id = self.pending_trades[f"order_{order_id}"]
                del self.pending_trades[f"order_{order_id}"]
            
            if trade_doc_id:
                update_data = {
                    'status': 'failed',
                    'error_code': error_info.get('error_code', ''),
                    'error_description': error_info.get('description', ''),
                    'failed_at': datetime.now()
                }
                
                self.db.collection('trades').document(trade_doc_id).update(update_data)
                print(f"✅ Trade marked as failed: {error_info.get('error_code', '')}")
                
        except Exception as e:
            print(f"❌ Error handling trade error: {e}")
    
    # ===============================
    # ACCOUNT & PERFORMANCE TRACKING
    # ===============================
    
    def update_daily_summary(self, pnl, is_winner):
        """Update daily performance summary"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            doc_ref = self.db.collection('account').document(f'daily_summary_{today}')
            
            # Get existing summary or create new
            doc = doc_ref.get()
            if doc.exists:
                summary = doc.to_dict()
            else:
                summary = {
                    'date': today,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl': 0.0,
                    'biggest_win': 0.0,
                    'biggest_loss': 0.0,
                    'created_at': datetime.now()
                }
            
            # Update metrics
            summary['total_trades'] += 1
            summary['total_pnl'] += pnl
            
            if is_winner:
                summary['winning_trades'] += 1
                summary['biggest_win'] = max(summary['biggest_win'], pnl)
            else:
                summary['losing_trades'] += 1
                summary['biggest_loss'] = min(summary['biggest_loss'], pnl)
            
            # Calculate win rate
            if summary['total_trades'] > 0:
                summary['win_rate'] = (summary['winning_trades'] / summary['total_trades']) * 100
            
            summary['last_updated'] = datetime.now()
            
            doc_ref.set(summary)
            print(f"✅ Daily summary updated: {summary['total_trades']} trades, {summary['win_rate']:.1f}% win rate")
            
        except Exception as e:
            print(f"❌ Error updating daily summary: {e}")
    
    def save_session_info(self, session_data):
        """Save trading session information"""
        try:
            session_id = f"{datetime.now().strftime('%Y-%m-%d')}_session"
            
            session_info = {
                'session_id': session_id,
                'start_time': session_data.get('start_time', datetime.now()),
                'end_time': datetime.now(),
                'pairs_analyzed': session_data.get('pairs_analyzed', []),
                'trades_executed': session_data.get('trades_executed', 0),
                'errors_encountered': session_data.get('errors', []),
                'total_runtime_minutes': session_data.get('runtime_minutes', 0),
                'bot_version': session_data.get('version', 'v2.1')
            }
            
            self.db.collection('sessions').document(session_id).set(session_info)
            print(f"✅ Session info saved: {session_id}")
            
        except Exception as e:
            print(f"❌ Error saving session info: {e}")
    
    # ===============================
    # ENHANCED STORAGE OPERATIONS
    # ===============================
    
    def upload_trade_report_by_symbol(self, local_excel_path, symbol):
        """Upload Excel trade report organized by symbol"""
        try:
            date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Organize by symbol in storage
            symbol_folder = symbol.replace('/', '_')
            remote_path = f"reports/trade_reports/{symbol_folder}/DETAILED_TRADES_{symbol_folder}_{date_str}.xlsx"
            
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_excel_path)
            
            print(f"✅ Trade report uploaded: {remote_path}")
            return blob.public_url
        except Exception as e:
            print(f"❌ Error uploading trade report: {e}")
            return None
    
    def log_error(self, error_data):
        """Log errors to both Firestore and Storage"""
        try:
            # Save to Firestore for querying
            error_doc = {
                'timestamp': datetime.now(),
                'error_type': error_data.get('error_type', ''),
                'pair': error_data.get('pair', ''),
                'details': error_data.get('details', ''),
                'session_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            self.db.collection('errors').add(error_doc)
            
            # Also append to daily error log file
            self.append_to_error_log(error_data)
            
        except Exception as e:
            print(f"❌ Error logging error: {e}")
    
    def append_to_error_log(self, error_data):
        """Append error to daily error log file"""
        try:
            import tempfile
            import os
            
            date_str = datetime.now().strftime('%Y-%m-%d')
            log_entry = f"[{datetime.now().isoformat()}] {error_data.get('error_type', '')} - {error_data.get('pair', '')} - {error_data.get('details', '')}\n"
            
            # Create temp file and upload
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as temp_file:
                temp_file.write(log_entry)
                temp_file_path = temp_file.name
            
            # Upload to storage
            remote_path = f"logs/error_logs/errors_{date_str}.log"
            blob = self.bucket.blob(remote_path)
            
            # Append mode: download existing, append, upload
            try:
                existing_content = blob.download_as_text()
                with open(temp_file_path, 'w') as f:
                    f.write(existing_content + log_entry)
            except:
                # File doesn't exist yet, just upload new content
                pass
            
            blob.upload_from_filename(temp_file_path)
            os.unlink(temp_file_path)
            
        except Exception as e:
            print(f"❌ Error appending to error log: {e}")
    
    # ===============================
    # QUERY METHODS
    # ===============================
    
    def get_active_trades(self):
        """Get all currently active trades"""
        try:
            trades = self.db.collection('trades').where('status', '==', 'active').stream()
            return [{'id': trade.id, **trade.to_dict()} for trade in trades]
        except Exception as e:
            print(f"❌ Error getting active trades: {e}")
            return []
    
    def get_trade_by_position_id(self, position_id):
        """Get trade by position ID"""
        try:
            trades = self.db.collection('trades').where('position_id', '==', position_id).limit(1).stream()
            for trade in trades:
                return {'id': trade.id, **trade.to_dict()}
            return None
        except Exception as e:
            print(f"❌ Error getting trade by position ID: {e}")
            return None
    
    def get_daily_performance(self, date=None):
        """Get daily performance summary"""
        try:
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            doc = self.db.collection('account').document(f'daily_summary_{date}').get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            print(f"❌ Error getting daily performance: {e}")
            return None

    # === ENHANCED METHODS FOR 500 TRENDBAR DATA ===
    
    def save_complete_trade_package(self, trade_data: Dict, trendbar_data: List[Dict], analysis_data: Dict) -> str:
        """
        Save complete trade package with 500 trendbars and analysis
        
        Args:
            trade_data: Complete trade information
            trendbar_data: List of 500 trendbar dictionaries
            analysis_data: Market analysis results
            
        Returns:
            trade_id: Unique trade identifier
        """
        try:
            # 1. Generate unique trade ID with sequence
            base_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol_clean = trade_data['symbol'].replace('/', '_')
            
            # Get sequence number for this symbol today
            sequence = self._get_next_sequence_number(symbol_clean, base_timestamp[:8])
            trade_id = f"trade_{base_timestamp}_{symbol_clean}_{sequence:03d}"
            
            logger.info(f"💾 Saving complete trade package: {trade_id}")
            
            # 2. Prepare trade document
            trade_doc = {
                # Trade identification
                'trade_id': trade_id,
                'timestamp': datetime.now(),
                'symbol': trade_data['symbol'],
                'decision': trade_data.get('decision', 'BUY'),
                'status': 'pending',
                
                # Position details
                'position_id': trade_data.get('position_id'),
                'order_id': trade_data.get('order_id'),
                'volume_lots': trade_data.get('volume_lots', 0.0),
                'volume_units': trade_data.get('volume_units', 0),
                
                # Price levels
                'entry_price': trade_data.get('entry_price', 0.0),
                'stop_loss': trade_data.get('stop_loss', 0.0),
                'take_profit': trade_data.get('take_profit', 0.0),
                
                # Performance (to be updated on close)
                'exit_price': 0.0,
                'actual_pnl_usd': 0.0,
                'risk_reward_ratio': trade_data.get('risk_reward_ratio', 0.0),
                'risk_pips': trade_data.get('risk_pips', 0.0),
                'reward_pips': trade_data.get('reward_pips', 0.0),
                'is_winner': None,
                
                # Market context
                'trendbar_count': len(trendbar_data),
                'analysis_timeframe': 'M30',
                'market_session': analysis_data.get('market_session', 'unknown'),
                'volatility_level': analysis_data.get('volatility_level', 'medium'),
                
                # Strategy details
                'strategy_name': trade_data.get('strategy_name', ''),
                'zone_type': trade_data.get('zone_type', ''),
                'zone_high': trade_data.get('zone_high', 0.0),
                'zone_low': trade_data.get('zone_low', 0.0),
                'confidence_level': trade_data.get('confidence_level', 'medium'),
                'trade_reason': trade_data.get('trade_reason', ''),
                
                # File references
                'excel_file_path': f"reports/individual_trades/{symbol_clean}/{trade_id}.xlsx",
                'trendbar_data_path': f"market_data/trendbars/{symbol_clean}/{trade_id}.json",
                'session_id': self.session_id,
                'created_at': datetime.now()
            }
            
            # 3. Save trade document to Firestore
            self.db.collection('trades').document(trade_id).set(trade_doc)
            logger.info(f"✅ Trade document saved: {trade_id}")
            
            # 4. Save trendbar data separately (large dataset)
            trendbar_doc = {
                'trade_id': trade_id,
                'symbol': trade_data['symbol'],
                'timeframe': 'M30',
                'data_timestamp': datetime.now(),
                'trendbar_count': len(trendbar_data),
                'trendbars': trendbar_data,
                'market_analysis': analysis_data
            }
            self.db.collection('trendbar_data').document(trade_id).set(trendbar_doc)
            logger.info(f"📊 Trendbar data saved: {len(trendbar_data)} bars")
            
            # 5. Generate individual Excel file
            excel_path = self.create_individual_trade_excel(trade_id, trade_data, trendbar_data, analysis_data)
            
            # 6. Upload Excel to Firebase Storage
            storage_path = f"reports/individual_trades/{symbol_clean}/{trade_id}.xlsx"
            self.upload_file_to_storage(excel_path, storage_path)
            
            # 7. Update daily session summary
            self.update_session_summary(trade_id, trade_data)
            
            logger.info(f"🎯 Complete trade package saved successfully: {trade_id}")
            return trade_id
            
        except Exception as e:
            logger.error(f"❌ Error saving complete trade package: {e}")
            raise
    
    def create_individual_trade_excel(self, trade_id: str, trade_data: Dict, trendbar_data: List[Dict], analysis_data: Dict) -> str:
        """
        Create individual Excel file for a single trade with 500 trendbars
        
        Args:
            trade_id: Unique trade identifier
            trade_data: Trade information
            trendbar_data: List of 500 trendbar dictionaries
            analysis_data: Market analysis results
            
        Returns:
            excel_path: Path to created Excel file
        """
        try:
            filename = f"temp/{trade_id}.xlsx"
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet 1: Trade Details
                trade_details = {
                    'Trade ID': trade_id,
                    'Symbol': trade_data.get('symbol', ''),
                    'Entry Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Exit Time': 'Pending',
                    'Direction': trade_data.get('decision', 'BUY'),
                    'Position ID': trade_data.get('position_id', 'Pending'),
                    'Order ID': trade_data.get('order_id', 'Pending'),
                    'Entry Price': trade_data.get('entry_price', 0.0),
                    'Exit Price': 'Pending',
                    'Stop Loss': trade_data.get('stop_loss', 0.0),
                    'Take Profit': trade_data.get('take_profit', 0.0),
                    'Position Size': trade_data.get('volume_lots', 0.0),
                    'Risk Amount': '$50.00',  # Standard risk
                    'Actual P&L': 'Pending',
                    'R:R Ratio': trade_data.get('risk_reward_ratio', 0.0),
                    'Result': 'Pending',
                    'Strategy': trade_data.get('strategy_name', ''),
                    'Zone Type': trade_data.get('zone_type', ''),
                    'Zone High': trade_data.get('zone_high', 0.0),
                    'Zone Low': trade_data.get('zone_low', 0.0),
                    'Trade Reason': trade_data.get('trade_reason', ''),
                    'Market Session': analysis_data.get('market_session', 'unknown'),
                    'Volatility': analysis_data.get('volatility_level', 'medium'),
                    'Confidence Level': trade_data.get('confidence_level', 'medium')
                }
                
                trade_details_df = pd.DataFrame([trade_details])
                trade_details_df.to_excel(writer, sheet_name='Trade_Details', index=False)
                
                # Sheet 2: 500 Trendbar Data
                if trendbar_data:
                    trendbar_df = pd.DataFrame(trendbar_data)
                    # Ensure columns are in correct order
                    if 'timestamp' in trendbar_df.columns:
                        trendbar_df = trendbar_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    trendbar_df.to_excel(writer, sheet_name='Market_Data_500_Bars', index=False)
                
                # Sheet 3: Technical Analysis
                analysis_details = {
                    'Trend Direction': analysis_data.get('trend_direction', 'Unknown'),
                    'Support Levels': ', '.join(map(str, analysis_data.get('support_levels', []))),
                    'Resistance Levels': ', '.join(map(str, analysis_data.get('resistance_levels', []))),
                    'Zones Identified': analysis_data.get('key_zones_identified', 0),
                    'Market Session': analysis_data.get('market_session', 'unknown'),
                    'Volatility': analysis_data.get('volatility_level', 'medium'),
                    'Volume Profile': analysis_data.get('volume_profile', 'normal'),
                    'Signal Strength': analysis_data.get('signal_strength', 'medium'),
                    'Entry Confirmation': analysis_data.get('entry_confirmation', 'pending'),
                    'Risk Level': analysis_data.get('risk_level', 'medium')
                }
                
                analysis_df = pd.DataFrame([analysis_details])
                analysis_df.to_excel(writer, sheet_name='Technical_Analysis', index=False)
            
            logger.info(f"📄 Individual Excel file created: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Error creating individual Excel file: {e}")
            raise
    
    def update_trade_on_close(self, trade_id: str, exit_data: Dict) -> bool:
        """
        Update trade when position is closed
        
        Args:
            trade_id: Unique trade identifier
            exit_data: Exit information (price, P&L, etc.)
            
        Returns:
            success: True if update successful
        """
        try:
            # Update trade document
            trade_ref = self.db.collection('trades').document(trade_id)
            
            update_data = {
                'status': 'completed',
                'exit_price': exit_data.get('exit_price', 0.0),
                'actual_pnl_usd': exit_data.get('pnl_usd', 0.0),
                'is_winner': exit_data.get('pnl_usd', 0.0) > 0,
                'exit_time': datetime.now(),
                'completed_at': datetime.now()
            }
            
            trade_ref.update(update_data)
            
            # Update Excel file with final results
            self._update_excel_with_results(trade_id, exit_data)
            
            logger.info(f"🎯 Trade completed and updated: {trade_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating trade on close: {e}")
            return False
    
    def _get_next_sequence_number(self, symbol: str, date_str: str) -> int:
        """Get next sequence number for trades on this symbol/date"""
        try:
            # Query trades for this symbol and date
            trades_ref = self.db.collection('trades')
            query = trades_ref.where('symbol', '==', symbol.replace('_', '/')).where('timestamp', '>=', 
                                   datetime.strptime(date_str, '%Y%m%d')).where('timestamp', '<', 
                                   datetime.strptime(date_str, '%Y%m%d') + timedelta(days=1))
            
            trades = query.get()
            return len(trades) + 1
            
        except Exception as e:
            logger.error(f"❌ Error getting sequence number: {e}")
            return 1
    
    def _update_excel_with_results(self, trade_id: str, exit_data: Dict):
        """Update Excel file with final trade results"""
        try:
            # This would download the Excel, update it, and re-upload
            # For now, we'll create a new version with results
            logger.info(f"📊 Excel update queued for: {trade_id}")
            
        except Exception as e:
            logger.error(f"❌ Error updating Excel with results: {e}")
    
    def upload_file_to_storage(self, local_path: str, storage_path: str) -> bool:
        """Upload file to Firebase Storage"""
        try:
            blob = self.bucket.blob(storage_path)
            blob.upload_from_filename(local_path)
            
            # Make file publicly accessible (optional)
            # blob.make_public()
            
            logger.info(f"☁️ File uploaded to: {storage_path}")
            
            # Clean up local temp file
            if os.path.exists(local_path):
                os.remove(local_path)
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Error uploading file to storage: {e}")
            return False
    
    def update_session_summary(self, trade_id: str, trade_data: Dict):
        """Update daily session summary"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            session_ref = self.db.collection('sessions').document(f"session_{today}")
            
            # Use atomic transaction to update session
            session_ref.set({
                'date': today,
                'session_id': self.session_id,
                'last_updated': datetime.now(),
                'trades_count': firestore.Increment(1),
                'symbols_traded': firestore.ArrayUnion([trade_data['symbol']])
            }, merge=True)
            
            logger.info(f"📈 Session summary updated for: {today}")
            
        except Exception as e:
            logger.error(f"❌ Error updating session summary: {e}")

# Example integration with cTrader bot lifecycle
def example_ctrader_integration():
    """Example of how to integrate with cTrader bot"""
    
    firebase = FirebaseTrader()
    
    # 1. When order is sent (in sendOrderReq)
    pending_order = {
        "symbol": "EUR/USD",
        "volume": 15000,  # cTrader format
        "stop_loss": 1.08200,
        "take_profit": 1.08900,
        "decision": "BUY",
        "entry_price": 1.08457,
        "reason": "Strong demand zone rejection...",
        "risk_reward_ratio": "2.85",
        "potential_loss_usd": "$50.00",
        "potential_win_usd": "$142.50"
    }
    
    trade_doc_id = firebase.save_pending_trade(pending_order, order_id=87654321)
    
    # 2. When position is created (in onOrderSent)
    firebase.update_trade_with_position(
        position_id=12345678, 
        order_id=87654321, 
        actual_entry_price=1.08462
    )
    
    # 3. When SL/TP are set (in onAmendSent)
    firebase.update_trade_sl_tp_set(position_id=12345678)
    
    # 4. When position closes (in onPositionClosed)
    final_data = {
        'entry_price': 1.08462,
        'exit_price': 1.08612,
        'pnl': 142.50,
        'decision': 'BUY',
        'volume_lots': 0.15,
        'execution_time_ms': 245
    }
    
    firebase.complete_trade(position_id=12345678, final_data=final_data)

if __name__ == "__main__":
    example_ctrader_integration() 