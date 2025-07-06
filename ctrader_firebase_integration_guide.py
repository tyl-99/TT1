# 🔥 cTrader.py Firebase Integration Guide
# How to integrate the enhanced Firebase structure with 300 trendbar data

"""
INTEGRATION STEPS:
1. Add Firebase import and initialization
2. Modify onTrendbarDataReceived() to store trendbar data  
3. Enhance analyze_with_our_strategy() to capture analysis data
4. Update sendOrderReq() to save complete trade package
5. Modify onOrderSent() to link position ID
6. Update onPositionClosed() to complete trade record
"""

# =====================================
# 1. ADD FIREBASE IMPORT (top of ctrader.py)
# =====================================

from firebase_trader import FirebaseTrader
import logging

# Initialize Firebase trader
firebase_trader = FirebaseTrader()
logger = logging.getLogger(__name__)

# =====================================  
# 2. MODIFY onTrendbarDataReceived() METHOD
# =====================================

def onTrendbarDataReceived(self, response):
    """Enhanced trendbar data processing with Firebase storage"""
    print("Trendbar data received:")
    
    # Reset API retry count on successful response
    self.reset_api_retry_state()
    
    try:
        parsed = Protobuf.extract(response)
        trendbars = parsed.trendbar  # This is a list of trendbar objects
        
        if not trendbars:
            logger.warning(f"⚠️ No trendbar data received for {self.current_pair}")
            self.move_to_next_pair()
            return
        
        # Convert trendbars to DataFrame
        data = []
        for tb in trendbars:
            data.append({
                'timestamp': datetime.datetime.utcfromtimestamp(tb.utcTimestampInMinutes * 60),
                'open': (tb.low + tb.deltaOpen) / 1e5,
                'high': (tb.low + tb.deltaHigh) / 1e5,
                'low': tb.low / 1e5,
                'close': (tb.low + tb.deltaClose) / 1e5,
                'volume': tb.volume
            })
        
        df = pd.DataFrame(data)
        df.sort_values('timestamp', inplace=True, ascending=False)
        
        if self.trendbar.empty:
            # First call - store M30 data as base
            df['timestamp'] = df['timestamp'].astype(str)
            self.trendbar = df
        
        self.trendbar.sort_values('timestamp', inplace=True, ascending=True)
        print(f"\n📊 {self.current_pair} - Trendbar data after sorting (showing last 5 rows):")
        print(self.trendbar.tail().to_string())
        
        # 🔥 NEW: Store trendbar data for Firebase (300 bars)
        self.current_trendbar_data = self.trendbar.to_dict('records')
        logger.info(f"📊 Stored {len(self.current_trendbar_data)} trendbar records for {self.current_pair}")
        
        # Continue with strategy analysis
        self.analyze_with_our_strategy()
        
    except Exception as e:
        logger.error(f"❌ Error processing trendbar data for {self.current_pair}: {e}")
        self.move_to_next_pair()

# =====================================
# 3. ENHANCE analyze_with_our_strategy() METHOD  
# =====================================

def analyze_with_our_strategy(self):
    """Analyze market data using our optimized supply/demand strategies"""
    try:
        # Get the appropriate strategy for this pair
        strategy = self.strategies.get(self.current_pair)
        if not strategy:
            logger.warning(f"No strategy found for {self.current_pair}")
            self.move_to_next_pair()
            return
        
        # Analyze the market data
        signal = strategy.analyze_trade_signal(self.trendbar, self.current_pair)
        
        logger.info(f"\n=== Strategy Decision for {self.current_pair} ===")
        logger.info(f"Decision: {signal.get('decision')}")
        
        # 🔥 NEW: Capture market analysis data for Firebase
        self.current_analysis_data = {
            'trend_direction': signal.get('trend_direction', 'unknown'),
            'support_levels': signal.get('support_levels', []),
            'resistance_levels': signal.get('resistance_levels', []),
            'key_zones_identified': signal.get('zones_count', 0),
            'market_session': self.get_market_session(),
            'volatility_level': self.get_volatility_level(),
            'volume_profile': signal.get('volume_profile', 'normal'),
            'signal_strength': signal.get('confidence', 'medium'),
            'entry_confirmation': signal.get('entry_confirmation', 'pending'),
            'risk_level': self.calculate_risk_level(signal)
        }
        
        if signal.get("decision") == "NO TRADE":
            logger.info(f"No trade signal for {self.current_pair}")
            self.move_to_next_pair()
        else:
            # CENTRALIZED R:R FILTER - Check R:R ratio before executing trade
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            
            # Calculate R:R using direct price distances
            risk_distance = abs(entry_price - stop_loss)
            reward_distance = abs(take_profit - entry_price)
            
            rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
            
            # Still need pip calculations for minimum stop loss check
            pip_size = 0.01 if 'JPY' in self.current_pair else 0.0001
            risk_pips = risk_distance / pip_size
            
            # MINIMUM STOP LOSS FILTER - Check if stop loss is at least 5 pips
            if risk_pips < 5.0:
                logger.info(f"❌ Trade REJECTED for {self.current_pair}: Stop loss {risk_pips:.1f} pips < 5 pips minimum")
                print(f"⚠️ {self.current_pair}: Stop loss {risk_pips:.1f} pips too tight, minimum required: 5 pips")
                self.move_to_next_pair()
                return
            
            if rr_ratio < self.min_rr_ratio:
                reward_pips = reward_distance / pip_size
                logger.info(f"❌ Trade REJECTED for {self.current_pair}: R:R {rr_ratio:.2f} < {self.min_rr_ratio}")
                logger.info(f"   Risk: {risk_pips:.1f} pips | Reward: {reward_pips:.1f} pips")
                print(f"⚠️ {self.current_pair}: R:R {rr_ratio:.2f} too low, minimum required: {self.min_rr_ratio}")
                self.move_to_next_pair()
                return
                
            # Trade passes all filters - proceed with execution
            logger.info(f"✅ Trade APPROVED for {self.current_pair}: R:R {rr_ratio:.2f} | Risk: {risk_pips:.1f} pips")
            print(f"🎯 {self.current_pair}: Executing trade with R:R {rr_ratio:.2f}")
            
            # Format and send the trade
            formatted_data = self.format_trade_data(signal)
            self.sendOrderReq(self.current_pair, formatted_data)
            
    except Exception as e:
        logger.error(f"❌ Error in strategy analysis for {self.current_pair}: {e}")
        self.move_to_next_pair()

# =====================================
# 4. UPDATE sendOrderReq() METHOD
# =====================================

def sendOrderReq(self, symbol, trade_data):
    """Enhanced order request with Firebase integration"""
    try:
        print(f"\n🚀 Sending Order Request for {symbol}")
        
        # Extract data from the trade_data object
        order_id = trade_data.order_id
        symbol_name = trade_data.symbol_name
        order_type = trade_data.order_type
        trade_side = trade_data.trade_side
        volume = trade_data.volume
        limit_price = trade_data.limit_price
        stop_price = trade_data.stop_price
        
        print(f"📋 Order Details:")
        print(f"   Symbol: {symbol_name}")
        print(f"   Side: {trade_side}")
        print(f"   Volume: {volume}")
        print(f"   Entry: {limit_price}")
        print(f"   Stop Loss: {stop_price}")
        
        # 🔥 NEW: Prepare complete trade data with 300 trendbars
        complete_trade_data = {
            'symbol': symbol_name,
            'decision': trade_side,
            'volume_lots': volume / 100000,  # Convert to lots
            'volume_units': volume,
            'entry_price': limit_price,
            'stop_loss': stop_price,
            'take_profit': trade_data.take_profit_price,
            'risk_reward_ratio': trade_data.risk_reward_ratio,
            'risk_pips': trade_data.risk_pips,
            'reward_pips': trade_data.reward_pips,
            'strategy_name': f"{symbol.replace('/', '')}SupplyDemandStrategy",
            'zone_type': trade_data.zone_type,
            'zone_high': trade_data.zone_high,
            'zone_low': trade_data.zone_low,
            'confidence_level': trade_data.confidence_level,
            'trade_reason': trade_data.trade_reason,
            'order_id': order_id
        }
        
        # 🔥 NEW: Save complete trade package to Firebase
        try:
            trade_id = firebase_trader.save_complete_trade_package(
                trade_data=complete_trade_data,
                trendbar_data=self.current_trendbar_data,
                analysis_data=self.current_analysis_data
            )
            logger.info(f"🔥 Complete trade package saved: {trade_id}")
            
            # Store trade_id for later reference
            self.current_trade_id = trade_id
            
        except Exception as firebase_error:
            logger.error(f"❌ Firebase save error: {firebase_error}")
            # Continue with trade execution even if Firebase fails
        
        # Create and send the actual order request
        msg = ProtoOANewOrderReq()
        msg.ctidTraderAccountId = self.trader_account_id
        msg.symbolId = symbol
        msg.orderType = order_type
        msg.tradeSide = trade_side
        msg.volume = volume
        msg.timeInForce = ProtoOATimeInForce.GOOD_TILL_CANCEL
        
        if limit_price is not None:
            msg.limitPrice = limit_price
        if stop_price is not None:
            msg.stopPrice = stop_price
            
        # Store pending order details
        self.pending_order = trade_data
        
        serialized_msg = msg.SerializeToString()
        self.send_message(serialized_msg, ProtoOAPayloadType.PROTO_OA_NEW_ORDER_REQ)
        
        logger.info(f"📤 Order request sent for {symbol_name}")
        
    except Exception as e:
        logger.error(f"❌ Error sending order request: {e}")

# =====================================
# 5. UPDATE onOrderSent() METHOD
# =====================================

def onOrderSent(self, response):
    """Enhanced order sent handler with Firebase position linking"""
    try:
        parsed = Protobuf.extract(response)
        order = parsed.order
        
        print(f"\n✅ Order Sent Successfully!")
        print(f"   Order ID: {order.orderId}")
        print(f"   Symbol: {order.symbolId}")
        print(f"   Volume: {order.requestedVolume}")
        print(f"   Status: {order.orderStatus}")
        
        # 🔥 NEW: Update Firebase trade with order details
        if hasattr(self, 'current_trade_id') and self.current_trade_id:
            try:
                # Update trade document with order ID
                trade_ref = firebase_trader.db.collection('trades').document(self.current_trade_id)
                trade_ref.update({
                    'order_id': order.orderId,
                    'order_status': str(order.orderStatus),
                    'order_sent_time': datetime.now(),
                    'status': 'order_sent'
                })
                logger.info(f"🔥 Trade updated with order ID: {order.orderId}")
                
            except Exception as firebase_error:
                logger.error(f"❌ Firebase update error: {firebase_error}")
        
        # Continue with existing logic...
        if order.orderStatus == ProtoOAOrderStatus.ORDER_STATUS_FILLED:
            print("🎯 Order filled immediately!")
            self.handle_filled_order(order)
        else:
            print(f"⏳ Order pending: {order.orderStatus}")
            
        # Move to next pair after successful order
        self.move_to_next_pair()
        
    except Exception as e:
        logger.error(f"❌ Error processing order sent response: {e}")
        self.move_to_next_pair()

# =====================================
# 6. UPDATE onPositionClosed() METHOD
# =====================================

def onPositionClosed(self, response):
    """Enhanced position closed handler with Firebase completion"""
    try:
        parsed = Protobuf.extract(response)
        position = parsed.position
        
        print(f"\n🏁 Position Closed!")
        print(f"   Position ID: {position.positionId}")
        print(f"   Symbol: {position.symbolId}")
        print(f"   Volume: {position.volume}")
        print(f"   P&L: {position.moneyGross}")
        
        # 🔥 NEW: Complete Firebase trade record
        if hasattr(self, 'current_trade_id') and self.current_trade_id:
            try:
                exit_data = {
                    'position_id': position.positionId,
                    'exit_price': position.price,
                    'pnl_usd': position.moneyGross,
                    'exit_time': datetime.now(),
                    'position_status': str(position.positionStatus)
                }
                
                firebase_trader.update_trade_on_close(self.current_trade_id, exit_data)
                logger.info(f"🔥 Trade completed in Firebase: {self.current_trade_id}")
                
                # Clear current trade ID
                self.current_trade_id = None
                
            except Exception as firebase_error:
                logger.error(f"❌ Firebase completion error: {firebase_error}")
        
        # Continue with existing notification logic...
        self.send_pushover_notification()
        
    except Exception as e:
        logger.error(f"❌ Error processing position closed: {e}")

# =====================================
# 7. HELPER METHODS FOR ANALYSIS DATA
# =====================================

def get_market_session(self):
    """Determine current market session"""
    current_hour = datetime.now().hour
    
    if 7 <= current_hour < 16:
        return "london_session"
    elif 13 <= current_hour < 22:
        return "new_york_session"
    elif 21 <= current_hour or current_hour < 6:
        return "sydney_session"
    else:
        return "overlap_session"

def get_volatility_level(self):
    """Calculate volatility level from recent price action"""
    if hasattr(self, 'trendbar') and not self.trendbar.empty:
        recent_bars = self.trendbar.tail(20)
        if not recent_bars.empty:
            volatility = recent_bars['high'].std() + recent_bars['low'].std()
            
            if volatility > 0.002:
                return "high"
            elif volatility > 0.001:
                return "medium"
            else:
                return "low"
    
    return "medium"  # Default

def calculate_risk_level(self, signal):
    """Calculate risk level based on market conditions"""
    risk_factors = 0
    
    # Check various risk factors
    if signal.get('confidence', 'medium') == 'low':
        risk_factors += 1
    if self.get_volatility_level() == 'high':
        risk_factors += 1
    if len(signal.get('support_levels', [])) < 2:
        risk_factors += 1
        
    if risk_factors >= 2:
        return "high"
    elif risk_factors == 1:
        return "medium"
    else:
        return "low"

# =====================================
# 8. CLASS INITIALIZATION UPDATES
# =====================================

class Trader:
    def __init__(self):
        # ... existing initialization ...
        
        # 🔥 NEW: Initialize Firebase and data storage
        self.firebase_trader = FirebaseTrader()
        self.current_trendbar_data = []
        self.current_analysis_data = {}
        self.current_trade_id = None
        
        logger.info("🔥 cTrader bot initialized with Firebase integration")

# =====================================
# 9. USAGE EXAMPLE  
# =====================================

"""
COMPLETE WORKFLOW:

1. Bot receives 300 trendbar data points
   → onTrendbarDataReceived() stores self.current_trendbar_data

2. Strategy analyzes market 
   → analyze_with_our_strategy() stores self.current_analysis_data

3. Trade signal generated and approved
   → sendOrderReq() calls firebase_trader.save_complete_trade_package()
   → Creates individual Excel file with 300 bars
   → Uploads to Firebase Storage
   → Saves to Firestore collections: 'trades' and 'trendbar_data'

4. Order confirmed by broker
   → onOrderSent() updates trade with order_id and position_id

5. Position closed
   → onPositionClosed() calls firebase_trader.update_trade_on_close()
   → Updates final P&L and results

RESULT: Complete trade documentation with full market context! 🚀
""" 