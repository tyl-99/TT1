import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

class MarketConditionAnalyzer:
    """
    Analyzes market conditions to identify suitable/unsuitable trading environments
    for the EUR/USD Fast Momentum Strategy
    """
    
    def __init__(self):
        self.pip_size = 0.00001
        
    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive market condition analysis
        Returns detailed analysis of current market state
        """
        if len(df) < 50:
            return {'error': 'Insufficient data for analysis'}
        
        # Calculate basic indicators
        df = self._calculate_indicators(df)
        
        # Get current market state
        current = df.iloc[-1]
        
        analysis = {
            'timestamp': current.name,
            'current_price': current['close'],
            'market_condition': self._classify_market_condition(df),
            'volatility_state': self._analyze_volatility(df),
            'trend_state': self._analyze_trend(df),
            'momentum_state': self._analyze_momentum(df),
            'session_analysis': self._analyze_session(df),
            'risk_level': self._assess_risk_level(df),
            'trading_recommendation': self._get_trading_recommendation(df),
            'filters_passed': self._check_strategy_filters(df)
        }
        
        return analysis
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for analysis"""
        # EMA
        df['ema_fast'] = df['close'].ewm(span=8).mean()
        df['ema_slow'] = df['close'].ewm(span=21).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        df['tr'] = np.maximum(df['high'] - df['low'], 
                             np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                       abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(window=14).mean()
        df['atr_pips'] = df['atr'] / self.pip_size
        
        return df
    
    def _classify_market_condition(self, df: pd.DataFrame) -> str:
        """Classify overall market condition"""
        current = df.iloc[-1]
        recent = df.tail(20)
        
        # Volatility analysis
        current_atr = current['atr_pips']
        
        # Trend analysis
        ema_trend = current['ema_fast'] > current['ema_slow']
        ema_separation = abs(current['ema_fast'] - current['ema_slow']) / self.pip_size
        
        # Price range analysis
        recent_range = (recent['high'].max() - recent['low'].min()) / self.pip_size
        
        # RSI analysis
        rsi = current['rsi']
        
        # Classification logic
        if current_atr > 25:
            return "EXTREME_VOLATILITY"
        elif current_atr < 5:
            return "LOW_VOLATILITY"
        elif recent_range < 10:
            return "CONSOLIDATION"
        elif recent_range > 50:
            return "GAP_OR_EXTREME_MOVE"
        elif ema_separation < 1:
            return "WEAK_TREND"
        elif rsi < 20 or rsi > 80:
            return "EXTREME_MOMENTUM"
        elif ema_trend and ema_separation > 3:
            return "STRONG_TREND"
        else:
            return "NORMAL_MARKET"
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility conditions"""
        current = df.iloc[-1]
        recent = df.tail(20)
        
        avg_atr = recent['atr_pips'].mean()
        current_atr = current['atr_pips']
        
        return {
            'current_atr_pips': current_atr,
            'average_atr_pips': avg_atr,
            'volatility_state': 'HIGH' if current_atr > 15 else 'LOW' if current_atr < 8 else 'NORMAL',
            'volatility_trend': 'INCREASING' if current_atr > avg_atr * 1.2 else 'DECREASING' if current_atr < avg_atr * 0.8 else 'STABLE',
            'suitable_for_strategy': 5.5 <= current_atr <= 25.0
        }
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend conditions"""
        current = df.iloc[-1]
        recent = df.tail(10)
        
        ema_trend = current['ema_fast'] > current['ema_slow']
        ema_separation = abs(current['ema_fast'] - current['ema_slow']) / self.pip_size
        
        # Trend strength
        trend_strength = 'STRONG' if ema_separation > 3 else 'WEAK' if ema_separation < 1 else 'MODERATE'
        
        # Trend consistency
        trend_changes = 0
        for i in range(1, len(recent)):
            prev_trend = recent.iloc[i-1]['ema_fast'] > recent.iloc[i-1]['ema_slow']
            curr_trend = recent.iloc[i]['ema_fast'] > recent.iloc[i]['ema_slow']
            if prev_trend != curr_trend:
                trend_changes += 1
        
        trend_consistency = 'STABLE' if trend_changes <= 1 else 'CHOPPY' if trend_changes <= 3 else 'UNSTABLE'
        
        return {
            'direction': 'BULLISH' if ema_trend else 'BEARISH',
            'strength': trend_strength,
            'consistency': trend_consistency,
            'ema_separation_pips': ema_separation,
            'trend_changes_recent': trend_changes,
            'suitable_for_strategy': ema_separation >= 1.0 and trend_changes <= 3
        }
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum conditions"""
        current = df.iloc[-1]
        recent = df.tail(10)
        
        rsi = current['rsi']
        rsi_trend = 'RISING' if current['rsi'] > recent.iloc[-2]['rsi'] else 'FALLING'
        
        # Check for extreme conditions
        extreme_oversold = rsi < 20
        extreme_overbought = rsi > 80
        moderate_oversold = 30 <= rsi <= 40
        moderate_overbought = 60 <= rsi <= 70
        
        return {
            'current_rsi': rsi,
            'rsi_trend': rsi_trend,
            'momentum_state': 'EXTREME_OVERSOLD' if extreme_oversold else 'EXTREME_OVERBOUGHT' if extreme_overbought else 'MODERATE',
            'reversal_opportunity': moderate_oversold or moderate_overbought,
            'suitable_for_strategy': 15 <= rsi <= 85 and (moderate_oversold or moderate_overbought)
        }
    
    def _analyze_session(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading session conditions"""
        current_time = df.index[-1]
        hour = current_time.hour
        
        # Session classification
        if 2 <= hour < 10:
            session = 'ASIAN'
            volatility_expected = 'LOW'
        elif 8 <= hour < 16:
            session = 'LONDON'
            volatility_expected = 'HIGH'
        elif 13 <= hour < 21:
            session = 'NEW_YORK'
            volatility_expected = 'HIGH'
        elif 20 <= hour < 2:
            session = 'LATE_NY'
            volatility_expected = 'LOW'
        else:
            session = 'OVERNIGHT'
            volatility_expected = 'LOW'
        
        # Check if it's overlap period
        is_overlap = (8 <= hour < 12) or (13 <= hour < 17)
        
        return {
            'current_session': session,
            'expected_volatility': volatility_expected,
            'is_overlap': is_overlap,
            'suitable_for_strategy': session in ['LONDON', 'NEW_YORK'] or is_overlap
        }
    
    def _assess_risk_level(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall market risk level"""
        current = df.iloc[-1]
        recent = df.tail(20)
        
        risk_factors = []
        risk_score = 0
        
        # Volatility risk
        if current['atr_pips'] > 20:
            risk_factors.append('HIGH_VOLATILITY')
            risk_score += 3
        elif current['atr_pips'] < 6:
            risk_factors.append('LOW_VOLATILITY')
            risk_score += 1
        
        # Trend risk
        ema_separation = abs(current['ema_fast'] - current['ema_slow']) / self.pip_size
        if ema_separation < 0.5:
            risk_factors.append('WEAK_TREND')
            risk_score += 2
        
        # Momentum risk
        if current['rsi'] < 15 or current['rsi'] > 85:
            risk_factors.append('EXTREME_MOMENTUM')
            risk_score += 3
        
        # Price range risk
        recent_range = (recent['high'].max() - recent['low'].min()) / self.pip_size
        if recent_range > 40:
            risk_factors.append('LARGE_PRICE_RANGE')
            risk_score += 2
        
        # Risk level classification
        if risk_score >= 6:
            risk_level = 'HIGH'
        elif risk_score >= 3:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'suitable_for_strategy': risk_level in ['LOW', 'MEDIUM']
        }
    
    def _get_trading_recommendation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get trading recommendation based on market conditions"""
        analysis = {
            'volatility': self._analyze_volatility(df),
            'trend': self._analyze_trend(df),
            'momentum': self._analyze_momentum(df),
            'session': self._analyze_session(df),
            'risk': self._assess_risk_level(df)
        }
        
        # Decision logic
        suitable_conditions = 0
        total_conditions = 5
        
        if analysis['volatility']['suitable_for_strategy']:
            suitable_conditions += 1
        if analysis['trend']['suitable_for_strategy']:
            suitable_conditions += 1
        if analysis['momentum']['suitable_for_strategy']:
            suitable_conditions += 1
        if analysis['session']['suitable_for_strategy']:
            suitable_conditions += 1
        if analysis['risk']['suitable_for_strategy']:
            suitable_conditions += 1
        
        suitability_percentage = (suitable_conditions / total_conditions) * 100
        
        if suitability_percentage >= 80:
            recommendation = 'STRONG_BUY'
            confidence = 'HIGH'
        elif suitability_percentage >= 60:
            recommendation = 'BUY'
            confidence = 'MEDIUM'
        elif suitability_percentage >= 40:
            recommendation = 'HOLD'
            confidence = 'LOW'
        else:
            recommendation = 'AVOID'
            confidence = 'HIGH'
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'suitability_percentage': suitability_percentage,
            'suitable_conditions': suitable_conditions,
            'total_conditions': total_conditions,
            'reasoning': self._get_recommendation_reasoning(analysis)
        }
    
    def _get_recommendation_reasoning(self, analysis: Dict) -> str:
        """Get detailed reasoning for trading recommendation"""
        reasons = []
        
        if not analysis['volatility']['suitable_for_strategy']:
            reasons.append("Volatility outside optimal range")
        if not analysis['trend']['suitable_for_strategy']:
            reasons.append("Weak or choppy trend")
        if not analysis['momentum']['suitable_for_strategy']:
            reasons.append("Extreme momentum conditions")
        if not analysis['session']['suitable_for_strategy']:
            reasons.append("Low activity session")
        if not analysis['risk']['suitable_for_strategy']:
            reasons.append("High risk market conditions")
        
        if not reasons:
            return "All market conditions are suitable for the strategy"
        else:
            return f"Market conditions not optimal: {', '.join(reasons)}"
    
    def _check_strategy_filters(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check which strategy filters would pass"""
        current = df.iloc[-1]
        recent = df.tail(20)
        
        # Filter 1: EMA trend
        ema_trend = current['ema_fast'] > current['ema_slow']
        ema_separation = abs(current['ema_fast'] - current['ema_slow']) / self.pip_size
        filter_1 = ema_trend and ema_separation >= 1.0
        
        # Filter 2: RSI conditions (simplified check)
        rsi = current['rsi']
        filter_2 = 15 <= rsi <= 85
        
        # Filter 3: ATR volatility
        atr = current['atr_pips']
        filter_3 = 5.5 <= atr <= 25.0
        
        # Filter 4: Market conditions
        recent_range = (recent['high'].max() - recent['low'].min()) / self.pip_size
        filter_4 = 8 <= recent_range <= 50
        
        return {
            'ema_trend_filter': filter_1,
            'rsi_filter': filter_2,
            'atr_filter': filter_3,
            'market_condition_filter': filter_4,
            'all_filters_passed': filter_1 and filter_2 and filter_3 and filter_4
        }
    
    def print_market_analysis(self, analysis: Dict[str, Any]):
        """Print formatted market analysis"""
        print("\n" + "="*60)
        print("📊 MARKET CONDITION ANALYSIS")
        print("="*60)
        
        print(f"🕐 Timestamp: {analysis['timestamp']}")
        print(f"💰 Current Price: {analysis['current_price']:.5f}")
        print(f"🎯 Market Condition: {analysis['market_condition']}")
        
        print(f"\n📈 VOLATILITY ANALYSIS:")
        vol = analysis['volatility_state']
        print(f"   Current ATR: {vol['current_atr_pips']:.1f} pips")
        print(f"   Average ATR: {vol['average_atr_pips']:.1f} pips")
        print(f"   State: {vol['volatility_state']} ({vol['volatility_trend']})")
        print(f"   Suitable: {'✅' if vol['suitable_for_strategy'] else '❌'}")
        
        print(f"\n📊 TREND ANALYSIS:")
        trend = analysis['trend_state']
        print(f"   Direction: {trend['direction']}")
        print(f"   Strength: {trend['strength']}")
        print(f"   Consistency: {trend['consistency']}")
        print(f"   EMA Separation: {trend['ema_separation_pips']:.1f} pips")
        print(f"   Suitable: {'✅' if trend['suitable_for_strategy'] else '❌'}")
        
        print(f"\n⚡ MOMENTUM ANALYSIS:")
        mom = analysis['momentum_state']
        print(f"   RSI: {mom['current_rsi']:.1f} ({mom['rsi_trend']})")
        print(f"   State: {mom['momentum_state']}")
        print(f"   Reversal Opportunity: {'✅' if mom['reversal_opportunity'] else '❌'}")
        print(f"   Suitable: {'✅' if mom['suitable_for_strategy'] else '❌'}")
        
        print(f"\n🌍 SESSION ANALYSIS:")
        session = analysis['session_analysis']
        print(f"   Current Session: {session['current_session']}")
        print(f"   Expected Volatility: {session['expected_volatility']}")
        print(f"   Overlap Period: {'✅' if session['is_overlap'] else '❌'}")
        print(f"   Suitable: {'✅' if session['suitable_for_strategy'] else '❌'}")
        
        print(f"\n⚠️  RISK ASSESSMENT:")
        risk = analysis['risk_level']
        print(f"   Risk Level: {risk['risk_level']} (Score: {risk['risk_score']})")
        print(f"   Risk Factors: {', '.join(risk['risk_factors']) if risk['risk_factors'] else 'None'}")
        print(f"   Suitable: {'✅' if risk['suitable_for_strategy'] else '❌'}")
        
        print(f"\n🎯 TRADING RECOMMENDATION:")
        rec = analysis['trading_recommendation']
        print(f"   Recommendation: {rec['recommendation']}")
        print(f"   Confidence: {rec['confidence']}")
        print(f"   Suitability: {rec['suitability_percentage']:.1f}%")
        print(f"   Reasoning: {rec['reasoning']}")
        
        print(f"\n🔍 STRATEGY FILTERS:")
        filters = analysis['filters_passed']
        print(f"   EMA Trend: {'✅' if filters['ema_trend_filter'] else '❌'}")
        print(f"   RSI: {'✅' if filters['rsi_filter'] else '❌'}")
        print(f"   ATR: {'✅' if filters['atr_filter'] else '❌'}")
        print(f"   Market Conditions: {'✅' if filters['market_condition_filter'] else '❌'}")
        print(f"   All Filters Passed: {'✅' if filters['all_filters_passed'] else '❌'}")
        
        print("="*60) 