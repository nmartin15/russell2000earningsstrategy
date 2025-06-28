from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

class StrategyType(Enum):
    SHORT_VERTICAL = "short_vertical"
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"
    LONG_BUTTERFLY = "long_butterfly"

class OptionsManager:
    """
    Options Strategy Manager for Russell 2000 Earnings Strategy
    
    Implements:
    1. Short Vertical Spreads (Bear Call Spreads)
    2. Broken Wing Butterfly Spreads
    3. Dynamic position management and adjustments
    4. Risk management and profit taking
    """
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        
        # Strategy parameters
        self.days_to_expiration_min = 30
        self.days_to_expiration_max = 60
        self.target_dte = 45
        
        # Greeks targets
        self.target_delta = 0.20  # Target delta for short strikes
        self.max_gamma_exposure = 0.05
        self.max_vega_exposure = 0.10
        
        # Profit/Loss management
        self.profit_target = 0.50  # Close at 50% of max profit
        self.stop_loss_threshold = 0.25  # Stop loss at 25% of max risk
        self.time_decay_close = 7  # Close 7 days before expiration
        
        # Position sizing
        self.max_risk_per_position = 0.05  # 5% max risk per position
        self.min_premium_threshold = 0.10  # Minimum $0.10 premium
        
        # Active positions tracking
        self.positions = {}
        self.position_history = []
        
        # Options data cache
        self.options_cache = defaultdict(dict)
        
    def get_options_filter(self):
        """
        Return options filter function for QuantConnect
        
        Returns:
            function: Options filter function
        """
        def options_filter(universe):
            return (universe
                    .strikes(-15, 15)  # 15 strikes above and below ATM
                    .expiration(self.days_to_expiration_min, self.days_to_expiration_max)
                    .include_weeklys()
                    .contracts_limit(50))  # Limit contracts for performance
        
        return options_filter
    
    def evaluate_strategy_for_symbol(self, symbol, sue_score):
        """
        Evaluate which options strategy to use for a symbol based on SUE score and market conditions
        
        Args:
            symbol: Stock symbol
            sue_score: Standardized Unexpected Earnings score
            
        Returns:
            StrategyType: Recommended strategy type
        """
        if symbol not in self.algorithm.securities:
            return None
        
        security = self.algorithm.securities[symbol]
        current_price = security.price
        
        # Get implied volatility if available
        iv_rank = self._get_iv_rank(symbol)
        
        # Strategy selection logic based on SUE score and market conditions
        if sue_score > 2.0:  # Very strong earnings surprise
            if iv_rank and iv_rank > 0.7:  # High IV environment
                return StrategyType.SHORT_VERTICAL  # Sell premium
            else:
                return StrategyType.BROKEN_WING_BUTTERFLY  # Limited risk, asymmetric payoff
        elif sue_score > 1.0:  # Strong earnings surprise
            return StrategyType.SHORT_VERTICAL  # Conservative premium selling
        else:  # Moderate earnings surprise
            return StrategyType.BROKEN_WING_BUTTERFLY  # Balanced risk/reward
    
    def open_position(self, symbol, strategy_type, sue_score):
        """
        Open an options position for a symbol
        
        Args:
            symbol: Stock symbol
            strategy_type: Type of options strategy to implement
            sue_score: SUE score for position sizing
            
        Returns:
            bool: True if position opened successfully
        """
        try:
            if symbol in self.positions:
                self.algorithm.debug(f"Position already exists for {symbol}")
                return False
            
            # Get options chain
            option_chain = self.algorithm.option_chain(symbol)
            if not option_chain or symbol not in option_chain:
                return False
            
            chain = option_chain[symbol]
            
            if strategy_type == StrategyType.SHORT_VERTICAL:
                return self._open_short_vertical(symbol, chain, sue_score)
            elif strategy_type == StrategyType.BROKEN_WING_BUTTERFLY:
                return self._open_broken_wing_butterfly(symbol, chain, sue_score)
            else:
                return False
                
        except Exception as e:
            self.algorithm.debug(f"Error opening position for {symbol}: {e}")
            return False
    
    def _open_short_vertical(self, symbol, chain, sue_score):
        """
        Open a short vertical spread (bear call spread) position
        
        Args:
            symbol: Stock symbol
            chain: Options chain
            sue_score: SUE score
            
        Returns:
            bool: Success status
        """
        try:
            current_price = self.algorithm.securities[symbol].price
            
            # Select expiration closest to target DTE
            expiration = self._select_optimal_expiration(chain)
            if not expiration:
                return False
            
            # Filter calls by expiration
            calls = [x for x in chain if x.expiry == expiration and x.right == OptionRight.CALL]
            calls.sort(key=lambda x: x.strike)
            
            if len(calls) < 2:
                return False
            
            # Find strikes for the spread
            short_strike, long_strike = self._find_vertical_strikes(calls, current_price)
            
            if not short_strike or not long_strike:
                return False
            
            # Validate spread economics
            short_call = next(x for x in calls if x.strike == short_strike)
            long_call = next(x for x in calls if x.strike == long_strike)
            
            if not self._validate_spread_prices(short_call, long_call):
                return False
            
            # Calculate position size
            net_credit = short_call.bid_price - long_call.ask_price
            max_risk = long_strike - short_strike - net_credit
            
            if max_risk <= 0 or net_credit <= self.min_premium_threshold:
                return False
            
            contracts = self._calculate_position_size(max_risk, sue_score)
            
            if contracts <= 0:
                return False
            
            # Place orders
            short_ticket = self.algorithm.sell(short_call.symbol, contracts)
            long_ticket = self.algorithm.buy(long_call.symbol, contracts)
            
            # Store position
            position = {
                'symbol': symbol,
                'strategy_type': StrategyType.SHORT_VERTICAL,
                'short_call': short_call.symbol,
                'long_call': long_call.symbol,
                'contracts': contracts,
                'entry_date': self.algorithm.time,
                'expiration': expiration,
                'entry_credit': net_credit,
                'max_risk': max_risk,
                'short_strike': short_strike,
                'long_strike': long_strike,
                'sue_score': sue_score,
                'short_ticket': short_ticket,
                'long_ticket': long_ticket,
                'status': 'open'
            }
            
            self.positions[symbol] = position
            
            self.algorithm.debug(f"Opened short vertical for {symbol}: "
                               f"Sold ${short_strike} call, Bought ${long_strike} call, "
                               f"Credit: ${net_credit:.2f}, Contracts: {contracts}")
            
            return True
            
        except Exception as e:
            self.algorithm.debug(f"Error opening short vertical for {symbol}: {e}")
            return False
    
    def _open_broken_wing_butterfly(self, symbol, chain, sue_score):
        """
        Open a broken wing butterfly spread position
        
        Args:
            symbol: Stock symbol
            chain: Options chain
            sue_score: SUE score
            
        Returns:
            bool: Success status
        """
        try:
            current_price = self.algorithm.securities[symbol].price
            
            # Select expiration
            expiration = self._select_optimal_expiration(chain)
            if not expiration:
                return False
            
            # Filter calls by expiration
            calls = [x for x in chain if x.expiry == expiration and x.right == OptionRight.CALL]
            calls.sort(key=lambda x: x.strike)
            
            if len(calls) < 4:
                return False
            
            # Find strikes for broken wing butterfly
            strikes = self._find_butterfly_strikes(calls, current_price, broken_wing=True)
            
            if len(strikes) != 3:
                return False
            
            lower_strike, middle_strike, upper_strike = strikes
            
            # Get option contracts
            lower_call = next(x for x in calls if x.strike == lower_strike)
            middle_calls = [x for x in calls if x.strike == middle_strike]
            upper_call = next(x for x in calls if x.strike == upper_strike)
            
            if len(middle_calls) < 2:
                return False
            
            middle_call = middle_calls[0]  # We'll trade 2 contracts of this strike
            
            # Validate pricing
            if (lower_call.ask_price <= 0 or middle_call.bid_price <= 0 or 
                upper_call.ask_price <= 0):
                return False
            
            # Calculate net debit/credit
            net_cost = (lower_call.ask_price + upper_call.ask_price - 
                       2 * middle_call.bid_price)
            
            # For broken wing, we want a small debit or credit
            if abs(net_cost) > 0.50:  # Avoid expensive butterflies
                return False
            
            # Calculate max risk and reward
            wing_width_lower = middle_strike - lower_strike
            wing_width_upper = upper_strike - middle_strike
            
            if net_cost > 0:  # Net debit
                max_risk = net_cost
                max_reward = min(wing_width_lower, wing_width_upper) - net_cost
            else:  # Net credit
                max_risk = max(wing_width_lower, wing_width_upper) - abs(net_cost)
                max_reward = abs(net_cost)
            
            if max_risk <= 0 or max_reward <= 0:
                return False
            
            # Calculate position size
            contracts = self._calculate_position_size(max_risk, sue_score)
            
            if contracts <= 0:
                return False
            
            # Place orders for butterfly
            buy_lower_ticket = self.algorithm.buy(lower_call.symbol, contracts)
            sell_middle_ticket = self.algorithm.sell(middle_call.symbol, 2 * contracts)
            buy_upper_ticket = self.algorithm.buy(upper_call.symbol, contracts)
            
            # Store position
            position = {
                'symbol': symbol,
                'strategy_type': StrategyType.BROKEN_WING_BUTTERFLY,
                'lower_call': lower_call.symbol,
                'middle_call': middle_call.symbol,
                'upper_call': upper_call.symbol,
                'contracts': contracts,
                'entry_date': self.algorithm.time,
                'expiration': expiration,
                'entry_cost': net_cost,
                'max_risk': max_risk,
                'max_reward': max_reward,
                'lower_strike': lower_strike,
                'middle_strike': middle_strike,
                'upper_strike': upper_strike,
                'sue_score': sue_score,
                'buy_lower_ticket': buy_lower_ticket,
                'sell_middle_ticket': sell_middle_ticket,
                'buy_upper_ticket': buy_upper_ticket,
                'status': 'open'
            }
            
            self.positions[symbol] = position
            
            self.algorithm.debug(f"Opened broken wing butterfly for {symbol}: "
                               f"${lower_strike}/${middle_strike}/${upper_strike}, "
                               f"Cost: ${net_cost:.2f}, Contracts: {contracts}")
            
            return True
            
        except Exception as e:
            self.algorithm.debug(f"Error opening butterfly for {symbol}: {e}")
            return False
    
    def manage_positions(self):
        """
        Daily management of all active positions
        """
        positions_to_close = []
        positions_to_adjust = []
        
        for symbol, position in self.positions.items():
            try:
                # Check if position should be closed
                close_reason = self._should_close_position(symbol, position)
                if close_reason:
                    positions_to_close.append((symbol, close_reason))
                    continue
                
                # Check if position should be adjusted
                adjustment = self._should_adjust_position(symbol, position)
                if adjustment:
                    positions_to_adjust.append((symbol, adjustment))
                    
            except Exception as e:
                self.algorithm.debug(f"Error managing position for {symbol}: {e}")
                positions_to_close.append((symbol, "error"))
        
        # Execute position closures
        for symbol, reason in positions_to_close:
            self.close_position(symbol, reason)
        
        # Execute position adjustments
        for symbol, adjustment in positions_to_adjust:
            self._adjust_position(symbol, adjustment)
    
    def close_position(self, symbol, reason="manual"):
        """
        Close a position
        
        Args:
            symbol: Stock symbol
            reason: Reason for closure
        """
        if symbol not in self.positions:
            return False
        
        try:
            position = self.positions[symbol]
            
            if position['strategy_type'] == StrategyType.SHORT_VERTICAL:
                self._close_short_vertical(position)
            elif position['strategy_type'] == StrategyType.BROKEN_WING_BUTTERFLY:
                self._close_butterfly(position)
            
            # Move to history
            position['close_date'] = self.algorithm.time
            position['close_reason'] = reason
            position['status'] = 'closed'
            self.position_history.append(position)
            
            # Remove from active positions
            del self.positions[symbol]
            
            self.algorithm.debug(f"Closed position for {symbol}: {reason}")
            return True
            
        except Exception as e:
            self.algorithm.debug(f"Error closing position for {symbol}: {e}")
            return False
    
    def _close_short_vertical(self, position):
        """Close a short vertical spread"""
        # Buy back short call, sell long call
        self.algorithm.buy(position['short_call'], position['contracts'])
        self.algorithm.sell(position['long_call'], position['contracts'])
    
    def _close_butterfly(self, position):
        """Close a butterfly spread"""
        # Reverse the original trades
        self.algorithm.sell(position['lower_call'], position['contracts'])
        self.algorithm.buy(position['middle_call'], 2 * position['contracts'])
        self.algorithm.sell(position['upper_call'], position['contracts'])
    
    def _should_close_position(self, symbol, position):
        """
        Determine if a position should be closed
        
        Returns:
            str: Reason for closure or None
        """
        try:
            # Time-based closure
            days_to_expiry = (position['expiration'] - self.algorithm.time).days
            if days_to_expiry <= self.time_decay_close:
                return "time_decay"
            
            # Check if underlying still exists
            if symbol not in self.algorithm.securities:
                return "delisted"
            
            current_price = self.algorithm.securities[symbol].price
            
            if position['strategy_type'] == StrategyType.SHORT_VERTICAL:
                return self._check_vertical_close_conditions(position, current_price)
            elif position['strategy_type'] == StrategyType.BROKEN_WING_BUTTERFLY:
                return self._check_butterfly_close_conditions(position, current_price)
            
            return None
            
        except Exception as e:
            self.algorithm.debug(f"Error checking close conditions for {symbol}: {e}")
            return "error"
    
    def _check_vertical_close_conditions(self, position, current_price):
        """Check close conditions for vertical spreads"""
        try:
            # Get current option prices
            short_option = self.algorithm.securities.get(position['short_call'])
            long_option = self.algorithm.securities.get(position['long_call'])
            
            if not short_option or not long_option:
                return "no_data"
            
            current_short_price = short_option.price
            current_long_price = long_option.price
            
            # Current spread value
            current_spread_value = current_short_price - current_long_price
            entry_credit = position['entry_credit']
            
            # Profit target (50% of max profit)
            if current_spread_value <= entry_credit * (1 - self.profit_target):
                return "profit_target"
            
            # Stop loss
            unrealized_loss = current_spread_value - entry_credit
            if unrealized_loss >= position['max_risk'] * self.stop_loss_threshold:
                return "stop_loss"
            
            # Underlying moved significantly above short strike (early assignment risk)
            if current_price > position['short_strike'] * 1.1:
                return "assignment_risk"
            
            return None
            
        except Exception as e:
            return "error"
    
    def _check_butterfly_close_conditions(self, position, current_price):
        """Check close conditions for butterfly spreads"""
        try:
            # Get current option prices
            lower_option = self.algorithm.securities.get(position['lower_call'])
            middle_option = self.algorithm.securities.get(position['middle_call'])
            upper_option = self.algorithm.securities.get(position['upper_call'])
            
            if not all([lower_option, middle_option, upper_option]):
                return "no_data"
            
            # Calculate current butterfly value
            current_value = (lower_option.price + upper_option.price - 
                           2 * middle_option.price)
            
            entry_cost = position['entry_cost']
            
            # Profit target
            max_reward = position['max_reward']
            if entry_cost > 0:  # Net debit butterfly
                profit = current_value - entry_cost
                if profit >= max_reward * self.profit_target:
                    return "profit_target"
            else:  # Net credit butterfly
                profit = abs(entry_cost) - current_value
                if profit >= max_reward * self.profit_target:
                    return "profit_target"
            
            # Stop loss
            if entry_cost > 0:  # Net debit
                if current_value <= entry_cost * (1 - self.stop_loss_threshold):
                    return "stop_loss"
            
            return None
            
        except Exception as e:
            return "error"
    
    def _should_adjust_position(self, symbol, position):
        """
        Check if position needs adjustment based on underlying movement
        
        Returns:
            dict: Adjustment parameters or None
        """
        # For now, we'll keep it simple and not implement adjustments
        # In practice, you might roll positions, add hedges, etc.
        return None
    
    def _adjust_position(self, symbol, adjustment):
        """Execute position adjustment"""
        # Placeholder for future adjustment logic
        pass
    
    def _select_optimal_expiration(self, chain):
        """Select the best expiration date"""
        target_expiration = self.algorithm.time + timedelta(days=self.target_dte)
        expirations = sorted(chain.expiration_dates)
        
        if not expirations:
            return None
        
        return min(expirations, key=lambda x: abs((x - target_expiration).days))
    
    def _find_vertical_strikes(self, calls, current_price):
        """Find optimal strikes for vertical spread"""
        # Target: Short strike around current price + 5%, long strike 5-10% higher
        target_short_strike = current_price * 1.05
        
        # Find closest strike to target
        short_candidates = [c for c in calls if c.strike >= target_short_strike]
        if not short_candidates:
            return None, None
        
        short_strike = min(short_candidates, key=lambda x: abs(x.strike - target_short_strike)).strike
        
        # Find long strike (5-15% higher than short)
        target_long_strike = short_strike * 1.1
        long_candidates = [c for c in calls if c.strike > short_strike * 1.05 and c.strike <= short_strike * 1.15]
        
        if not long_candidates:
            return None, None
        
        long_strike = min(long_candidates, key=lambda x: abs(x.strike - target_long_strike)).strike
        
        return short_strike, long_strike
    
    def _find_butterfly_strikes(self, calls, current_price, broken_wing=False):
        """Find optimal strikes for butterfly spread"""
        # For broken wing, we want asymmetric wings with bias toward upside
        strikes = [c.strike for c in calls]
        
        if len(strikes) < 3:
            return []
        
        # Center strike around current price
        middle_strike = min(strikes, key=lambda x: abs(x - current_price))
        middle_index = strikes.index(middle_strike)
        
        if broken_wing:
            # Broken wing: smaller downside wing, larger upside wing
            # This gives us more upside profit potential
            if middle_index >= 2 and middle_index < len(strikes) - 3:
                lower_strike = strikes[middle_index - 2]  # 2 strikes below
                upper_strike = strikes[middle_index + 3]  # 3 strikes above
                return [lower_strike, middle_strike, upper_strike]
        else:
            # Regular butterfly: symmetric wings
            wing_size = 2
            if middle_index >= wing_size and middle_index < len(strikes) - wing_size:
                lower_strike = strikes[middle_index - wing_size]
                upper_strike = strikes[middle_index + wing_size]
                return [lower_strike, middle_strike, upper_strike]
        
        return []
    
    def _validate_spread_prices(self, short_option, long_option):
        """Validate that spread has valid bid/ask prices"""
        return (short_option.bid_price > 0 and 
                long_option.ask_price > 0 and
                short_option.bid_price > long_option.ask_price)
    
    def _calculate_position_size(self, max_risk, sue_score):
        """
        Calculate position size based on risk and SUE score
        
        Args:
            max_risk: Maximum risk per contract
            sue_score: SUE score for sizing bias
            
        Returns:
            int: Number of contracts
        """
        # Base position size on portfolio value and max risk per position
        portfolio_value = self.algorithm.portfolio.total_portfolio_value
        max_position_risk = portfolio_value * self.max_risk_per_position
        
        # Adjust based on SUE score (higher SUE = larger position)
        sue_multiplier = min(2.0, 1.0 + (sue_score - 1.0) * 0.2)  # Cap at 2x normal size
        adjusted_max_risk = max_position_risk * sue_multiplier
        
        # Calculate contracts
        contracts = int(adjusted_max_risk / (max_risk * 100))  # 100 shares per contract
        
        return max(1, contracts)  # Minimum 1 contract
    
    def _get_iv_rank(self, symbol):
        """
        Get implied volatility rank for a symbol
        
        Returns:
            float: IV rank (0-1) or None
        """
        # Placeholder - would need historical IV data
        # For now, return None to indicate no IV data available
        return None
    
    def get_portfolio_summary(self):
        """
        Get summary of all positions
        
        Returns:
            dict: Portfolio summary
        """
        summary = {
            'active_positions': len(self.positions),
            'total_positions_opened': len(self.position_history) + len(self.positions),
            'strategies_used': {},
            'total_risk': 0,
            'total_premium': 0
        }
        
        # Analyze active positions
        for position in self.positions.values():
            strategy = position['strategy_type'].value
            summary['strategies_used'][strategy] = summary['strategies_used'].get(strategy, 0) + 1
            summary['total_risk'] += position.get('max_risk', 0) * position['contracts'] * 100
            
            if position['strategy_type'] == StrategyType.SHORT_VERTICAL:
                summary['total_premium'] += position.get('entry_credit', 0) * position['contracts'] * 100
        
        return summary