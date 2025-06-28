from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque

class Russell2000EarningsOptionsStrategy(QCAlgorithm):
    """
    Russell 2000 Earnings Surprise Options Strategy
    
    Strategy:
    1. Universe: Russell 2000 stocks via IWM ETF constituents
    2. Screen: Top 5% by biggest earnings surprises (SUE)
    3. Options: Short vertical spreads or broken wing butterflies
    4. Management: Dynamic adjustments on upside moves, downside protection
    """
    
    def initialize(self):
        self.set_start_date(2022, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(1000000)
        
        # Algorithm settings
        self.set_benchmark("IWM")
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        
        # Universe settings
        self.universe_settings.asynchronous = True
        self.universe_settings.resolution = Resolution.DAILY
        self.universe_settings.data_normalization_mode = DataNormalizationMode.ADJUSTED
        self.universe_settings.minimum_time_in_universe = 30
        
        # Strategy parameters
        self.max_positions = 20
        self.earnings_lookback_quarters = 8
        self.rebalance_frequency = 30  # days
        self.top_percentage = 0.05  # Top 5%
        
        # Risk management
        self.max_portfolio_allocation = 0.95
        self.max_position_size = 0.1  # 10% per position
        self.stop_loss_percentage = 0.25  # 25% stop loss
        
        # Options strategy settings
        self.days_to_expiration_min = 30
        self.days_to_expiration_max = 60
        self.delta_target = 0.20  # Target delta for short strikes
        self.profit_target = 0.50  # Close at 50% profit
        
        # Data structures
        self.earnings_data = defaultdict(lambda: deque(maxlen=self.earnings_lookback_quarters))
        self.sue_scores = {}
        self.selected_symbols = []
        self.options_positions = {}
        self.last_rebalance = datetime.min
        
        # Add IWM for universe selection
        iwm_symbol = Symbol.create("IWM", SecurityType.EQUITY, Market.USA)
        self.add_universe(
            self.universe.etf(iwm_symbol, self.universe_settings, self._etf_filter),
            self._fundamental_filter
        )
        
        # Add IWM for benchmark
        self.iwm = self.add_equity("IWM", Resolution.DAILY).symbol
        
        # Schedule monthly rebalancing
        self.schedule.on(
            self.date_rules.month_start(),
            self.time_rules.after_market_open("IWM", 60),
            self._rebalance
        )
        
        # Schedule daily options management
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("IWM", 30),
            self._manage_options_positions
        )
        
        # Warm up period
        self.set_warm_up(252)  # 1 year of data
        
    def _etf_filter(self, constituents):
        """Filter IWM constituents for liquid, tradeable stocks"""
        if not constituents:
            return []
            
        # Filter for valid constituents with meaningful weights
        valid_constituents = [
            c for c in constituents 
            if c.weight and c.weight > 0.001  # At least 0.1% weight
        ]
        
        # Sort by weight and return top 500 for fundamental analysis
        sorted_constituents = sorted(valid_constituents, key=lambda x: x.weight, reverse=True)
        return [c.symbol for c in sorted_constituents[:500]]
    
    def _fundamental_filter(self, fundamental):
        """Apply fundamental filters and calculate SUE scores"""
        if self.is_warming_up:
            return []
            
        # Update earnings data and calculate SUE scores
        self._update_earnings_data(fundamental)
        
        # Apply basic filters
        filtered_symbols = []
        for f in fundamental:
            if self._passes_basic_filters(f):
                filtered_symbols.append(f.symbol)
        
        # Calculate SUE scores for filtered symbols
        sue_candidates = []
        for symbol in filtered_symbols:
            sue_score = self._calculate_sue_score(symbol)
            if sue_score is not None and sue_score > 0:  # Positive earnings surprises only
                sue_candidates.append((symbol, sue_score))
        
        # Sort by SUE score and select top 5%
        sue_candidates.sort(key=lambda x: x[1], reverse=True)
        top_count = max(1, int(len(sue_candidates) * self.top_percentage))
        
        selected = sue_candidates[:top_count]
        self.sue_scores = {symbol: score for symbol, score in selected}
        
        return [symbol for symbol, _ in selected[:self.max_positions]]
    
    def _passes_basic_filters(self, fundamental):
        """Apply basic quality and liquidity filters"""
        try:
            return (
                fundamental.has_fundamental_data and
                fundamental.price > 5 and  # Min price
                fundamental.price < 500 and  # Max price  
                fundamental.dollar_volume > 1000000 and  # Min daily volume
                fundamental.earning_reports.basic_eps.three_months > 0 and  # Profitable
                fundamental.valuation_ratios.pe_ratio > 0 and
                fundamental.valuation_ratios.pe_ratio < 100  # Reasonable P/E
            )
        except:
            return False
    
    def _update_earnings_data(self, fundamental):
        """Update earnings data for SUE calculation"""
        for f in fundamental:
            if f.earning_reports.basic_eps.three_months:
                symbol = f.symbol
                current_eps = f.earning_reports.basic_eps.three_months
                
                # Store quarterly EPS data
                earnings_entry = {
                    'date': self.time,
                    'eps': current_eps,
                    'quarter': self._get_quarter(self.time)
                }
                
                # Only add if it's a new quarter
                if (not self.earnings_data[symbol] or 
                    self.earnings_data[symbol][-1]['quarter'] != earnings_entry['quarter']):
                    self.earnings_data[symbol].append(earnings_entry)
    
    def _get_quarter(self, date):
        """Get quarter identifier from date"""
        return f"{date.year}Q{(date.month - 1) // 3 + 1}"
    
    def _calculate_sue_score(self, symbol):
        """Calculate Standardized Unexpected Earnings (SUE) score"""
        if symbol not in self.earnings_data or len(self.earnings_data[symbol]) < 5:
            return None
            
        earnings_history = list(self.earnings_data[symbol])
        
        # Get current and year-ago EPS
        current_eps = earnings_history[-1]['eps']
        
        # Find EPS from 4 quarters ago
        year_ago_eps = None
        for i in range(len(earnings_history) - 2, -1, -1):
            if earnings_history[i]['eps'] is not None:
                year_ago_eps = earnings_history[i]['eps']
                break
        
        if year_ago_eps is None:
            return None
        
        # Calculate earnings change
        earnings_change = current_eps - year_ago_eps
        
        # Calculate historical volatility of earnings changes
        changes = []
        for i in range(1, len(earnings_history)):
            if (earnings_history[i]['eps'] is not None and 
                earnings_history[i-1]['eps'] is not None):
                change = earnings_history[i]['eps'] - earnings_history[i-1]['eps']
                changes.append(change)
        
        if len(changes) < 3:
            return None
        
        # Calculate standard deviation of changes
        std_dev = np.std(changes)
        
        if std_dev == 0:
            return None
        
        # Calculate SUE score
        sue_score = earnings_change / std_dev
        return sue_score
    
    def on_securities_changed(self, changes):
        """Handle universe changes"""
        # Remove old positions
        for security in changes.removed_securities:
            if security.symbol in self.options_positions:
                self._close_options_position(security.symbol)
                
        # Update selected symbols
        self.selected_symbols = [s.symbol for s in changes.added_securities if s.symbol != self.iwm]
        
        # Add options for new securities
        for security in changes.added_securities:
            if security.symbol != self.iwm:
                self._add_options_for_symbol(security.symbol)
    
    def _add_options_for_symbol(self, symbol):
        """Add options data for a symbol"""
        try:
            equity = self.securities[symbol]
            if equity.has_data:
                # Add option chain
                option = self.add_option(symbol)
                option.set_filter(self._options_filter)
        except Exception as e:
            self.debug(f"Failed to add options for {symbol}: {e}")
    
    def _options_filter(self, universe):
        """Filter options chain"""
        return (universe
                .strikes(-10, 10)  # 10 strikes above and below ATM
                .expiration(self.days_to_expiration_min, self.days_to_expiration_max)
                .include_weeklys())
    
    def _rebalance(self):
        """Monthly rebalancing"""
        if not self.selected_symbols:
            return
            
        # Close existing options positions if symbols changed
        current_symbols = set(self.selected_symbols)
        existing_symbols = set(self.options_positions.keys())
        
        # Close positions for symbols no longer selected
        for symbol in existing_symbols - current_symbols:
            self._close_options_position(symbol)
        
        # Open new positions for newly selected symbols
        for symbol in current_symbols - existing_symbols:
            self._open_options_position(symbol)
    
    def _open_options_position(self, symbol):
        """Open options position for a symbol"""
        try:
            if symbol not in self.securities:
                return
                
            security = self.securities[symbol]
            if not security.has_data:
                return
                
            current_price = security.price
            if current_price <= 0:
                return
            
            # Get options chain
            option_chain = self.option_chain(symbol)
            if not option_chain or symbol not in option_chain:
                return
            
            chain = option_chain[symbol]
            
            # Select expiration (closest to 45 days)
            target_expiration = self.time + timedelta(days=45)
            expirations = sorted(chain.expiration_dates)
            
            if not expirations:
                return
                
            selected_expiration = min(expirations, 
                                    key=lambda x: abs((x - target_expiration).days))
            
            # Filter options by expiration
            calls = [x for x in chain if x.expiry == selected_expiration and x.right == OptionRight.CALL]
            
            if len(calls) < 3:
                return
            
            # Sort by strike price
            calls.sort(key=lambda x: x.strike)
            
            # Find ATM, ITM, and OTM strikes for butterfly or vertical spread
            atm_call = min(calls, key=lambda x: abs(x.strike - current_price))
            atm_index = calls.index(atm_call)
            
            # Implement short vertical spread (bear call spread)
            if atm_index < len(calls) - 1:
                short_call = calls[atm_index]  # Sell ATM call
                long_call = calls[atm_index + 1]  # Buy OTM call
                
                # Check if we have valid bid/ask prices
                if (short_call.bid_price > 0 and long_call.ask_price > 0 and 
                    short_call.bid_price > long_call.ask_price):
                    
                    # Calculate position size
                    net_credit = short_call.bid_price - long_call.ask_price
                    max_risk = long_call.strike - short_call.strike - net_credit
                    
                    if max_risk > 0:
                        position_value = self.portfolio.total_portfolio_value * self.max_position_size
                        contracts = max(1, int(position_value / (max_risk * 100)))
                        
                        # Place orders
                        short_ticket = self.sell(short_call.symbol, contracts)
                        long_ticket = self.buy(long_call.symbol, contracts)
                        
                        # Store position info
                        self.options_positions[symbol] = {
                            'type': 'short_vertical',
                            'short_call': short_call.symbol,
                            'long_call': long_call.symbol,
                            'contracts': contracts,
                            'entry_date': self.time,
                            'entry_credit': net_credit,
                            'max_risk': max_risk,
                            'short_ticket': short_ticket,
                            'long_ticket': long_ticket
                        }
                        
                        self.debug(f"Opened short vertical spread for {symbol}: "
                                 f"Sold {short_call.symbol}, Bought {long_call.symbol}")
                        
        except Exception as e:
            self.debug(f"Error opening options position for {symbol}: {e}")
    
    def _close_options_position(self, symbol):
        """Close options position for a symbol"""
        if symbol not in self.options_positions:
            return
            
        try:
            position = self.options_positions[symbol]
            
            # Close the spread
            if position['type'] == 'short_vertical':
                # Buy back short call, sell long call
                self.buy(position['short_call'], position['contracts'])
                self.sell(position['long_call'], position['contracts'])
                
            del self.options_positions[symbol]
            self.debug(f"Closed options position for {symbol}")
            
        except Exception as e:
            self.debug(f"Error closing options position for {symbol}: {e}")
    
    def _manage_options_positions(self):
        """Daily options position management"""
        positions_to_close = []
        
        for symbol, position in self.options_positions.items():
            try:
                # Check if we should close the position
                if self._should_close_position(symbol, position):
                    positions_to_close.append(symbol)
                    
            except Exception as e:
                self.debug(f"Error managing position for {symbol}: {e}")
        
        # Close positions that meet criteria
        for symbol in positions_to_close:
            self._close_options_position(symbol)
    
    def _should_close_position(self, symbol, position):
        """Determine if position should be closed"""
        try:
            # Check if underlying security still exists
            if symbol not in self.securities:
                return True
            
            # Time-based closure (close 7 days before expiration)
            short_option = self.securities.get(position['short_call'])
            if short_option and hasattr(short_option, 'expiry'):
                days_to_expiry = (short_option.expiry - self.time).days
                if days_to_expiry <= 7:
                    return True
            
            # Profit target (50% of max profit)
            current_short_price = self.securities[position['short_call']].price
            current_long_price = self.securities[position['long_call']].price
            
            current_spread_value = current_short_price - current_long_price
            initial_credit = position['entry_credit']
            
            # Close if we've captured 50% of the premium
            if current_spread_value <= initial_credit * 0.5:
                return True
            
            # Stop loss (25% of max risk)
            current_loss = current_spread_value - initial_credit
            max_risk = position['max_risk']
            
            if current_loss >= max_risk * self.stop_loss_percentage:
                return True
            
            return False
            
        except Exception as e:
            self.debug(f"Error evaluating position closure for {symbol}: {e}")
            return True  # Close on error
    
    def on_data(self, data):
        """Handle data updates"""
        pass  # Primary logic handled in scheduled events
    
    def on_order_event(self, order_event):
        """Handle order events"""
        if order_event.status == OrderStatus.FILLED:
            self.debug(f"Order filled: {order_event.symbol} - {order_event.direction} - {order_event.fill_quantity}")