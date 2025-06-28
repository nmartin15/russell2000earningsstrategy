from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Import our custom modules
from earnings_screener import EarningsScreener
from options_manager import OptionsManager, StrategyType
from risk_manager import RiskManager, RiskLevel

class Russell2000EarningsOptionsStrategy(QCAlgorithm):
    """
    Integrated Russell 2000 Earnings Surprise Options Strategy
    
    This is the main algorithm that integrates all components:
    1. Russell 2000 universe selection via IWM ETF
    2. Earnings surprise screening using SUE
    3. Options strategies (vertical spreads & butterflies)
    4. Comprehensive risk management
    5. Dynamic position management with upside adjustments
    """
    
    def initialize(self):
        # Algorithm setup
        self.set_start_date(2022, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(1000000)
        self.set_benchmark("IWM")
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)
        
        # Universe settings
        self.universe_settings.asynchronous = True
        self.universe_settings.resolution = Resolution.DAILY
        self.universe_settings.data_normalization_mode = DataNormalizationMode.ADJUSTED
        self.universe_settings.minimum_time_in_universe = 30
        
        # Strategy parameters
        self.max_positions = 20
        self.rebalance_frequency = 30  # days
        self.top_percentage = 0.05  # Top 5% by SUE
        
        # Initialize components
        self.earnings_screener = EarningsScreener(self, lookback_quarters=8)
        self.options_manager = OptionsManager(self)
        self.risk_manager = RiskManager(self)
        
        # Add benchmark
        self.iwm = self.add_equity("IWM", Resolution.DAILY).symbol
        
        # Universe selection
        iwm_symbol = Symbol.create("IWM", SecurityType.EQUITY, Market.USA)
        self.add_universe(
            self.universe.etf(iwm_symbol, self.universe_settings, self._etf_filter),
            self._fundamental_filter
        )
        
        # Scheduling
        self.schedule.on(
            self.date_rules.month_start(),
            self.time_rules.after_market_open("IWM", 60),
            self._monthly_rebalance
        )
        
        self.schedule.on(
            self.date_rules.every_day(),
            self.time_rules.after_market_open("IWM", 30),
            self._daily_management
        )
        
        # Performance tracking
        self.monthly_performance = []
        self.trade_log = []
        
        # Warm up
        self.set_warm_up(252)
        
    def _etf_filter(self, constituents):
        """Filter IWM constituents for initial screening"""
        if not constituents:
            return []
        
        # Filter for liquid constituents
        valid_constituents = [
            c for c in constituents 
            if c.weight and c.weight > 0.001  # Minimum 0.1% weight
        ]
        
        # Return top 500 by weight for fundamental analysis
        sorted_constituents = sorted(valid_constituents, key=lambda x: x.weight, reverse=True)
        return [c.symbol for c in sorted_constituents[:500]]
    
    def _fundamental_filter(self, fundamental):
        """Apply fundamental filters and SUE screening"""
        if self.is_warming_up:
            return []
        
        try:
            # Update earnings data
            self.earnings_screener.update_earnings_data(fundamental)
            
            # Apply basic quality filters
            filtered_symbols = []
            for f in fundamental:
                if self._passes_quality_filters(f):
                    filtered_symbols.append(f.symbol)
            
            if not filtered_symbols:
                return []
            
            # Calculate SUE scores and rank
            sue_rankings = self.earnings_screener.rank_by_sue(
                symbols=filtered_symbols,
                top_percentage=self.top_percentage,
                min_sue_score=0.5  # Minimum SUE threshold
            )
            
            # Return top candidates up to max positions
            selected_symbols = [symbol for symbol, score in sue_rankings[:self.max_positions]]
            
            self.debug(f"SUE screening selected {len(selected_symbols)} symbols from {len(filtered_symbols)} candidates")
            
            return selected_symbols
            
        except Exception as e:
            self.debug(f"Error in fundamental filter: {e}")
            return []
    
    def _passes_quality_filters(self, fundamental):
        """Apply basic quality and liquidity filters"""
        try:
            return (
                fundamental.has_fundamental_data and
                fundamental.price > 5 and
                fundamental.price < 500 and
                fundamental.dollar_volume > 1000000 and
                fundamental.earning_reports.basic_eps.three_months > 0 and
                fundamental.valuation_ratios.pe_ratio > 0 and
                fundamental.valuation_ratios.pe_ratio < 100 and
                fundamental.operation_ratios.roe.one_year > 0.05
            )
        except:
            return False
    
    def on_securities_changed(self, changes):
        """Handle universe changes"""
        try:
            # Remove options for securities leaving universe
            for security in changes.removed_securities:
                if security.symbol != self.iwm:
                    self.options_manager.close_position(security.symbol, "removed_from_universe")
            
            # Add options for new securities
            for security in changes.added_securities:
                if security.symbol != self.iwm:
                    self._add_options_for_symbol(security.symbol)
            
            self.debug(f"Universe changed: +{len(changes.added_securities)} -{len(changes.removed_securities)}")
            
        except Exception as e:
            self.debug(f"Error handling securities changes: {e}")
    
    def _add_options_for_symbol(self, symbol):
        """Add options data subscription for a symbol"""
        try:
            option = self.add_option(symbol)
            option.set_filter(self.options_manager.get_options_filter())
        except Exception as e:
            self.debug(f"Failed to add options for {symbol}: {e}")
    
    def _monthly_rebalance(self):
        """Monthly portfolio rebalancing"""
        try:
            self.debug("Starting monthly rebalance...")
            
            # Get current SUE rankings
            current_symbols = [s for s in self.active_securities.keys() if s != self.iwm]
            sue_rankings = self.earnings_screener.rank_by_sue(
                symbols=current_symbols,
                top_percentage=self.top_percentage,
                min_sue_score=0.5
            )
            
            if not sue_rankings:
                self.debug("No symbols passed SUE screening")
                return
            
            # Close positions not in top rankings
            top_symbols = {symbol for symbol, score in sue_rankings[:self.max_positions]}
            current_positions = set(self.options_manager.positions.keys())
            
            positions_to_close = current_positions - top_symbols
            for symbol in positions_to_close:
                self.options_manager.close_position(symbol, "rebalance")
            
            # Open new positions
            positions_to_open = top_symbols - current_positions
            for symbol in positions_to_open:
                self._evaluate_and_open_position(symbol, dict(sue_rankings)[symbol])
            
            # Log rebalance results
            self.debug(f"Rebalance complete: {len(positions_to_open)} new positions, "
                      f"{len(positions_to_close)} closed positions")
            
        except Exception as e:
            self.debug(f"Error in monthly rebalance: {e}")
    
    def _evaluate_and_open_position(self, symbol, sue_score):
        """Evaluate and potentially open a new options position"""
        try:
            # Determine strategy type
            strategy_type = self.options_manager.evaluate_strategy_for_symbol(symbol, sue_score)
            
            if not strategy_type:
                return
            
            # Estimate position risk for pre-trade check
            estimated_risk = self._estimate_position_risk(symbol, strategy_type)
            estimated_contracts = self._estimate_contracts(symbol, sue_score)
            
            # Pre-trade risk check
            risk_check = self.risk_manager.pre_trade_risk_check(
                symbol, strategy_type, estimated_contracts, estimated_risk
            )
            
            if not risk_check['approved']:
                self.debug(f"Position rejected for {symbol}: {risk_check['warnings']}")
                return
            
            # Open position with risk-adjusted size
            contracts = risk_check['adjusted_position_size']
            if contracts > 0:
                success = self.options_manager.open_position(symbol, strategy_type, sue_score)
                
                if success:
                    # Update risk tracking
                    position_info = self.options_manager.positions.get(symbol)
                    if position_info:
                        self.risk_manager.post_trade_update(symbol, position_info)
                    
                    self.debug(f"Opened {strategy_type.value} position for {symbol} "
                              f"(SUE: {sue_score:.2f}, Contracts: {contracts})")
                
        except Exception as e:
            self.debug(f"Error evaluating position for {symbol}: {e}")
    
    def _daily_management(self):
        """Daily position and risk management"""
        try:
            # Risk monitoring
            self.risk_manager.monitor_portfolio_risk()
            
            # Options position management
            self.options_manager.manage_positions()
            
            # Dynamic adjustments based on underlying moves
            self._manage_upside_moves()
            
            # Update performance tracking
            self._update_performance_metrics()
            
        except Exception as e:
            self.debug(f"Error in daily management: {e}")
    
    def _manage_upside_moves(self):
        """Manage positions when underlying stocks move up significantly"""
        try:
            for symbol, position in self.options_manager.positions.items():
                if symbol not in self.securities:
                    continue
                
                current_price = self.securities[symbol].price
                
                # Check for significant upside moves (>10% from entry)
                entry_date = position.get('entry_date')
                if not entry_date:
                    continue
                
                # Get entry price
                history = self.history(symbol, 1, Resolution.DAILY, entry_date)
                if history.empty:
                    continue
                
                entry_price = history.iloc[0]['close']
                price_change = (current_price - entry_price) / entry_price
                
                # Significant upside move detected
                if price_change > 0.10:  # 10% move up
                    self._handle_upside_move(symbol, position, price_change)
                
        except Exception as e:
            self.debug(f"Error managing upside moves: {e}")
    
    def _handle_upside_move(self, symbol, position, price_change):
        """Handle specific upside move in a position"""
        try:
            strategy_type = position.get('strategy_type')
            
            if strategy_type == StrategyType.SHORT_VERTICAL:
                # For short verticals, consider closing early to lock in profits
                # or rolling the spread higher
                if price_change > 0.15:  # 15% move
                    self.options_manager.close_position(symbol, "large_upside_move")
                    self.debug(f"Closed short vertical for {symbol} due to {price_change:.1%} upside move")
                
            elif strategy_type == StrategyType.BROKEN_WING_BUTTERFLY:
                # Broken wing butterflies can benefit from upside moves
                # Consider partial profit taking
                if price_change > 0.20:  # 20% move
                    # Check if we're profitable and consider partial close
                    current_value = self._estimate_butterfly_value(symbol, position)
                    if current_value and current_value > position.get('entry_cost', 0) * 1.5:
                        # Take partial profits
                        self.debug(f"Taking partial profits on butterfly for {symbol} "
                                  f"after {price_change:.1%} upside move")
                
        except Exception as e:
            self.debug(f"Error handling upside move for {symbol}: {e}")
    
    def _estimate_position_risk(self, symbol, strategy_type):
        """Estimate maximum risk per contract for a strategy"""
        # Conservative estimates for risk checking
        if strategy_type == StrategyType.SHORT_VERTICAL:
            return 5.0  # Assume $5 max risk per contract
        elif strategy_type == StrategyType.BROKEN_WING_BUTTERFLY:
            return 3.0  # Assume $3 max risk per contract
        else:
            return 4.0  # Default estimate
    
    def _estimate_contracts(self, symbol, sue_score):
        """Estimate number of contracts based on SUE score"""
        # Base on portfolio size and SUE strength
        portfolio_value = self.portfolio.total_portfolio_value
        base_allocation = portfolio_value * 0.03  # 3% base allocation
        
        # Adjust for SUE strength
        sue_multiplier = min(2.0, 1.0 + (sue_score - 1.0) * 0.2)
        adjusted_allocation = base_allocation * sue_multiplier
        
        # Estimate contracts (assuming $4 average risk per contract)
        estimated_contracts = int(adjusted_allocation / 400)  # $4 * 100 shares
        
        return max(1, min(10, estimated_contracts))  # Between 1-10 contracts
    
    def _estimate_butterfly_value(self, symbol, position):
        """Estimate current value of a butterfly position"""
        try:
            lower_option = self.securities.get(position['lower_call'])
            middle_option = self.securities.get(position['middle_call'])
            upper_option = self.securities.get(position['upper_call'])
            
            if not all([lower_option, middle_option, upper_option]):
                return None
            
            current_value = (lower_option.price + upper_option.price - 
                           2 * middle_option.price)
            
            return current_value
            
        except Exception as e:
            return None
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        try:
            # Monthly performance tracking
            if self.time.day == 1:  # First day of month
                portfolio_value = self.portfolio.total_portfolio_value
                self.monthly_performance.append({
                    'date': self.time,
                    'portfolio_value': portfolio_value,
                    'active_positions': len(self.options_manager.positions),
                    'cash': self.portfolio.cash
                })
            
        except Exception as e:
            self.debug(f"Error updating performance metrics: {e}")
    
    def on_data(self, data):
        """Handle incoming data"""
        pass  # All logic handled in scheduled functions
    
    def on_order_event(self, order_event):
        """Handle order events"""
        if order_event.status == OrderStatus.FILLED:
            # Log successful trades
            self.trade_log.append({
                'date': self.time,
                'symbol': order_event.symbol,
                'direction': order_event.direction,
                'quantity': order_event.fill_quantity,
                'price': order_event.fill_price
            })
            
            self.debug(f"Order filled: {order_event.symbol} {order_event.direction} "
                      f"{order_event.fill_quantity} @ {order_event.fill_price}")
    
    def on_end_of_algorithm(self):
        """End of algorithm cleanup and summary"""
        try:
            # Final performance summary
            final_value = self.portfolio.total_portfolio_value
            total_return = (final_value - 1000000) / 1000000
            
            self.debug(f"\n=== STRATEGY PERFORMANCE SUMMARY ===")
            self.debug(f"Final Portfolio Value: ${final_value:,.2f}")
            self.debug(f"Total Return: {total_return:.2%}")
            self.debug(f"Total Trades: {len(self.trade_log)}")
            
            # Risk summary
            risk_dashboard = self.risk_manager.get_risk_dashboard()
            self.debug(f"Maximum Drawdown: {risk_dashboard.get('current_drawdown', 0):.2%}")
            self.debug(f"Final Risk Level: {risk_dashboard.get('risk_level', 'Unknown')}")
            
            # SUE screening summary
            screening_stats = self.earnings_screener.get_summary_statistics()
            self.debug(f"Symbols Tracked: {screening_stats.get('total_symbols_tracked', 0)}")
            self.debug(f"Average SUE Score: {screening_stats.get('avg_sue_score', 0):.2f}")
            
            # Options portfolio summary
            options_summary = self.options_manager.get_portfolio_summary()
            self.debug(f"Total Positions Opened: {options_summary.get('total_positions_opened', 0)}")
            self.debug(f"Strategies Used: {options_summary.get('strategies_used', {})}")
            
        except Exception as e:
            self.debug(f"Error in end of algorithm summary: {e}")

# Alias for backwards compatibility
Russell2000EarningsStrategy = Russell2000EarningsOptionsStrategy