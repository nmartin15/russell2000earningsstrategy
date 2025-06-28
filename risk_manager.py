from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskManager:
    """
    Comprehensive Risk Management System for Russell 2000 Earnings Options Strategy
    
    Manages:
    1. Portfolio-level risk limits
    2. Position-level risk controls
    3. Greeks exposure monitoring
    4. Drawdown protection
    5. Correlation risk management
    6. Volatility-based position sizing
    """
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        
        # Portfolio risk limits
        self.max_portfolio_risk = 0.20  # Maximum 20% portfolio at risk
        self.max_position_risk = 0.05   # Maximum 5% per position
        self.max_sector_concentration = 0.30  # Maximum 30% in any sector
        self.max_daily_loss = 0.03      # Maximum 3% daily loss
        self.max_drawdown = 0.15        # Maximum 15% drawdown
        
        # Greeks limits
        self.max_portfolio_delta = 0.10  # Net delta exposure limit
        self.max_portfolio_gamma = 0.05  # Gamma exposure limit
        self.max_portfolio_vega = 0.15   # Vega exposure limit
        self.max_portfolio_theta = 0.05  # Theta exposure limit
        
        # Volatility and correlation
        self.volatility_lookback = 30    # Days for volatility calculation
        self.correlation_lookback = 60   # Days for correlation calculation
        self.max_correlation = 0.7       # Maximum correlation between positions
        
        # Risk monitoring
        self.risk_history = deque(maxlen=252)  # 1 year of daily risk metrics
        self.drawdown_history = deque(maxlen=252)
        self.var_history = deque(maxlen=30)    # Value at Risk history
        
        # Position tracking
        self.position_risks = {}
        self.sector_exposure = defaultdict(float)
        self.correlation_matrix = {}
        
        # Emergency controls
        self.emergency_stop_triggered = False
        self.risk_reduction_mode = False
        self.last_risk_check = datetime.min
        
        # Performance tracking
        self.daily_pnl_history = deque(maxlen=252)
        self.peak_portfolio_value = 0
        self.current_drawdown = 0
        
    def pre_trade_risk_check(self, symbol, strategy_type, contracts, max_risk_per_contract):
        """
        Perform comprehensive risk check before opening a new position
        
        Args:
            symbol: Stock symbol
            strategy_type: Type of options strategy
            contracts: Number of contracts
            max_risk_per_contract: Maximum risk per contract
            
        Returns:
            dict: Risk check results with approval status and recommendations
        """
        risk_check = {
            'approved': False,
            'risk_level': RiskLevel.LOW,
            'warnings': [],
            'recommendations': [],
            'max_allowed_contracts': 0,
            'adjusted_position_size': 0
        }
        
        try:
            # Calculate position risk
            total_position_risk = contracts * max_risk_per_contract * 100  # Convert to dollars
            portfolio_value = self.algorithm.portfolio.total_portfolio_value
            position_risk_percentage = total_position_risk / portfolio_value
            
            # Check position size limits
            if position_risk_percentage > self.max_position_risk:
                max_allowed_contracts = int((portfolio_value * self.max_position_risk) / (max_risk_per_contract * 100))
                risk_check['max_allowed_contracts'] = max_allowed_contracts
                risk_check['warnings'].append(f"Position size exceeds limit. Max allowed: {max_allowed_contracts} contracts")
                
                if max_allowed_contracts == 0:
                    risk_check['risk_level'] = RiskLevel.CRITICAL
                    return risk_check
                else:
                    contracts = max_allowed_contracts
                    total_position_risk = contracts * max_risk_per_contract * 100
                    position_risk_percentage = total_position_risk / portfolio_value
            
            # Check portfolio risk limits
            current_portfolio_risk = self._calculate_current_portfolio_risk()
            total_portfolio_risk = current_portfolio_risk + position_risk_percentage
            
            if total_portfolio_risk > self.max_portfolio_risk:
                risk_check['warnings'].append(f"Total portfolio risk would exceed limit: {total_portfolio_risk:.2%}")
                risk_check['risk_level'] = RiskLevel.HIGH
                
                # Calculate max allowed based on remaining risk budget
                remaining_risk_budget = self.max_portfolio_risk - current_portfolio_risk
                if remaining_risk_budget <= 0:
                    return risk_check  # No risk budget left
                
                max_contracts_by_portfolio = int((portfolio_value * remaining_risk_budget) / (max_risk_per_contract * 100))
                contracts = min(contracts, max_contracts_by_portfolio)
            
            # Check sector concentration
            sector = self._get_sector(symbol)
            if sector:
                current_sector_exposure = self.sector_exposure.get(sector, 0)
                new_sector_exposure = current_sector_exposure + position_risk_percentage
                
                if new_sector_exposure > self.max_sector_concentration:
                    risk_check['warnings'].append(f"Sector concentration risk: {sector} exposure would be {new_sector_exposure:.2%}")
                    risk_check['risk_level'] = max(risk_check['risk_level'], RiskLevel.MEDIUM)
            
            # Check correlation risk
            correlation_risk = self._assess_correlation_risk(symbol)
            if correlation_risk > self.max_correlation:
                risk_check['warnings'].append(f"High correlation risk: {correlation_risk:.2f}")
                risk_check['risk_level'] = max(risk_check['risk_level'], RiskLevel.MEDIUM)
            
            # Check volatility conditions
            volatility_assessment = self._assess_volatility_conditions(symbol)
            if volatility_assessment['risk_level'] != RiskLevel.LOW:
                risk_check['warnings'].extend(volatility_assessment['warnings'])
                risk_check['risk_level'] = max(risk_check['risk_level'], volatility_assessment['risk_level'])
            
            # Check current drawdown
            if self.current_drawdown > self.max_drawdown * 0.8:  # 80% of max drawdown
                risk_check['warnings'].append(f"High drawdown: {self.current_drawdown:.2%}")
                risk_check['risk_level'] = RiskLevel.HIGH
                contracts = int(contracts * 0.5)  # Reduce position size by 50%
            
            # Emergency stop check
            if self.emergency_stop_triggered:
                risk_check['warnings'].append("Emergency stop is active")
                return risk_check
            
            # Final approval
            if contracts > 0 and risk_check['risk_level'] != RiskLevel.CRITICAL:
                risk_check['approved'] = True
                risk_check['adjusted_position_size'] = contracts
            
            # Add recommendations
            risk_check['recommendations'] = self._generate_risk_recommendations(risk_check, symbol)
            
            return risk_check
            
        except Exception as e:
            self.algorithm.debug(f"Error in pre-trade risk check: {e}")
            risk_check['warnings'].append(f"Error in risk calculation: {e}")
            return risk_check
    
    def monitor_portfolio_risk(self):
        """
        Daily portfolio risk monitoring and emergency controls
        """
        try:
            current_time = self.algorithm.time
            
            # Skip if already checked today
            if self.last_risk_check.date() == current_time.date():
                return
            
            self.last_risk_check = current_time
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            # Check emergency conditions
            self._check_emergency_conditions()
            
            # Update risk history
            risk_metrics = self._calculate_risk_metrics()
            self.risk_history.append({
                'date': current_time,
                'metrics': risk_metrics
            })
            
            # Log risk status
            if len(self.risk_history) % 30 == 0:  # Log monthly
                self._log_risk_summary()
                
        except Exception as e:
            self.algorithm.debug(f"Error in portfolio risk monitoring: {e}")
    
    def post_trade_update(self, symbol, position_info):
        """
        Update risk tracking after a trade is executed
        
        Args:
            symbol: Stock symbol
            position_info: Dictionary containing position details
        """
        try:
            # Update position risk tracking
            self.position_risks[symbol] = {
                'max_risk': position_info.get('max_risk', 0),
                'contracts': position_info.get('contracts', 0),
                'entry_date': position_info.get('entry_date'),
                'strategy_type': position_info.get('strategy_type'),
                'current_risk': position_info.get('max_risk', 0)  # Initially same as max risk
            }
            
            # Update sector exposure
            sector = self._get_sector(symbol)
            if sector:
                portfolio_value = self.algorithm.portfolio.total_portfolio_value
                position_risk = position_info.get('max_risk', 0) * position_info.get('contracts', 0) * 100
                risk_percentage = position_risk / portfolio_value
                self.sector_exposure[sector] += risk_percentage
            
        except Exception as e:
            self.algorithm.debug(f"Error in post-trade update: {e}")
    
    def position_closed_update(self, symbol):
        """
        Update risk tracking when a position is closed
        
        Args:
            symbol: Stock symbol
        """
        try:
            if symbol in self.position_risks:
                # Update sector exposure
                sector = self._get_sector(symbol)
                if sector:
                    portfolio_value = self.algorithm.portfolio.total_portfolio_value
                    position_info = self.position_risks[symbol]
                    position_risk = position_info['max_risk'] * position_info['contracts'] * 100
                    risk_percentage = position_risk / portfolio_value
                    self.sector_exposure[sector] = max(0, self.sector_exposure[sector] - risk_percentage)
                
                # Remove from position tracking
                del self.position_risks[symbol]
                
        except Exception as e:
            self.algorithm.debug(f"Error in position closed update: {e}")
    
    def calculate_var(self, confidence_level=0.05, lookback_days=30):
        """
        Calculate Value at Risk (VaR) for the portfolio
        
        Args:
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            lookback_days: Number of days to look back
            
        Returns:
            float: VaR value
        """
        try:
            if len(self.daily_pnl_history) < lookback_days:
                return 0
            
            recent_returns = list(self.daily_pnl_history)[-lookback_days:]
            portfolio_value = self.algorithm.portfolio.total_portfolio_value
            
            # Convert to percentage returns
            percentage_returns = [pnl / portfolio_value for pnl in recent_returns]
            
            # Calculate VaR
            var_percentile = np.percentile(percentage_returns, confidence_level * 100)
            var_dollar = var_percentile * portfolio_value
            
            return abs(var_dollar)
            
        except Exception as e:
            self.algorithm.debug(f"Error calculating VaR: {e}")
            return 0
    
    def get_position_sizing_multiplier(self, symbol, sue_score):
        """
        Get position sizing multiplier based on current risk conditions
        
        Args:
            symbol: Stock symbol
            sue_score: SUE score
            
        Returns:
            float: Position sizing multiplier (0.0 to 2.0)
        """
        try:
            base_multiplier = 1.0
            
            # Adjust based on current drawdown
            if self.current_drawdown > 0.10:  # Above 10% drawdown
                base_multiplier *= 0.7
            elif self.current_drawdown > 0.05:  # Above 5% drawdown
                base_multiplier *= 0.85
            
            # Adjust based on portfolio risk utilization
            current_risk = self._calculate_current_portfolio_risk()
            risk_utilization = current_risk / self.max_portfolio_risk
            
            if risk_utilization > 0.8:  # High risk utilization
                base_multiplier *= 0.6
            elif risk_utilization > 0.6:  # Medium risk utilization
                base_multiplier *= 0.8
            
            # Adjust based on volatility regime
            volatility_multiplier = self._get_volatility_multiplier(symbol)
            base_multiplier *= volatility_multiplier
            
            # Adjust based on SUE score strength
            sue_multiplier = min(1.5, 1.0 + (sue_score - 1.0) * 0.1)
            base_multiplier *= sue_multiplier
            
            # Cap the multiplier
            return max(0.1, min(2.0, base_multiplier))
            
        except Exception as e:
            self.algorithm.debug(f"Error calculating position sizing multiplier: {e}")
            return 0.5  # Conservative default
    
    def _calculate_current_portfolio_risk(self):
        """Calculate current portfolio risk percentage"""
        total_risk = 0
        portfolio_value = self.algorithm.portfolio.total_portfolio_value
        
        for position_info in self.position_risks.values():
            position_risk = position_info['current_risk'] * position_info['contracts'] * 100
            total_risk += position_risk
        
        return total_risk / portfolio_value if portfolio_value > 0 else 0
    
    def _get_sector(self, symbol):
        """Get sector for a symbol"""
        try:
            if symbol in self.algorithm.securities:
                security = self.algorithm.securities[symbol]
                if hasattr(security, 'fundamentals') and security.fundamentals:
                    return security.fundamentals.asset_classification.morningstar_sector_code
        except:
            pass
        return "Unknown"
    
    def _assess_correlation_risk(self, symbol):
        """Assess correlation risk with existing positions"""
        # Simplified correlation assessment
        # In practice, you'd calculate actual correlations between returns
        
        if len(self.position_risks) == 0:
            return 0
        
        # For now, return a conservative estimate based on sector overlap
        symbol_sector = self._get_sector(symbol)
        same_sector_positions = 0
        
        for existing_symbol in self.position_risks.keys():
            if self._get_sector(existing_symbol) == symbol_sector:
                same_sector_positions += 1
        
        # Return correlation proxy based on sector concentration
        return min(0.9, same_sector_positions * 0.2)
    
    def _assess_volatility_conditions(self, symbol):
        """Assess current volatility conditions for a symbol"""
        assessment = {
            'risk_level': RiskLevel.LOW,
            'warnings': []
        }
        
        try:
            if symbol not in self.algorithm.securities:
                return assessment
            
            # Get recent price history
            history = self.algorithm.history(symbol, self.volatility_lookback, Resolution.DAILY)
            
            if history.empty or len(history) < 10:
                assessment['warnings'].append("Insufficient price history")
                assessment['risk_level'] = RiskLevel.MEDIUM
                return assessment
            
            # Calculate volatility
            returns = history['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Assess volatility level
            if volatility > 0.6:  # High volatility
                assessment['warnings'].append(f"High volatility: {volatility:.2%}")
                assessment['risk_level'] = RiskLevel.HIGH
            elif volatility > 0.4:  # Medium volatility
                assessment['warnings'].append(f"Elevated volatility: {volatility:.2%}")
                assessment['risk_level'] = RiskLevel.MEDIUM
            
            return assessment
            
        except Exception as e:
            assessment['warnings'].append(f"Error assessing volatility: {e}")
            assessment['risk_level'] = RiskLevel.MEDIUM
            return assessment
    
    def _update_portfolio_metrics(self):
        """Update daily portfolio performance metrics"""
        try:
            current_value = self.algorithm.portfolio.total_portfolio_value
            
            # Update peak and drawdown
            if current_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_value
                self.current_drawdown = 0
            else:
                self.current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
            
            self.drawdown_history.append({
                'date': self.algorithm.time,
                'drawdown': self.current_drawdown,
                'portfolio_value': current_value
            })
            
            # Calculate daily P&L
            if len(self.drawdown_history) > 1:
                previous_value = self.drawdown_history[-2]['portfolio_value']
                daily_pnl = current_value - previous_value
                self.daily_pnl_history.append(daily_pnl)
            
        except Exception as e:
            self.algorithm.debug(f"Error updating portfolio metrics: {e}")
    
    def _check_emergency_conditions(self):
        """Check for emergency stop conditions"""
        try:
            # Daily loss check
            if len(self.daily_pnl_history) > 0:
                daily_pnl = self.daily_pnl_history[-1]
                portfolio_value = self.algorithm.portfolio.total_portfolio_value
                daily_loss_percentage = abs(daily_pnl) / portfolio_value
                
                if daily_pnl < 0 and daily_loss_percentage > self.max_daily_loss:
                    self.emergency_stop_triggered = True
                    self.algorithm.debug(f"EMERGENCY STOP: Daily loss {daily_loss_percentage:.2%} exceeds limit")
            
            # Drawdown check
            if self.current_drawdown > self.max_drawdown:
                self.emergency_stop_triggered = True
                self.algorithm.debug(f"EMERGENCY STOP: Drawdown {self.current_drawdown:.2%} exceeds limit")
            
            # Risk reduction mode
            if self.current_drawdown > self.max_drawdown * 0.7:  # 70% of max drawdown
                self.risk_reduction_mode = True
                self.algorithm.debug("RISK REDUCTION MODE: High drawdown detected")
            elif self.current_drawdown < self.max_drawdown * 0.3:  # Below 30% of max
                self.risk_reduction_mode = False
            
        except Exception as e:
            self.algorithm.debug(f"Error checking emergency conditions: {e}")
    
    def _calculate_risk_metrics(self):
        """Calculate comprehensive risk metrics"""
        metrics = {}
        
        try:
            portfolio_value = self.algorithm.portfolio.total_portfolio_value
            
            metrics['portfolio_risk'] = self._calculate_current_portfolio_risk()
            metrics['drawdown'] = self.current_drawdown
            metrics['var_5_percent'] = self.calculate_var(0.05, 30)
            metrics['num_positions'] = len(self.position_risks)
            metrics['sector_concentration'] = max(self.sector_exposure.values()) if self.sector_exposure else 0
            metrics['emergency_stop'] = self.emergency_stop_triggered
            metrics['risk_reduction_mode'] = self.risk_reduction_mode
            
            return metrics
            
        except Exception as e:
            self.algorithm.debug(f"Error calculating risk metrics: {e}")
            return {}
    
    def _get_volatility_multiplier(self, symbol):
        """Get position sizing multiplier based on volatility"""
        try:
            history = self.algorithm.history(symbol, 20, Resolution.DAILY)
            if history.empty:
                return 1.0
            
            returns = history['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Reduce position size in high volatility environments
            if volatility > 0.5:
                return 0.7
            elif volatility > 0.3:
                return 0.85
            else:
                return 1.0
                
        except Exception as e:
            return 1.0
    
    def _generate_risk_recommendations(self, risk_check, symbol):
        """Generate risk management recommendations"""
        recommendations = []
        
        if risk_check['risk_level'] == RiskLevel.HIGH:
            recommendations.append("Consider reducing position size")
            recommendations.append("Monitor position closely")
        
        if risk_check['risk_level'] == RiskLevel.CRITICAL:
            recommendations.append("Do not open position")
            recommendations.append("Review risk management parameters")
        
        if self.current_drawdown > 0.05:
            recommendations.append("Currently in drawdown - consider conservative sizing")
        
        if len(self.position_risks) > 15:
            recommendations.append("High number of positions - ensure diversification")
        
        return recommendations
    
    def _log_risk_summary(self):
        """Log periodic risk summary"""
        try:
            current_risk = self._calculate_current_portfolio_risk()
            var_95 = self.calculate_var(0.05, 30)
            
            self.algorithm.debug(f"RISK SUMMARY:")
            self.algorithm.debug(f"Portfolio Risk: {current_risk:.2%}")
            self.algorithm.debug(f"Current Drawdown: {self.current_drawdown:.2%}")
            self.algorithm.debug(f"95% VaR: ${var_95:,.0f}")
            self.algorithm.debug(f"Active Positions: {len(self.position_risks)}")
            self.algorithm.debug(f"Emergency Stop: {self.emergency_stop_triggered}")
            
        except Exception as e:
            self.algorithm.debug(f"Error logging risk summary: {e}")
    
    def get_risk_dashboard(self):
        """
        Get comprehensive risk dashboard data
        
        Returns:
            dict: Risk dashboard information
        """
        dashboard = {}
        
        try:
            dashboard['portfolio_risk'] = self._calculate_current_portfolio_risk()
            dashboard['max_portfolio_risk'] = self.max_portfolio_risk
            dashboard['current_drawdown'] = self.current_drawdown
            dashboard['max_drawdown'] = self.max_drawdown
            dashboard['var_95'] = self.calculate_var(0.05, 30)
            dashboard['active_positions'] = len(self.position_risks)
            dashboard['sector_exposure'] = dict(self.sector_exposure)
            dashboard['emergency_stop'] = self.emergency_stop_triggered
            dashboard['risk_reduction_mode'] = self.risk_reduction_mode
            dashboard['risk_level'] = self._get_overall_risk_level()
            
            return dashboard
            
        except Exception as e:
            self.algorithm.debug(f"Error creating risk dashboard: {e}")
            return {}
    
    def _get_overall_risk_level(self):
        """Determine overall portfolio risk level"""
        if self.emergency_stop_triggered:
            return RiskLevel.CRITICAL
        
        current_risk = self._calculate_current_portfolio_risk()
        risk_utilization = current_risk / self.max_portfolio_risk
        
        if risk_utilization > 0.8 or self.current_drawdown > 0.1:
            return RiskLevel.HIGH
        elif risk_utilization > 0.6 or self.current_drawdown > 0.05:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW