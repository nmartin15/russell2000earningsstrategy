# Russell 2000 Earnings Surprise Options Strategy

A comprehensive quantitative trading strategy for QuantConnect that targets Russell 2000 stocks with the biggest earnings surprises and implements sophisticated options strategies with dynamic risk management.

## Strategy Overview

This algorithm implements a multi-component trading strategy:

1. **Universe Selection**: Russell 2000 stocks via IWM ETF constituents
2. **Earnings Screening**: Standardized Unexpected Earnings (SUE) to identify top 5% performers
3. **Options Strategies**: Short vertical spreads and broken wing butterflies
4. **Risk Management**: Comprehensive portfolio and position-level risk controls
5. **Dynamic Management**: Automatic adjustments for upside moves and changing conditions

## Files Structure

### Core Strategy Files
- `integrated_strategy.py` - Main algorithm integrating all components
- `russell_2000_earnings_options_strategy.py` - Original standalone implementation

### Component Modules
- `earnings_screener.py` - SUE calculation and earnings surprise screening
- `options_manager.py` - Options strategy implementation and management
- `risk_manager.py` - Comprehensive risk management system

## Key Features

### Earnings Surprise Screening (SUE)
- **Formula**: `SUE = (EPS_current - EPS_4quarters_ago) / σ(EPS_changes)`
- Tracks quarterly earnings data with 8-quarter lookback
- Ranks stocks by SUE score and selects top 5%
- Filters for quality and liquidity requirements

### Options Strategies

#### Short Vertical Spreads (Bear Call Spreads)
- **Use Case**: High SUE scores with elevated implied volatility
- **Structure**: Sell ATM call, buy OTM call
- **Profit**: Premium collected from time decay
- **Risk**: Limited to strike width minus premium

#### Broken Wing Butterfly Spreads
- **Use Case**: Moderate to high SUE scores
- **Structure**: Asymmetric butterfly with upside bias
- **Profit**: Benefits from limited stock movement with upside potential
- **Risk**: Limited maximum loss with defined profit zones

### Risk Management

#### Portfolio Level
- Maximum 20% portfolio at risk
- Maximum 5% risk per position  
- Maximum 30% sector concentration
- Daily loss limits (3%) and drawdown limits (15%)

#### Position Level
- Greeks monitoring (Delta, Gamma, Vega, Theta)
- Correlation risk assessment
- Volatility-based position sizing
- Emergency stop mechanisms

#### Dynamic Management
- 50% profit targets
- 25% stop losses
- Time decay management (close 7 days before expiration)
- Upside move adjustments (>10% underlying moves)

## Usage Instructions

### For QuantConnect Platform

1. **Upload Files**: Upload all Python files to your QuantConnect algorithm folder

2. **Main Algorithm**: Use `integrated_strategy.py` as your main algorithm file

3. **Configuration**: Adjust parameters in the `initialize()` method:
   ```python
   self.max_positions = 20          # Maximum concurrent positions
   self.top_percentage = 0.05       # Top 5% by SUE score
   self.max_portfolio_risk = 0.20   # 20% max portfolio risk
   ```

4. **Backtest Period**: Default 2022-2024, adjust as needed:
   ```python
   self.set_start_date(2022, 1, 1)
   self.set_end_date(2024, 12, 31)
   ```

5. **Capital**: Default $1M, adjust based on your account:
   ```python
   self.set_cash(1000000)
   ```

### Strategy Parameters

#### Earnings Screening
- `lookback_quarters = 8` - Historical earnings data periods
- `min_sue_score = 0.5` - Minimum SUE threshold
- `earnings_lookback_quarters = 8` - Quarters for SUE calculation

#### Options Selection
- `days_to_expiration_min = 30` - Minimum DTE
- `days_to_expiration_max = 60` - Maximum DTE
- `target_dte = 45` - Target days to expiration
- `target_delta = 0.20` - Target delta for short strikes

#### Risk Controls
- `max_portfolio_risk = 0.20` - Maximum portfolio risk
- `max_position_risk = 0.05` - Maximum per position risk
- `profit_target = 0.50` - Close at 50% profit
- `stop_loss_threshold = 0.25` - Stop loss at 25% of max risk

## Algorithm Logic Flow

### Monthly Rebalancing
1. **Universe Update**: Get current IWM constituents
2. **Fundamental Screening**: Apply quality filters
3. **SUE Calculation**: Calculate earnings surprises
4. **Ranking**: Rank by SUE score, select top 5%
5. **Position Management**: Close old positions, open new ones
6. **Risk Validation**: Verify all positions pass risk checks

### Daily Management  
1. **Risk Monitoring**: Check portfolio risk metrics
2. **Position Management**: Monitor existing options positions
3. **Profit/Loss Management**: Close positions meeting targets
4. **Upside Adjustments**: Handle significant underlying moves
5. **Emergency Controls**: Trigger stops if risk limits exceeded

### Options Strategy Selection
- **High SUE (>2.0) + High IV**: Short Vertical Spreads
- **High SUE (>2.0) + Low IV**: Broken Wing Butterfly
- **Moderate SUE (1-2)**: Short Vertical Spreads
- **Lower SUE (<1)**: Broken Wing Butterfly

## Performance Expectations

### Target Metrics
- **Sharpe Ratio**: 0.8-1.2 (based on SUE research)
- **Maximum Drawdown**: <15%
- **Win Rate**: 60-70% (premium selling strategies)
- **Average Hold Time**: 30-45 days

### Risk Characteristics
- **Market Neutral**: Limited directional exposure
- **Volatility Sensitive**: Benefits from time decay
- **Earnings Driven**: Alpha from earnings surprises
- **Small Cap Focus**: Russell 2000 universe

## Customization Options

### Modify Universe Size
```python
# In _etf_filter method
return [c.symbol for c in sorted_constituents[:1000]]  # Increase from 500
```

### Adjust SUE Selection
```python
# In _fundamental_filter method
self.top_percentage = 0.03  # Top 3% instead of 5%
min_sue_score = 1.0         # Higher SUE threshold
```

### Change Options Strategy Mix
```python
# In evaluate_strategy_for_symbol method
if sue_score > 1.5:  # Lower threshold for verticals
    return StrategyType.SHORT_VERTICAL
```

### Modify Risk Parameters
```python
# In RiskManager initialization
self.max_portfolio_risk = 0.15      # More conservative
self.profit_target = 0.40           # Earlier profit taking
self.stop_loss_threshold = 0.20     # Tighter stops
```

## Monitoring and Debugging

### Key Debug Messages
- Universe selection results
- SUE screening statistics  
- Position opening/closing events
- Risk limit breaches
- Performance summaries

### Performance Tracking
- Monthly portfolio snapshots
- Trade log with fill details
- Risk metrics history
- SUE score distributions

### Risk Dashboard
Access comprehensive risk metrics:
```python
dashboard = self.risk_manager.get_risk_dashboard()
```

## Important Notes

### QuantConnect Requirements
- **Data**: US Equity + Options + Fundamentals
- **Tier**: Professional tier recommended for options data
- **Memory**: Strategy uses significant memory for universe selection

### Limitations
- Requires options trading permissions
- Higher data costs due to options chains
- Complex strategy may have execution delays
- Backtesting differs from live trading conditions

### Best Practices
- Start with paper trading to validate execution
- Monitor risk metrics closely during live trading  
- Adjust position sizes based on account size
- Review and update parameters periodically

## Support and Development

This strategy is designed for educational and research purposes. Users should:
- Thoroughly backtest before live implementation
- Understand options trading risks
- Monitor positions actively
- Adjust parameters for their risk tolerance

For questions or improvements, review the code comments and QuantConnect documentation.