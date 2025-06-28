from AlgorithmImports import *
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque

class EarningsScreener:
    """
    Earnings Surprise Screener using Standardized Unexpected Earnings (SUE)
    
    SUE Formula: (EPS_current - EPS_4quarters_ago) / σ(EPS_changes)
    
    This class handles:
    - Earnings data collection and storage
    - SUE score calculation
    - Ranking and filtering of stocks by earnings surprise
    """
    
    def __init__(self, algorithm, lookback_quarters=8, min_data_points=4):
        self.algorithm = algorithm
        self.lookback_quarters = lookback_quarters
        self.min_data_points = min_data_points
        
        # Data storage
        self.earnings_data = defaultdict(lambda: deque(maxlen=self.lookback_quarters))
        self.sue_scores = {}
        self.earnings_estimates = defaultdict(dict)
        
        # Cache for performance
        self.last_calculation_date = None
        self.cached_rankings = []
    
    def update_earnings_data(self, fundamental_data):
        """
        Update earnings data from fundamental data feed
        
        Args:
            fundamental_data: List of fundamental data objects
        """
        current_date = self.algorithm.time
        current_quarter = self._get_quarter(current_date)
        
        for f in fundamental_data:
            symbol = f.symbol
            
            try:
                # Get current quarter EPS
                current_eps = f.earning_reports.basic_eps.three_months
                if current_eps is None or current_eps == 0:
                    continue
                
                # Get earnings estimate data if available
                estimate_data = self._extract_estimate_data(f)
                
                # Create earnings entry
                earnings_entry = {
                    'date': current_date,
                    'quarter': current_quarter,
                    'eps_actual': current_eps,
                    'eps_estimate': estimate_data.get('eps_estimate'),
                    'eps_surprise': estimate_data.get('eps_surprise'),
                    'revenue': f.earning_reports.total_revenue.three_months,
                    'shares_outstanding': f.earning_reports.basic_average_shares.three_months
                }
                
                # Only add if it's a new quarter or updated data
                if (not self.earnings_data[symbol] or 
                    self.earnings_data[symbol][-1]['quarter'] != current_quarter):
                    self.earnings_data[symbol].append(earnings_entry)
                else:
                    # Update the latest entry
                    self.earnings_data[symbol][-1] = earnings_entry
                    
            except Exception as e:
                self.algorithm.debug(f"Error updating earnings data for {symbol}: {e}")
                continue
    
    def _extract_estimate_data(self, fundamental):
        """
        Extract earnings estimate data from fundamental object
        
        Args:
            fundamental: Fundamental data object
            
        Returns:
            dict: Dictionary containing estimate data
        """
        try:
            # Try to get earnings estimate data
            estimate_data = {}
            
            # Check if earnings estimates are available
            if hasattr(fundamental, 'earnings_estimates'):
                estimates = fundamental.earnings_estimates
                if estimates and hasattr(estimates, 'quarter_earnings_estimate'):
                    estimate_data['eps_estimate'] = estimates.quarter_earnings_estimate
                    
                    # Calculate surprise if both actual and estimate are available
                    actual_eps = fundamental.earning_reports.basic_eps.three_months
                    if actual_eps and estimate_data['eps_estimate']:
                        surprise = actual_eps - estimate_data['eps_estimate']
                        estimate_data['eps_surprise'] = surprise
                        estimate_data['eps_surprise_percent'] = (surprise / abs(estimate_data['eps_estimate'])) * 100
            
            return estimate_data
            
        except Exception as e:
            return {}
    
    def calculate_sue_scores(self, symbols=None):
        """
        Calculate SUE scores for given symbols or all symbols with sufficient data
        
        Args:
            symbols: List of symbols to calculate SUE for. If None, calculate for all.
            
        Returns:
            dict: Dictionary mapping symbols to SUE scores
        """
        if symbols is None:
            symbols = list(self.earnings_data.keys())
        
        sue_scores = {}
        
        for symbol in symbols:
            try:
                sue_score = self._calculate_single_sue_score(symbol)
                if sue_score is not None:
                    sue_scores[symbol] = sue_score
            except Exception as e:
                self.algorithm.debug(f"Error calculating SUE for {symbol}: {e}")
                continue
        
        self.sue_scores = sue_scores
        return sue_scores
    
    def _calculate_single_sue_score(self, symbol):
        """
        Calculate SUE score for a single symbol
        
        Args:
            symbol: Symbol to calculate SUE for
            
        Returns:
            float: SUE score or None if insufficient data
        """
        if symbol not in self.earnings_data:
            return None
        
        earnings_history = list(self.earnings_data[symbol])
        
        if len(earnings_history) < self.min_data_points:
            return None
        
        # Get the most recent EPS
        current_eps = earnings_history[-1]['eps_actual']
        if current_eps is None:
            return None
        
        # Find EPS from 4 quarters ago (year-over-year comparison)
        year_ago_eps = self._find_year_ago_eps(earnings_history)
        if year_ago_eps is None:
            return None
        
        # Calculate the earnings surprise (unexpected earnings)
        earnings_surprise = current_eps - year_ago_eps
        
        # Calculate the standard deviation of historical earnings changes
        earnings_changes = self._calculate_historical_changes(earnings_history)
        
        if len(earnings_changes) < 3:
            return None
        
        std_dev = np.std(earnings_changes, ddof=1)  # Sample standard deviation
        
        if std_dev == 0 or np.isnan(std_dev):
            return None
        
        # Calculate SUE score
        sue_score = earnings_surprise / std_dev
        
        return sue_score
    
    def _find_year_ago_eps(self, earnings_history):
        """
        Find EPS from approximately 4 quarters ago
        
        Args:
            earnings_history: List of earnings data
            
        Returns:
            float: EPS from 4 quarters ago or None
        """
        if len(earnings_history) < 2:
            return None
        
        # Try to find exact 4-quarter match first
        current_quarter = earnings_history[-1]['quarter']
        current_year = int(current_quarter.split('Q')[0])
        current_qtr = int(current_quarter.split('Q')[1])
        
        target_year = current_year - 1
        target_quarter = f"{target_year}Q{current_qtr}"
        
        # Look for exact match
        for entry in reversed(earnings_history[:-1]):
            if entry['quarter'] == target_quarter and entry['eps_actual'] is not None:
                return entry['eps_actual']
        
        # If no exact match, find the closest historical EPS (at least 3 quarters back)
        for entry in reversed(earnings_history[:-3]):
            if entry['eps_actual'] is not None:
                return entry['eps_actual']
        
        return None
    
    def _calculate_historical_changes(self, earnings_history):
        """
        Calculate historical quarter-over-quarter earnings changes
        
        Args:
            earnings_history: List of earnings data
            
        Returns:
            list: List of earnings changes
        """
        changes = []
        
        for i in range(1, len(earnings_history)):
            current_eps = earnings_history[i]['eps_actual']
            previous_eps = earnings_history[i-1]['eps_actual']
            
            if current_eps is not None and previous_eps is not None:
                change = current_eps - previous_eps
                changes.append(change)
        
        return changes
    
    def rank_by_sue(self, symbols=None, top_percentage=0.05, min_sue_score=0):
        """
        Rank symbols by SUE score and return top performers
        
        Args:
            symbols: List of symbols to rank. If None, use all available.
            top_percentage: Percentage of top performers to return
            min_sue_score: Minimum SUE score threshold
            
        Returns:
            list: List of (symbol, sue_score) tuples, sorted by SUE score
        """
        # Calculate SUE scores if not already done
        if not self.sue_scores:
            self.calculate_sue_scores(symbols)
        
        # Filter by symbols if provided
        if symbols:
            filtered_scores = {s: score for s, score in self.sue_scores.items() 
                             if s in symbols and score >= min_sue_score}
        else:
            filtered_scores = {s: score for s, score in self.sue_scores.items() 
                             if score >= min_sue_score}
        
        # Sort by SUE score (descending)
        ranked_symbols = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top percentage
        if top_percentage > 0:
            top_count = max(1, int(len(ranked_symbols) * top_percentage))
            ranked_symbols = ranked_symbols[:top_count]
        
        self.cached_rankings = ranked_symbols
        self.last_calculation_date = self.algorithm.time
        
        return ranked_symbols
    
    def get_earnings_quality_metrics(self, symbol):
        """
        Get additional earnings quality metrics for a symbol
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            dict: Dictionary of earnings quality metrics
        """
        if symbol not in self.earnings_data:
            return {}
        
        earnings_history = list(self.earnings_data[symbol])
        
        if len(earnings_history) < 2:
            return {}
        
        metrics = {}
        
        try:
            # Earnings growth rate
            recent_eps = [e['eps_actual'] for e in earnings_history[-4:] if e['eps_actual'] is not None]
            if len(recent_eps) >= 2:
                earnings_growth = (recent_eps[-1] - recent_eps[0]) / abs(recent_eps[0]) if recent_eps[0] != 0 else 0
                metrics['earnings_growth_rate'] = earnings_growth
            
            # Earnings stability (coefficient of variation)
            if len(recent_eps) >= 3:
                eps_mean = np.mean(recent_eps)
                eps_std = np.std(recent_eps)
                if eps_mean != 0:
                    metrics['earnings_stability'] = eps_std / abs(eps_mean)
            
            # Consecutive positive surprises
            consecutive_positives = 0
            for entry in reversed(earnings_history):
                if entry.get('eps_surprise') and entry['eps_surprise'] > 0:
                    consecutive_positives += 1
                else:
                    break
            metrics['consecutive_positive_surprises'] = consecutive_positives
            
            # Average surprise magnitude
            surprises = [e.get('eps_surprise', 0) for e in earnings_history if e.get('eps_surprise') is not None]
            if surprises:
                metrics['avg_surprise_magnitude'] = np.mean(np.abs(surprises))
                metrics['avg_surprise'] = np.mean(surprises)
            
        except Exception as e:
            self.algorithm.debug(f"Error calculating quality metrics for {symbol}: {e}")
        
        return metrics
    
    def _get_quarter(self, date):
        """
        Get quarter identifier from date
        
        Args:
            date: datetime object
            
        Returns:
            str: Quarter identifier (e.g., "2023Q1")
        """
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}Q{quarter}"
    
    def get_summary_statistics(self):
        """
        Get summary statistics of the earnings screening process
        
        Returns:
            dict: Summary statistics
        """
        stats = {
            'total_symbols_tracked': len(self.earnings_data),
            'symbols_with_sue_scores': len(self.sue_scores),
            'avg_sue_score': np.mean(list(self.sue_scores.values())) if self.sue_scores else 0,
            'max_sue_score': max(self.sue_scores.values()) if self.sue_scores else 0,
            'min_sue_score': min(self.sue_scores.values()) if self.sue_scores else 0,
            'last_calculation_date': self.last_calculation_date
        }
        
        return stats