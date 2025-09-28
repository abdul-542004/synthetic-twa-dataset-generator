"""
Digital Twin Framework for Personalized Wellness Optimization
Builds on the synthetic TWA dataset to create individual digital twins for wellness journey optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
import copy
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import sys
import os

# Add parent directory to path for dataset generator imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from enhanced_longitudinal_generator import EnhancedLongitudinalTWADataGenerator
from wellness_aging_outcomes import WellnessAgingOutcomeGenerator
from twa_behavior_generator import ResearchValidatedTWAGenerator
from demographics_generator import EnhancedDemographicGenerator


class PersonalDigitalTwin:
    """
    Individual digital twin for personalized wellness optimization
    Uses synthetic dataset patterns to model and optimize real person's wellness journey
    """
    
    def __init__(self, person_id: str, demographics: Dict, initial_behaviors: Dict, 
                 initial_outcomes: Dict = None, random_seed: int = 42):
        """
        Initialize personal digital twin
        
        Args:
            person_id: Unique identifier for this person
            demographics: Person's demographic information
            initial_behaviors: Current/baseline TWA behaviors
            initial_outcomes: Current wellness outcomes (if available)
            random_seed: For reproducible modeling
        """
        self.person_id = person_id
        self.demographics = demographics
        self.current_behaviors = initial_behaviors.copy()
        self.initial_behaviors = initial_behaviors.copy()
        self.history = []  # Track changes over time
        
        # Initialize generators for modeling
        np.random.seed(random_seed)
        self.outcome_generator = WellnessAgingOutcomeGenerator(random_seed=random_seed)
        self.twa_generator = ResearchValidatedTWAGenerator(random_seed=random_seed)
        
        # Current wellness state
        if initial_outcomes:
            self.current_outcomes = initial_outcomes.copy()
        else:
            # Estimate initial outcomes from behaviors and demographics
            self.current_outcomes = self.outcome_generator.generate_aging_wellness_outcomes(
                self.demographics, self.current_behaviors, months_elapsed=0
            )
        
        # Modeling parameters
        self.behavior_change_capacity = self._assess_change_capacity()
        self.risk_tolerance = 0.5  # Default moderate risk tolerance
        self.priority_outcomes = ['biological_age_acceleration', 'life_satisfaction_score']
        
        # Track optimization history
        self.optimization_history = []
        
        print(f"Digital Twin initialized for {person_id}")
        print(f"Baseline wellness profile created from demographics and behaviors")
    
    def _assess_change_capacity(self) -> Dict[str, float]:
        """
        Assess person's capacity for behavior change based on demographics
        Returns likelihood of successful change for each behavior category
        """
        capacity = {}
        
        # Age effects on change capacity
        age = self.demographics.get('age_numeric', 45)
        if age < 30:
            base_capacity = 0.8  # High adaptability
        elif age < 50:
            base_capacity = 0.7  # Good adaptability
        elif age < 65:
            base_capacity = 0.6  # Moderate adaptability
        else:
            base_capacity = 0.5  # Lower but still meaningful adaptability
        
        # Education and income boost change capacity
        education_boost = {
            'Less than HS': 0.0, 'High School': 0.1, 
            'Some College': 0.15, 'Bachelor+': 0.2
        }
        income_boost = {
            '<$35k': 0.0, '$35-50k': 0.05, '$50-75k': 0.1, 
            '$75-100k': 0.15, '$100-150k': 0.2, '>$150k': 0.25
        }
        
        education_factor = education_boost.get(self.demographics.get('education', 'High School'), 0.1)
        income_factor = income_boost.get(self.demographics.get('income_bracket', '$50-75k'), 0.1)
        
        adjusted_capacity = min(0.95, base_capacity + education_factor + income_factor)
        
        # Different behaviors have different change difficulties
        capacity = {
            'motion_days_week': adjusted_capacity * 0.9,  # Moderate difficulty
            'diet_mediterranean_score': adjusted_capacity * 0.7,  # Harder to change
            'meditation_minutes_week': adjusted_capacity * 0.8,  # Moderate-hard
            'sleep_hours': adjusted_capacity * 0.6,  # Hard to change
            'smoking_status': adjusted_capacity * 0.4,  # Very hard
            'alcohol_drinks_week': adjusted_capacity * 0.6,  # Hard
            'social_connections_count': adjusted_capacity * 0.5,  # Hard
            'purpose_meaning_score': adjusted_capacity * 0.3,  # Very hard
        }
        
        return capacity
    
    def predict_outcome_changes(self, behavior_changes: Dict, months_ahead: int = 3) -> Dict:
        """
        Predict wellness outcomes from proposed behavior changes
        
        Args:
            behavior_changes: Dict of proposed changes to current behaviors
            months_ahead: How many months to project outcomes
            
        Returns:
            Dict with predicted outcomes and confidence intervals
        """
        # Create hypothetical future behaviors
        future_behaviors = self.current_behaviors.copy()
        
        # Apply proposed changes with change capacity constraints
        for behavior, change in behavior_changes.items():
            if behavior in self.behavior_change_capacity:
                # Limit change based on person's capacity
                max_change = self.behavior_change_capacity[behavior]
                if isinstance(change, (int, float)):
                    actual_change = change * max_change
                    future_behaviors[behavior] = self.current_behaviors.get(behavior, 0) + actual_change
                else:
                    # For categorical changes (like smoking status)
                    if np.random.random() < max_change:
                        future_behaviors[behavior] = change
        
        # Generate predicted outcomes
        predicted_outcomes = self.outcome_generator.generate_aging_wellness_outcomes(
            self.demographics, future_behaviors, months_elapsed=months_ahead
        )
        
        # Calculate confidence intervals (simplified approach)
        confidence_intervals = {}
        for outcome, value in predicted_outcomes.items():
            if isinstance(value, (int, float)):
                # Uncertainty increases with time and complexity
                uncertainty = 0.1 + (months_ahead * 0.02)  # 10% base + 2% per month
                std_error = abs(value) * uncertainty
                confidence_intervals[outcome] = {
                    'predicted': value,
                    'lower_95': value - (1.96 * std_error),
                    'upper_95': value + (1.96 * std_error),
                    'confidence': 0.95 - (months_ahead * 0.05)  # Decreasing confidence over time
                }
        
        return {
            'predicted_outcomes': predicted_outcomes,
            'confidence_intervals': confidence_intervals,
            'behavior_changes_applied': {k: v for k, v in future_behaviors.items() 
                                       if k in behavior_changes}
        }
    
    def optimize_activities(self, target_outcomes: Dict, constraints: Dict = None, 
                          optimization_horizon: int = 6) -> Dict:
        """
        Optimize TWA activities to achieve target wellness outcomes
        
        Args:
            target_outcomes: Dict of desired outcome values
            constraints: Limits on behavior changes (optional)
            optimization_horizon: Months to optimize over
            
        Returns:
            Optimized activity plan with expected outcomes
        """
        
        print(f"Optimizing activities for {self.person_id}...")
        print(f"Target outcomes: {target_outcomes}")
        
        # Define optimization bounds for each behavior
        behavior_bounds = self._get_optimization_bounds(constraints)
        
        # Define objective function (minimize distance to targets)
        def objective_function(behavior_vector):
            # Convert vector back to behavior dict
            behavior_changes = self._vector_to_behaviors(behavior_vector)
            
            # Predict outcomes
            predictions = self.predict_outcome_changes(behavior_changes, optimization_horizon)
            predicted = predictions['predicted_outcomes']
            
            # Calculate weighted distance to targets with better scaling
            total_error = 0
            target_count = 0
            
            for outcome, target in target_outcomes.items():
                if outcome in predicted:
                    current_value = self.current_outcomes.get(outcome, 0)
                    predicted_value = predicted[outcome]
                    
                    # Calculate improvement toward target
                    current_distance = abs(current_value - target)
                    predicted_distance = abs(predicted_value - target)
                    
                    # Penalize if we're moving away from target, reward if moving toward it
                    if predicted_distance > current_distance:
                        error = (predicted_distance - current_distance) * 10  # Heavy penalty for moving away
                    else:
                        error = predicted_distance  # Reward moving toward target
                    
                    total_error += error ** 2
                    target_count += 1
            
            # Normalize by number of targets
            if target_count > 0:
                total_error = total_error / target_count
            
            # Much lighter penalty for behavior changes to encourage meaningful recommendations
            change_penalty = 0
            for behavior, change in behavior_changes.items():
                if isinstance(change, (int, float)):
                    # Light penalty for very extreme changes only
                    if abs(change) > 10:  # Only penalize truly extreme changes
                        change_penalty += (abs(change) - 10) ** 2 * 0.01
            
            return total_error + change_penalty
        
        # Try multiple initial guesses to avoid local minima
        best_result = None
        best_objective = float('inf')
        
        # Initial guesses: no change, small positive changes, small negative changes
        initial_guesses = [
            self._behaviors_to_vector({}),  # No change
            self._behaviors_to_vector({k: 0.5 for k in behavior_bounds.keys()}),  # Small increases
            self._behaviors_to_vector({k: -0.5 for k in behavior_bounds.keys()}),  # Small decreases
        ]
        
        bounds_list = [(bound[0], bound[1]) for bound in behavior_bounds.values()]
        
        # Run optimization with multiple starting points
        try:
            for initial_vector in initial_guesses:
                result = minimize(
                    objective_function,
                    initial_vector,
                    method='L-BFGS-B',
                    bounds=bounds_list,
                    options={'maxiter': 200, 'ftol': 1e-9}
                )
                
                if result.fun < best_objective:
                    best_objective = result.fun
                    best_result = result
            
            # Use the best result
            result = best_result if best_result else result
            
            # Convert result back to behaviors
            optimal_changes = self._vector_to_behaviors(result.x)
            
            # If optimization produced minimal changes, use rule-based approach
            max_change = max(abs(change) for change in optimal_changes.values() if isinstance(change, (int, float)))
            if max_change < 0.1:  # Very small changes, use rule-based approach
                print("Optimization produced minimal changes, using rule-based recommendations...")
                optimal_changes = self._generate_rule_based_recommendations(target_outcomes)
            
            # Get final predictions
            final_predictions = self.predict_outcome_changes(optimal_changes, optimization_horizon)
            
            optimization_result = {
                'person_id': self.person_id,
                'optimization_successful': result.success,
                'optimal_behavior_changes': optimal_changes,
                'predicted_outcomes': final_predictions['predicted_outcomes'],
                'confidence_intervals': final_predictions['confidence_intervals'],
                'target_outcomes': target_outcomes,
                'optimization_horizon_months': optimization_horizon,
                'objective_value': result.fun,
                'change_feasibility': self._assess_change_feasibility(optimal_changes)
            }
            
            # Store in history
            self.optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'optimization_result': optimization_result
            })
            
            return optimization_result
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            return {
                'person_id': self.person_id,
                'optimization_successful': False,
                'error': str(e),
                'fallback_recommendations': self._generate_fallback_recommendations(target_outcomes)
            }
    
    def _get_optimization_bounds(self, constraints: Dict = None) -> Dict:
        """Get reasonable bounds for behavior optimization"""
        
        # Default bounds based on realistic behavior ranges
        default_bounds = {
            'motion_days_week': (-2, 4),  # Can reduce by 2 or increase by 4
            'diet_mediterranean_score': (-1, 3),  # Diet changes are gradual
            'meditation_minutes_week': (-30, 120),  # Meditation can change more dramatically
            'sleep_hours': (-0.5, 1.5),  # Sleep changes are limited
            'alcohol_drinks_week': (-10, 0),  # Usually trying to reduce alcohol
            'social_connections_count': (-1, 3),  # Social connections change slowly
        }
        
        # Apply person's change capacity (ensure minimum meaningful bounds)
        adjusted_bounds = {}
        for behavior, (lower, upper) in default_bounds.items():
            capacity = self.behavior_change_capacity.get(behavior, 0.5)
            
            # For optimization to be meaningful, use much more generous bounds
            # The capacity will be applied during actual behavior change implementation
            generous_lower = lower * 0.8  # Use 80% of default bounds for optimization
            generous_upper = upper * 0.8  # The capacity mainly affects implementation success
            
            adjusted_bounds[behavior] = (generous_lower, generous_upper)
        
        # Apply any specific constraints
        if constraints:
            for behavior, constraint in constraints.items():
                if behavior in adjusted_bounds:
                    if 'min_change' in constraint:
                        adjusted_bounds[behavior] = (
                            max(adjusted_bounds[behavior][0], constraint['min_change']),
                            adjusted_bounds[behavior][1]
                        )
                    if 'max_change' in constraint:
                        adjusted_bounds[behavior] = (
                            adjusted_bounds[behavior][0],
                            min(adjusted_bounds[behavior][1], constraint['max_change'])
                        )
        
        return adjusted_bounds
    
    def _behaviors_to_vector(self, behavior_changes: Dict) -> np.ndarray:
        """Convert behavior changes dict to optimization vector"""
        behavior_order = ['motion_days_week', 'diet_mediterranean_score', 
                         'meditation_minutes_week', 'sleep_hours', 
                         'alcohol_drinks_week', 'social_connections_count']
        
        vector = []
        for behavior in behavior_order:
            vector.append(behavior_changes.get(behavior, 0))
        
        return np.array(vector)
    
    def _vector_to_behaviors(self, vector: np.ndarray) -> Dict:
        """Convert optimization vector to behavior changes dict"""
        behavior_order = ['motion_days_week', 'diet_mediterranean_score', 
                         'meditation_minutes_week', 'sleep_hours', 
                         'alcohol_drinks_week', 'social_connections_count']
        
        behavior_changes = {}
        for i, behavior in enumerate(behavior_order):
            if i < len(vector):
                behavior_changes[behavior] = vector[i]
        
        return behavior_changes
    
    def _calculate_wellness_score(self) -> float:
        """Calculate comprehensive wellness score (0-100)"""
        
        # Behavioral component (40% of total score)
        behavior_score = 0
        behavior_weights = {
            'motion_days_week': 0.25,
            'diet_mediterranean_score': 0.25,
            'meditation_minutes_week': 0.15,
            'sleep_hours': 0.20,
            'social_connections_count': 0.15
        }
        
        for behavior, weight in behavior_weights.items():
            current_value = self.current_behaviors.get(behavior, 0)
            
            # Normalize each behavior to 0-1 scale
            if behavior == 'motion_days_week':
                normalized = min(current_value / 7, 1.0)  # 7 days = perfect
            elif behavior == 'diet_mediterranean_score':
                normalized = min(current_value / 10, 1.0)  # 10 = perfect Mediterranean diet
            elif behavior == 'meditation_minutes_week':
                normalized = min(current_value / 150, 1.0)  # 150 min/week = excellent
            elif behavior == 'sleep_hours':
                normalized = max(0, 1 - abs(current_value - 8) / 4)  # 8 hours optimal, ±4 range
            elif behavior == 'social_connections_count':
                normalized = min(current_value / 10, 1.0)  # 10 connections = excellent
            else:
                normalized = 0.5  # Default middle score
            
            behavior_score += normalized * weight
        
        # Outcome component (40% of total score)
        outcome_score = 0
        outcome_weights = {
            'biological_age_acceleration': 0.4,
            'life_satisfaction_score': 0.3,
            'mortality_risk_score': 0.3
        }
        
        for outcome, weight in outcome_weights.items():
            current_value = self.current_outcomes.get(outcome, 0)
            
            if outcome == 'biological_age_acceleration':
                # Better if more negative (younger biological age)
                normalized = max(0, min(1, (5 - current_value) / 10))  # -5 to +5 range
            elif outcome == 'life_satisfaction_score':
                normalized = min(current_value / 10, 1.0)  # 10 = perfect satisfaction
            elif outcome == 'mortality_risk_score':
                # Better if lower risk
                normalized = max(0, 1 - min(current_value / 0.1, 1))  # 0.1 = high risk
            else:
                normalized = 0.5
            
            outcome_score += normalized * weight
        
        # Risk factors penalty (20% of total score)
        risk_penalty = 0
        harmful_behaviors = {
            'alcohol_drinks_week': 14,  # Above 14 drinks/week is risky
            'processed_food_servings_week': 10,  # Above 10 servings/week
            'added_sugar_grams_day': 50,  # Above 50g/day
            'sodium_grams_day': 6  # Above 6g/day
        }
        
        for behavior, threshold in harmful_behaviors.items():
            current_value = self.current_behaviors.get(behavior, 0)
            if current_value > threshold:
                risk_penalty += (current_value - threshold) / threshold * 0.05  # 5% penalty per threshold
        
        risk_penalty = min(risk_penalty, 0.2)  # Cap at 20% penalty
        
        # Final wellness score (0-100)
        final_score = (behavior_score * 40 + outcome_score * 40 + (0.2 - risk_penalty) * 100)
        return max(0, min(100, final_score))
    
    def _assess_change_feasibility(self, behavior_changes: Dict) -> Dict:
        """Assess how feasible the proposed changes are for this person"""
        feasibility = {}
        
        for behavior, change in behavior_changes.items():
            if behavior in self.behavior_change_capacity:
                capacity = self.behavior_change_capacity[behavior]
                
                # Assess feasibility based on magnitude of change and capacity
                if isinstance(change, (int, float)):
                    change_magnitude = abs(change)
                    current_value = abs(self.current_behaviors.get(behavior, 1))
                    relative_change = change_magnitude / max(current_value, 1)
                    
                    if relative_change <= 0.2:  # Small change
                        feasibility_score = capacity
                    elif relative_change <= 0.5:  # Moderate change
                        feasibility_score = capacity * 0.7
                    else:  # Large change
                        feasibility_score = capacity * 0.4
                    
                    feasibility[behavior] = {
                        'feasibility_score': feasibility_score,
                        'change_magnitude': change_magnitude,
                        'relative_change': relative_change,
                        'recommendation': self._get_feasibility_recommendation(feasibility_score),
                        'change_capacity': capacity
                    }
        
        return feasibility
    
    def _get_feasibility_recommendation(self, score: float) -> str:
        """Get recommendation based on feasibility score"""
        if score >= 0.8:
            return "Highly feasible - good chance of success"
        elif score >= 0.6:
            return "Moderately feasible - requires commitment"
        elif score >= 0.4:
            return "Challenging - consider gradual approach"
        else:
            return "Very challenging - may need support/intervention"
    
    def _generate_rule_based_recommendations(self, target_outcomes: Dict) -> Dict:
        """Generate meaningful rule-based recommendations when optimization fails"""
        recommendations = {}
        
        for outcome, target in target_outcomes.items():
            current = self.current_outcomes.get(outcome, 0)
            
            if outcome == 'biological_age_acceleration':
                if current > target:  # Need to reduce biological age acceleration (get younger)
                    # Increase beneficial behaviors
                    current_motion = self.current_behaviors.get('motion_days_week', 0)
                    if current_motion < 5:
                        recommendations['motion_days_week'] = min(2, 5 - current_motion)
                    
                    current_diet = self.current_behaviors.get('diet_mediterranean_score', 0)
                    if current_diet < 8:
                        recommendations['diet_mediterranean_score'] = min(2, 8 - current_diet)
                    
                    current_meditation = self.current_behaviors.get('meditation_minutes_week', 0)
                    if current_meditation < 90:
                        recommendations['meditation_minutes_week'] = min(60, 90 - current_meditation)
                    
                    # Reduce harmful behaviors
                    current_alcohol = self.current_behaviors.get('alcohol_drinks_week', 0)
                    if current_alcohol > 7:
                        recommendations['alcohol_drinks_week'] = max(-3, 7 - current_alcohol)
            
            elif outcome == 'life_satisfaction_score':
                if current < target:  # Need to increase life satisfaction
                    current_social = self.current_behaviors.get('social_connections_count', 0)
                    if current_social < 8:
                        recommendations['social_connections_count'] = min(2, 8 - current_social)
                    
                    current_meditation = self.current_behaviors.get('meditation_minutes_week', 0)
                    if current_meditation < 60:
                        recommendations['meditation_minutes_week'] = min(45, 60 - current_meditation)
                    
                    current_sleep = self.current_behaviors.get('sleep_hours', 0)
                    if current_sleep < 7.5:
                        recommendations['sleep_hours'] = min(1, 7.5 - current_sleep)
            
            elif outcome == 'mortality_risk_score':
                if current > target:  # Need to reduce mortality risk
                    # Focus on high-impact changes
                    current_motion = self.current_behaviors.get('motion_days_week', 0)
                    if current_motion < 4:
                        recommendations['motion_days_week'] = min(2, 4 - current_motion)
                    
                    current_alcohol = self.current_behaviors.get('alcohol_drinks_week', 0)
                    if current_alcohol > 10:
                        recommendations['alcohol_drinks_week'] = max(-4, 10 - current_alcohol)
        
        # Ensure recommendations are within reasonable bounds
        for behavior, change in recommendations.items():
            bounds = self._get_optimization_bounds().get(behavior, (-5, 5))
            recommendations[behavior] = max(bounds[0], min(bounds[1], change))
        
        return recommendations
    
    def _generate_fallback_recommendations(self, target_outcomes: Dict) -> Dict:
        """Generate simple rule-based recommendations if optimization fails"""
        recommendations = {}
        
        # Simple heuristics for common targets
        for outcome, target in target_outcomes.items():
            current = self.current_outcomes.get(outcome, 0)
            
            if outcome == 'biological_age_acceleration':
                if current > target:  # Need to reduce biological age acceleration
                    recommendations.update({
                        'motion_days_week': 1,  # Increase exercise
                        'diet_mediterranean_score': 1,  # Improve diet
                        'meditation_minutes_week': 30  # Add meditation
                    })
            
            elif outcome == 'life_satisfaction_score':
                if current < target:  # Need to increase life satisfaction
                    recommendations.update({
                        'social_connections_count': 1,  # Improve social connections
                        'purpose_meaning_score': 0.5,  # Enhance purpose
                        'meditation_minutes_week': 20  # Add mindfulness
                    })
        
        return recommendations
    
    def update_actual_behaviors(self, new_behaviors: Dict, date: datetime = None):
        """Update digital twin with person's actual behavior data"""
        if date is None:
            date = datetime.now()
        
        # Store historical data
        self.history.append({
            'date': date.isoformat(),
            'behaviors': self.current_behaviors.copy(),
            'outcomes': self.current_outcomes.copy()
        })
        
        # Update current behaviors
        self.current_behaviors.update(new_behaviors)
        
        # Recalculate outcomes based on new behaviors
        months_since_baseline = len(self.history)
        self.current_outcomes = self.outcome_generator.generate_aging_wellness_outcomes(
            self.demographics, self.current_behaviors, months_elapsed=months_since_baseline
        )
        
        print(f"Digital twin updated for {self.person_id} on {date.strftime('%Y-%m-%d')}")
    
    def get_progress_report(self) -> Dict:
        """Generate comprehensive progress report"""
        if not self.history:
            return {"error": "No historical data available"}
        
        # Calculate trends
        behavior_trends = {}
        outcome_trends = {}
        
        # Get behavior changes over time
        for behavior in self.current_behaviors.keys():
            values = [self.initial_behaviors.get(behavior, 0)]
            values.extend([h['behaviors'].get(behavior, 0) for h in self.history])
            values.append(self.current_behaviors.get(behavior, 0))
            
            # Only calculate trends for numeric values
            numeric_values = []
            for v in values:
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    # Skip non-numeric values like 'Never', 'Former', etc.
                    continue
            
            if len(numeric_values) > 1:
                trend = np.polyfit(range(len(numeric_values)), numeric_values, 1)[0]  # Linear trend
                
                # Define which behaviors are "good" (higher = better) vs "bad" (lower = better)
                harmful_behaviors = {
                    'alcohol_drinks_week', 'processed_food_servings_week', 'added_sugar_grams_day',
                    'sodium_grams_day', 'smoking_status'  # For harmful behaviors, less is better
                }
                
                # Determine trajectory based on behavior type
                if behavior in harmful_behaviors:
                    # For harmful behaviors: negative trend = improving, positive trend = declining
                    trajectory = 'improving' if trend < 0 else 'declining' if trend > 0 else 'stable'
                else:
                    # For beneficial behaviors: positive trend = improving, negative trend = declining  
                    trajectory = 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable'
                
                behavior_trends[behavior] = {
                    'initial': numeric_values[0] if numeric_values else 0,
                    'current': numeric_values[-1] if numeric_values else 0,
                    'change': (numeric_values[-1] - numeric_values[0]) if len(numeric_values) > 1 else 0,
                    'trend': trend,
                    'trajectory': trajectory
                }
            elif len(values) > 0:
                # For non-numeric categorical values, just track changes
                behavior_trends[behavior] = {
                    'initial': values[0],
                    'current': values[-1],
                    'change': 'changed' if values[0] != values[-1] else 'no change',
                    'trend': 0,
                    'trajectory': 'categorical'
                }
        
        # Get outcome changes over time
        initial_outcomes = self.outcome_generator.generate_aging_wellness_outcomes(
            self.demographics, self.initial_behaviors, months_elapsed=0
        )
        
        for outcome in self.current_outcomes.keys():
            initial_value = initial_outcomes.get(outcome, 0)
            current_value = self.current_outcomes.get(outcome, 0)
            
            outcome_trends[outcome] = {
                'initial': initial_value,
                'current': current_value,
                'change': current_value - initial_value,
                'percent_change': ((current_value - initial_value) / max(abs(initial_value), 1)) * 100
            }
        
        # Calculate comprehensive wellness score
        wellness_score = self._calculate_wellness_score()
        
        return {
            'person_id': self.person_id,
            'report_date': datetime.now().isoformat(),
            'tracking_period_months': len(self.history),
            'behavior_trends': behavior_trends,
            'outcome_trends': outcome_trends,
            'optimization_count': len(self.optimization_history),
            'current_wellness_score': wellness_score
        }


class DigitalTwinOrchestrator:
    """
    Manages multiple digital twins and provides population-level insights
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize the orchestrator"""
        self.twins = {}  # person_id -> PersonalDigitalTwin
        self.population_model = EnhancedLongitudinalTWADataGenerator(random_seed=random_seed)
        self.random_seed = random_seed
        
    def create_digital_twin(self, person_id: str, demographics: Dict, 
                          initial_behaviors: Dict, initial_outcomes: Dict = None) -> PersonalDigitalTwin:
        """Create and register a new digital twin"""
        
        twin = PersonalDigitalTwin(
            person_id=person_id,
            demographics=demographics,
            initial_behaviors=initial_behaviors,
            initial_outcomes=initial_outcomes,
            random_seed=self.random_seed
        )
        
        self.twins[person_id] = twin
        return twin
    
    def get_twin(self, person_id: str) -> Optional[PersonalDigitalTwin]:
        """Get a specific digital twin"""
        return self.twins.get(person_id)
    
    def bulk_optimize(self, target_outcomes: Dict, person_ids: List[str] = None) -> Dict:
        """Optimize activities for multiple people"""
        
        if person_ids is None:
            person_ids = list(self.twins.keys())
        
        results = {}
        
        for person_id in person_ids:
            twin = self.twins.get(person_id)
            if twin:
                try:
                    result = twin.optimize_activities(target_outcomes)
                    results[person_id] = result
                except Exception as e:
                    results[person_id] = {'error': str(e)}
        
        return results
    
    def get_population_insights(self) -> Dict:
        """Get insights across all digital twins"""
        
        if not self.twins:
            return {"error": "No digital twins available"}
        
        # Aggregate statistics
        all_demographics = []
        all_behaviors = []
        all_outcomes = []
        
        for twin in self.twins.values():
            all_demographics.append(twin.demographics)
            all_behaviors.append(twin.current_behaviors)
            all_outcomes.append(twin.current_outcomes)
        
        # Convert to DataFrames for analysis
        demo_df = pd.DataFrame(all_demographics)
        behavior_df = pd.DataFrame(all_behaviors)
        outcome_df = pd.DataFrame(all_outcomes)
        
        insights = {
            'population_size': len(self.twins),
            'demographic_distribution': {
                col: demo_df[col].value_counts().to_dict() 
                for col in demo_df.select_dtypes(include=['object']).columns
            },
            'behavior_averages': behavior_df.select_dtypes(include=['number']).mean().to_dict(),
            'outcome_averages': outcome_df.mean().to_dict(),
            'behavior_correlations': behavior_df.select_dtypes(include=['number']).corr().to_dict(),
            'top_optimization_targets': self._get_common_optimization_targets()
        }
        
        return insights
    
    def _get_common_optimization_targets(self) -> Dict:
        """Identify most common optimization targets across population"""
        
        target_counts = {}
        
        for twin in self.twins.values():
            for opt_history in twin.optimization_history:
                targets = opt_history['optimization_result'].get('target_outcomes', {})
                for target in targets.keys():
                    target_counts[target] = target_counts.get(target, 0) + 1
        
        # Sort by frequency
        sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'most_common_targets': sorted_targets[:10],
            'total_optimizations': sum(target_counts.values())
        }
    
    def export_population_data(self, filepath: str):
        """Export all digital twin data for analysis"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'population_size': len(self.twins),
            'twins': {}
        }
        
        for person_id, twin in self.twins.items():
            export_data['twins'][person_id] = {
                'demographics': twin.demographics,
                'current_behaviors': twin.current_behaviors,
                'current_outcomes': twin.current_outcomes,
                'initial_behaviors': twin.initial_behaviors,
                'behavior_change_capacity': twin.behavior_change_capacity,
                'history_length': len(twin.history),
                'optimization_count': len(twin.optimization_history)
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Population data exported to {filepath}")


if __name__ == "__main__":
    # Demonstration of the Digital Twin Framework
    
    print("="*80)
    print("DIGITAL TWIN FRAMEWORK DEMONSTRATION")
    print("Personalized Wellness Optimization System")
    print("="*80)
    
    # Initialize orchestrator
    orchestrator = DigitalTwinOrchestrator(random_seed=42)
    
    # Example person 1: Young professional wanting to improve aging outcomes
    person1_demographics = {
        'age_numeric': 32,
        'gender': 'Female',
        'education': 'Bachelor+',
        'income_bracket': '$75-100k',
        'urban_rural': 'Urban',
        'fitness_level': 'Medium'
    }
    
    person1_behaviors = {
        'motion_days_week': 2,
        'diet_mediterranean_score': 5.5,
        'meditation_minutes_week': 0,
        'sleep_hours': 6.5,
        'alcohol_drinks_week': 8,
        'social_connections_count': 4
    }
    
    # Create digital twin
    twin1 = orchestrator.create_digital_twin(
        person_id="person_001",
        demographics=person1_demographics,
        initial_behaviors=person1_behaviors
    )
    
    print(f"\nDigital Twin created for person_001")
    print(f"Current biological age acceleration: {twin1.current_outcomes['biological_age_acceleration']:.2f} years")
    print(f"Current life satisfaction: {twin1.current_outcomes['life_satisfaction_score']:.1f}/10")
    
    # Optimize activities to reduce biological aging and increase life satisfaction
    optimization_targets = {
        'biological_age_acceleration': -1.0,  # Target 1 year younger biological age
        'life_satisfaction_score': 8.0       # Target high life satisfaction
    }
    
    print(f"\nOptimizing activities...")
    optimization_result = twin1.optimize_activities(optimization_targets)
    
    if optimization_result['optimization_successful']:
        print(f"Optimization successful!")
        print(f"Recommended behavior changes:")
        for behavior, change in optimization_result['optimal_behavior_changes'].items():
            current = twin1.current_behaviors.get(behavior, 0)
            new_value = current + change
            print(f"  {behavior}: {current:.1f} -> {new_value:.1f} (change: {change:+.1f})")
        
        print(f"\nPredicted outcomes:")
        for outcome, value in optimization_result['predicted_outcomes'].items():
            if outcome in optimization_targets:
                print(f"  {outcome}: {value:.2f} (target: {optimization_targets[outcome]})")
        
        print(f"\nChange feasibility assessment:")
        for behavior, feasibility in optimization_result['change_feasibility'].items():
            score = feasibility['feasibility_score']
            recommendation = feasibility['recommendation']
            print(f"  {behavior}: {score:.2f} - {recommendation}")
    
    # Simulate person following recommendations for 3 months
    print(f"\n" + "="*50)
    print("SIMULATING 3 MONTHS OF PROGRESS")
    print("="*50)
    
    # Month 1 - partial implementation
    month1_behaviors = twin1.current_behaviors.copy()
    for behavior, change in optimization_result['optimal_behavior_changes'].items():
        if behavior in month1_behaviors:
            month1_behaviors[behavior] += change * 0.3  # 30% implementation
    
    twin1.update_actual_behaviors(month1_behaviors, datetime.now() - timedelta(days=60))
    
    # Month 2 - better implementation
    month2_behaviors = twin1.current_behaviors.copy()
    for behavior, change in optimization_result['optimal_behavior_changes'].items():
        if behavior in month2_behaviors:
            month2_behaviors[behavior] += change * 0.3  # Additional 30%
    
    twin1.update_actual_behaviors(month2_behaviors, datetime.now() - timedelta(days=30))
    
    # Month 3 - full implementation
    month3_behaviors = twin1.current_behaviors.copy()
    for behavior, change in optimization_result['optimal_behavior_changes'].items():
        if behavior in month3_behaviors:
            month3_behaviors[behavior] += change * 0.4  # Final 40%
    
    twin1.update_actual_behaviors(month3_behaviors, datetime.now())
    
    # Generate progress report
    progress_report = twin1.get_progress_report()
    
    print(f"\nPROGRESS REPORT for {progress_report['person_id']}:")
    print(f"Tracking Period: {progress_report['tracking_period_months']} months")
    print(f"Current Wellness Score: {progress_report['current_wellness_score']:.1f}")
    
    print(f"\nKey Behavior Changes:")
    for behavior, trend in progress_report['behavior_trends'].items():
        if abs(trend['change']) > 0.1:  # Only show meaningful changes
            print(f"  {behavior}: {trend['initial']:.1f} -> {trend['current']:.1f} "
                  f"({trend['change']:+.1f}, {trend['trajectory']})")
    
    print(f"\nKey Outcome Changes:")
    for outcome, trend in progress_report['outcome_trends'].items():
        if outcome in optimization_targets:
            print(f"  {outcome}: {trend['initial']:.2f} -> {trend['current']:.2f} "
                  f"({trend['change']:+.2f}, {trend['percent_change']:+.1f}%)")
    
    # Population insights
    print(f"\n" + "="*50)
    print("POPULATION INSIGHTS")
    print("="*50)
    
    population_insights = orchestrator.get_population_insights()
    print(f"Population Size: {population_insights['population_size']}")
    print(f"Average Behaviors:")
    for behavior, avg in list(population_insights['behavior_averages'].items())[:5]:
        print(f"  {behavior}: {avg:.1f}")
    
    print(f"\nFramework demonstration completed successfully!")
    print(f"Digital twin ready for real-world deployment.")