"""
Wellness & Aging Outcome Generator
Scientific outcome modeling using expert consensus biomarkers and validated effect sizes
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple
from scipy.stats import norm, truncnorm, gamma, lognorm
import math


class WellnessAgingOutcomeGenerator:
    """
    Generates scientifically-grounded aging and wellness outcomes based on TWA behaviors
    Uses validated effect sizes from longevity research and expert consensus biomarkers
    """
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Research-validated effect sizes from longevity studies
        self.research_effects = {
            'biological_age_effects': {
                # Protective effects (negative = age reduction)
                'motion_high': -1.2,           # Regular exercise (3+ days/week)
                'diet_mediterranean': -2.3,    # High Mediterranean diet adherence
                'meditation_regular': -1.8,    # Regular mindfulness practice (150+ min/week)
                'sleep_quality': -0.5,         # Per quality sleep hour above 6
                'purpose_high': -3.1,          # Strong life purpose (8+ score)
                'social_connected': -1.5,      # Strong social connections (4+ close)
                'nature_connection': -0.8,     # Regular nature exposure
                'cultural_engagement': -0.6,   # Active cultural participation
                
                # Risk effects (positive = age acceleration)
                'smoking_current': +5.3,       # Current smoking
                'alcohol_excess': +2.1,        # Heavy drinking (>14 drinks/week)
                'processed_foods': +1.7,       # High ultra-processed food consumption
                'sugar_excess': +1.2,          # High added sugar (>50g/day)
                'sodium_excess': +0.9,         # High sodium (>6g/day)
                'social_isolated': +2.8,       # Social isolation (<2 connections)
                'sedentary': +1.5,             # Very low activity (<1 day/week)
                'sleep_poor': +1.1             # Poor sleep quality (<6 hours, <5 quality)
            },
            
            'mortality_risk_effects': {
                # Hazard ratios from epidemiological studies
                'purpose_high': 0.57,          # HR from Boyle et al. (40% reduction)
                'social_isolated': 1.91,       # HR from meta-analysis (91% increase)
                'smoking_current': 2.24,       # HR from epidemiological studies
                'exercise_regular': 0.72,      # HR from exercise studies (28% reduction)
                'diet_quality_high': 0.70,     # HR from dietary studies (30% reduction)
                'meditation_practice': 0.82,   # HR from mindfulness studies (18% reduction)
                'alcohol_excess': 1.31,        # HR from alcohol studies (31% increase)
                'sleep_quality_good': 0.85,    # HR from sleep studies (15% reduction)
                'nature_connection': 0.88,     # HR from green space studies (12% reduction)
                'social_connected': 0.50       # HR from social connection meta-analyses
            },
            
            'biomarker_effects': {
                # Inflammatory markers (% change)
                'crp_reduction_exercise': -0.25,    # 25% CRP reduction from exercise
                'crp_increase_smoking': +0.60,      # 60% CRP increase from smoking
                'il6_reduction_meditation': -0.30,  # 30% IL-6 reduction from meditation
                'il6_increase_isolation': +0.40,    # 40% IL-6 increase from isolation
                
                # Growth factors and aging markers
                'igf1_reduction_diet': -0.18,       # 18% IGF-1 reduction from caloric restriction
                'gdf15_increase_aging': +0.15,      # 15% GDF-15 increase per decade
                'cortisol_reduction_nature': -0.30, # 30% cortisol reduction from nature
                'cortisol_increase_stress': +0.45,  # 45% cortisol increase from chronic stress
                
                # Cellular aging markers
                'telomere_exercise': +0.15,         # 15% telomere length increase
                'telomere_smoking': -0.25,          # 25% telomere shortening from smoking
                'dna_methylation_diet': -0.12       # 12% improvement in methylation age
            }
        }
        
        # Reference ranges for biomarkers (healthy adult ranges)
        self.biomarker_ranges = {
            'crp': {'mean': 1.5, 'std': 1.2, 'min': 0.1, 'max': 10.0, 'units': 'mg/L'},
            'il6': {'mean': 2.8, 'std': 1.8, 'min': 0.5, 'max': 15.0, 'units': 'pg/mL'},
            'igf1': {'mean': 180, 'std': 60, 'min': 50, 'max': 400, 'units': 'ng/mL'},
            'gdf15': {'mean': 800, 'std': 400, 'min': 200, 'max': 3000, 'units': 'pg/mL'},
            'cortisol': {'mean': 12, 'std': 4, 'min': 3, 'max': 25, 'units': 'μg/dL'},
            'grip_strength': {'mean': 35, 'std': 12, 'min': 10, 'max': 70, 'units': 'kg'},
            'gait_speed': {'mean': 1.3, 'std': 0.3, 'min': 0.4, 'max': 2.0, 'units': 'm/s'},
            'balance_score': {'mean': 75, 'std': 15, 'min': 20, 'max': 100, 'units': 'score'},
            'frailty_index': {'mean': 0.15, 'std': 0.12, 'min': 0.0, 'max': 0.7, 'units': 'ratio'}
        }
        
        # Life expectancy baseline by demographics (US 2023 data)
        self.life_expectancy_base = {
            'Male': {'White': 76.4, 'Black': 71.5, 'Hispanic': 78.8, 'Asian': 81.2, 'Other': 75.0},
            'Female': {'White': 81.1, 'Black': 76.1, 'Hispanic': 83.8, 'Asian': 85.5, 'Other': 80.0},
            'Non-binary': {'White': 78.8, 'Black': 73.8, 'Hispanic': 81.3, 'Asian': 83.4, 'Other': 77.5}
        }
    
    def _calculate_biological_age_base(self, chronological_age: int, demographics: Dict) -> float:
        """Calculate baseline biological age with demographic adjustments"""
        
        # Start with chronological age
        bio_age = float(chronological_age)
        
        # Gender effects (women age more slowly on average)
        if demographics['gender'] == 'Female':
            bio_age -= 0.5
        elif demographics['gender'] == 'Male':
            bio_age += 0.3
        
        # Ethnicity effects (based on health disparities research)
        ethnicity_effects = {
            'White': 0.0, 'Asian': -0.8, 'Hispanic': -0.3, 
            'Black': +1.2, 'Other': +0.1
        }
        bio_age += ethnicity_effects.get(demographics['ethnicity'], 0.0)
        
        # Socioeconomic effects
        education_effects = {
            'Less than HS': +2.0, 'High School': +0.5, 
            'Some College': -0.2, 'Bachelor+': -1.5
        }
        bio_age += education_effects.get(demographics['education'], 0.0)
        
        income_effects = {
            '<$35k': +1.5, '$35-50k': +0.8, '$50-75k': +0.2,
            '$75-100k': -0.3, '$100-150k': -0.8, '>$150k': -1.2
        }
        bio_age += income_effects.get(demographics['income_bracket'], 0.0)
        
        return bio_age
    
    def _calculate_biological_age_from_behaviors(self, base_bio_age: float, 
                                               twa_behaviors: Dict, months_elapsed: int) -> float:
        """Calculate biological age modification from TWA behaviors"""
        
        effects = self.research_effects['biological_age_effects']
        annual_age_modification = 0.0
        
        # Apply protective effects (Do More behaviors) - FIXED THRESHOLDS FOR REALISTIC PREVALENCE
        if twa_behaviors['motion_days_week'] >= 5:  # Increased from 3 to match 28% prevalence target
            annual_age_modification += effects['motion_high']
            
        if twa_behaviors['diet_mediterranean_score'] >= 7:  # Increased from 6 to match 22% prevalence target
            annual_age_modification += effects['diet_mediterranean']
            
        if twa_behaviors['meditation_minutes_week'] >= 150:  # Increased from 60 to match 15% prevalence target
            annual_age_modification += effects['meditation_regular']
            
        if twa_behaviors['sleep_hours'] >= 7 and twa_behaviors['sleep_quality_score'] >= 6:
            sleep_bonus_hours = max(0, twa_behaviors['sleep_quality_score'] - 6)
            annual_age_modification += effects['sleep_quality'] * sleep_bonus_hours
            
        if twa_behaviors['purpose_meaning_score'] >= 8:  # Increased from 6 to match realistic prevalence
            annual_age_modification += effects['purpose_high']
            
        if twa_behaviors['social_connections_count'] >= 4:  # Increased from 3 to match realistic prevalence
            annual_age_modification += effects['social_connected']
            
        if twa_behaviors['nature_minutes_week'] >= 120:  # 2+ hours/week
            annual_age_modification += effects['nature_connection']
            
        if twa_behaviors['cultural_hours_week'] >= 5:  # Increased from 3
            annual_age_modification += effects['cultural_engagement']
        
        # Apply risk effects (Do Less behaviors) - FIXED SIGNS
        if twa_behaviors['smoking_status'] == 'Current':
            annual_age_modification += effects['smoking_current']  # Positive = aging acceleration
            
        if twa_behaviors['alcohol_drinks_week'] > 14:
            annual_age_modification += effects['alcohol_excess']
            
        if twa_behaviors['processed_food_servings_week'] > 10:
            annual_age_modification += effects['processed_foods']
            
        if twa_behaviors['added_sugar_grams_day'] > 50:
            annual_age_modification += effects['sugar_excess']
            
        if twa_behaviors['sodium_grams_day'] > 6:
            annual_age_modification += effects['sodium_excess']
        
        # Social isolation and sedentary penalties
        if twa_behaviors['social_connections_count'] < 2:
            annual_age_modification += effects['social_isolated']
            
        if twa_behaviors['motion_days_week'] < 1:
            annual_age_modification += effects['sedentary']
            
        if (twa_behaviors['sleep_hours'] < 6 or twa_behaviors['sleep_quality_score'] < 5):
            annual_age_modification += effects['sleep_poor']
        
        # FIXED: Convert annual effect to monthly and apply diminishing returns
        # Use logarithmic scaling to prevent extreme effects
        if annual_age_modification != 0:
            # Apply diminishing returns for multiple interventions
            sign = 1 if annual_age_modification > 0 else -1
            abs_effect = abs(annual_age_modification)
            # Stronger diminishing returns: each additional intervention has much less impact
            adjusted_effect = sign * (abs_effect ** 0.5)  # Square root for stronger diminishing returns
            
            monthly_effect = adjusted_effect / 12  # Convert annual to monthly
            cumulative_effect = monthly_effect * months_elapsed
        else:
            cumulative_effect = 0
        
        # Natural aging progression (reduced to account for intervention effects)
        natural_aging = months_elapsed / 12 * 0.8  # Slightly less than 1 year per year
        
        biological_age = base_bio_age + natural_aging + cumulative_effect
        
        return max(18, biological_age)  # Minimum biological age of 18
    
    def _calculate_mortality_risk_score(self, demographics: Dict, twa_behaviors: Dict, 
                                      biological_age: float) -> float:
        """Calculate mortality risk score based on behaviors and bioage"""
        
        # Start with baseline risk (age-adjusted)
        chronological_age = demographics['age_numeric']
        base_risk = 0.01 * math.exp((chronological_age - 50) / 25)  # Exponential age effect
        
        # Apply behavior-based hazard ratio modifications
        hr_effects = self.research_effects['mortality_risk_effects']
        risk_multiplier = 1.0
        
        # Protective factors
        if twa_behaviors['purpose_meaning_score'] >= 8:
            risk_multiplier *= hr_effects['purpose_high']
            
        if twa_behaviors['social_connections_count'] >= 4:
            risk_multiplier *= hr_effects['social_connected']
            
        if twa_behaviors['motion_days_week'] >= 3:
            risk_multiplier *= hr_effects['exercise_regular']
            
        if twa_behaviors['diet_mediterranean_score'] >= 7:
            risk_multiplier *= hr_effects['diet_quality_high']
            
        if twa_behaviors['meditation_minutes_week'] >= 150:
            risk_multiplier *= hr_effects['meditation_practice']
            
        if (twa_behaviors['sleep_hours'] >= 7 and twa_behaviors['sleep_quality_score'] >= 6):
            risk_multiplier *= hr_effects['sleep_quality_good']
            
        if twa_behaviors['nature_minutes_week'] >= 120:
            risk_multiplier *= hr_effects['nature_connection']
        
        # Risk factors
        if twa_behaviors['social_connections_count'] < 2:
            risk_multiplier *= hr_effects['social_isolated']
            
        if twa_behaviors['smoking_status'] == 'Current':
            risk_multiplier *= hr_effects['smoking_current']
            
        if twa_behaviors['alcohol_drinks_week'] > 14:
            risk_multiplier *= hr_effects['alcohol_excess']
        
        # Biological age acceleration effect
        age_acceleration = biological_age - chronological_age
        if age_acceleration > 0:
            risk_multiplier *= (1 + age_acceleration * 0.08)  # 8% increase per year of acceleration
        else:
            risk_multiplier *= (1 + age_acceleration * 0.05)  # 5% decrease per year younger bio age
        
        final_risk = base_risk * risk_multiplier
        return min(1.0, max(0.001, final_risk))  # Bounded between 0.1% and 100%
    
    def _calculate_estimated_lifespan(self, demographics: Dict, mortality_risk: float) -> float:
        """Calculate estimated lifespan based on demographics and risk factors"""
        
        # Base life expectancy from demographics
        gender = demographics['gender']
        ethnicity = demographics['ethnicity']
        base_lifespan = self.life_expectancy_base[gender][ethnicity]
        
        # Adjust for current age (conditional life expectancy)
        current_age = demographics['age_numeric']
        if current_age > 65:
            # Adjust upward for survival to current age
            base_lifespan += (current_age - 65) * 0.2
        
        # Apply mortality risk modification
        # Convert annual mortality risk to lifespan impact
        if mortality_risk > 0.001:
            risk_factor = -math.log(1 - mortality_risk) * 20  # Approximate conversion
            lifespan_adjustment = -risk_factor
        else:
            lifespan_adjustment = 5  # Bonus for very low risk
        
        estimated_lifespan = base_lifespan + lifespan_adjustment
        return max(current_age + 1, min(120, estimated_lifespan))  # Reasonable bounds
    
    def _generate_biomarkers(self, demographics: Dict, twa_behaviors: Dict, 
                           biological_age: float) -> Dict:
        """Generate biomarkers based on behaviors and biological age"""
        
        biomarkers = {}
        
        # Inflammatory markers (CRP)
        crp_base = self.biomarker_ranges['crp']['mean']
        
        # Exercise effect on CRP
        if twa_behaviors['motion_days_week'] >= 3:
            crp_base *= (1 + self.research_effects['biomarker_effects']['crp_reduction_exercise'])
            
        # Smoking effect on CRP
        if twa_behaviors['smoking_status'] == 'Current':
            crp_base *= (1 + self.research_effects['biomarker_effects']['crp_increase_smoking'])
        
        # Age effect
        age_effect = (biological_age - 40) / 40 * 0.3  # 30% increase per 40 years
        crp_base *= (1 + age_effect)
        
        # Add random variation
        crp_value = max(0.1, crp_base * np.random.lognormal(0, 0.4))
        biomarkers['crp'] = min(10.0, crp_value)
        
        # IL-6 (similar pattern)
        il6_base = self.biomarker_ranges['il6']['mean']
        
        if twa_behaviors['meditation_minutes_week'] >= 150:
            il6_base *= (1 + self.research_effects['biomarker_effects']['il6_reduction_meditation'])
            
        if twa_behaviors['social_connections_count'] < 2:
            il6_base *= (1 + self.research_effects['biomarker_effects']['il6_increase_isolation'])
        
        il6_base *= (1 + age_effect)
        il6_value = max(0.5, il6_base * np.random.lognormal(0, 0.3))
        biomarkers['il6'] = min(15.0, il6_value)
        
        # IGF-1 (decreases with age and caloric restriction)
        igf1_base = self.biomarker_ranges['igf1']['mean']
        
        # Diet quality effect
        if twa_behaviors['diet_mediterranean_score'] >= 7:
            igf1_base *= (1 + self.research_effects['biomarker_effects']['igf1_reduction_diet'])
        
        # Age effect (IGF-1 decreases with age)
        igf1_base *= (1 - (biological_age - 30) / 100)
        
        igf1_value = max(50, igf1_base + np.random.normal(0, 30))
        biomarkers['igf1'] = min(400, igf1_value)
        
        # GDF-15 (increases with age and poor health)
        gdf15_base = self.biomarker_ranges['gdf15']['mean']
        
        # Age effect
        gdf15_base *= (1 + (biological_age - 40) / 50 * 
                      self.research_effects['biomarker_effects']['gdf15_increase_aging'])
        
        # Poor behaviors increase GDF-15
        if twa_behaviors['smoking_status'] == 'Current':
            gdf15_base *= 1.4
        if twa_behaviors['motion_days_week'] < 2:
            gdf15_base *= 1.2
        
        gdf15_value = max(200, gdf15_base * np.random.lognormal(0, 0.3))
        biomarkers['gdf15'] = min(3000, gdf15_value)
        
        # Cortisol (stress hormone)
        cortisol_base = self.biomarker_ranges['cortisol']['mean']
        
        # Nature connection reduces cortisol
        if twa_behaviors['nature_minutes_week'] >= 120:
            cortisol_base *= (1 + self.research_effects['biomarker_effects']['cortisol_reduction_nature'])
        
        # High stress occupation effect
        if demographics.get('occupation') in ['Management', 'Healthcare', 'Professional/Technical']:
            cortisol_base *= (1 + self.research_effects['biomarker_effects']['cortisol_increase_stress'])
        
        # Meditation reduces cortisol
        if twa_behaviors['meditation_minutes_week'] >= 150:
            cortisol_base *= 0.8
        
        cortisol_value = max(3, cortisol_base + np.random.normal(0, 2))
        biomarkers['cortisol'] = min(25, cortisol_value)
        
        return biomarkers
    
    def _generate_functional_measures(self, demographics: Dict, twa_behaviors: Dict, 
                                    biological_age: float) -> Dict:
        """Generate functional measures (grip strength, gait speed, etc.)"""
        
        measures = {}
        current_age = demographics['age_numeric']
        
        # Grip strength
        grip_base = self.biomarker_ranges['grip_strength']['mean']
        
        # Gender effect
        if demographics['gender'] == 'Male':
            grip_base *= 1.3
        elif demographics['gender'] == 'Female':
            grip_base *= 0.7
        
        # Age decline (about 1% per year after 40)
        if current_age > 40:
            age_decline = (current_age - 40) * 0.01
            grip_base *= (1 - age_decline)
        
        # Exercise benefit
        if twa_behaviors['motion_days_week'] >= 3:
            grip_base *= 1.15
        
        # Biological age effect
        bio_age_effect = (biological_age - current_age) * 0.02
        grip_base *= (1 - bio_age_effect)
        
        grip_value = max(10, grip_base + np.random.normal(0, 4))
        measures['grip_strength'] = min(70, grip_value)
        
        # Gait speed
        gait_base = self.biomarker_ranges['gait_speed']['mean']
        
        # Age decline
        if current_age > 50:
            age_decline = (current_age - 50) * 0.008
            gait_base *= (1 - age_decline)
        
        # Exercise benefit
        if twa_behaviors['motion_days_week'] >= 3:
            gait_base *= 1.10
        
        # Biological age effect
        gait_base *= (1 - bio_age_effect)
        
        gait_value = max(0.4, gait_base + np.random.normal(0, 0.1))
        measures['gait_speed'] = min(2.0, gait_value)
        
        # Balance score
        balance_base = self.biomarker_ranges['balance_score']['mean']
        
        # Age decline
        if current_age > 60:
            age_decline = (current_age - 60) * 0.015
            balance_base *= (1 - age_decline)
        
        # Exercise benefit
        if twa_behaviors['motion_days_week'] >= 3:
            balance_base *= 1.12
        
        balance_value = max(20, balance_base + np.random.normal(0, 8))
        measures['balance'] = min(100, balance_value)
        
        # Frailty index (0-1, higher = more frail)
        frailty_base = self.biomarker_ranges['frailty_index']['mean']
        
        # Age increase
        frailty_base += (current_age - 50) * 0.003
        
        # Biological age effect
        frailty_base += (biological_age - current_age) * 0.01
        
        # Protective behaviors
        if twa_behaviors['motion_days_week'] >= 3:
            frailty_base *= 0.85
        if twa_behaviors['social_connections_count'] >= 4:
            frailty_base *= 0.90
        if twa_behaviors['diet_mediterranean_score'] >= 7:
            frailty_base *= 0.88
        
        # Risk behaviors
        if twa_behaviors['smoking_status'] == 'Current':
            frailty_base *= 1.3
        if twa_behaviors['social_connections_count'] < 2:
            frailty_base *= 1.2
        
        frailty_value = max(0.0, frailty_base + np.random.normal(0, 0.05))
        measures['frailty'] = min(0.7, frailty_value)
        
        # Cognitive composite score
        cognitive_base = 85  # Average cognitive composite score
        
        # Age decline
        if current_age > 60:
            cognitive_base -= (current_age - 60) * 0.5
        
        # Education benefit
        education_benefits = {
            'Less than HS': -5, 'High School': 0, 
            'Some College': +3, 'Bachelor+': +8
        }
        cognitive_base += education_benefits.get(demographics['education'], 0)
        
        # Protective behaviors
        if twa_behaviors['motion_days_week'] >= 3:
            cognitive_base += 3
        if twa_behaviors['cultural_hours_week'] >= 5:
            cognitive_base += 4
        if twa_behaviors['meditation_minutes_week'] >= 150:
            cognitive_base += 2
        if twa_behaviors['social_connections_count'] >= 4:
            cognitive_base += 3
        
        # Risk behaviors
        if twa_behaviors['smoking_status'] == 'Current':
            cognitive_base -= 5
        if twa_behaviors['alcohol_drinks_week'] > 14:
            cognitive_base -= 3
        
        cognitive_value = max(20, cognitive_base + np.random.normal(0, 8))
        measures['cognitive'] = min(130, cognitive_value)
        
        # Processing speed (related to cognitive but separate)
        processing_base = cognitive_base * 0.9  # Slightly correlated
        
        # Age has stronger effect on processing speed
        if current_age > 50:
            processing_base -= (current_age - 50) * 0.8
        
        processing_value = max(20, processing_base + np.random.normal(0, 6))
        measures['processing_speed'] = min(130, processing_value)
        
        return measures
    
    def _generate_psychosocial_outcomes(self, twa_behaviors: Dict, biomarkers: Dict) -> Dict:
        """Generate psychosocial wellbeing outcomes"""
        
        outcomes = {}
        
        # Life satisfaction (strongly correlated with purpose and social connections)
        life_sat_base = 6.5  # Average life satisfaction
        
        # Strong predictors
        life_sat_base += (twa_behaviors['purpose_meaning_score'] - 6) * 0.4
        life_sat_base += (twa_behaviors['social_connections_count'] - 3) * 0.3
        
        # Health-related predictors
        life_sat_base += (twa_behaviors['sleep_quality_score'] - 6) * 0.2
        life_sat_base += min(0, (twa_behaviors['motion_days_week'] - 2) * 0.2)
        
        # Nature and cultural engagement
        if twa_behaviors['nature_minutes_week'] >= 120:
            life_sat_base += 0.5
        if twa_behaviors['cultural_hours_week'] >= 5:
            life_sat_base += 0.4
        
        life_sat_value = life_sat_base + np.random.normal(0, 0.8)
        outcomes['life_satisfaction'] = np.clip(life_sat_value, 1, 10)
        
        # Stress level (inverse of life satisfaction, but independent factors)
        stress_base = 5.5  # Moderate stress baseline
        
        # Stress reducers
        if twa_behaviors['meditation_minutes_week'] >= 150:
            stress_base -= 1.2
        if twa_behaviors['nature_minutes_week'] >= 120:
            stress_base -= 0.8
        if twa_behaviors['motion_days_week'] >= 3:
            stress_base -= 0.6
        
        # Stress increasers
        if biomarkers['cortisol'] > 15:  # High cortisol
            stress_base += 1.0
        if twa_behaviors['sleep_quality_score'] < 6:
            stress_base += 1.2
        if twa_behaviors['social_connections_count'] < 2:
            stress_base += 1.5
        
        stress_value = stress_base + np.random.normal(0, 0.7)
        outcomes['stress'] = np.clip(stress_value, 1, 10)
        
        # Depression risk score
        depression_base = 3.0  # Low baseline risk
        
        # Strong protective factors
        if twa_behaviors['social_connections_count'] >= 4:
            depression_base -= 1.5
        if twa_behaviors['purpose_meaning_score'] >= 8:
            depression_base -= 1.2
        if twa_behaviors['motion_days_week'] >= 3:
            depression_base -= 0.8
        
        # Risk factors
        if twa_behaviors['social_connections_count'] < 2:
            depression_base += 2.5
        if biomarkers['il6'] > 4:  # High inflammation linked to depression
            depression_base += 1.0
        if twa_behaviors['smoking_status'] == 'Current':
            depression_base += 0.8
        
        depression_value = depression_base + np.random.normal(0, 0.8)
        outcomes['depression_risk'] = np.clip(depression_value, 1, 10)
        
        # Social support score (related to but distinct from connections count)
        support_base = twa_behaviors['social_connections_count'] * 1.8 + 2
        
        # Quality matters, not just quantity
        if twa_behaviors['purpose_meaning_score'] >= 7:  # Purpose helps social quality
            support_base += 1.0
        if twa_behaviors['cultural_hours_week'] >= 5:  # Cultural activities → social quality
            support_base += 0.8
        
        support_value = support_base + np.random.normal(0, 1.0)
        outcomes['social_support'] = np.clip(support_value, 1, 10)
        
        return outcomes
    
    def generate_aging_wellness_outcomes(self, demographics: Dict, twa_behaviors: Dict, 
                                       months_elapsed: int) -> Dict:
        """
        Generate scientifically-grounded aging and wellness outcomes
        """
        
        chronological_age = demographics['age_numeric']
        
        # Calculate biological age
        base_bio_age = self._calculate_biological_age_base(chronological_age, demographics)
        biological_age = self._calculate_biological_age_from_behaviors(
            base_bio_age, twa_behaviors, months_elapsed
        )
        
        # Calculate mortality risk score
        mortality_risk = self._calculate_mortality_risk_score(
            demographics, twa_behaviors, biological_age
        )
        
        # Estimate lifespan
        estimated_lifespan = self._calculate_estimated_lifespan(demographics, mortality_risk)
        
        # Generate biomarkers
        biomarkers = self._generate_biomarkers(demographics, twa_behaviors, biological_age)
        
        # Generate functional measures
        functional_measures = self._generate_functional_measures(
            demographics, twa_behaviors, biological_age
        )
        
        # Generate psychosocial outcomes
        psychosocial_outcomes = self._generate_psychosocial_outcomes(twa_behaviors, biomarkers)
        
        return {
            # Primary aging outcomes
            'biological_age_years': round(biological_age, 1),
            'biological_age_acceleration': round(biological_age - chronological_age, 1),
            'mortality_risk_score': round(mortality_risk, 4),
            'estimated_lifespan_years': round(estimated_lifespan, 1),
            
            # Biomarkers (expert consensus validated)
            'crp_mg_l': round(biomarkers['crp'], 2),
            'il6_pg_ml': round(biomarkers['il6'], 1),
            'igf1_ng_ml': round(biomarkers['igf1'], 0),
            'gdf15_pg_ml': round(biomarkers['gdf15'], 0),
            'cortisol_ug_dl': round(biomarkers['cortisol'], 1),
            
            # Functional measures
            'grip_strength_kg': round(functional_measures['grip_strength'], 1),
            'gait_speed_ms': round(functional_measures['gait_speed'], 2),
            'balance_score': round(functional_measures['balance'], 1),
            'frailty_index': round(functional_measures['frailty'], 3),
            
            # Cognitive measures
            'cognitive_composite_score': round(functional_measures['cognitive'], 1),
            'processing_speed_score': round(functional_measures['processing_speed'], 1),
            
            # Psychosocial wellbeing
            'life_satisfaction_score': round(psychosocial_outcomes['life_satisfaction'], 1),
            'stress_level_score': round(psychosocial_outcomes['stress'], 1),
            'depression_risk_score': round(psychosocial_outcomes['depression_risk'], 1),
            'social_support_score': round(psychosocial_outcomes['social_support'], 1)
        }


if __name__ == "__main__":
    # Test the outcome generator
    from demographics_generator import EnhancedDemographicGenerator
    from twa_behavior_generator import ResearchValidatedTWAGenerator, _month_to_season
    
    # Generate test data
    demo_gen = EnhancedDemographicGenerator(random_seed=42)
    twa_gen = ResearchValidatedTWAGenerator(random_seed=42)
    outcome_gen = WellnessAgingOutcomeGenerator(random_seed=42)
    
    demographics = demo_gen.generate_correlated_demographics(n_samples=5)
    
    print("Testing Wellness & Aging Outcome Generator:")
    print("=" * 50)
    
    for i, person in enumerate(demographics):
        # Generate behaviors for month 6 (summer)
        behaviors = twa_gen.generate_monthly_twa_behaviors(person, 6, 'Summer')
        
        # Generate outcomes after 6 months
        outcomes = outcome_gen.generate_aging_wellness_outcomes(person, behaviors, 6)
        
        print(f"\nSubject {person['subject_id']} ({person['age_numeric']} yrs, {person['gender']}, {person['education']}):")
        print(f"  Biological Age: {outcomes['biological_age_years']} (acceleration: {outcomes['biological_age_acceleration']:+.1f})")
        print(f"  Mortality Risk: {outcomes['mortality_risk_score']:.3f}")
        print(f"  Estimated Lifespan: {outcomes['estimated_lifespan_years']:.1f} years")
        print(f"  CRP: {outcomes['crp_mg_l']} mg/L, IL-6: {outcomes['il6_pg_ml']} pg/mL")
        print(f"  Grip Strength: {outcomes['grip_strength_kg']} kg, Gait Speed: {outcomes['gait_speed_ms']} m/s")
        print(f"  Life Satisfaction: {outcomes['life_satisfaction_score']}/10, Stress: {outcomes['stress_level_score']}/10")
        print(f"  Frailty Index: {outcomes['frailty_index']:.3f}")
    
    print(f"\nOutcome Generation Test Completed Successfully!")