"""
Enhanced Longitudinal TWA Dataset Generator
Improved persona consistency, demographic alignment, and longitudinal coherence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import math
from scipy import stats

from demographics_generator import EnhancedDemographicGenerator
from twa_behavior_generator import ResearchValidatedTWAGenerator
from wellness_aging_outcomes import WellnessAgingOutcomeGenerator


class EnhancedLongitudinalTWADataGenerator:
    """
    Enhanced master class with improved consistency and persona alignment
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize with enhanced consistency features"""
        np.random.seed(random_seed)
        
        # Initialize component generators
        self.demo_generator = EnhancedDemographicGenerator(random_seed=random_seed)
        self.twa_generator = ResearchValidatedTWAGenerator(random_seed=random_seed)
        self.outcome_generator = WellnessAgingOutcomeGenerator(random_seed=random_seed)
        
        # Enhanced consistency parameters
        self.consistency_config = {
            'behavioral_stability': 0.85,  # High stability for most behaviors
            'demographic_influence': 0.75,  # Strong demographic influence
            'seasonal_effect': 0.15,       # Moderate seasonal effects
            'outcome_responsiveness': 0.65, # Outcomes respond to behavior changes
            'biomarker_persistence': 0.80   # Biomarkers change slowly
        }
    
    def generate_complete_dataset(self, n_subjects: int = 1000, months: int = 12,
                                include_life_events: bool = True) -> pd.DataFrame:
        """Generate complete enhanced dataset with improved consistency"""
        
        print(f"Generating enhanced dataset with improved consistency...")
        print(f"Subjects: {n_subjects:,}, Months: {months}")
        
        # Generate base demographics
        print("1. Generating demographics...")
        demographics_list = self.demo_generator.generate_correlated_demographics(n_subjects)
        
        # Generate complete longitudinal data
        all_data = []
        
        for i, demographics in enumerate(demographics_list):
            if i % 100 == 0:
                print(f"   Processing subject {i+1:,}/{n_subjects:,}")
            
            # Generate person-specific traits (stable characteristics)
            person_traits = self._generate_person_traits(demographics)
            
            # Generate baseline behaviors aligned with demographics
            baseline_behaviors = self._generate_demographic_aligned_behaviors(demographics, person_traits)
            
            # Generate longitudinal trajectory for this person
            person_data = self._generate_person_longitudinal_data(
                demographics, person_traits, baseline_behaviors, months
            )
            
            # Add life events if requested
            if include_life_events:
                person_data = self._add_realistic_life_events(person_data, demographics)
            
            all_data.extend(person_data)
        
        print(f"2. Converting to DataFrame...")
        dataset = pd.DataFrame(all_data)
        
        print(f"3. Final processing and validation...")
        dataset = self._add_final_processing(dataset)
        
        return dataset
    
    def _generate_person_traits(self, demographics: Dict) -> Dict:
        """Generate stable person-level traits that drive consistency"""
        
        traits = {}
        
        # Behavioral consistency (some people are more routine-oriented)
        base_consistency = 0.6
        
        # Demographics influence consistency
        if demographics['age_numeric'] > 50:
            base_consistency += 0.25  # Older adults more consistent
        elif demographics['age_numeric'] < 30:
            base_consistency -= 0.15  # Younger adults more variable
        
        if demographics['education'] in ['Bachelor+']:
            base_consistency += 0.15  # Higher education → more consistent
        
        if demographics['marital_status'] in ['Married']:
            base_consistency += 0.1   # Married people more routine
        
        traits['behavioral_consistency'] = np.clip(
            base_consistency + np.random.normal(0, 0.1), 0.3, 0.95
        )
        
        # Health motivation (drives healthy behaviors)
        base_motivation = 0.5
        
        if demographics['education'] in ['Bachelor+']:
            base_motivation += 0.2
        if demographics['income_bracket'] in ['$75-100k', '$100-150k', '>$150k']:
            base_motivation += 0.15
        if demographics['age_numeric'] > 40:
            base_motivation += 0.15  # Health awareness increases with age
        
        traits['health_motivation'] = np.clip(
            base_motivation + np.random.normal(0, 0.15), 0.1, 0.9
        )
        
        # Stress susceptibility
        base_stress = 0.5
        
        if demographics['age_numeric'] < 35:
            base_stress += 0.2  # Young adults more stressed
        if demographics['income_bracket'] in ['<$35k', '$35-50k']:
            base_stress += 0.15  # Lower income → more stress
        if demographics['occupation'] in ['Management', 'Professional']:
            base_stress += 0.1   # High-pressure jobs
        
        traits['stress_susceptibility'] = np.clip(
            base_stress + np.random.normal(0, 0.1), 0.1, 0.9
        )
        
        # Social orientation
        base_social = 0.5
        
        if demographics['gender'] == 'Female':
            base_social += 0.15
        if demographics['marital_status'] == 'Married':
            base_social += 0.1
        if demographics['age_numeric'] > 65:
            base_social += 0.1  # Social connections more important with age
        
        traits['social_orientation'] = np.clip(
            base_social + np.random.normal(0, 0.12), 0.1, 0.9
        )
        
        return traits
    
    def _generate_demographic_aligned_behaviors(self, demographics: Dict, traits: Dict) -> Dict:
        """Generate baseline behaviors aligned with person's demographics and traits"""
        
        behaviors = {}
        
        # Motion/Exercise aligned with demographics
        base_motion = 2.0  # Base 2 days/week
        
        # Age effects
        if demographics['age_numeric'] < 35:
            base_motion += 1.0
        elif demographics['age_numeric'] > 65:
            base_motion -= 0.5
        
        # Education/Income effects
        if demographics['education'] in ['Bachelor+']:
            base_motion += 0.8
        if demographics['income_bracket'] in ['$75-100k', '$100-150k', '>$150k']:
            base_motion += 0.5
        
        # Fitness level effects
        fitness_effects = {'High': 1.5, 'Medium': 0.0, 'Low': -1.0}
        base_motion += fitness_effects.get(demographics['fitness_level'], 0)
        
        # Health motivation effects
        base_motion += (traits['health_motivation'] - 0.5) * 2.0
        
        behaviors['motion_days_week'] = np.clip(
            base_motion + np.random.normal(0, 0.8), 0, 7
        )
        
        # Sleep hours aligned with demographics
        base_sleep = 7.5  # Base hours
        
        # Age effects
        if demographics['age_numeric'] > 65:
            base_sleep -= 0.5
        elif demographics['age_numeric'] < 30:
            base_sleep -= 0.3
        
        # Occupation effects
        if demographics['occupation'] in ['Management', 'Professional']:
            base_sleep -= 0.2  # Slightly less sleep due to stress
        
        behaviors['sleep_hours'] = np.clip(
            base_sleep + np.random.normal(0, 0.5), 4, 12
        )
        
        # Sleep quality aligned with stress and health
        base_quality = 7.0
        base_quality -= traits['stress_susceptibility'] * 2.0
        base_quality += traits['health_motivation'] * 1.5
        
        behaviors['sleep_quality_score'] = np.clip(
            base_quality + np.random.normal(0, 0.8), 1, 10
        )
        
        # Diet quality aligned with education and income
        base_diet = 5.0
        
        if demographics['education'] in ['Bachelor+']:
            base_diet += 1.5
        if demographics['income_bracket'] in ['$100-150k', '>$150k']:
            base_diet += 1.0
        elif demographics['income_bracket'] in ['<$35k']:
            base_diet -= 1.0
        
        # Health motivation strongly influences diet
        base_diet += (traits['health_motivation'] - 0.5) * 3.0
        
        behaviors['diet_mediterranean_score'] = np.clip(
            base_diet + np.random.normal(0, 0.8), 0, 10
        )
        
        # Meditation aligned with education and stress
        base_meditation = 30  # Base minutes per week
        
        if demographics['education'] in ['Bachelor+']:
            base_meditation += 40
        if demographics['age_numeric'] > 50:
            base_meditation += 30
        
        # High stress people may or may not meditate more
        stress_effect = (traits['stress_susceptibility'] - 0.5) * 50
        if traits['health_motivation'] > 0.6:
            base_meditation += abs(stress_effect)  # High motivation uses meditation for stress
        else:
            base_meditation -= stress_effect  # Low motivation → less meditation when stressed
        
        behaviors['meditation_minutes_week'] = np.clip(
            base_meditation + np.random.normal(0, 40), 0, 600
        )
        
        # Hydration aligned with health awareness
        base_hydration = 8.0
        base_hydration += (traits['health_motivation'] - 0.5) * 3.0
        
        behaviors['hydration_cups_day'] = np.clip(
            base_hydration + np.random.normal(0, 1.5), 4, 16
        )
        
        # Social connections aligned with social orientation
        base_social = 4.0
        base_social += (traits['social_orientation'] - 0.5) * 6.0
        
        if demographics['marital_status'] == 'Married':
            base_social += 1.0
        if demographics['age_numeric'] > 65:
            base_social -= 0.5  # Slightly fewer connections with age
        
        behaviors['social_connections_count'] = np.clip(
            int(base_social + np.random.normal(0, 1.5)), 0, 12
        )
        
        # Nature time aligned with region and lifestyle
        base_nature = 120  # Base minutes per week
        
        if demographics['urban_rural'] == 'Rural':
            base_nature += 60
        if demographics['region'] in ['West']:
            base_nature += 30
        
        base_nature += (traits['health_motivation'] - 0.5) * 80
        
        behaviors['nature_minutes_week'] = np.clip(
            int(base_nature + np.random.normal(0, 40)), 0, 600
        )
        
        # Cultural activities aligned with education and income
        base_cultural = 3.0
        
        if demographics['education'] in ['Bachelor+']:
            base_cultural += 2.0
        if demographics['income_bracket'] in ['$75-100k', '$100-150k', '>$150k']:
            base_cultural += 1.5
        
        behaviors['cultural_hours_week'] = np.clip(
            base_cultural + np.random.normal(0, 1.5), 0, 30
        )
        
        # Purpose aligned with education and social connections
        base_purpose = 6.0
        
        if demographics['education'] in ['Bachelor+']:
            base_purpose += 0.8
        
        # Health motivation effects
        base_purpose += (traits['health_motivation'] - 0.5) * 2.0
        
        behaviors['purpose_meaning_score'] = np.clip(
            base_purpose + np.random.normal(0, 1.0), 1, 10
        )
        
        # Smoking aligned with demographics (very stable trait)
        smoking_probs = self._get_smoking_probabilities(demographics)
        behaviors['smoking_status'] = np.random.choice(
            ['Never', 'Former', 'Current'], 
            p=smoking_probs
        )
        
        # Alcohol consumption aligned with demographics
        base_alcohol = 5.0  # Base drinks per week
        
        # Age effects
        if demographics['age_numeric'] < 30:
            base_alcohol += 3.0
        elif demographics['age_numeric'] > 65:
            base_alcohol -= 2.0
        
        # Gender effects
        if demographics['gender'] == 'Female':
            base_alcohol -= 1.5
        
        # Education effects
        if demographics['education'] in ['Bachelor+']:
            base_alcohol += 1.0
        
        behaviors['alcohol_drinks_week'] = np.clip(
            base_alcohol + np.random.normal(0, 2.5), 0, 35
        )
        
        # Add other behavioral variables with demographic alignment
        behaviors.update(self._generate_remaining_behaviors(demographics, traits))
        
        return behaviors
    
    def _get_smoking_probabilities(self, demographics: Dict) -> List[float]:
        """Get smoking status probabilities based on demographics"""
        
        base_current = 0.14  # 14% current smoking rate
        
        # Education effect (strongest predictor)
        edu_effects = {
            'Less than HS': 2.5, 'High School': 1.8, 
            'Some College': 1.2, 'Bachelor+': 0.4
        }
        education_mult = edu_effects.get(demographics['education'], 1.5)
        
        # Income effect
        income_effects = {
            '<$35k': 1.8, '$35-50k': 1.4, '$50-75k': 1.1,
            '$75-100k': 0.8, '$100-150k': 0.6, '>$150k': 0.4
        }
        income_mult = income_effects.get(demographics['income_bracket'], 1.2)
        
        # Age effect
        if demographics['age_numeric'] < 30:
            age_mult = 1.3
        elif demographics['age_numeric'] > 65:
            age_mult = 0.6
        else:
            age_mult = 1.0
        
        current_prob = base_current * education_mult * income_mult * age_mult
        current_prob = min(0.50, current_prob)  # Cap at 50%
        
        # Former smoker probability
        former_prob = 0.22 * (1 + (demographics['age_numeric'] - 40) / 100)
        former_prob = max(0.10, min(0.35, former_prob))
        
        never_prob = max(0.15, 1 - current_prob - former_prob)
        
        # Normalize
        total = current_prob + former_prob + never_prob
        return [never_prob/total, former_prob/total, current_prob/total]
    
    def _generate_remaining_behaviors(self, demographics: Dict, traits: Dict) -> Dict:
        """Generate remaining behavioral variables with demographic alignment"""
        
        behaviors = {}
        
        # Added sugar aligned with diet quality and education
        base_sugar = 50  # Base grams per day
        
        if demographics['education'] in ['Bachelor+']:
            base_sugar -= 15
        elif demographics['education'] in ['Less than HS']:
            base_sugar += 20
        
        if demographics['income_bracket'] in ['<$35k', '$35-50k']:
            base_sugar += 15
        
        # Inverse of health motivation
        base_sugar += (0.5 - traits['health_motivation']) * 40
        
        behaviors['added_sugar_grams_day'] = np.clip(
            base_sugar + np.random.normal(0, 20), 10, 200
        )
        
        # Sodium aligned with processed food consumption
        base_sodium = 3.5  # Base grams per day
        
        if demographics['education'] in ['Bachelor+']:
            base_sodium -= 0.5
        
        # Health motivation reduces sodium
        base_sodium -= (traits['health_motivation'] - 0.5) * 1.0
        
        behaviors['sodium_grams_day'] = np.clip(
            base_sodium + np.random.normal(0, 0.6), 1.5, 8.0
        )
        
        # Processed foods aligned with demographics
        base_processed = 15  # Base servings per week
        
        if demographics['education'] in ['Bachelor+']:
            base_processed -= 5
        if demographics['income_bracket'] in ['<$35k']:
            base_processed += 5
        
        # Health motivation reduces processed foods
        base_processed -= (traits['health_motivation'] - 0.5) * 8
        
        behaviors['processed_food_servings_week'] = np.clip(
            int(base_processed + np.random.normal(0, 4)), 0, 40
        )
        
        return behaviors
    
    def _generate_person_longitudinal_data(self, demographics: Dict, traits: Dict, 
                                         baseline_behaviors: Dict, months: int) -> List[Dict]:
        """Generate longitudinal data for one person with enhanced consistency"""
        
        person_data = []
        
        # Initialize stable characteristics that persist across months
        stable_outcomes = self._initialize_stable_outcomes(demographics, baseline_behaviors, traits)
        
        for month in range(months):
            month_data = demographics.copy()
            month_data.update({
                'month': month,
                'season': self._month_to_season(month),
                'observation_date': pd.Timestamp('2024-01-15') + pd.DateOffset(months=month),
                'months_since_baseline': month,
                'age_at_observation': demographics['age_numeric'] + month/12.0
            })
            
            # Generate behaviors for this month with high consistency
            if month == 0:
                # First month uses baseline behaviors
                month_behaviors = baseline_behaviors.copy()
            else:
                # Subsequent months are highly consistent with gradual changes
                month_behaviors = self._generate_consistent_monthly_behaviors(
                    person_data[month-1], demographics, traits, month
                )
            
            month_data.update(month_behaviors)
            
            # Generate outcomes aligned with behaviors and demographics
            month_outcomes = self._generate_aligned_outcomes(
                month_behaviors, demographics, traits, stable_outcomes, month
            )
            month_data.update(month_outcomes)
            
            # Generate composite scores
            month_composites = self._generate_composite_scores(month_behaviors, month_outcomes, demographics)
            month_data.update(month_composites)
            
            # Add derived metrics
            month_data.update(self._generate_derived_metrics(month_behaviors))
            
            person_data.append(month_data)
        
        return person_data
    
    def _generate_consistent_monthly_behaviors(self, prev_month: Dict, demographics: Dict, 
                                             traits: Dict, month: int) -> Dict:
        """Generate behaviors with high consistency and gradual realistic changes"""
        
        new_behaviors = {}
        
        # High stability behaviors (almost never change)
        stable_behaviors = ['smoking_status']
        for behavior in stable_behaviors:
            if np.random.random() < 0.98:  # 98% stability
                new_behaviors[behavior] = prev_month[behavior]
            else:
                # Very rare change (like quitting smoking)
                if behavior == 'smoking_status' and prev_month[behavior] == 'Current':
                    new_behaviors[behavior] = 'Former' if np.random.random() < 0.3 else 'Current'
                else:
                    new_behaviors[behavior] = prev_month[behavior]
        
        # Medium stability behaviors (gradual changes)
        medium_stability = [
            'motion_days_week', 'diet_mediterranean_score', 'social_connections_count',
            'purpose_meaning_score', 'meditation_minutes_week', 'cultural_hours_week'
        ]
        
        consistency = traits['behavioral_consistency']
        
        for behavior in medium_stability:
            if behavior in prev_month:
                prev_value = prev_month[behavior]
                
                # Seasonal effects
                seasonal_modifier = self._get_seasonal_modifier(behavior, self._month_to_season(month))
                
                # Small random drift
                if isinstance(prev_value, (int, float)):
                    drift = np.random.normal(0, 0.1) * (1 - consistency)
                    new_value = prev_value * (1 + drift) * seasonal_modifier
                    
                    # Keep within reasonable bounds
                    if behavior == 'motion_days_week':
                        new_value = np.clip(new_value, 0, 7)
                    elif behavior == 'diet_mediterranean_score':
                        new_value = np.clip(new_value, 0, 10)
                    elif behavior == 'purpose_meaning_score':
                        new_value = np.clip(new_value, 1, 10)
                    elif behavior == 'meditation_minutes_week':
                        new_value = np.clip(new_value, 0, 600)
                    elif behavior == 'social_connections_count':
                        new_value = np.clip(int(new_value), 0, 12)
                    elif behavior == 'cultural_hours_week':
                        new_value = np.clip(new_value, 0, 30)
                    
                    new_behaviors[behavior] = type(prev_value)(new_value)
                else:
                    new_behaviors[behavior] = prev_value
        
        # Variable behaviors (more change but still consistent)
        variable_behaviors = [
            'sleep_hours', 'sleep_quality_score', 'hydration_cups_day',
            'alcohol_drinks_week', 'nature_minutes_week'
        ]
        
        for behavior in variable_behaviors:
            if behavior in prev_month:
                prev_value = prev_month[behavior]
                seasonal_modifier = self._get_seasonal_modifier(behavior, self._month_to_season(month))
                
                # More variation allowed
                drift = np.random.normal(0, 0.2) * (1 - consistency * 0.7)
                new_value = prev_value * (1 + drift) * seasonal_modifier
                
                # Keep within bounds
                if behavior == 'sleep_hours':
                    new_value = np.clip(new_value, 4, 12)
                elif behavior == 'sleep_quality_score':
                    new_value = np.clip(new_value, 1, 10)
                elif behavior == 'hydration_cups_day':
                    new_value = np.clip(new_value, 4, 16)
                elif behavior == 'alcohol_drinks_week':
                    new_value = np.clip(new_value, 0, 35)
                elif behavior == 'nature_minutes_week':
                    new_value = np.clip(new_value, 0, 600)
                
                new_behaviors[behavior] = type(prev_value)(new_value)
        
        # Dependent behaviors (depend on diet quality and other factors)
        diet_score = new_behaviors.get('diet_mediterranean_score', prev_month.get('diet_mediterranean_score', 5))
        
        # Sugar inversely related to diet quality
        base_sugar = prev_month.get('added_sugar_grams_day', 50)
        diet_effect = (5 - diet_score) * 3  # Lower diet quality → more sugar
        new_behaviors['added_sugar_grams_day'] = np.clip(
            base_sugar + diet_effect + np.random.normal(0, 5), 10, 200
        )
        
        # Sodium related to processed foods
        base_sodium = prev_month.get('sodium_grams_day', 3.5)
        new_behaviors['sodium_grams_day'] = np.clip(
            base_sodium + np.random.normal(0, 0.2), 1.5, 8.0
        )
        
        # Processed foods inverse to diet quality
        base_processed = prev_month.get('processed_food_servings_week', 15)
        processed_drift = (5 - diet_score) * 1.5
        new_behaviors['processed_food_servings_week'] = np.clip(
            int(base_processed + processed_drift + np.random.normal(0, 2)), 0, 40
        )
        
        return new_behaviors
    
    def _get_seasonal_modifier(self, behavior: str, season: str) -> float:
        """Get seasonal modifier for specific behaviors"""
        
        seasonal_effects = {
            'Winter': {
                'motion_days_week': 0.9,     # Less exercise in winter
                'nature_minutes_week': 0.7,   # Much less nature time
                'alcohol_drinks_week': 1.1    # Slightly more drinking
            },
            'Spring': {
                'motion_days_week': 1.1,     # More exercise
                'nature_minutes_week': 1.3,   # More nature time
                'alcohol_drinks_week': 1.0
            },
            'Summer': {
                'motion_days_week': 1.2,     # Peak exercise
                'nature_minutes_week': 1.4,   # Peak nature time  
                'alcohol_drinks_week': 1.2,   # Summer drinking
                'hydration_cups_day': 1.15    # More hydration needed
            },
            'Fall': {
                'motion_days_week': 1.0,
                'nature_minutes_week': 1.1,
                'alcohol_drinks_week': 1.0
            }
        }
        
        return seasonal_effects.get(season, {}).get(behavior, 1.0)
    
    def _initialize_stable_outcomes(self, demographics: Dict, behaviors: Dict, traits: Dict) -> Dict:
        """Initialize stable outcome characteristics for this person"""
        
        stable = {}
        
        # Genetic/constitutional factors that affect biomarkers
        stable['inflammation_tendency'] = np.random.normal(0, 0.3)  # Some people more inflammatory
        stable['metabolic_efficiency'] = np.random.normal(0, 0.2)   # Metabolic differences
        stable['stress_physiology'] = np.random.normal(0, 0.25)     # Stress response differences
        stable['baseline_fitness'] = demographics.get('fitness_level', 'Medium')
        
        # Calculate person's biological age offset (stable component)
        age_offset = 0
        
        # Education effect on biological age
        if demographics['education'] in ['Bachelor+']:
            age_offset -= 1.5
        elif demographics['education'] in ['Less than HS']:
            age_offset += 1.5
        
        # Income effect
        if demographics['income_bracket'] in ['>$150k', '$100-150k']:
            age_offset -= 1.0
        elif demographics['income_bracket'] in ['<$35k']:
            age_offset += 1.5
        
        # Add individual variation
        age_offset += np.random.normal(0, 2.0)
        
        stable['biological_age_offset'] = age_offset
        
        return stable
    
    def _generate_aligned_outcomes(self, behaviors: Dict, demographics: Dict, 
                                 traits: Dict, stable: Dict, month: int) -> Dict:
        """Generate outcomes that align with behaviors and demographics"""
        
        outcomes = {}
        
        # Calculate biological age based on behaviors and demographics
        base_bio_age = demographics['age_numeric'] + month/12.0
        bio_age_adjustment = stable['biological_age_offset']
        
        # Exercise effect (strong)
        exercise_days = behaviors.get('motion_days_week', 2)
        bio_age_adjustment -= (exercise_days - 2) * 0.3  # Each extra day reduces bio age
        
        # Diet effect (strong)  
        diet_score = behaviors.get('diet_mediterranean_score', 5)
        bio_age_adjustment -= (diet_score - 5) * 0.4  # Good diet reduces bio age
        
        # Sleep effect
        sleep_hours = behaviors.get('sleep_hours', 7.5)
        sleep_quality = behaviors.get('sleep_quality_score', 7)
        optimal_sleep = 1.0 - abs(sleep_hours - 7.5) / 7.5  # Penalty for too little/much sleep
        bio_age_adjustment -= (sleep_quality * optimal_sleep - 5) * 0.2
        
        # Smoking effect (very strong)
        if behaviors.get('smoking_status') == 'Current':
            bio_age_adjustment += 4.0
        elif behaviors.get('smoking_status') == 'Former':
            bio_age_adjustment += 1.0
        
        # Alcohol effect
        alcohol = behaviors.get('alcohol_drinks_week', 5)
        if alcohol > 14:  # Heavy drinking
            bio_age_adjustment += (alcohol - 14) * 0.15
        elif alcohol < 7 and alcohol > 1:  # Moderate drinking beneficial
            bio_age_adjustment -= 0.5
        
        # Social connection effect
        social_connections = behaviors.get('social_connections_count', 4)
        bio_age_adjustment -= (social_connections - 4) * 0.15
        
        # Purpose effect
        purpose = behaviors.get('purpose_meaning_score', 6)
        bio_age_adjustment -= (purpose - 6) * 0.2
        
        # Meditation effect
        meditation = behaviors.get('meditation_minutes_week', 60)
        bio_age_adjustment -= min(meditation / 60, 2) * 0.3  # Capped benefit
        
        # Calculate final biological age
        outcomes['biological_age_years'] = base_bio_age + bio_age_adjustment
        outcomes['biological_age_acceleration'] = bio_age_adjustment
        
        # Generate biomarkers aligned with biological age and behaviors
        outcomes.update(self._generate_aligned_biomarkers(behaviors, demographics, bio_age_adjustment, stable, traits))
        
        # Generate functional measures aligned with age and fitness
        outcomes.update(self._generate_aligned_functional_measures(behaviors, demographics, bio_age_adjustment))
        
        # Generate mortality risk and lifespan estimates
        mortality_risk = self._calculate_mortality_risk(bio_age_adjustment, demographics['age_numeric'])
        outcomes['mortality_risk_score'] = mortality_risk
        outcomes['estimated_lifespan_years'] = self._estimate_lifespan(mortality_risk, demographics['age_numeric'])
        
        # Generate psychosocial measures
        outcomes.update(self._generate_psychosocial_measures(behaviors, demographics, traits))
        
        return outcomes
    
    def _generate_aligned_biomarkers(self, behaviors: Dict, demographics: Dict, 
                                   bio_age_adj: float, stable: Dict, traits: Dict) -> Dict:
        """Generate biomarkers aligned with behaviors and biological age"""
        
        biomarkers = {}
        
        # CRP (inflammation marker)
        base_crp = 1.5  # mg/L
        
        # Age effect
        base_crp += demographics['age_numeric'] * 0.02
        
        # Exercise reduces inflammation strongly
        exercise_days = behaviors.get('motion_days_week', 2)
        base_crp *= (0.7 ** (exercise_days / 2))  # Strong anti-inflammatory effect
        
        # Diet effect
        diet_score = behaviors.get('diet_mediterranean_score', 5)
        base_crp *= (1.3 - diet_score * 0.06)  # Better diet → lower inflammation
        
        # Smoking increases inflammation
        if behaviors.get('smoking_status') == 'Current':
            base_crp *= 2.0
        
        # Weight/BMI proxy (higher BMI → more inflammation)
        if demographics.get('fitness_level') == 'Low':
            base_crp *= 1.4
        
        # Individual variation
        base_crp *= (1 + stable['inflammation_tendency'])
        
        biomarkers['crp_mg_l'] = np.clip(base_crp, 0.1, 10.0)
        
        # IL-6 (similar pattern to CRP but lower correlation)
        base_il6 = 2.0  # pg/mL
        base_il6 += demographics['age_numeric'] * 0.03
        base_il6 *= (0.8 ** (exercise_days / 2))
        base_il6 *= (1.2 - diet_score * 0.04)
        
        if behaviors.get('smoking_status') == 'Current':
            base_il6 *= 1.8
        
        base_il6 *= (1 + stable['inflammation_tendency'] * 0.8)
        
        biomarkers['il6_pg_ml'] = np.clip(base_il6, 0.5, 15.0)
        
        # IGF-1 (growth factor, peaks in youth, affected by exercise)
        base_igf1 = 200  # ng/mL
        
        # Age effect (declines with age)
        age_factor = max(0.3, 1 - (demographics['age_numeric'] - 25) * 0.015)
        base_igf1 *= age_factor
        
        # Exercise increases IGF-1 moderately
        base_igf1 *= (1 + exercise_days * 0.05)
        
        # Diet effect
        base_igf1 *= (0.9 + diet_score * 0.02)
        
        # Individual variation
        base_igf1 *= (1 + stable['metabolic_efficiency'])
        
        biomarkers['igf1_ng_ml'] = np.clip(base_igf1, 50.0, 400.0)
        
        # GDF-15 (stress/aging marker)
        base_gdf15 = 800  # pg/mL
        
        # Age effect (increases with age)
        base_gdf15 += demographics['age_numeric'] * 15
        
        # Exercise reduces GDF-15
        base_gdf15 *= (0.85 ** (exercise_days / 2))
        
        # Smoking increases
        if behaviors.get('smoking_status') == 'Current':
            base_gdf15 *= 1.6
        
        # Stress effect
        stress_level = demographics.get('stress_level_score', 5)
        base_gdf15 *= (1 + (stress_level - 5) * 0.08)
        
        biomarkers['gdf15_pg_ml'] = np.clip(base_gdf15, 269.0, 3000.0)
        
        # Cortisol (stress hormone)
        base_cortisol = 12.0  # μg/dL
        
        # Stress effect
        stress_multiplier = 1 + traits.get('stress_susceptibility', 0.5) * 0.6
        base_cortisol *= stress_multiplier
        
        # Sleep effect (poor sleep → higher cortisol)
        sleep_quality = behaviors.get('sleep_quality_score', 7)
        base_cortisol *= (1.4 - sleep_quality * 0.05)
        
        # Meditation reduces cortisol
        meditation = behaviors.get('meditation_minutes_week', 60)
        base_cortisol *= (1 - min(meditation / 300, 0.3))
        
        # Exercise effect (moderate exercise reduces, too much increases)
        exercise_days = behaviors.get('motion_days_week', 2)
        if exercise_days <= 5:
            base_cortisol *= (0.9 ** exercise_days)
        else:
            base_cortisol *= 1.2  # Overtraining effect
        
        biomarkers['cortisol_ug_dl'] = np.clip(base_cortisol, 3.0, 25.0)
        
        return biomarkers
    
    def _generate_aligned_functional_measures(self, behaviors: Dict, demographics: Dict, 
                                            bio_age_adj: float) -> Dict:
        """Generate functional measures aligned with age, fitness, and behaviors"""
        
        measures = {}
        
        # Grip strength
        age = demographics['age_numeric']
        
        # Base grip strength by age and gender
        if demographics['gender'] == 'Male':
            base_grip = 46 - (age - 30) * 0.4  # Men start higher, decline faster
        else:
            base_grip = 28 - (age - 30) * 0.3  # Women start lower, decline slower
        
        # Exercise strongly affects grip strength
        exercise_days = behaviors.get('motion_days_week', 2)
        base_grip += exercise_days * 2.0
        
        # Biological age adjustment
        base_grip -= bio_age_adj * 1.5
        
        # Individual variation
        measures['grip_strength_kg'] = np.clip(base_grip + np.random.normal(0, 3), 10.0, 70.0)
        
        # Gait speed
        base_gait = 1.3  # m/s
        
        # Age effect
        base_gait -= (age - 30) * 0.008
        
        # Exercise effect
        base_gait += exercise_days * 0.04
        
        # Biological age adjustment
        base_gait -= bio_age_adj * 0.03
        
        measures['gait_speed_ms'] = np.clip(base_gait + np.random.normal(0, 0.1), 0.4, 2.0)
        
        # Balance score
        base_balance = 85  # out of 100
        
        # Age effect
        base_balance -= (age - 30) * 0.5
        
        # Exercise helps balance significantly
        base_balance += exercise_days * 2.5
        
        # Biological age adjustment
        base_balance -= bio_age_adj * 2.0
        
        measures['balance_score'] = np.clip(base_balance + np.random.normal(0, 5), 20.0, 100.0)
        
        # Frailty index (0-1, higher is worse)
        base_frailty = 0.1
        
        # Age effect
        base_frailty += (age - 30) * 0.005
        
        # Exercise reduces frailty
        base_frailty -= exercise_days * 0.02
        
        # Biological age adjustment
        base_frailty += bio_age_adj * 0.015
        
        # Diet effect
        diet_score = behaviors.get('diet_mediterranean_score', 5)
        base_frailty -= (diet_score - 5) * 0.01
        
        measures['frailty_index'] = np.clip(base_frailty + np.random.normal(0, 0.05), 0.0, 0.7)
        
        # Cognitive composite score
        base_cognitive = 95
        
        # Age effect (declines with age)
        base_cognitive -= (age - 30) * 0.3
        
        # Education effect (strong)
        if demographics['education'] in ['Bachelor+']:
            base_cognitive += 8
        elif demographics['education'] in ['Less than HS']:
            base_cognitive -= 8
        
        # Exercise helps cognition
        base_cognitive += exercise_days * 1.5
        
        # Diet effect
        diet_score = behaviors.get('diet_mediterranean_score', 5)
        base_cognitive += (diet_score - 5) * 1.2
        
        # Sleep effect
        sleep_quality = behaviors.get('sleep_quality_score', 7)
        base_cognitive += (sleep_quality - 7) * 0.8
        
        # Biological age adjustment
        base_cognitive -= bio_age_adj * 1.8
        
        measures['cognitive_composite_score'] = np.clip(base_cognitive + np.random.normal(0, 6), 46.0, 127.3)
        
        # Processing speed (similar to cognitive but more age-sensitive)
        base_processing = 85
        base_processing -= (age - 30) * 0.4  # Declines faster with age
        
        if demographics['education'] in ['Bachelor+']:
            base_processing += 6
        
        base_processing += exercise_days * 1.0
        base_processing += (diet_score - 5) * 0.8
        base_processing -= bio_age_adj * 1.5
        
        measures['processing_speed_score'] = np.clip(base_processing + np.random.normal(0, 5), 20.0, 112.1)
        
        return measures
    
    def _calculate_mortality_risk(self, bio_age_adj: float, chronological_age: float) -> float:
        """Calculate annual mortality risk based on biological age acceleration"""
        
        # Base mortality risk by age (CDC data approximation)
        if chronological_age < 25:
            base_risk = 0.001
        elif chronological_age < 35:
            base_risk = 0.0015
        elif chronological_age < 45:
            base_risk = 0.003
        elif chronological_age < 55:
            base_risk = 0.007
        elif chronological_age < 65:
            base_risk = 0.016
        elif chronological_age < 75:
            base_risk = 0.035
        else:
            base_risk = 0.08
        
        # Biological age acceleration multiplier
        risk_multiplier = 1.15 ** bio_age_adj  # Each year of acceleration increases risk by 15%
        
        final_risk = base_risk * risk_multiplier
        return np.clip(final_risk, 0.001, 0.200)
    
    def _estimate_lifespan(self, mortality_risk: float, current_age: float) -> float:
        """Estimate total lifespan based on mortality risk"""
        
        # Simple life expectancy calculation based on current mortality risk
        # This is a simplified model
        if mortality_risk <= 0.005:
            life_expectancy = 88
        elif mortality_risk <= 0.010:
            life_expectancy = 82
        elif mortality_risk <= 0.020:
            life_expectancy = 78
        elif mortality_risk <= 0.040:
            life_expectancy = 74
        else:
            life_expectancy = 70
        
        # Adjust for current age
        remaining_years = max(5, life_expectancy - current_age)
        estimated_lifespan = current_age + remaining_years
        
        return np.clip(estimated_lifespan, 60.0, 120.0)
    
    def _generate_psychosocial_measures(self, behaviors: Dict, demographics: Dict, traits: Dict) -> Dict:
        """Generate psychosocial measures aligned with behaviors and traits"""
        
        measures = {}
        
        # Life satisfaction
        base_satisfaction = 6.5
        
        # Purpose strongly affects satisfaction
        purpose = behaviors.get('purpose_meaning_score', 6)
        base_satisfaction += (purpose - 6) * 0.4
        
        # Social connections affect satisfaction
        social = behaviors.get('social_connections_count', 4)
        base_satisfaction += (social - 4) * 0.2
        
        # Exercise affects mood
        exercise_days = behaviors.get('motion_days_week', 2)
        base_satisfaction += exercise_days * 0.15
        
        # Sleep quality affects satisfaction
        sleep_quality = behaviors.get('sleep_quality_score', 7)
        base_satisfaction += (sleep_quality - 7) * 0.2
        
        measures['life_satisfaction_score'] = np.clip(base_satisfaction + np.random.normal(0, 0.5), 1.0, 10.0)
        
        # Stress level (inverse of many positive factors)
        base_stress = 5.0
        
        # Stress susceptibility trait
        base_stress += (traits.get('stress_susceptibility', 0.5) - 0.5) * 4
        
        # Exercise reduces stress
        base_stress -= exercise_days * 0.3
        
        # Meditation reduces stress significantly
        meditation = behaviors.get('meditation_minutes_week', 60)
        base_stress -= min(meditation / 60, 3) * 0.4
        
        # Sleep quality affects stress
        sleep_quality = behaviors.get('sleep_quality_score', 7)
        base_stress -= (sleep_quality - 7) * 0.3
        
        # Purpose reduces stress
        base_stress -= (purpose - 6) * 0.2
        
        measures['stress_level_score'] = np.clip(base_stress + np.random.normal(0, 0.4), 1.0, 10.0)
        
        # Depression risk
        base_depression = 3.0
        
        # Social connections protect against depression
        base_depression -= (social - 4) * 0.15
        
        # Exercise protects against depression
        base_depression -= exercise_days * 0.2
        
        # Purpose protects against depression
        base_depression -= (purpose - 6) * 0.25
        
        # Sleep affects depression risk
        base_depression -= (sleep_quality - 7) * 0.15
        
        measures['depression_risk_score'] = np.clip(base_depression + np.random.normal(0, 0.3), 1.0, 8.5)
        
        # Social support score
        base_support = 6.0
        
        # Directly related to social connections
        base_support += (social - 4) * 0.3
        
        # Social orientation trait
        base_support += (traits.get('social_orientation', 0.5) - 0.5) * 4
        
        measures['social_support_score'] = np.clip(base_support + np.random.normal(0, 0.4), 2.1, 10.0)
        
        return measures
    
    def _generate_composite_scores(self, behaviors: Dict, outcomes: Dict, demographics: Dict) -> Dict:
        """Generate composite scores that align with behaviors and outcomes"""
        
        composites = {}
        
        # Healthy aging profile (0-100 scale)
        score = 50  # Base score
        
        # TWA behaviors contribute
        exercise_days = behaviors.get('motion_days_week', 2)
        score += exercise_days * 3  # Up to 21 points for 7 days
        
        diet_score = behaviors.get('diet_mediterranean_score', 5)
        score += (diet_score - 5) * 4  # Up to 20 points for excellent diet
        
        sleep_quality = behaviors.get('sleep_quality_score', 7)
        score += (sleep_quality - 7) * 2  # Up to 6 points for excellent sleep
        
        meditation = behaviors.get('meditation_minutes_week', 60)
        score += min(meditation / 30, 10)  # Up to 10 points for regular meditation
        
        # Subtract for negative behaviors
        if behaviors.get('smoking_status') == 'Current':
            score -= 20
        elif behaviors.get('smoking_status') == 'Former':
            score -= 5
        
        alcohol = behaviors.get('alcohol_drinks_week', 5)
        if alcohol > 14:
            score -= (alcohol - 14) * 0.5
        
        # Social factors
        social = behaviors.get('social_connections_count', 4)
        score += (social - 4) * 1.5
        
        purpose = behaviors.get('purpose_meaning_score', 6)
        score += (purpose - 6) * 2
        
        composites['healthy_aging_profile'] = np.clip(score, 0.0, 100.0)
        
        # Blue Zone similarity score (0-100 scale)  
        blue_zone_score = 30  # Base score
        
        # Blue Zone characteristics
        # 1. Regular physical activity (not intense gym)
        if 3 <= exercise_days <= 6:  # Moderate regular activity
            blue_zone_score += 15
        elif exercise_days >= 1:
            blue_zone_score += 10
        
        # 2. Plant-based diet (Mediterranean-style)
        blue_zone_score += (diet_score - 2) * 8  # Heavy weight on diet
        
        # 3. Purpose (ikigai/plan de vida)
        blue_zone_score += (purpose - 4) * 6
        
        # 4. Social connections
        if social >= 3:
            blue_zone_score += 15
        else:
            blue_zone_score += social * 3
        
        # 5. Stress reduction (meditation proxy)
        if meditation > 0:
            blue_zone_score += min(meditation / 20, 12)
        
        # 6. Moderate alcohol consumption
        if 3 <= alcohol <= 10:  # Moderate consumption (like blue zones)
            blue_zone_score += 8
        elif alcohol == 0:
            blue_zone_score += 5
        elif alcohol > 14:
            blue_zone_score -= 10
        
        # 7. No smoking (critical)
        if behaviors.get('smoking_status') == 'Never':
            blue_zone_score += 10
        elif behaviors.get('smoking_status') == 'Former':
            blue_zone_score += 5
        else:
            blue_zone_score -= 15
        
        composites['blue_zone_similarity_score'] = np.clip(blue_zone_score, 0.0, 100.0)
        
        return composites
    
    def _generate_derived_metrics(self, behaviors: Dict) -> Dict:
        """Generate derived boolean and summary metrics"""
        
        metrics = {}
        
        # Boolean flags for meeting guidelines
        metrics['meets_exercise_guidelines'] = behaviors.get('motion_days_week', 0) >= 3
        
        sleep_hours = behaviors.get('sleep_hours', 7.5)
        metrics['meets_sleep_guidelines'] = 7 <= sleep_hours <= 9
        
        metrics['high_diet_quality'] = behaviors.get('diet_mediterranean_score', 5) >= 7
        
        metrics['regular_meditation'] = behaviors.get('meditation_minutes_week', 0) >= 60
        
        metrics['strong_social_support'] = behaviors.get('social_connections_count', 4) >= 6
        
        metrics['high_purpose'] = behaviors.get('purpose_meaning_score', 6) >= 8
        
        alcohol = behaviors.get('alcohol_drinks_week', 5)
        metrics['heavy_drinking'] = alcohol > 14
        
        metrics['current_smoker'] = behaviors.get('smoking_status') == 'Current'
        
        # Summary metrics
        exercise_days = behaviors.get('motion_days_week', 0)
        metrics['total_weekly_exercise_minutes'] = int(exercise_days * 45)  # Assume 45 min per session
        
        # Total wellness score (simplified composite)
        wellness = 0
        wellness += behaviors.get('motion_days_week', 0) * 2
        wellness += behaviors.get('diet_mediterranean_score', 5) * 3
        wellness += behaviors.get('sleep_quality_score', 7) * 2
        wellness += min(behaviors.get('meditation_minutes_week', 0) / 60, 5) * 3
        wellness += behaviors.get('social_connections_count', 4) * 1.5
        wellness += behaviors.get('purpose_meaning_score', 6) * 2
        wellness += behaviors.get('hydration_cups_day', 8)
        
        # Subtract negative factors
        if behaviors.get('smoking_status') == 'Current':
            wellness -= 15
        
        if behaviors.get('alcohol_drinks_week', 5) > 14:
            wellness -= 5
        
        metrics['total_wellness_score'] = max(0, wellness)
        
        return metrics
    
    def _add_realistic_life_events(self, person_data: List[Dict], demographics: Dict) -> List[Dict]:
        """Add realistic life events that impact behaviors"""
        
        # Simple life events that could happen during 12 months
        life_events = []
        
        # Job change (affects stress and income)
        if np.random.random() < 0.08:  # 8% chance per year
            event_month = np.random.randint(1, len(person_data))
            life_events.append(('job_change', event_month))
        
        # Health scare (motivates behavior change)
        if np.random.random() < 0.05:  # 5% chance per year
            event_month = np.random.randint(1, len(person_data))
            life_events.append(('health_scare', event_month))
        
        # Family event (affects social connections)
        if np.random.random() < 0.12:  # 12% chance per year
            event_month = np.random.randint(1, len(person_data))
            life_events.append(('family_event', event_month))
        
        # Apply life events
        for event_type, event_month in life_events:
            if event_type == 'health_scare':
                # Motivates healthier behaviors
                for month in range(event_month, len(person_data)):
                    person_data[month]['motion_days_week'] = min(7, 
                        person_data[month]['motion_days_week'] + 1)
                    person_data[month]['diet_mediterranean_score'] = min(10,
                        person_data[month]['diet_mediterranean_score'] + 1)
                    if person_data[month]['smoking_status'] == 'Current':
                        if np.random.random() < 0.3:  # 30% quit after health scare
                            person_data[month]['smoking_status'] = 'Former'
        
        return person_data
    
    def _add_final_processing(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add final processing and ensure data consistency"""
        
        # Ensure proper data types
        integer_cols = ['motion_days_week', 'meditation_minutes_week', 'social_connections_count',
                       'nature_minutes_week', 'processed_food_servings_week', 'household_size',
                       'months_since_baseline', 'total_weekly_exercise_minutes']
        
        for col in integer_cols:
            if col in dataset.columns:
                dataset[col] = dataset[col].astype(int)
        
        # Round float columns to appropriate precision
        float_2_cols = ['sleep_hours', 'sleep_quality_score', 'hydration_cups_day',
                       'diet_mediterranean_score', 'alcohol_drinks_week', 'cultural_hours_week',
                       'purpose_meaning_score', 'added_sugar_grams_day', 'sodium_grams_day']
        
        for col in float_2_cols:
            if col in dataset.columns:
                dataset[col] = dataset[col].round(1)
        
        # Ensure consistent subject ordering
        dataset = dataset.sort_values(['subject_id', 'month']).reset_index(drop=True)
        
        return dataset
    
    def _month_to_season(self, month: int) -> str:
        """Convert month number (0-11) to season"""
        if month in [11, 0, 1]:  # Dec, Jan, Feb
            return 'Winter'
        elif month in [2, 3, 4]:  # Mar, Apr, May
            return 'Spring'
        elif month in [5, 6, 7]:  # Jun, Jul, Aug
            return 'Summer'
        else:  # Sep, Oct, Nov
            return 'Fall'


if __name__ == "__main__":
    # Test the enhanced generator
    generator = EnhancedLongitudinalTWADataGenerator(random_seed=42)
    
    # Generate small test dataset
    test_data = generator.generate_complete_dataset(n_subjects=10, months=3, include_life_events=False)
    
    print("Enhanced Generator Test Results:")
    print(f"Dataset shape: {test_data.shape}")
    print(f"Columns: {len(test_data.columns)}")
    
    # Test consistency for one person
    person_data = test_data[test_data['subject_id'] == 'SYNTH_000000']
    print(f"\nConsistency test for {person_data['subject_id'].iloc[0]}:")
    print(f"Motion days: {person_data['motion_days_week'].tolist()}")
    print(f"Diet scores: {person_data['diet_mediterranean_score'].round(1).tolist()}")
    print(f"Purpose scores: {person_data['purpose_meaning_score'].round(1).tolist()}")
    print(f"Bio age acceleration: {person_data['biological_age_acceleration'].round(1).tolist()}")