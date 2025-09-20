"""
Longitudinal TWA Dataset Generator
Comprehensive system for generating 12-month longitudinal data with seasonal effects and aging trajectories
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple
import asyncio
from datetime import datetime, timedelta
import json

from demographics_generator import EnhancedDemographicGenerator
from twa_behavior_generator import ResearchValidatedTWAGenerator
from wellness_aging_outcomes import WellnessAgingOutcomeGenerator


class LongitudinalTWADataGenerator:
    """
    Master class that orchestrates the generation of complete longitudinal TWA dataset
    with demographics, behaviors, and wellness/aging outcomes
    """
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Initialize component generators
        self.demographic_gen = EnhancedDemographicGenerator(random_seed)
        self.twa_gen = ResearchValidatedTWAGenerator(random_seed)
        self.outcome_gen = WellnessAgingOutcomeGenerator(random_seed)
        
        # Season mapping
        self.month_to_season = {
            0: 'Winter', 1: 'Winter', 2: 'Spring',    # Jan, Feb, Mar
            3: 'Spring', 4: 'Spring', 5: 'Summer',    # Apr, May, Jun
            6: 'Summer', 7: 'Summer', 8: 'Fall',      # Jul, Aug, Sep
            9: 'Fall', 10: 'Fall', 11: 'Winter'       # Oct, Nov, Dec
        }
        
        # Behavioral trajectory parameters
        self.behavior_stability = {
            'high_stability': ['education', 'income_bracket', 'ethnicity', 'gender'],
            'medium_stability': ['smoking_status', 'social_connections_count', 'purpose_meaning_score'],
            'variable': ['motion_days_week', 'sleep_quality_score', 'diet_mediterranean_score', 
                        'meditation_minutes_week', 'alcohol_drinks_week', 'nature_minutes_week']
        }
        
        # Research validation targets
        self.validation_targets = {
            'exercise_prevalence': 0.28,        # ~28% meet exercise guidelines
            'high_diet_quality': 0.22,          # ~22% have high Mediterranean diet scores
            'regular_meditation': 0.15,         # ~15% meditate regularly
            'strong_social_support': 0.35,      # ~35% have strong social networks
            'high_purpose': 0.40,               # ~40% report high life purpose
            'current_smoking': 0.14,            # ~14% current smoking rate
            'heavy_drinking': 0.12              # ~12% heavy drinking (>14 drinks/week)
        }
    
    def _add_behavioral_consistency(self, person_behaviors: List[Dict], 
                                  demographics: Dict) -> List[Dict]:
        """Add realistic behavioral consistency and gradual changes over time"""
        
        if len(person_behaviors) <= 1:
            return person_behaviors
        
        # Create person-specific behavioral tendencies
        person_traits = self._generate_person_traits(demographics)
        
        # Apply consistency and gradual changes
        for month in range(1, len(person_behaviors)):
            prev_month = person_behaviors[month - 1]
            curr_month = person_behaviors[month]
            
            # High stability behaviors (rarely change)
            for behavior in ['smoking_status']:
                if np.random.random() > 0.05:  # 95% stability
                    curr_month[behavior] = prev_month[behavior]
            
            # Medium stability behaviors (gradual changes)
            for behavior in ['social_connections_count', 'purpose_meaning_score']:
                if behavior in curr_month and behavior in prev_month:
                    # Allow gradual drift with momentum
                    prev_value = prev_month[behavior]
                    curr_value = curr_month[behavior]
                    
                    # 70% weight to previous value, 30% to new random value
                    blended_value = 0.7 * prev_value + 0.3 * curr_value
                    curr_month[behavior] = type(curr_value)(blended_value)
            
            # Variable behaviors (allow more change but with momentum)
            for behavior in ['motion_days_week', 'diet_mediterranean_score', 'meditation_minutes_week']:
                if behavior in curr_month and behavior in prev_month:
                    prev_value = prev_month[behavior]
                    curr_value = curr_month[behavior]
                    
                    # Apply person traits (some people are more consistent)
                    consistency_factor = person_traits.get('behavioral_consistency', 0.5)
                    weight = 0.3 + consistency_factor * 0.4  # 30-70% weight to previous
                    
                    blended_value = weight * prev_value + (1 - weight) * curr_value
                    curr_month[behavior] = type(curr_value)(blended_value)
        
        return person_behaviors
    
    def _generate_person_traits(self, demographics: Dict) -> Dict:
        """Generate stable person-level traits that influence behavior consistency"""
        
        traits = {}
        
        # Behavioral consistency (some people are more routine-oriented)
        base_consistency = 0.5
        
        # Age effect (older adults more consistent)
        if demographics['age_numeric'] > 50:
            base_consistency += 0.2
        elif demographics['age_numeric'] < 30:
            base_consistency -= 0.1
        
        # Education effect (higher education → more health awareness)
        education_effects = {
            'Less than HS': -0.1, 'High School': 0.0, 
            'Some College': 0.1, 'Bachelor+': 0.2
        }
        base_consistency += education_effects.get(demographics['education'], 0.0)
        
        # Add random individual variation
        traits['behavioral_consistency'] = np.clip(
            base_consistency + np.random.normal(0, 0.15), 0.2, 0.8
        )
        
        # Health motivation level (affects adoption of healthy behaviors)
        base_motivation = 0.5
        
        # Fitness level baseline motivation
        fitness_effects = {'Low': -0.2, 'Medium': 0.0, 'High': 0.3}
        base_motivation += fitness_effects.get(demographics['fitness_level'], 0.0)
        
        traits['health_motivation'] = np.clip(
            base_motivation + np.random.normal(0, 0.2), 0.1, 0.9
        )
        
        # Social orientation (affects social behaviors)
        base_social = 0.5
        
        # Urban vs rural effect
        if demographics['urban_rural'] == 'Urban':
            base_social += 0.1
        elif demographics['urban_rural'] == 'Rural':
            base_social -= 0.1
        
        traits['social_orientation'] = np.clip(
            base_social + np.random.normal(0, 0.2), 0.2, 0.8
        )
        
        return traits
    
    def _calculate_healthy_aging_profile(self, behaviors: Dict, outcomes: Dict) -> float:
        """Calculate composite healthy aging score based on research"""
        
        score = 0.0
        
        # Behavioral components (40% weight)
        behavioral_score = 0.0
        
        # Exercise component
        behavioral_score += min(behaviors['motion_days_week'] / 7, 1.0) * 0.15
        
        # Diet quality component
        behavioral_score += min(behaviors['diet_mediterranean_score'] / 10, 1.0) * 0.15
        
        # Mindfulness component
        behavioral_score += min(behaviors['meditation_minutes_week'] / 300, 1.0) * 0.10
        
        # Biological components (40% weight)
        biological_score = 0.0
        
        # Biological age acceleration (inverted - negative acceleration is good)
        bio_age_component = max(0, (5 - outcomes['biological_age_acceleration']) / 10)
        biological_score += bio_age_component * 0.20
        
        # Mortality risk (inverted - lower risk is better)
        mortality_component = max(0, 1 - min(outcomes['mortality_risk_score'] * 100, 1))
        biological_score += mortality_component * 0.20
        
        # Psychosocial components (20% weight)
        psychosocial_score = 0.0
        
        # Life satisfaction component
        psychosocial_score += (outcomes['life_satisfaction_score'] / 10) * 0.10
        
        # Purpose component
        psychosocial_score += (behaviors['purpose_meaning_score'] / 10) * 0.10
        
        total_score = (behavioral_score + biological_score + psychosocial_score) * 100
        return min(100, max(0, total_score))
    
    def _calculate_blue_zone_similarity(self, behaviors: Dict, demographics: Dict) -> float:
        """Calculate similarity to Blue Zone lifestyle patterns"""
        
        similarity_factors = {}
        
        # Plant-rich diet (Mediterranean diet as proxy)
        similarity_factors['plant_rich_diet'] = min(behaviors['diet_mediterranean_score'] / 8, 1.0)
        
        # Natural movement (regular but not excessive exercise)
        optimal_motion = 4  # 4 days/week is ideal balance
        if behaviors['motion_days_week'] <= optimal_motion:
            similarity_factors['natural_movement'] = behaviors['motion_days_week'] / optimal_motion
        else:
            similarity_factors['natural_movement'] = 1.0 - (behaviors['motion_days_week'] - optimal_motion) / 3
        
        # Stress management (meditation and nature connection)
        stress_mgmt = (min(behaviors['meditation_minutes_week'] / 150, 1.0) * 0.6 + 
                      min(behaviors['nature_minutes_week'] / 120, 1.0) * 0.4)
        similarity_factors['stress_management'] = stress_mgmt
        
        # Strong social connections
        similarity_factors['social_connections'] = min(behaviors['social_connections_count'] / 4, 1.0)
        
        # Purpose driven life
        similarity_factors['purpose_driven'] = min(behaviors['purpose_meaning_score'] / 8, 1.0)
        
        # Moderate alcohol consumption (or none)
        if behaviors['alcohol_drinks_week'] <= 7:
            similarity_factors['moderate_alcohol'] = 1.0
        elif behaviors['alcohol_drinks_week'] <= 14:
            similarity_factors['moderate_alcohol'] = 0.5
        else:
            similarity_factors['moderate_alcohol'] = 0.0
        
        # No smoking
        similarity_factors['no_smoking'] = 1.0 if behaviors['smoking_status'] == 'Never' else 0.0
        
        # Quality sleep
        if behaviors['sleep_hours'] >= 7 and behaviors['sleep_quality_score'] >= 7:
            similarity_factors['quality_sleep'] = 1.0
        elif behaviors['sleep_hours'] >= 6 and behaviors['sleep_quality_score'] >= 6:
            similarity_factors['quality_sleep'] = 0.7
        else:
            similarity_factors['quality_sleep'] = 0.3
        
        # Calculate weighted average
        return sum(similarity_factors.values()) / len(similarity_factors) * 100
    
    def _add_life_events(self, person_data: List[Dict], demographics: Dict) -> List[Dict]:
        """Add realistic life events that can affect behaviors and outcomes"""
        
        # Potential life events with probabilities and effects
        life_events = {
            'health_scare': {
                'probability': 0.02,  # 2% chance per month
                'age_multiplier': lambda age: 1.0 if age < 50 else 2.0,
                'effects': {
                    'diet_mediterranean_score': +1.5,
                    'motion_days_week': +1,
                    'meditation_minutes_week': +50,
                    'smoking_status': 'quit_if_current'
                },
                'duration_months': 6
            },
            'job_stress': {
                'probability': 0.015,  # 1.5% chance per month
                'age_multiplier': lambda age: 1.5 if 30 <= age <= 55 else 0.5,
                'effects': {
                    'stress_level_score': +2.0,
                    'alcohol_drinks_week': +3,
                    'sleep_quality_score': -1.5,
                    'meditation_minutes_week': -20
                },
                'duration_months': 4
            },
            'relationship_change': {
                'probability': 0.01,  # 1% chance per month
                'age_multiplier': lambda age: 1.5 if age < 40 else 0.7,
                'effects': {
                    'social_connections_count': '+/-2',  # Can go either way
                    'life_satisfaction_score': '+/-2',
                    'purpose_meaning_score': '+/-1'
                },
                'duration_months': 3
            }
        }
        
        # Track active events for this person
        active_events = {}
        
        for month_idx, month_data in enumerate(person_data):
            # Check for new events
            age = demographics['age_numeric']
            
            for event_name, event_config in life_events.items():
                if event_name not in active_events:
                    base_prob = event_config['probability']
                    age_adj_prob = base_prob * event_config['age_multiplier'](age)
                    
                    if np.random.random() < age_adj_prob:
                        # Event occurs
                        active_events[event_name] = {
                            'start_month': month_idx,
                            'end_month': month_idx + event_config['duration_months'],
                            'effects': event_config['effects']
                        }
            
            # Apply effects of active events
            for event_name, event_data in list(active_events.items()):
                if month_idx >= event_data['start_month'] and month_idx < event_data['end_month']:
                    # Apply event effects
                    for variable, effect in event_data['effects'].items():
                        if variable in month_data:
                            current_value = month_data[variable]
                            
                            if isinstance(effect, (int, float)):
                                if isinstance(current_value, (int, float)):
                                    month_data[variable] = type(current_value)(
                                        max(0, current_value + effect)
                                    )
                            elif effect == 'quit_if_current' and variable == 'smoking_status':
                                if current_value == 'Current':
                                    month_data[variable] = 'Former'
                            elif '+/-' in str(effect):
                                # Bidirectional effect
                                magnitude = float(effect.replace('+/-', ''))
                                change = np.random.choice([-magnitude, magnitude])
                                if isinstance(current_value, (int, float)):
                                    month_data[variable] = type(current_value)(
                                        max(0, current_value + change)
                                    )
                
                # Remove expired events
                if month_idx >= event_data['end_month']:
                    del active_events[event_name]
        
        return person_data
    
    def generate_complete_dataset(self, n_subjects: int = 10000, months: int = 12,
                                include_life_events: bool = True) -> pd.DataFrame:
        """
        Generate complete longitudinal TWA dataset with outcomes
        """
        
        print(f"Generating longitudinal dataset: {n_subjects} subjects, {months} months")
        print("=" * 60)
        
        # Generate base demographics
        print("Phase 1: Generating demographic profiles...")
        demographics = self.demographic_gen.generate_correlated_demographics(n_subjects)
        
        complete_dataset = []
        subjects_processed = 0
        
        print("Phase 2: Generating longitudinal behavioral and outcome data...")
        
        for person in demographics:
            baseline_age = person['age_numeric']
            person_monthly_data = []
            
            # Generate monthly observations
            for month in range(months):
                season = self.month_to_season[month]
                observation_date = datetime(2024, month + 1, 15)
                
                # Generate TWA behaviors for this month
                twa_behaviors = self.twa_gen.generate_monthly_twa_behaviors(
                    person, month, season
                )
                
                # Generate wellness/aging outcomes
                outcomes = self.outcome_gen.generate_aging_wellness_outcomes(
                    person, twa_behaviors, month
                )
                
                # Compile complete monthly record
                monthly_record = {
                    # Identifiers and time
                    'subject_id': person['subject_id'],
                    'month': month,
                    'season': season,
                    'observation_date': observation_date.strftime('%Y-%m-%d'),
                    
                    # Demographics (time-invariant)
                    **person,
                    
                    # TWA Behaviors
                    **twa_behaviors,
                    
                    # Wellness & Aging Outcomes
                    **outcomes
                }
                
                person_monthly_data.append(monthly_record)
            
            # Add behavioral consistency across months
            person_monthly_data = self._add_behavioral_consistency(person_monthly_data, person)
            
            # Add life events if requested
            if include_life_events:
                person_monthly_data = self._add_life_events(person_monthly_data, person)
            
            # Calculate derived variables for each month
            for monthly_record in person_monthly_data:
                # Research validation variables
                monthly_record['meets_exercise_guidelines'] = monthly_record['motion_days_week'] >= 3
                monthly_record['meets_sleep_guidelines'] = (
                    monthly_record['sleep_hours'] >= 7 and 
                    monthly_record['sleep_quality_score'] >= 6
                )
                monthly_record['high_diet_quality'] = monthly_record['diet_mediterranean_score'] >= 7
                monthly_record['regular_meditation'] = monthly_record['meditation_minutes_week'] >= 150
                monthly_record['strong_social_support'] = monthly_record['social_connections_count'] >= 4
                monthly_record['high_purpose'] = monthly_record['purpose_meaning_score'] >= 8
                monthly_record['heavy_drinking'] = monthly_record['alcohol_drinks_week'] > 14
                monthly_record['current_smoker'] = monthly_record['smoking_status'] == 'Current'
                
                # Composite scores
                monthly_record['healthy_aging_profile'] = self._calculate_healthy_aging_profile(
                    {k: v for k, v in monthly_record.items() if k in [
                        'motion_days_week', 'diet_mediterranean_score', 'meditation_minutes_week',
                        'purpose_meaning_score'
                    ]},
                    {k: v for k, v in monthly_record.items() if k in [
                        'biological_age_acceleration', 'mortality_risk_score', 'life_satisfaction_score'
                    ]}
                )
                
                monthly_record['blue_zone_similarity_score'] = self._calculate_blue_zone_similarity(
                    {k: v for k, v in monthly_record.items() if k in [
                        'diet_mediterranean_score', 'motion_days_week', 'meditation_minutes_week',
                        'nature_minutes_week', 'social_connections_count', 'purpose_meaning_score',
                        'alcohol_drinks_week', 'smoking_status', 'sleep_hours', 'sleep_quality_score'
                    ]},
                    person
                )
            
            # Add to complete dataset
            complete_dataset.extend(person_monthly_data)
            subjects_processed += 1
            
            # Progress reporting
            if subjects_processed % 500 == 0:
                print(f"  Processed {subjects_processed}/{n_subjects} subjects ({subjects_processed/n_subjects*100:.1f}%)")
        
        print(f"Phase 3: Converting to DataFrame and final processing...")
        
        # Convert to DataFrame
        df = pd.DataFrame(complete_dataset)
        
        # Add additional derived variables
        df['months_since_baseline'] = df['month']
        df['age_at_observation'] = df['age_numeric'] + (df['month'] / 12)
        df['total_weekly_exercise_minutes'] = df['motion_days_week'] * 45  # Assume 45 min per session
        df['total_wellness_score'] = (
            (df['healthy_aging_profile'] + df['blue_zone_similarity_score']) / 2
        )
        
        print(f"Dataset generation completed successfully!")
        print(f"Final dataset: {len(df)} total observations")
        print(f"  {df['subject_id'].nunique()} unique subjects")
        print(f"  {df['month'].nunique()} months of data")
        print(f"  {len(df.columns)} total variables")
        
        return df
    
    def generate_validation_summary(self, df: pd.DataFrame) -> Dict:
        """Generate validation summary comparing to research benchmarks"""
        
        # Calculate prevalence rates from final month data
        final_month = df[df['month'] == df['month'].max()]
        
        validation_summary = {
            'sample_characteristics': {
                'total_subjects': df['subject_id'].nunique(),
                'total_observations': len(df),
                'months_of_data': df['month'].nunique(),
                'mean_age': final_month['age_numeric'].mean(),
                'age_range': f"{final_month['age_numeric'].min()}-{final_month['age_numeric'].max()}",
                'gender_distribution': final_month['gender'].value_counts().to_dict(),
                'education_distribution': final_month['education'].value_counts().to_dict()
            },
            'behavior_prevalence': {
                'meets_exercise_guidelines': final_month['meets_exercise_guidelines'].mean(),
                'high_diet_quality': final_month['high_diet_quality'].mean(),
                'regular_meditation': final_month['regular_meditation'].mean(),
                'strong_social_support': final_month['strong_social_support'].mean(),
                'high_purpose': final_month['high_purpose'].mean(),
                'current_smoking': final_month['current_smoker'].mean(),
                'heavy_drinking': final_month['heavy_drinking'].mean()
            },
            'outcome_characteristics': {
                'mean_biological_age_acceleration': final_month['biological_age_acceleration'].mean(),
                'mean_mortality_risk': final_month['mortality_risk_score'].mean(),
                'mean_healthy_aging_profile': final_month['healthy_aging_profile'].mean(),
                'mean_blue_zone_similarity': final_month['blue_zone_similarity_score'].mean(),
                'mean_life_satisfaction': final_month['life_satisfaction_score'].mean()
            },
            'research_benchmark_comparison': {}
        }
        
        # Compare to research targets
        for metric, target in self.validation_targets.items():
            if metric in validation_summary['behavior_prevalence']:
                actual = validation_summary['behavior_prevalence'][metric]
                difference = actual - target
                percent_diff = (difference / target) * 100
                
                validation_summary['research_benchmark_comparison'][metric] = {
                    'target': target,
                    'actual': actual,
                    'difference': difference,
                    'percent_difference': percent_diff,
                    'within_5_percent': abs(percent_diff) <= 5
                }
        
        return validation_summary


if __name__ == "__main__":
    # Test the complete longitudinal generator
    generator = LongitudinalTWADataGenerator(random_seed=42)
    
    print("Testing Longitudinal TWA Dataset Generator")
    print("=" * 50)
    
    # Generate small test dataset
    test_df = generator.generate_complete_dataset(n_subjects=100, months=3, include_life_events=False)
    
    print(f"\nTest Dataset Summary:")
    print(f"Shape: {test_df.shape}")
    print(f"Columns: {list(test_df.columns)}")
    print(f"Subjects: {test_df['subject_id'].nunique()}")
    print(f"Months: {test_df['month'].unique()}")
    
    # Show sample records
    print(f"\nSample Records (Subject SYNTH_000000):")
    sample_subject = test_df[test_df['subject_id'] == 'SYNTH_000000'][
        ['month', 'season', 'motion_days_week', 'diet_mediterranean_score', 
         'biological_age_acceleration', 'healthy_aging_profile']
    ]
    print(sample_subject.to_string(index=False))
    
    # Generate validation summary
    validation = generator.generate_validation_summary(test_df)
    print(f"\nValidation Summary:")
    print(f"Exercise guidelines met: {validation['behavior_prevalence']['meets_exercise_guidelines']:.1%}")
    print(f"High diet quality: {validation['behavior_prevalence']['high_diet_quality']:.1%}")
    print(f"Current smoking: {validation['behavior_prevalence']['current_smoking']:.1%}")
    print(f"Mean healthy aging profile: {validation['outcome_characteristics']['mean_healthy_aging_profile']:.1f}")