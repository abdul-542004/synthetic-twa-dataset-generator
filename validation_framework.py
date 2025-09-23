"""
Dataset Validation Framework
Comprehensive validation against research benchmarks to ensure dataset quality
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class DatasetValidator:
    """
    Validates synthetic TWA dataset against research benchmarks and expected correlations
    Ensures dataset maintains scientific rigor and realistic patterns
    """
    
    def __init__(self):
        # Research benchmarks from published studies
        self.research_benchmarks = {
            'demographic_distributions': {
                'age_18_24': 0.11, 'age_25_34': 0.14, 'age_35_44': 0.13,
                'age_45_54': 0.12, 'age_55_64': 0.13, 'age_65_74': 0.11,
                'education_bachelor_plus': 0.33,
                'income_over_75k': 0.37,
                'urban_suburban': 0.85
            },
            'health_behavior_prevalence': {
                'meets_exercise_guidelines': 0.28,      # CDC data
                'high_diet_quality': 0.22,              # Mediterranean diet adherence
                'regular_meditation': 0.15,             # Mindfulness practice prevalence  
                'adequate_sleep': 0.35,                 # 7+ hours good quality sleep
                'strong_social_support': 0.35,          # 4+ close relationships
                'high_life_purpose': 0.40,              # High purpose/meaning scores
                'current_smoking': 0.14,                # CDC smoking rates
                'heavy_drinking': 0.12,                 # >14 drinks/week prevalence
                'high_processed_food': 0.25             # High ultra-processed consumption
            },
            'health_outcome_ranges': {
                'biological_age_acceleration': {'mean': 0.0, 'std': 3.5, 'range': (-10, 15)},
                'mortality_risk_score': {'mean': 0.015, 'std': 0.025, 'range': (0.001, 0.2)},
                'crp_mg_l': {'mean': 1.5, 'std': 1.2, 'range': (0.1, 10)},
                'il6_pg_ml': {'mean': 2.8, 'std': 1.8, 'range': (0.5, 15)},
                'grip_strength_kg': {'mean': 35, 'std': 12, 'range': (10, 70)},
                'life_satisfaction_score': {'mean': 6.5, 'std': 1.8, 'range': (1, 10)}
            },
            'correlations': {
                # Established research correlations with acceptable ranges
                'education_income': {'target': 0.42, 'acceptable_range': (0.35, 0.50)},
                'exercise_sleep': {'target': 0.32, 'acceptable_range': (0.25, 0.40)},
                'diet_meditation': {'target': 0.28, 'acceptable_range': (0.20, 0.35)},
                'social_purpose': {'target': 0.31, 'acceptable_range': (0.25, 0.38)},
                'smoking_drinking': {'target': 0.48, 'acceptable_range': (0.40, 0.55)},
                'purpose_mortality_risk': {'target': -0.35, 'acceptable_range': (-0.45, -0.25)},
                'exercise_biological_age': {'target': -0.25, 'acceptable_range': (-0.35, -0.15)},
                'age_frailty': {'target': 0.45, 'acceptable_range': (0.35, 0.55)}
            },
            'biological_age_effects': {
                # Expected biological age effects from interventions
                'high_exercise_effect': {'target': -1.2, 'acceptable_range': (-2.0, -0.5)},
                'mediterranean_diet_effect': {'target': -2.3, 'acceptable_range': (-3.0, -1.5)},
                'smoking_effect': {'target': 5.3, 'acceptable_range': (4.0, 7.0)},
                'high_purpose_effect': {'target': -3.1, 'acceptable_range': (-4.0, -2.0)},
                'social_isolation_effect': {'target': 2.8, 'acceptable_range': (2.0, 4.0)}
            }
        }
        
        # Validation tolerances
        self.tolerance_levels = {
            'excellent': 0.02,    # Within 2%
            'good': 0.05,         # Within 5% 
            'acceptable': 0.10,   # Within 10%
            'poor': 0.20          # Within 20%
        }
    
    def validate_demographic_distributions(self, df: pd.DataFrame) -> Dict:
        """Validate demographic distributions against US Census data"""
        
        results = {'category': 'demographic_distributions', 'validations': {}}
        
        # Age distribution validation
        age_dist = df.groupby('age_group')['subject_id'].nunique() / df['subject_id'].nunique()
        
        for age_group in ['18-24', '25-34', '35-44', '45-54', '55-64', '65-74']:
            key = f'age_{age_group.replace("-", "_")}'
            if key in self.research_benchmarks['demographic_distributions']:
                expected = self.research_benchmarks['demographic_distributions'][key]
                actual = age_dist.get(age_group, 0.0)
                
                results['validations'][key] = {
                    'expected': expected,
                    'actual': actual,
                    'difference': actual - expected,
                    'percent_error': abs(actual - expected) / expected * 100,
                    'status': self._get_validation_status(actual, expected)
                }
        
        # Education distribution
        education_dist = df.groupby('education')['subject_id'].nunique() / df['subject_id'].nunique()
        bachelor_plus = education_dist.get('Bachelor+', 0.0)
        expected_bachelor = self.research_benchmarks['demographic_distributions']['education_bachelor_plus']
        
        results['validations']['education_bachelor_plus'] = {
            'expected': expected_bachelor,
            'actual': bachelor_plus,
            'difference': bachelor_plus - expected_bachelor,
            'percent_error': abs(bachelor_plus - expected_bachelor) / expected_bachelor * 100,
            'status': self._get_validation_status(bachelor_plus, expected_bachelor)
        }
        
        # Income distribution
        income_high = df[df['income_bracket'].isin(['$75-100k', '$100-150k', '>$150k'])]['subject_id'].nunique() / df['subject_id'].nunique()
        expected_income = self.research_benchmarks['demographic_distributions']['income_over_75k']
        
        results['validations']['income_over_75k'] = {
            'expected': expected_income,
            'actual': income_high,
            'difference': income_high - expected_income,
            'percent_error': abs(income_high - expected_income) / expected_income * 100,
            'status': self._get_validation_status(income_high, expected_income)
        }
        
        return results
    
    def validate_behavior_prevalence(self, df: pd.DataFrame) -> Dict:
        """Validate health behavior prevalence against research data"""
        
        results = {'category': 'health_behavior_prevalence', 'validations': {}}
        
        # Use final month data for prevalence calculations
        final_month = df[df['month'] == df['month'].max()]
        
        behavior_mappings = {
            'meets_exercise_guidelines': 'meets_exercise_guidelines',
            'high_diet_quality': 'high_diet_quality', 
            'regular_meditation': 'regular_meditation',
            'adequate_sleep': 'meets_sleep_guidelines',
            'strong_social_support': 'strong_social_support',
            'high_life_purpose': 'high_purpose',
            'current_smoking': 'current_smoker',
            'heavy_drinking': 'heavy_drinking'
        }
        
        for benchmark_key, df_column in behavior_mappings.items():
            if df_column in final_month.columns:
                expected = self.research_benchmarks['health_behavior_prevalence'][benchmark_key]
                actual = final_month[df_column].mean()
                
                results['validations'][benchmark_key] = {
                    'expected': expected,
                    'actual': actual,
                    'difference': actual - expected,
                    'percent_error': abs(actual - expected) / expected * 100,
                    'status': self._get_validation_status(actual, expected)
                }
        
        return results
    
    def validate_outcome_distributions(self, df: pd.DataFrame) -> Dict:
        """Validate health outcome distributions against research ranges"""
        
        results = {'category': 'health_outcome_ranges', 'validations': {}}
        
        # Use final month data
        final_month = df[df['month'] == df['month'].max()]
        
        for outcome, benchmarks in self.research_benchmarks['health_outcome_ranges'].items():
            if outcome in final_month.columns:
                data = final_month[outcome].dropna()
                
                actual_mean = data.mean()
                actual_std = data.std()
                actual_range = (data.min(), data.max())
                
                expected_mean = benchmarks['mean']
                expected_std = benchmarks['std']
                expected_range = benchmarks['range']
                
                # Validate mean
                mean_status = self._get_validation_status(actual_mean, expected_mean)
                
                # Validate standard deviation (more lenient)
                std_error = abs(actual_std - expected_std) / expected_std
                std_status = 'excellent' if std_error < 0.15 else 'good' if std_error < 0.30 else 'acceptable'
                
                # Validate range (check if actual range is within reasonable bounds)
                range_status = 'excellent' if (actual_range[0] >= expected_range[0] * 0.8 and 
                                             actual_range[1] <= expected_range[1] * 1.2) else 'acceptable'
                
                results['validations'][outcome] = {
                    'mean': {
                        'expected': expected_mean,
                        'actual': actual_mean,
                        'difference': actual_mean - expected_mean,
                        'percent_error': abs(actual_mean - expected_mean) / abs(expected_mean) * 100,
                        'status': mean_status
                    },
                    'std': {
                        'expected': expected_std,
                        'actual': actual_std,
                        'status': std_status
                    },
                    'range': {
                        'expected': expected_range,
                        'actual': actual_range,
                        'status': range_status
                    }
                }
        
        return results
    
    def validate_correlations(self, df: pd.DataFrame) -> Dict:
        """Validate key correlations against research literature"""
        
        results = {'category': 'correlations', 'validations': {}}
        
        # Use final month data for correlation analysis
        final_month = df[df['month'] == df['month'].max()]
        
        correlation_mappings = {
            'education_income': ('income_numeric', 'education'),
            'exercise_sleep': ('motion_days_week', 'sleep_quality_score'),
            'diet_meditation': ('diet_mediterranean_score', 'meditation_minutes_week'),
            'social_purpose': ('social_connections_count', 'purpose_meaning_score'),
            'smoking_drinking': ('smoking_status', 'alcohol_drinks_week'),
            'purpose_mortality_risk': ('purpose_meaning_score', 'mortality_risk_score'),
            'exercise_biological_age': ('motion_days_week', 'biological_age_acceleration'),
            'age_frailty': ('age_numeric', 'frailty_index')
        }
        
        for corr_name, (var1, var2) in correlation_mappings.items():
            if var1 in final_month.columns and var2 in final_month.columns:
                
                # Handle categorical variables
                if var1 == 'education':
                    # Convert education to numeric
                    education_map = {'Less than HS': 1, 'High School': 2, 'Some College': 3, 'Bachelor+': 4}
                    var1_data = final_month[var1].map(education_map)
                elif var1 == 'smoking_status':
                    # Convert smoking to numeric
                    smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
                    var1_data = final_month[var1].map(smoking_map)
                else:
                    var1_data = final_month[var1]
                
                if var2 == 'education':
                    # Convert education to numeric
                    education_map = {'Less than HS': 1, 'High School': 2, 'Some College': 3, 'Bachelor+': 4}
                    var2_data = final_month[var2].map(education_map)
                elif var2 == 'smoking_status':
                    # Convert smoking to numeric
                    smoking_map = {'Never': 0, 'Former': 1, 'Current': 2}
                    var2_data = final_month[var2].map(smoking_map)
                else:
                    var2_data = final_month[var2]
                
                # Ensure both variables are numeric and remove any non-numeric values
                var1_clean = pd.to_numeric(var1_data, errors='coerce').dropna()
                var2_clean = pd.to_numeric(var2_data, errors='coerce').dropna()
                
                # Align the data (keep only rows where both variables have valid values)
                valid_indices = var1_clean.index.intersection(var2_clean.index)
                if len(valid_indices) > 10:  # Need minimum sample size
                    var1_final = var1_clean.loc[valid_indices]
                    var2_final = var2_clean.loc[valid_indices]
                    
                    # Calculate correlation
                    correlation = stats.pearsonr(var1_final, var2_final)[0]
                    # Calculate correlation
                    correlation = stats.pearsonr(var1_final, var2_final)[0]
                    expected = self.research_benchmarks['correlations'][corr_name]['target']
                    acceptable_range = self.research_benchmarks['correlations'][corr_name]['acceptable_range']
                    
                    # Check if within acceptable range
                    within_range = acceptable_range[0] <= correlation <= acceptable_range[1]
                    status = 'excellent' if within_range else 'poor'
                    
                    results['validations'][corr_name] = {
                        'expected': expected,
                        'actual': correlation,
                        'difference': correlation - expected,
                        'acceptable_range': acceptable_range,
                        'within_range': within_range,
                        'status': status,
                        'sample_size': len(valid_indices)
                    }
                else:
                    # Insufficient data for correlation
                    results['validations'][corr_name] = {
                        'expected': self.research_benchmarks['correlations'][corr_name]['target'],
                        'actual': None,
                        'status': 'insufficient_data',
                        'sample_size': len(valid_indices) if 'valid_indices' in locals() else 0
                    }
        
        return results
    
    def validate_biological_age_effects(self, df: pd.DataFrame) -> Dict:
        """Validate biological age effects from lifestyle interventions"""
        
        results = {'category': 'biological_age_effects', 'validations': {}}
        
        # Use final month data
        final_month = df[df['month'] == df['month'].max()]
        
        # High exercise effect
        high_exercise = final_month[final_month['motion_days_week'] >= 4]
        low_exercise = final_month[final_month['motion_days_week'] <= 1]
        
        if len(high_exercise) > 10 and len(low_exercise) > 10:
            exercise_effect = high_exercise['biological_age_acceleration'].mean() - low_exercise['biological_age_acceleration'].mean()
            expected = self.research_benchmarks['biological_age_effects']['high_exercise_effect']['target']
            acceptable_range = self.research_benchmarks['biological_age_effects']['high_exercise_effect']['acceptable_range']
            
            results['validations']['high_exercise_effect'] = {
                'expected': expected,
                'actual': exercise_effect,
                'difference': exercise_effect - expected,
                'acceptable_range': acceptable_range,
                'within_range': acceptable_range[0] <= exercise_effect <= acceptable_range[1],
                'n_high_exercise': len(high_exercise),
                'n_low_exercise': len(low_exercise)
            }
        
        # Mediterranean diet effect
        high_diet = final_month[final_month['diet_mediterranean_score'] >= 8]
        low_diet = final_month[final_month['diet_mediterranean_score'] <= 4]
        
        if len(high_diet) > 10 and len(low_diet) > 10:
            diet_effect = high_diet['biological_age_acceleration'].mean() - low_diet['biological_age_acceleration'].mean()
            expected = self.research_benchmarks['biological_age_effects']['mediterranean_diet_effect']['target']
            acceptable_range = self.research_benchmarks['biological_age_effects']['mediterranean_diet_effect']['acceptable_range']
            
            results['validations']['mediterranean_diet_effect'] = {
                'expected': expected,
                'actual': diet_effect,
                'difference': diet_effect - expected,
                'acceptable_range': acceptable_range,
                'within_range': acceptable_range[0] <= diet_effect <= acceptable_range[1],
                'n_high_diet': len(high_diet),
                'n_low_diet': len(low_diet)
            }
        
        # Smoking effect
        current_smokers = final_month[final_month['smoking_status'] == 'Current']
        never_smokers = final_month[final_month['smoking_status'] == 'Never']
        
        if len(current_smokers) > 5 and len(never_smokers) > 10:
            smoking_effect = current_smokers['biological_age_acceleration'].mean() - never_smokers['biological_age_acceleration'].mean()
            expected = self.research_benchmarks['biological_age_effects']['smoking_effect']['target']
            acceptable_range = self.research_benchmarks['biological_age_effects']['smoking_effect']['acceptable_range']
            
            results['validations']['smoking_effect'] = {
                'expected': expected,
                'actual': smoking_effect,
                'difference': smoking_effect - expected,
                'acceptable_range': acceptable_range,
                'within_range': acceptable_range[0] <= smoking_effect <= acceptable_range[1],
                'n_smokers': len(current_smokers),
                'n_never_smokers': len(never_smokers)
            }
        
        return results
    
    def _get_validation_status(self, actual: float, expected: float) -> str:
        """Determine validation status based on percent error"""
        if expected == 0:
            return 'excellent' if abs(actual) < 0.01 else 'poor'
        
        percent_error = abs(actual - expected) / abs(expected)
        
        if percent_error <= self.tolerance_levels['excellent']:
            return 'excellent'
        elif percent_error <= self.tolerance_levels['good']:
            return 'good'
        elif percent_error <= self.tolerance_levels['acceptable']:
            return 'acceptable'
        else:
            return 'poor'
    
    def generate_comprehensive_validation_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive validation report"""
        
        print("Generating Comprehensive Validation Report...")
        print("=" * 50)
        
        validation_results = {
            'dataset_overview': {
                'total_subjects': df['subject_id'].nunique(),
                'total_observations': len(df),
                'months_of_data': df['month'].nunique(),
                'variables': len(df.columns)
            },
            'validations': {}
        }
        
        # Run all validation categories
        print("1. Validating demographic distributions...")
        validation_results['validations']['demographics'] = self.validate_demographic_distributions(df)
        
        print("2. Validating health behavior prevalence...")
        validation_results['validations']['behaviors'] = self.validate_behavior_prevalence(df)
        
        print("3. Validating outcome distributions...")
        validation_results['validations']['outcomes'] = self.validate_outcome_distributions(df)
        
        print("4. Validating correlations...")
        validation_results['validations']['correlations'] = self.validate_correlations(df)
        
        print("5. Validating biological age effects...")
        validation_results['validations']['bio_age_effects'] = self.validate_biological_age_effects(df)
        
        # Calculate overall validation summary
        print("6. Calculating overall validation summary...")
        validation_results['summary'] = self._calculate_validation_summary(validation_results['validations'])
        
        return validation_results
    
    def _calculate_validation_summary(self, validations: Dict) -> Dict:
        """Calculate overall validation summary statistics"""
        
        all_statuses = []
        
        for category, category_data in validations.items():
            for validation_name, validation_data in category_data.get('validations', {}).items():
                if isinstance(validation_data, dict):
                    if 'status' in validation_data:
                        all_statuses.append(validation_data['status'])
                    elif 'mean' in validation_data and 'status' in validation_data['mean']:
                        all_statuses.append(validation_data['mean']['status'])
        
        status_counts = {status: all_statuses.count(status) for status in ['excellent', 'good', 'acceptable', 'poor']}
        total_validations = len(all_statuses)
        
        summary = {
            'total_validations': total_validations,
            'status_counts': status_counts,
            'status_percentages': {status: (count / total_validations * 100) if total_validations > 0 else 0 
                                 for status, count in status_counts.items()},
            'overall_quality': 'excellent' if status_counts['excellent'] / total_validations > 0.7 else
                              'good' if (status_counts['excellent'] + status_counts['good']) / total_validations > 0.8 else
                              'acceptable' if status_counts['poor'] / total_validations < 0.2 else 'poor'
        }
        
        return summary
    
    def print_validation_report(self, validation_results: Dict):
        """Print formatted validation report"""
        
        print("\n" + "="*70)
        print("DATASET VALIDATION REPORT")
        print("="*70)
        
        # Dataset overview
        overview = validation_results['dataset_overview']
        print(f"\nDATASET OVERVIEW:")
        print(f"  Total Subjects: {overview['total_subjects']:,}")
        print(f"  Total Observations: {overview['total_observations']:,}")
        print(f"  Months of Data: {overview['months_of_data']}")
        print(f"  Variables: {overview['variables']}")
        
        # Overall summary
        summary = validation_results['summary']
        print(f"\nOVERALL VALIDATION SUMMARY:")
        print(f"  Total Validations: {summary['total_validations']}")
        print(f"  Overall Quality: {summary['overall_quality'].upper()}")
        print(f"  Status Distribution:")
        for status, percentage in summary['status_percentages'].items():
            print(f"    {status.capitalize()}: {percentage:.1f}%")
        
        # Category details
        print(f"\nVALIDATION DETAILS BY CATEGORY:")
        
        for category_name, category_data in validation_results['validations'].items():
            print(f"\n{category_name.upper()}:")
            
            for validation_name, validation_data in category_data.get('validations', {}).items():
                if isinstance(validation_data, dict):
                    if 'status' in validation_data:
                        status = validation_data['status']
                        expected = validation_data.get('expected', 'N/A')
                        actual = validation_data.get('actual', 'N/A')
                        print(f"  {validation_name}: {status.upper()} (Expected: {expected:.3f}, Actual: {actual:.3f})")
                    elif 'mean' in validation_data:
                        status = validation_data['mean']['status']
                        expected = validation_data['mean']['expected']
                        actual = validation_data['mean']['actual']
                        print(f"  {validation_name}: {status.upper()} (Expected: {expected:.3f}, Actual: {actual:.3f})")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    # Test validation framework
    from longitudinal_generator import LongitudinalTWADataGenerator
    
    print("Testing Dataset Validation Framework")
    print("=" * 50)
    
    # Generate test dataset
    generator = LongitudinalTWADataGenerator(random_seed=42)
    test_df = generator.generate_complete_dataset(n_subjects=1000, months=3, include_life_events=False)
    
    # Run validation
    validator = DatasetValidator()
    validation_results = validator.generate_comprehensive_validation_report(test_df)
    
    # Print report
    validator.print_validation_report(validation_results)
    
    print(f"\nValidation framework test completed successfully!")