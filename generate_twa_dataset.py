"""
Enhanced TWA Dataset Generation - Main Script
Complete research-grounded synthetic US wellness & aging dataset generation
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict
import os
import sys

from enhanced_longitudinal_generator import EnhancedLongitudinalTWADataGenerator
from validation_framework import DatasetValidator


def main():
    """
    Generate complete enhanced TWA dataset with research validation
    """
    
    print("="*80)
    print("ENHANCED TWA DATASET GENERATION")
    print("Research-Grounded Synthetic US Wellness & Aging Dataset")
    print("="*80)
    
    # Configuration
    config = {
        'n_subjects': 1000,  # Start with smaller sample for testing
        'months': 12,
        'include_life_events': True,
        'random_seed': 42,
        'output_dir': 'generated_datasets',
        'validate_dataset': True
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize generator
    print(f"\nInitializing enhanced dataset generator...")
    generator = EnhancedLongitudinalTWADataGenerator(random_seed=config['random_seed'])
    
    # Generate complete dataset
    print(f"\nGenerating complete dataset...")
    print(f"This may take 30-45 minutes for {config['n_subjects']:,} subjects...")
    
    try:
        dataset = generator.generate_complete_dataset(
            n_subjects=config['n_subjects'],
            months=config['months'],
            include_life_events=config['include_life_events']
        )
        
        print(f"\nDataset generation completed successfully!")
        print(f"Generated {len(dataset)} total observations")
        print(f"  {dataset['subject_id'].nunique()} unique subjects")
        print(f"  {dataset['month'].nunique()} months of data")
        print(f"  {len(dataset.columns)} variables")
        
    except Exception as e:
        print(f"\nError during dataset generation: {e}")
        return False
    
    # Generate validation report if requested
    if config['validate_dataset']:
        print(f"\nRunning comprehensive validation...")
        validator = DatasetValidator()
        
        try:
            validation_results = validator.generate_comprehensive_validation_report(dataset)
            
            # Print validation report
            validator.print_validation_report(validation_results)
            
            # Save validation results
            validation_file = os.path.join(config['output_dir'], 'validation_report.json')
            with open(validation_file, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.generic):
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    return obj
                
                json.dump(convert_numpy(validation_results), f, indent=2)
            
            print(f"\nValidation report saved to: {validation_file}")
            
        except Exception as e:
            print(f"\nWarning: Validation failed: {e}")
            print("Proceeding with dataset export...")
    
    # Export dataset
    print(f"\nExporting dataset to files...")
    
    # Export to CSV (main format)
    csv_file = os.path.join(config['output_dir'], 'enhanced_twa_dataset.csv')
    dataset.to_csv(csv_file, index=False)
    print(f"Full dataset exported to: {csv_file}")
    
    # Export to Excel with multiple sheets
    excel_file = os.path.join(config['output_dir'], 'enhanced_twa_dataset.xlsx')
    
    try:
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            # Full dataset (sample if too large)
            if len(dataset) > 100000:
                # Sample 10,000 random observations for Excel
                sample_dataset = dataset.sample(n=10000, random_state=config['random_seed'])
                sample_dataset.to_excel(writer, sheet_name='Dataset_Sample', index=False)
                print(f"Dataset sample (10,000 obs) exported to: {excel_file}")
            else:
                dataset.to_excel(writer, sheet_name='Full_Dataset', index=False)
                print(f"Full dataset exported to: {excel_file}")
            
            # Demographics summary
            demographics = dataset[dataset['month'] == 0].copy()  # Baseline demographics
            demo_summary = demographics.describe(include='all')
            demo_summary.to_excel(writer, sheet_name='Demographics_Summary')
            
            # Behavior summary by month
            behavior_cols = [
                'month', 'motion_days_week', 'diet_mediterranean_score', 
                'meditation_minutes_week', 'social_connections_count',
                'purpose_meaning_score', 'smoking_status', 'alcohol_drinks_week'
            ]
            behavior_summary = dataset[behavior_cols].groupby('month').agg({
                'motion_days_week': ['mean', 'std'],
                'diet_mediterranean_score': ['mean', 'std'],
                'meditation_minutes_week': ['mean', 'std'],
                'social_connections_count': ['mean', 'std'],
                'purpose_meaning_score': ['mean', 'std'],
                'alcohol_drinks_week': ['mean', 'std']
            }).round(2)
            behavior_summary.to_excel(writer, sheet_name='Behavior_Trends')
            
            # Outcome summary by month
            outcome_cols = [
                'month', 'biological_age_acceleration', 'mortality_risk_score',
                'healthy_aging_profile', 'blue_zone_similarity_score',
                'life_satisfaction_score'
            ]
            outcome_summary = dataset[outcome_cols].groupby('month').agg({
                'biological_age_acceleration': ['mean', 'std'],
                'mortality_risk_score': ['mean', 'std'],
                'healthy_aging_profile': ['mean', 'std'],
                'blue_zone_similarity_score': ['mean', 'std'],
                'life_satisfaction_score': ['mean', 'std']
            }).round(3)
            outcome_summary.to_excel(writer, sheet_name='Outcome_Trends')
            
        print(f"Excel workbook with multiple sheets exported to: {excel_file}")
        
    except Exception as e:
        print(f"Warning: Excel export failed: {e}")
        print("CSV export completed successfully.")
    
    # Generate dataset documentation
    print(f"\nGenerating dataset documentation...")
    doc_file = os.path.join(config['output_dir'], 'dataset_documentation.md')
    
    documentation = generate_dataset_documentation(dataset, config, validation_results if config['validate_dataset'] else None)
    
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(documentation)
    
    print(f"Dataset documentation saved to: {doc_file}")
    
    # Generate data dictionary
    data_dict_file = os.path.join(config['output_dir'], 'data_dictionary.csv')
    data_dictionary = generate_data_dictionary(dataset)
    data_dictionary.to_csv(data_dict_file, index=False)
    print(f"Data dictionary saved to: {data_dict_file}")
    
    # Final summary
    print(f"\n" + "="*80)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Output Directory: {config['output_dir']}")
    print(f"Files Generated:")
    print(f"  1. {csv_file} - Full dataset (CSV)")
    print(f"  2. {excel_file} - Dataset with summaries (Excel)")
    print(f"  3. {doc_file} - Complete documentation")
    print(f"  4. {data_dict_file} - Data dictionary")
    if config['validate_dataset']:
        print(f"  5. {validation_file} - Validation report")
    
    print(f"\nDataset Summary:")
    print(f"  Total Subjects: {dataset['subject_id'].nunique():,}")
    print(f"  Total Observations: {len(dataset):,}")
    print(f"  Time Period: {dataset['month'].nunique()} months")
    print(f"  Variables: {len(dataset.columns)}")
    print(f"  File Size: ~{os.path.getsize(csv_file) / (1024*1024):.1f} MB")
    
    return True


def generate_dataset_documentation(dataset: pd.DataFrame, config: Dict, validation_results: Dict = None) -> str:
    """Generate comprehensive dataset documentation"""
    
    doc = f"""# Enhanced TWA Dataset Documentation

## Dataset Overview

**Dataset Name**: Enhanced Synthetic US Wellness & Aging Dataset  
**Generation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Version**: 1.0  
**Total Subjects**: {dataset['subject_id'].nunique():,}  
**Total Observations**: {len(dataset):,}  
**Time Period**: {dataset['month'].nunique()} months  
**Variables**: {len(dataset.columns)}

## Purpose and Scope

This dataset was created to support research into Tiny Wellness Activities (TWAs) and their impact on aging and wellness outcomes. It integrates:

- **5 Do More Activities**: Motion, Sleep, Hydration, Diet Quality, Meditation
- **5 Do Less Activities**: Smoking, Excessive Alcohol, Added Sugars, Excess Sodium, Ultra-processed Foods  
- **Connection & Purpose**: Social Networks, Nature Connection, Cultural Engagement, Life Purpose

The dataset maintains research-validated correlations and effect sizes from longevity studies, Blue Zone research, and intervention trials.

## Scientific Foundation

### Research Sources
- Expert consensus biomarkers of aging
- Blue Zone lifestyle research  
- Mediterranean diet intervention studies
- Exercise and biological age research
- Social connection and mortality studies
- Purpose/meaning and longevity research

### Key Effect Sizes Implemented
- **Exercise**: -1.2 years biological age for regular exercisers
- **Mediterranean Diet**: -2.3 years biological age for high adherence
- **Purpose**: 40% mortality risk reduction (HR=0.60)
- **Social Connection**: 50% mortality risk reduction for well-connected individuals
- **Smoking**: +5.3 years biological age acceleration

## Dataset Structure

### Longitudinal Design
- **Baseline**: Month 0 demographics and initial measures
- **Follow-up**: Monthly assessments for {config['months']} months
- **Seasonal Effects**: Behavioral variations by season (Winter, Spring, Summer, Fall)
- **Life Events**: Realistic life event impacts on behaviors and outcomes

### Variable Categories

#### Demographics (Time-Invariant)
- Age, Gender, Ethnicity, Education, Income
- Geographic region, Urban/Rural status
- Occupation, Marital status, Household size

#### TWA Behaviors (Monthly)
- **Do More**: Motion days/week, Sleep hours/quality, Hydration, Diet quality, Meditation minutes
- **Do Less**: Smoking status, Alcohol consumption, Sugar/Sodium intake, Processed foods
- **Connection & Purpose**: Social connections, Nature time, Cultural activities, Life purpose

#### Wellness & Aging Outcomes (Monthly)
- **Biological Age**: Acceleration/deceleration relative to chronological age
- **Biomarkers**: CRP, IL-6, IGF-1, GDF-15, Cortisol (expert consensus validated)
- **Functional**: Grip strength, Gait speed, Balance, Frailty index
- **Cognitive**: Composite score, Processing speed
- **Psychosocial**: Life satisfaction, Stress, Depression risk, Social support

#### Composite Scores
- **Healthy Aging Profile**: Research-weighted composite of behaviors and outcomes
- **Blue Zone Similarity**: Lifestyle similarity to longevity populations

## Data Quality and Validation

### Demographic Accuracy
- US Census 2023 age, education, income distributions
- Regional and urban/rural patterns
- Occupation and household characteristics

### Behavioral Realism  
- CDC health behavior prevalence rates
- Seasonal variation patterns
- Behavioral consistency and change over time
- Life event impacts

### Outcome Validity
- Research-validated effect sizes
- Biomarker reference ranges
- Functional measure age-related changes
"""

    if validation_results:
        overall_quality = validation_results['summary']['overall_quality']
        excellent_pct = validation_results['summary']['status_percentages']['excellent']
        good_pct = validation_results['summary']['status_percentages']['good']
        
        doc += f"""
### Validation Results
- **Overall Quality**: {overall_quality.upper()}
- **Excellent Validations**: {excellent_pct:.1f}%
- **Good Validations**: {good_pct:.1f}%
- **Total Validations Performed**: {validation_results['summary']['total_validations']}
"""

    doc += f"""
## Usage Guidelines

### Research Applications
1. **TWA Impact Modeling**: Quantify lifestyle intervention effects on aging
2. **Personalized Health Planning**: Individual risk/benefit analysis
3. **Population Health Studies**: Demographic patterns in healthy aging
4. **Intervention Design**: Optimize multi-component wellness programs
5. **Biomarker Research**: Explore aging indicator relationships

### Statistical Considerations
- **Longitudinal Structure**: Account for within-subject correlation
- **Missing Data**: Minimal missing data by design
- **Outliers**: Realistic extreme values included
- **Sample Size**: Adequate power for most analyses

### Limitations
- **Synthetic Data**: Not suitable for FDA submissions or clinical guidelines
- **US Population**: Demographic patterns specific to United States
- **Time Period**: Cross-sectional snapshot of 2024 health patterns
- **Causality**: Correlational relationships, not experimental evidence

## File Structure

### Primary Dataset (`enhanced_twa_dataset.csv`)
- Complete longitudinal dataset
- {len(dataset):,} observations × {len(dataset.columns)} variables
- CSV format for universal compatibility

### Excel Workbook (`enhanced_twa_dataset.xlsx`)
- Multiple sheets with summaries and trends
- Demographic, behavioral, and outcome summaries
- Monthly trend analyses

### Documentation
- `dataset_documentation.md` - This comprehensive guide
- `data_dictionary.csv` - Variable definitions and ranges
- `validation_report.json` - Technical validation results

## Citation

When using this dataset, please cite:

*Enhanced Synthetic US Wellness & Aging Dataset: Research-Grounded TWA Impact Framework. Generated {datetime.now().strftime('%Y-%m-%d')}. Integrates expert consensus biomarkers of aging with validated lifestyle intervention effect sizes from longevity research.*

## Contact and Support

For questions about dataset generation methods, validation procedures, or research applications, please refer to the technical documentation and validation reports included with this dataset.

---

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Version**: 1.0  
**Format**: Longitudinal research-grade synthetic dataset
"""

    return doc


def generate_data_dictionary(dataset: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive data dictionary"""
    
    # Define variable metadata
    variable_info = {
        # Identifiers
        'subject_id': {'category': 'Identifier', 'description': 'Unique subject identifier', 'type': 'String', 'range': 'SYNTH_000000 to SYNTH_999999'},
        'month': {'category': 'Time', 'description': 'Month since baseline (0-11)', 'type': 'Integer', 'range': '0-11'},
        'season': {'category': 'Time', 'description': 'Season of observation', 'type': 'String', 'range': 'Winter, Spring, Summer, Fall'},
        'observation_date': {'category': 'Time', 'description': 'Date of observation', 'type': 'Date', 'range': '2024-01-15 to 2024-12-15'},
        
        # Demographics
        'age_group': {'category': 'Demographics', 'description': 'Age group category', 'type': 'String', 'range': '18-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75+'},
        'age_numeric': {'category': 'Demographics', 'description': 'Age in years', 'type': 'Integer', 'range': '18-90'},
        'gender': {'category': 'Demographics', 'description': 'Gender identity', 'type': 'String', 'range': 'Male, Female, Non-binary'},
        'ethnicity': {'category': 'Demographics', 'description': 'Ethnic background', 'type': 'String', 'range': 'White, Hispanic, Black, Asian, Other'},
        'education': {'category': 'Demographics', 'description': 'Highest education level', 'type': 'String', 'range': 'Less than HS, High School, Some College, Bachelor+'},
        'income_bracket': {'category': 'Demographics', 'description': 'Annual household income bracket', 'type': 'String', 'range': '<$35k, $35-50k, $50-75k, $75-100k, $100-150k, >$150k'},
        'income_numeric': {'category': 'Demographics', 'description': 'Annual income (midpoint estimate)', 'type': 'Integer', 'range': '25000-200000'},
        
        # TWA Behaviors - Do More
        'motion_days_week': {'category': 'TWA - Do More', 'description': 'Days per week of vigorous physical activity', 'type': 'Integer', 'range': '0-7'},
        'sleep_hours': {'category': 'TWA - Do More', 'description': 'Average sleep hours per night', 'type': 'Float', 'range': '4.0-12.0'},
        'sleep_quality_score': {'category': 'TWA - Do More', 'description': 'Sleep quality rating (1-10 scale)', 'type': 'Float', 'range': '1.0-10.0'},
        'hydration_cups_day': {'category': 'TWA - Do More', 'description': 'Cups of water per day', 'type': 'Float', 'range': '4.0-16.0'},
        'diet_mediterranean_score': {'category': 'TWA - Do More', 'description': 'Mediterranean diet adherence (0-10 scale)', 'type': 'Float', 'range': '0.0-10.0'},
        'meditation_minutes_week': {'category': 'TWA - Do More', 'description': 'Minutes of meditation/mindfulness per week', 'type': 'Integer', 'range': '0-600'},
        
        # TWA Behaviors - Do Less
        'smoking_status': {'category': 'TWA - Do Less', 'description': 'Current smoking status', 'type': 'String', 'range': 'Never, Former, Current'},
        'alcohol_drinks_week': {'category': 'TWA - Do Less', 'description': 'Alcoholic drinks per week', 'type': 'Float', 'range': '0.0-35.0'},
        'added_sugar_grams_day': {'category': 'TWA - Do Less', 'description': 'Added sugar grams per day', 'type': 'Float', 'range': '10.0-200.0'},
        'sodium_grams_day': {'category': 'TWA - Do Less', 'description': 'Sodium grams per day', 'type': 'Float', 'range': '1.5-8.0'},
        'processed_food_servings_week': {'category': 'TWA - Do Less', 'description': 'Ultra-processed food servings per week', 'type': 'Integer', 'range': '0-40'},
        
        # Connection & Purpose
        'social_connections_count': {'category': 'Connection & Purpose', 'description': 'Number of close social relationships', 'type': 'Integer', 'range': '0-12'},
        'nature_minutes_week': {'category': 'Connection & Purpose', 'description': 'Minutes spent in nature per week', 'type': 'Integer', 'range': '0-600'},
        'cultural_hours_week': {'category': 'Connection & Purpose', 'description': 'Hours of cultural activities per week', 'type': 'Float', 'range': '0.0-30.0'},
        'purpose_meaning_score': {'category': 'Connection & Purpose', 'description': 'Life purpose and meaning (1-10 scale)', 'type': 'Float', 'range': '1.0-10.0'},
        
        # Aging & Wellness Outcomes
        'biological_age_years': {'category': 'Aging Outcomes', 'description': 'Biological age in years', 'type': 'Float', 'range': '18.0-120.0'},
        'biological_age_acceleration': {'category': 'Aging Outcomes', 'description': 'Biological age minus chronological age', 'type': 'Float', 'range': '-10.0-15.0'},
        'mortality_risk_score': {'category': 'Aging Outcomes', 'description': 'Annual mortality risk probability', 'type': 'Float', 'range': '0.001-0.200'},
        'estimated_lifespan_years': {'category': 'Aging Outcomes', 'description': 'Estimated total lifespan', 'type': 'Float', 'range': '60.0-120.0'},
        
        # Biomarkers
        'crp_mg_l': {'category': 'Biomarkers', 'description': 'C-reactive protein (mg/L)', 'type': 'Float', 'range': '0.1-10.0'},
        'il6_pg_ml': {'category': 'Biomarkers', 'description': 'Interleukin-6 (pg/mL)', 'type': 'Float', 'range': '0.5-15.0'},
        'igf1_ng_ml': {'category': 'Biomarkers', 'description': 'IGF-1 (ng/mL)', 'type': 'Float', 'range': '50.0-400.0'},
        'cortisol_ug_dl': {'category': 'Biomarkers', 'description': 'Cortisol (μg/dL)', 'type': 'Float', 'range': '3.0-25.0'},
        
        # Functional Measures
        'grip_strength_kg': {'category': 'Functional', 'description': 'Grip strength (kg)', 'type': 'Float', 'range': '10.0-70.0'},
        'gait_speed_ms': {'category': 'Functional', 'description': 'Gait speed (m/s)', 'type': 'Float', 'range': '0.4-2.0'},
        'frailty_index': {'category': 'Functional', 'description': 'Frailty index (0-1 scale)', 'type': 'Float', 'range': '0.0-0.7'},
        
        # Psychosocial
        'life_satisfaction_score': {'category': 'Psychosocial', 'description': 'Life satisfaction (1-10 scale)', 'type': 'Float', 'range': '1.0-10.0'},
        'stress_level_score': {'category': 'Psychosocial', 'description': 'Perceived stress level (1-10 scale)', 'type': 'Float', 'range': '1.0-10.0'},
        
        # Composite Scores
        'healthy_aging_profile': {'category': 'Composite', 'description': 'Healthy aging composite score (0-100)', 'type': 'Float', 'range': '0.0-100.0'},
        'blue_zone_similarity_score': {'category': 'Composite', 'description': 'Blue Zone lifestyle similarity (0-100)', 'type': 'Float', 'range': '0.0-100.0'}
    }
    
    # Create data dictionary DataFrame
    data_dict_list = []
    
    for col in dataset.columns:
        if col in variable_info:
            info = variable_info[col]
            row = {
                'Variable': col,
                'Category': info['category'],
                'Description': info['description'],
                'Data_Type': info['type'],
                'Range/Values': info['range']
            }
        else:
            # Generic info for variables not explicitly defined
            row = {
                'Variable': col,
                'Category': 'Other',
                'Description': 'Variable description not specified',
                'Data_Type': str(dataset[col].dtype),
                'Range/Values': f"{dataset[col].min()} to {dataset[col].max()}" if dataset[col].dtype in ['int64', 'float64'] else "Various"
            }
        
        data_dict_list.append(row)
    
    return pd.DataFrame(data_dict_list)


if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 Dataset generation completed successfully!")
        print(f"Ready for TWA research and model development.")
    else:
        print(f"\n❌ Dataset generation failed.")
        sys.exit(1)