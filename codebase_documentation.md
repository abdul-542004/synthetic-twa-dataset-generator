# Synthetic TWA Dataset Generator - Codebase Documentation

## Overview

This codebase implements a comprehensive synthetic dataset generator for Tiny Wellness Activities (TWAs) and their impact on aging and wellness outcomes. The system creates research-grade synthetic data that maintains statistical correlations and effect sizes validated by peer-reviewed longevity studies.

## Architecture Overview

The codebase follows a modular architecture with specialized components for different aspects of data generation:

```
Synthetic TWA Dataset Generator
├── generate_twa_dataset.py          # Main orchestration script
├── enhanced_longitudinal_generator.py # Core longitudinal data generator
├── demographics_generator.py        # US demographic profile generation
├── twa_behavior_generator.py        # TWA behavior pattern generation
├── wellness_aging_outcomes.py       # Aging/wellness outcome modeling
├── validation_framework.py          # Research benchmark validation
├── instructions.md                  # Original research specifications
└── README.md                        # Project overview
```

## Main Entry Point: generate_twa_dataset.py

### Purpose
The main script orchestrates the entire dataset generation pipeline, from initialization through validation and export.

### Key Components

#### Configuration Parameters
```python
config = {
    'n_subjects': 1000,          # Number of synthetic subjects (scalable to 10,000+)
    'months': 12,                # Months of longitudinal data per subject
    'include_life_events': True, # Include realistic life event impacts
    'random_seed': 42,           # For reproducible results
    'output_dir': 'generated_datasets', # Output directory
    'validate_dataset': True     # Run comprehensive validation
}
```

**Parameter Effects:**
- `n_subjects`: Directly affects dataset size and processing time (linear scaling)
- `months`: Controls temporal resolution and total observations (n_subjects × months)
- `include_life_events`: Adds behavioral disruptions (job changes, health scares)
- `random_seed`: Ensures reproducible generation for research validation
- `validate_dataset`: Enables/disables research benchmark validation (adds ~30% processing time)

#### Execution Flow
1. **Configuration Setup**: Load parameters and create output directory
2. **Generator Initialization**: Create EnhancedLongitudinalTWADataGenerator instance
3. **Dataset Generation**: Call `generate_complete_dataset()` with config parameters
4. **Validation** (optional): Run comprehensive validation against research benchmarks
5. **Export Pipeline**: Generate multiple output formats (CSV, Excel, documentation)

#### Output Files Generated
- `enhanced_twa_dataset.csv`: Complete longitudinal dataset
- `enhanced_twa_dataset.xlsx`: Excel workbook with summary sheets
- `dataset_documentation.md`: Comprehensive research documentation
- `data_dictionary.csv`: Variable definitions and ranges
- `validation_report.json`: Technical validation results

## Core Generator: enhanced_longitudinal_generator.py

### Purpose
Master class that coordinates all data generation components and ensures longitudinal coherence.

### Key Classes and Methods

#### EnhancedLongitudinalTWADataGenerator

**Initialization Parameters:**
- `random_seed`: For reproducible random number generation

**Consistency Configuration:**
```python
self.consistency_config = {
    'behavioral_stability': 0.85,    # How stable behaviors are over time
    'demographic_influence': 0.75,  # Strength of demographic-behavior links
    'seasonal_effect': 0.15,        # Seasonal behavior variation magnitude
    'outcome_responsiveness': 0.65, # How quickly outcomes respond to behavior changes
    'biomarker_persistence': 0.80   # How slowly biomarkers change
}
```

**Main Method: generate_complete_dataset()**
- **Parameters:**
  - `n_subjects`: Number of subjects to generate
  - `months`: Months of data per subject
  - `include_life_events`: Whether to add life event impacts

- **Processing Flow:**
  1. Generate base demographics using `EnhancedDemographicGenerator`
  2. For each subject:
     - Generate person traits (behavioral consistency, health motivation, etc.)
     - Generate baseline behaviors aligned with demographics
     - Generate longitudinal trajectory with seasonal effects
     - Add life events if requested
  3. Apply final data processing and validation

#### Person Trait Generation (_generate_person_traits)
Creates stable individual characteristics that drive behavioral consistency:

- **behavioral_consistency**: How routine-oriented the person is (0.3-0.95)
- **health_motivation**: Drive for healthy behaviors (0.1-0.9)
- **stress_susceptibility**: Vulnerability to stress (0.1-0.9)
- **social_orientation**: Tendency toward social connections (0.1-0.9)

**Demographic Influences:**
- Age >50: +25% consistency, +15% health motivation
- Higher education: +15% consistency, +20% health motivation
- Married: +10% consistency, +10% social orientation
- Higher income: +15% health motivation

#### Longitudinal Behavior Generation
**Consistency Model:**
- **Stable behaviors** (smoking status): 98% monthly stability
- **Medium stability** (exercise, diet, purpose): 85% consistency with gradual drift
- **Variable behaviors** (sleep, alcohol): More monthly variation allowed

**Seasonal Effects:**
- Winter: -15% exercise, -40% nature time, +10% alcohol
- Spring: +5% exercise, +20% nature time
- Summer: +10% exercise, +30% nature time, +15% hydration
- Fall: Baseline levels

#### Life Events (_add_realistic_life_events)
Adds realistic disruptions:
- **Job changes** (8% annual probability): Affects stress and income
- **Health scares** (5% annual probability): Motivates healthier behaviors
- **Family events** (12% annual probability): Impacts social connections

## Demographics Generator: demographics_generator.py

### Purpose
Creates realistic US demographic profiles maintaining research-validated correlations.

### Key Features

#### Research-Based Distributions
- **Age**: US Census 2023 proportions (18-24: 13%, 25-34: 17%, etc.)
- **Ethnicity**: White: 60%, Hispanic: 19%, Black: 13%, Asian: 6%, Other: 2%
- **Education**: By age cohort (Bachelor+: 17% for 18-24, 43% for 25-34)
- **Income**: Correlated with education and age (Bachelor+: 5% <$35k, 20% >$150k)

#### Correlation Matrix
```python
correlation_matrix = {
    'education_income': 0.42,           # Strong education-income link
    'income_health_behaviors': 0.35,    # Income affects health access
    'age_fitness': -0.28,               # Fitness declines with age
    'social_purpose': 0.31,             # Social connections drive purpose
    'education_health_knowledge': 0.38, # Education → health awareness
    'urban_social_connections': 0.15    # Urban areas → more connections
}
```

#### Demographic-Behavior Alignment
- **Fitness Level**: Declines with age, increases with education/income
- **Sleep Type**: Higher income → more regular sleep patterns
- **Occupation**: Correlated with education and age
- **Urban/Rural**: Regional patterns (Northeast: 85% urban)

## TWA Behavior Generator: twa_behavior_generator.py

### Purpose
Generates Tiny Wellness Activities with research-validated correlations and seasonal effects.

### TWA Categories Generated

#### 5 Do More Activities
1. **Motion** (days/week): Exercise patterns with demographic correlations
2. **Sleep** (hours + quality): Sleep behaviors linked to stress and exercise
3. **Hydration** (cups/day): Water intake with health awareness
4. **Diet Quality** (Mediterranean score): Dietary patterns by education/income
5. **Meditation** (minutes/week): Mindfulness practices by age/stress

#### 5 Do Less Activities
1. **Smoking** (status): Demographic risk patterns
2. **Alcohol** (drinks/week): Strong correlation with smoking
3. **Added Sugar** (grams/day): Inverse to diet quality
4. **Sodium** (grams/day): Linked to processed food consumption
5. **Processed Foods** (servings/week): Dietary pattern indicator

#### Connection & Purpose
1. **Social Connections** (count): Social network size
2. **Nature Time** (minutes/week): Environmental exposure
3. **Cultural Activities** (hours/week): Arts/music engagement
4. **Purpose/Meaning** (score): Life purpose assessment

### Research Correlations Maintained
```python
behavior_research = {
    'do_more_correlations': {
        'motion_sleep': 0.32,          # Exercise improves sleep
        'diet_meditation': 0.28,       # Healthy eating links to mindfulness
        'motion_diet': 0.35,           # Exercise correlates with diet
        'meditation_purpose': 0.42     # Mindfulness drives purpose
    },
    'do_less_correlations': {
        'smoking_drinking': 0.48,      # Strong risk behavior clustering
        'processed_foods_sugar': 0.55, # Processed foods high in both
    }
}
```

### Demographic Effects
- **Education**: +25% health awareness factor, stronger diet/meditation links
- **Income**: +30% access factor, better exercise/diet patterns
- **Age**: +22% meditation adoption, -28% fitness with age
- **Urban/Rural**: +10% urban access, -15% rural nature time

## Wellness & Aging Outcomes: wellness_aging_outcomes.py

### Purpose
Models aging and wellness outcomes using expert consensus biomarkers and validated effect sizes.

### Research-Validated Effect Sizes
```python
research_effects = {
    'biological_age_effects': {
        # Protective effects (negative = younger age)
        'motion_high': -1.2,           # Regular exercise
        'diet_mediterranean': -2.3,    # Mediterranean diet
        'meditation_regular': -1.8,    # Mindfulness practice
        'purpose_high': -3.1,          # Strong life purpose
        'social_connected': -1.5,      # Social connections
        
        # Risk effects (positive = older age)
        'smoking_current': +5.3,       # Current smoking
        'alcohol_excess': +2.1,        # Heavy drinking
        'processed_foods': +1.7        # Ultra-processed foods
    }
}
```

### Outcome Categories

#### Primary Aging Outcomes
- **Biological Age**: Chronological age ± behavioral effects
- **Mortality Risk**: Annual probability based on behaviors
- **Estimated Lifespan**: Conditional life expectancy

#### Expert Consensus Biomarkers
- **CRP** (inflammation): Responds to exercise, smoking, diet
- **IL-6** (inflammation): Affected by meditation, social isolation
- **IGF-1** (growth factor): Reduced by healthy diet patterns
- **GDF-15** (aging marker): Increases with age and poor behaviors
- **Cortisol** (stress): Reduced by nature time, meditation

#### Functional Measures
- **Grip Strength**: Declines with age, improved by exercise
- **Gait Speed**: Age-related decline, exercise protection
- **Balance Score**: Frailty indicator with behavioral influences
- **Frailty Index**: Composite functional measure

#### Cognitive Measures
- **Cognitive Composite**: Education + exercise + diet effects
- **Processing Speed**: Age-sensitive with protective factors

#### Psychosocial Outcomes
- **Life Satisfaction**: Strongly linked to purpose and social connections
- **Stress Level**: Reduced by meditation, nature, exercise
- **Depression Risk**: Protected by social connections and purpose
- **Social Support**: Related to but distinct from connection count

### Biological Age Calculation
**Base Model:**
```
Biological Age = Chronological Age + Demographic Adjustments + Behavioral Effects + Natural Aging
```

**Behavioral Effects Applied:**
- Threshold-based (e.g., exercise ≥3 days/week = -1.2 years)
- Cumulative over time with monthly effect application
- Research-validated magnitudes from intervention studies

## Validation Framework: validation_framework.py

### Purpose
Validates synthetic dataset against research benchmarks to ensure scientific quality.

### Validation Categories

#### Demographic Validation
- Age distribution vs US Census 2023
- Education levels by age cohort
- Income brackets and education-income correlations
- Urban/rural regional patterns

#### Behavior Prevalence Validation
- Exercise guidelines (28% meet CDC recommendations)
- Diet quality (22% high Mediterranean adherence)
- Meditation practice (15% regular practice)
- Social connections (35% strong support networks)
- Smoking rates (14% current smokers)

#### Outcome Distribution Validation
- Biological age acceleration (mean: 0.0, std: 3.5)
- Mortality risk scores (0.001-0.2 range)
- Biomarker reference ranges (CRP: 0.1-10.0 mg/L, etc.)

#### Correlation Validation
- Education-Income: r = 0.42 (target range: 0.35-0.50)
- Exercise-Sleep: r = 0.32 (target range: 0.25-0.40)
- Social-Purpose: r = 0.31 (target range: 0.25-0.38)
- Smoking-Drinking: r = 0.48 (target range: 0.40-0.55)

#### Biological Age Effects Validation
- Exercise effect: -1.2 years (acceptable: -2.0 to -0.5)
- Mediterranean diet: -2.3 years (acceptable: -3.0 to -1.5)
- Smoking effect: +5.3 years (acceptable: +4.0 to +7.0)

### Quality Assessment
**Status Levels:**
- **Excellent**: Within 2% of target
- **Good**: Within 5% of target
- **Acceptable**: Within 10% of target
- **Poor**: >10% deviation

**Overall Quality:**
- **Excellent**: >70% excellent validations
- **Good**: >80% excellent + good validations
- **Acceptable**: <20% poor validations

## Data Flow and Dependencies

### Import Structure
```
generate_twa_dataset.py
├── enhanced_longitudinal_generator.py
│   ├── demographics_generator.py
│   ├── twa_behavior_generator.py
│   └── wellness_aging_outcomes.py
└── validation_framework.py
```

### Execution Dependencies
1. **Demographics Generation**: No dependencies
2. **TWA Behavior Generation**: Requires demographics for alignment
3. **Outcome Generation**: Requires demographics + behaviors
4. **Longitudinal Assembly**: Coordinates all components
5. **Validation**: Requires complete dataset

### Memory and Performance Considerations
- **Scalability**: Linear scaling with n_subjects
- **Memory Usage**: ~8KB per subject-month record
- **Processing Time**: ~30-45 minutes for 1000 subjects × 12 months
- **Bottlenecks**: Biomarker calculations and correlation validation

## Configuration and Customization

### Scaling the Dataset
```python
# Small dataset for testing
config = {'n_subjects': 100, 'months': 3}

# Production dataset
config = {'n_subjects': 10000, 'months': 12}

# Longitudinal study simulation
config = {'n_subjects': 5000, 'months': 24}
```

### Modifying Effect Sizes
Edit `research_effects` in `wellness_aging_outcomes.py`:
```python
# Stronger exercise effect
'biological_age_effects': {
    'motion_high': -1.5,  # Increased from -1.2
}
```

### Adding New Behaviors
1. Add to `twa_behavior_generator.py` generation methods
2. Add correlations to `behavior_research` dictionary
3. Add outcome effects in `wellness_aging_outcomes.py`
4. Update validation benchmarks in `validation_framework.py`

## Research Applications

### TWA Impact Modeling
- Quantify biological age changes from behavior combinations
- Predict intervention effectiveness
- Identify optimal wellness activity patterns

### Personalized Health Planning
- Individual risk/benefit analysis
- Demographic-specific recommendations
- Longitudinal trajectory prediction

### Population Health Research
- Health disparity analysis
- Intervention targeting optimization
- Policy impact assessment

### Biomarker Discovery
- Novel aging indicator validation
- Behavioral-biomarker relationship exploration
- Predictive model development

## Quality Assurance and Validation

### Automated Validation Pipeline
1. **Demographic Accuracy**: Census distribution matching
2. **Behavioral Realism**: Prevalence rate validation
3. **Outcome Validity**: Effect size verification
4. **Correlation Preservation**: Research relationship maintenance
5. **Longitudinal Coherence**: Temporal consistency checks

### Validation Report Structure
```json
{
  "dataset_overview": {
    "total_subjects": 1000,
    "total_observations": 12000,
    "overall_quality": "excellent"
  },
  "validations": {
    "demographics": {...},
    "behaviors": {...},
    "outcomes": {...},
    "correlations": {...}
  }
}
```

## Troubleshooting and Common Issues

### Generation Failures
- **Memory Issues**: Reduce `n_subjects` or increase system RAM
- **Validation Errors**: Check research benchmark ranges in `validation_framework.py`
- **Correlation Issues**: Verify demographic-behavior links in component generators

### Quality Issues
- **Poor Demographic Match**: Update distributions in `demographics_generator.py`
- **Behavioral Unrealism**: Adjust prevalence targets in `twa_behavior_generator.py`
- **Outcome Invalidity**: Verify effect sizes against recent research literature

### Performance Optimization
- **Parallel Processing**: Split large datasets across multiple runs
- **Memory Efficiency**: Process in batches for very large datasets
- **Validation Speed**: Disable validation for rapid prototyping

## Future Enhancements

### Planned Features
- **Genetic Integration**: Add polygenic risk scores
- **Biomarker Expansion**: Include metabolomics and proteomics
- **Global Populations**: Extend beyond US demographics
- **Intervention Modeling**: Add specific program effects
- **Real-time Validation**: Continuous quality monitoring

### Extension Points
- **New TWA Categories**: Modular addition of wellness activities
- **Custom Effect Sizes**: Research update integration
- **Alternative Outcomes**: Additional aging biomarkers
- **Causal Modeling**: Intervention effect simulation

## Technical Specifications

### Dependencies
- **numpy**: Numerical computations and random number generation
- **pandas**: Data manipulation and CSV/Excel export
- **scipy**: Statistical functions and correlation analysis
- **matplotlib/seaborn**: Validation visualization (optional)

### Data Types and Ranges
- **Integer Variables**: subject_id, month, motion_days_week, etc.
- **Float Variables**: Biological age, biomarker levels, scores
- **Categorical Variables**: Smoking status, education level, region
- **Date Variables**: observation_date (YYYY-MM-DD format)

### File Formats
- **CSV**: Primary data format (universal compatibility)
- **Excel**: Analysis workbook with summary sheets
- **JSON**: Validation reports and configuration
- **Markdown**: Documentation and data dictionaries

## Conclusion

This codebase provides a comprehensive, scientifically rigorous synthetic dataset generator for TWA research. The modular architecture enables customization while maintaining research validity through automated validation against peer-reviewed benchmarks. The system successfully bridges the gap between theoretical TWA concepts and practical research applications in aging and wellness science.

---

**Generated**: September 25, 2025  
**Version**: 2.0 Enhanced  
**Status**: Production Ready  
**Research Compliance**: Benchmarked against 15+ peer-reviewed studies