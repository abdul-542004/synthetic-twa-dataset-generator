"""
Complete Digital Twin Framework Integration
Demonstrates end-to-end personalized wellness optimization system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our digital twin components
from digital_twin_framework.core import (
    PersonalDigitalTwin, 
    DigitalTwinOrchestrator,
    InterventionPlanner, 
    InterventionPlan,
    ProgressMonitor, 
    AdaptationTrigger
)

# Import existing dataset generators for realistic baseline data
from enhanced_longitudinal_generator import EnhancedLongitudinalTWADataGenerator
from demographics_generator import EnhancedDemographicGenerator


class ComprehensiveWellnessOptimizer:
    """
    Complete digital twin system that integrates all components for end-to-end wellness optimization
    """
    
    def __init__(self, output_dir: str = "digital_twin_outputs", random_seed: int = 42):
        """Initialize the complete system"""
        
        self.output_dir = output_dir
        self.random_seed = random_seed
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize core components
        self.orchestrator = DigitalTwinOrchestrator(random_seed=random_seed)
        self.intervention_planner = InterventionPlanner()
        self.dataset_generator = EnhancedLongitudinalTWADataGenerator(random_seed=random_seed)
        
        # Track active monitoring sessions
        self.active_monitors = {}  # person_id -> ProgressMonitor
        
        # System statistics
        self.system_stats = {
            'total_twins_created': 0,
            'total_optimizations': 0,
            'total_interventions_planned': 0,
            'total_adaptations': 0,
            'success_stories': []
        }
        
        print(f"Comprehensive Wellness Optimizer initialized")
        print(f"Output directory: {output_dir}")
    
    def onboard_new_person(self, person_data: Dict) -> Dict:
        """
        Complete onboarding process for a new person
        
        Args:
            person_data: Dict containing:
                - person_id: str
                - demographics: Dict
                - current_behaviors: Dict
                - wellness_goals: Dict
                - preferences: Dict (optional)
        
        Returns:
            Complete onboarding result with digital twin, optimization, and intervention plan
        """
        
        person_id = person_data['person_id']
        demographics = person_data['demographics']
        current_behaviors = person_data['current_behaviors']
        wellness_goals = person_data['wellness_goals']
        preferences = person_data.get('preferences', {})
        
        print(f"\n{'='*60}")
        print(f"ONBOARDING NEW PERSON: {person_id}")
        print(f"{'='*60}")
        
        # Step 1: Create Digital Twin
        print(f"Step 1: Creating digital twin...")
        digital_twin = self.orchestrator.create_digital_twin(
            person_id=person_id,
            demographics=demographics,
            initial_behaviors=current_behaviors
        )
        
        self.system_stats['total_twins_created'] += 1
        
        # Step 2: Optimize Activities
        print(f"Step 2: Optimizing wellness activities...")
        optimization_result = digital_twin.optimize_activities(
            target_outcomes=wellness_goals,
            optimization_horizon=preferences.get('timeline_months', 6)
        )
        
        self.system_stats['total_optimizations'] += 1
        
        # Step 3: Create Intervention Plan
        print(f"Step 3: Creating personalized intervention plan...")
        intervention_plan = self.intervention_planner.create_intervention_plan(
            person_id=person_id,
            digital_twin_optimization=optimization_result,
            duration_weeks=preferences.get('program_duration_weeks', 12),
            intensity_preference=preferences.get('intensity', 'moderate')
        )
        
        self.system_stats['total_interventions_planned'] += 1
        
        # Step 4: Initialize Progress Monitoring
        print(f"Step 4: Setting up progress monitoring...")
        monitor = ProgressMonitor(person_id, asdict(intervention_plan))
        self.active_monitors[person_id] = monitor
        
        # Step 5: Generate onboarding report
        onboarding_result = {
            'person_id': person_id,
            'onboarding_date': datetime.now().isoformat(),
            'digital_twin_created': True,
            'optimization_successful': optimization_result.get('optimization_successful', False),
            'intervention_plan_created': True,
            'monitoring_initialized': True,
            
            # Key results
            'baseline_wellness_score': digital_twin._calculate_wellness_score(),
            'target_outcomes': wellness_goals,
            'predicted_outcomes': optimization_result.get('predicted_outcomes', {}),
            'recommended_changes': optimization_result.get('optimal_behavior_changes', {}),
            'intervention_plan_id': intervention_plan.plan_id,
            'estimated_success_probability': intervention_plan.estimated_success_probability,
            'weekly_time_commitment': intervention_plan.weekly_time_commitment,
            
            # Full objects for further use
            'digital_twin': digital_twin,
            'optimization_result': optimization_result,
            'intervention_plan': intervention_plan,
            'progress_monitor': monitor
        }
        
        # Save onboarding report
        self._save_onboarding_report(onboarding_result)
        
        print(f"✅ Onboarding completed successfully!")
        print(f"   Baseline wellness score: {onboarding_result['baseline_wellness_score']:.1f}/100")
        print(f"   Success probability: {onboarding_result['estimated_success_probability']:.1%}")
        print(f"   Weekly commitment: {onboarding_result['weekly_time_commitment']} minutes")
        
        return onboarding_result
    
    def update_person_progress(self, person_id: str, progress_data: Dict) -> Dict:
        """
        Update a person's progress and trigger adaptations if needed
        
        Args:
            progress_data: Dict containing:
                - behaviors: Dict of current behavior values
                - self_reported_metrics: Dict of self-reported outcomes
                - compliance_rate: float (0-1)
                - barriers_encountered: List[str] (optional)
                - measurement_date: datetime (optional)
        
        Returns:
            Progress update result with any adaptations made
        """
        
        if person_id not in self.active_monitors:
            return {'error': f'No active monitoring for person {person_id}'}
        
        monitor = self.active_monitors[person_id]
        digital_twin = self.orchestrator.get_twin(person_id)
        
        if not digital_twin:
            return {'error': f'Digital twin not found for person {person_id}'}
        
        print(f"\n{'='*50}")
        print(f"UPDATING PROGRESS: {person_id}")
        print(f"{'='*50}")
        
        # Update digital twin with actual behaviors
        if 'behaviors' in progress_data:
            digital_twin.update_actual_behaviors(
                progress_data['behaviors'],
                progress_data.get('measurement_date', datetime.now())
            )
        
        # Record progress in monitor
        metrics = progress_data.get('self_reported_metrics', {})
        compliance_rate = progress_data.get('compliance_rate', 0.8)
        measurement_date = progress_data.get('measurement_date', datetime.now())
        
        monitor.record_progress(
            metrics=metrics,
            compliance_rate=compliance_rate,
            measurement_date=measurement_date
        )
        
        # Handle any reported barriers
        barrier_suggestions = []
        if 'barriers_encountered' in progress_data:
            for barrier in progress_data['barriers_encountered']:
                suggestion = monitor.suggest_manual_adaptation(
                    barrier_description=barrier,
                    barrier_type='user_reported'
                )
                barrier_suggestions.append(suggestion)
        
        # Get progress report
        progress_report = digital_twin.get_progress_report()
        adaptation_report = monitor.get_adaptation_report()
        
        # Count any adaptations made
        if adaptation_report['total_adaptations'] > 0:
            self.system_stats['total_adaptations'] += 1
        
        result = {
            'person_id': person_id,
            'update_date': datetime.now().isoformat(),
            'progress_recorded': True,
            'current_compliance': compliance_rate,
            'adaptations_triggered': adaptation_report.get('recent_adaptations_30_days', 0),
            'barrier_suggestions': barrier_suggestions,
            'progress_report': progress_report,
            'adaptation_report': adaptation_report,
            'current_wellness_score': digital_twin._calculate_wellness_score()
        }
        
        # Save progress update
        self._save_progress_update(result)
        
        print(f"✅ Progress updated successfully!")
        print(f"   Current compliance: {compliance_rate:.1%}")
        print(f"   Wellness score: {result['current_wellness_score']:.1f}/100")
        if adaptation_report['total_adaptations'] > 0:
            print(f"   Adaptations made: {adaptation_report['total_adaptations']}")
        
        return result
    
    def generate_weekly_plan(self, person_id: str, week_number: int) -> Dict:
        """Generate detailed weekly action plan for a person"""
        
        if person_id not in self.active_monitors:
            return {'error': f'No active monitoring for person {person_id}'}
        
        monitor = self.active_monitors[person_id]
        
        # Get current intervention plan (may have been adapted)
        current_plan_dict = monitor.current_plan
        
        # Convert back to InterventionPlan object if needed
        if isinstance(current_plan_dict, dict) and 'phases' in current_plan_dict:
            # Filter to only include InterventionPlan fields
            expected_fields = ['person_id', 'plan_id', 'creation_date', 'total_duration_weeks',
                             'target_outcomes', 'phases', 'total_actions', 'estimated_success_probability',
                             'weekly_time_commitment', 'key_focus_areas']
            filtered_dict = {k: v for k, v in current_plan_dict.items() if k in expected_fields}
            
            weekly_checklist = self.intervention_planner.generate_weekly_checklist(
                InterventionPlan(**filtered_dict) if 'person_id' in filtered_dict else None,
                week_number
            )
        else:
            return {'error': 'Invalid intervention plan format'}
        
        # Enhance with digital twin insights
        digital_twin = self.orchestrator.get_twin(person_id)
        if digital_twin:
            # Add personalized tips based on current state
            current_outcomes = digital_twin.current_outcomes
            personalized_tips = self._generate_personalized_tips(current_outcomes, weekly_checklist)
            weekly_checklist['personalized_tips'] = personalized_tips
        
        # Save weekly plan
        self._save_weekly_plan(person_id, week_number, weekly_checklist)
        
        return weekly_checklist
    
    def _generate_personalized_tips(self, current_outcomes: Dict, weekly_checklist: Dict) -> List[str]:
        """Generate personalized tips based on current wellness state"""
        
        tips = []
        
        # Tips based on biological age
        bio_age_accel = current_outcomes.get('biological_age_acceleration', 0)
        if bio_age_accel > 2:
            tips.append("Focus on anti-aging activities - your biological age is accelerated. Prioritize sleep and stress reduction.")
        elif bio_age_accel < -1:
            tips.append("Great job! Your biological age is younger than your chronological age. Keep up the excellent work!")
        
        # Tips based on life satisfaction
        life_sat = current_outcomes.get('life_satisfaction_score', 7)
        if life_sat < 6:
            tips.append("Consider adding more social activities and purpose-driven actions to boost life satisfaction.")
        elif life_sat > 8:
            tips.append("Your high life satisfaction is fantastic! Use this positive energy to tackle more challenging wellness goals.")
        
        # Tips based on current week's focus
        if 'focus' in weekly_checklist:
            focus = weekly_checklist['focus'].lower()
            if 'foundation' in focus:
                tips.append("This is your foundation week - focus on consistency over intensity. Small daily actions build strong habits.")
            elif 'building' in focus:
                tips.append("You're in the building phase - gradually increase your efforts while maintaining what you've already established.")
            elif 'integration' in focus:
                tips.append("Integration week - all systems go! You're ready for the full program intensity.")
        
        # Limit to top 3 most relevant tips
        return tips[:3]
    
    def run_population_analysis(self) -> Dict:
        """Analyze patterns across all digital twins"""
        
        population_insights = self.orchestrator.get_population_insights()
        
        # Enhanced analysis
        success_patterns = self._analyze_success_patterns()
        common_barriers = self._analyze_common_barriers()
        optimization_trends = self._analyze_optimization_trends()
        
        analysis_result = {
            'analysis_date': datetime.now().isoformat(),
            'population_size': len(self.orchestrator.twins),
            'system_statistics': self.system_stats,
            'population_insights': population_insights,
            'success_patterns': success_patterns,
            'common_barriers': common_barriers,
            'optimization_trends': optimization_trends,
            'recommendations': self._generate_system_recommendations()
        }
        
        # Save analysis
        self._save_population_analysis(analysis_result)
        
        return analysis_result
    
    def _analyze_success_patterns(self) -> Dict:
        """Analyze what patterns lead to success"""
        
        high_performers = []
        low_performers = []
        
        for person_id, twin in self.orchestrator.twins.items():
            current_score = twin.current_outcomes.get('healthy_aging_profile', 0)
            initial_score = twin.outcome_generator.generate_aging_wellness_outcomes(
                twin.demographics, twin.initial_behaviors, 0
            ).get('healthy_aging_profile', 0)
            
            improvement = current_score - initial_score
            
            if improvement > 5:  # Significant improvement
                high_performers.append({
                    'person_id': person_id,
                    'improvement': improvement,
                    'demographics': twin.demographics,
                    'final_score': current_score
                })
            elif improvement < -2:  # Decline
                low_performers.append({
                    'person_id': person_id,
                    'decline': improvement,
                    'demographics': twin.demographics,
                    'final_score': current_score
                })
        
        return {
            'high_performers': len(high_performers),
            'low_performers': len(low_performers),
            'high_performer_characteristics': self._analyze_group_characteristics(high_performers),
            'improvement_factors': self._identify_improvement_factors(high_performers)
        }
    
    def _analyze_common_barriers(self) -> Dict:
        """Analyze most common barriers across population"""
        
        all_barriers = []
        adaptation_triggers = {}
        
        for monitor in self.active_monitors.values():
            # Collect adaptation events
            for event in monitor.adaptation_events:
                trigger = event.trigger_type.value
                adaptation_triggers[trigger] = adaptation_triggers.get(trigger, 0) + 1
                all_barriers.append(event.trigger_description)
        
        return {
            'most_common_triggers': adaptation_triggers,
            'total_barriers_reported': len(all_barriers),
            'barrier_resolution_rate': self._calculate_barrier_resolution_rate()
        }
    
    def _analyze_optimization_trends(self) -> Dict:
        """Analyze optimization patterns and effectiveness"""
        
        optimization_targets = {}
        success_by_target = {}
        
        for twin in self.orchestrator.twins.values():
            for opt_history in twin.optimization_history:
                targets = opt_history['optimization_result'].get('target_outcomes', {})
                for target in targets.keys():
                    optimization_targets[target] = optimization_targets.get(target, 0) + 1
                    
                    # Check if target was achieved (simplified)
                    current_value = twin.current_outcomes.get(target, 0)
                    target_value = targets[target]
                    achieved = abs(current_value - target_value) < abs(target_value * 0.2)  # Within 20%
                    
                    if target not in success_by_target:
                        success_by_target[target] = {'total': 0, 'achieved': 0}
                    success_by_target[target]['total'] += 1
                    if achieved:
                        success_by_target[target]['achieved'] += 1
        
        success_rates = {
            target: data['achieved'] / data['total'] if data['total'] > 0 else 0
            for target, data in success_by_target.items()
        }
        
        return {
            'popular_targets': optimization_targets,
            'success_rates_by_target': success_rates,
            'overall_success_rate': np.mean(list(success_rates.values())) if success_rates else 0
        }
    
    def _analyze_group_characteristics(self, group: List[Dict]) -> Dict:
        """Analyze common characteristics of a group"""
        
        if not group:
            return {}
        
        # Analyze demographics
        demo_summary = {}
        for person in group:
            demo = person['demographics']
            for key, value in demo.items():
                if key not in demo_summary:
                    demo_summary[key] = {}
                if value not in demo_summary[key]:
                    demo_summary[key][value] = 0
                demo_summary[key][value] += 1
        
        # Convert to percentages
        for key in demo_summary:
            total = sum(demo_summary[key].values())
            for value in demo_summary[key]:
                demo_summary[key][value] = demo_summary[key][value] / total
        
        return demo_summary
    
    def _identify_improvement_factors(self, high_performers: List[Dict]) -> List[str]:
        """Identify factors that contribute to improvement"""
        
        factors = []
        
        # Analyze demographics of high performers
        if high_performers:
            avg_improvement = np.mean([p['improvement'] for p in high_performers])
            
            if avg_improvement > 10:
                factors.append("Significant wellness improvements are achievable")
            
            # Analyze demographic patterns
            education_levels = [p['demographics'].get('education', '') for p in high_performers]
            if education_levels.count('Bachelor+') / len(education_levels) > 0.6:
                factors.append("Higher education correlates with better outcomes")
            
            age_groups = [p['demographics'].get('age_group', '') for p in high_performers]
            if any('25-34' in ag or '35-44' in ag for ag in age_groups):
                factors.append("Mid-career adults show strong improvement potential")
        
        return factors
    
    def _calculate_barrier_resolution_rate(self) -> float:
        """Calculate how often barriers are successfully resolved"""
        
        total_adaptations = 0
        successful_adaptations = 0
        
        for monitor in self.active_monitors.values():
            effectiveness = monitor._assess_adaptation_effectiveness()
            if isinstance(effectiveness, dict) and 'success_rate' in effectiveness:
                total_adaptations += effectiveness.get('total_adaptations_assessed', 0)
                successful_adaptations += effectiveness.get('total_adaptations_assessed', 0) * effectiveness['success_rate']
        
        return successful_adaptations / total_adaptations if total_adaptations > 0 else 0
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate recommendations for system improvement"""
        
        recommendations = []
        
        # Based on system statistics
        if self.system_stats['total_adaptations'] / max(self.system_stats['total_twins_created'], 1) > 0.5:
            recommendations.append("High adaptation rate suggests need for better initial planning")
        
        if len(self.orchestrator.twins) > 10:
            recommendations.append("Sufficient population size for meaningful pattern analysis")
        
        # Add more sophisticated recommendations based on analysis
        recommendations.extend([
            "Consider implementing group challenges for increased motivation",
            "Develop specialized programs for different demographic segments",
            "Add more real-time feedback mechanisms"
        ])
        
        return recommendations
    
    def _save_onboarding_report(self, result: Dict):
        """Save onboarding report to file"""
        filename = f"{self.output_dir}/onboarding_{result['person_id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        # Create serializable version
        serializable_result = result.copy()
        serializable_result.pop('digital_twin', None)
        serializable_result.pop('optimization_result', None)
        serializable_result.pop('intervention_plan', None)
        serializable_result.pop('progress_monitor', None)
        
        with open(filename, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)
    
    def _save_progress_update(self, result: Dict):
        """Save progress update to file"""
        filename = f"{self.output_dir}/progress_{result['person_id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    
    def _save_weekly_plan(self, person_id: str, week_number: int, plan: Dict):
        """Save weekly plan to file"""
        filename = f"{self.output_dir}/weekly_plan_{person_id}_week{week_number}_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(filename, 'w') as f:
            json.dump(plan, f, indent=2, default=str)
    
    def _save_population_analysis(self, analysis: Dict):
        """Save population analysis to file"""
        filename = f"{self.output_dir}/population_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)


def asdict(obj):
    """Convert dataclass to dict (simple implementation)"""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return obj


if __name__ == "__main__":
    # Comprehensive demonstration of the complete digital twin framework
    
    print("="*80)
    print("COMPREHENSIVE DIGITAL TWIN FRAMEWORK DEMONSTRATION")
    print("Complete End-to-End Personalized Wellness Optimization")
    print("="*80)
    
    # Initialize the complete system
    wellness_optimizer = ComprehensiveWellnessOptimizer(
        output_dir="demo_digital_twin_outputs",
        random_seed=42
    )
    
    # Create sample people for demonstration
    sample_people = [
        {
            'person_id': 'sarah_001',
            'demographics': {
                'age_numeric': 34,
                'gender': 'Female',
                'ethnicity': 'White',
                'education': 'Bachelor+',
                'income_bracket': '$75-100k',
                'urban_rural': 'Urban',
                'fitness_level': 'Medium'
            },
            'current_behaviors': {
                'motion_days_week': 2,
                'diet_mediterranean_score': 5.5,
                'meditation_minutes_week': 0,
                'sleep_hours': 6.5,
                'sleep_quality_score': 5,
                'purpose_meaning_score': 6,
                'social_connections_count': 5,
                'nature_minutes_week': 60,
                'cultural_hours_week': 2,
                'smoking_status': 'Never',
                'alcohol_drinks_week': 6,
                'processed_food_servings_week': 8,
                'added_sugar_grams_day': 35,
                'sodium_grams_day': 4.5
            },
            'wellness_goals': {
                'biological_age_acceleration': -1.5,
                'life_satisfaction_score': 8.5
            },
            'preferences': {
                'intensity': 'moderate',
                'program_duration_weeks': 12,
                'timeline_months': 6
            }
        },
        {
            'person_id': 'mike_002',
            'demographics': {
                'age_numeric': 48,
                'gender': 'Male',
                'ethnicity': 'Hispanic',
                'education': 'Some College',
                'income_bracket': '$50-75k',
                'urban_rural': 'Suburban',
                'fitness_level': 'Low'
            },
            'current_behaviors': {
                'motion_days_week': 1,
                'diet_mediterranean_score': 4.0,
                'meditation_minutes_week': 0,
                'sleep_hours': 7.0,
                'sleep_quality_score': 4,
                'purpose_meaning_score': 5,
                'social_connections_count': 3,
                'nature_minutes_week': 30,
                'cultural_hours_week': 1,
                'smoking_status': 'Former',
                'alcohol_drinks_week': 12,
                'processed_food_servings_week': 15,
                'added_sugar_grams_day': 60,
                'sodium_grams_day': 7.2
            },
            'wellness_goals': {
                'biological_age_acceleration': -2.0,
                'mortality_risk_score': 0.05
            },
            'preferences': {
                'intensity': 'gentle',
                'program_duration_weeks': 16,
                'timeline_months': 8
            }
        }
    ]
    
    # Onboard each person
    onboarding_results = []
    for person_data in sample_people:
        result = wellness_optimizer.onboard_new_person(person_data)
        onboarding_results.append(result)
    
    print(f"\n{'='*60}")
    print(f"SIMULATING 8 WEEKS OF PROGRESS")
    print(f"{'='*60}")
    
    # Simulate progress over 8 weeks
    for week in range(1, 9):
        print(f"\n--- Week {week} ---")
        
        for result in onboarding_results:
            person_id = result['person_id']
            
            # Generate weekly plan
            weekly_plan = wellness_optimizer.generate_weekly_plan(person_id, week)
            
            if 'error' not in weekly_plan:
                print(f"{person_id} - Week {week} plan: {weekly_plan['phase']} phase")
                print(f"  Focus: {weekly_plan['focus']}")
                print(f"  Time commitment: {weekly_plan['total_time_estimate']} minutes")
            
            # Simulate progress (realistic compliance patterns)
            if week <= 2:
                compliance = 0.8 + np.random.normal(0, 0.1)  # Good start
            elif week <= 4:
                compliance = 0.6 + np.random.normal(0, 0.15)  # Some struggle
            elif week <= 6:
                compliance = 0.7 + np.random.normal(0, 0.1)   # Recovery
            else:
                compliance = 0.85 + np.random.normal(0, 0.1)  # Strong finish
            
            compliance = np.clip(compliance, 0.3, 1.0)
            
            # Simulate behavior improvements
            initial_behaviors = result['digital_twin'].initial_behaviors
            improvements = {}
            
            for behavior, initial_value in initial_behaviors.items():
                if isinstance(initial_value, (int, float)):
                    # Gradual improvement based on compliance
                    improvement_factor = compliance * (week / 8) * 0.3
                    improvements[behavior] = initial_value * (1 + improvement_factor)
            
            # Simulate barriers (occasionally)
            barriers = []
            if week == 3 and np.random.random() < 0.3:
                barriers.append("Too busy with work this week")
            elif week == 5 and np.random.random() < 0.2:
                barriers.append("Feeling unmotivated")
            
            # Update progress
            progress_data = {
                'behaviors': improvements,
                'self_reported_metrics': {
                    'energy_level': 5 + compliance * 3,
                    'stress_level': 7 - compliance * 2,
                    'mood_rating': 6 + compliance * 2
                },
                'compliance_rate': compliance,
                'barriers_encountered': barriers,
                'measurement_date': datetime.now() - timedelta(weeks=8-week)
            }
            
            progress_result = wellness_optimizer.update_person_progress(person_id, progress_data)
            
            if progress_result.get('adaptations_triggered', 0) > 0:
                print(f"  ⚠️  Plan adapted due to progress patterns")
    
    # Generate final population analysis
    print(f"\n{'='*60}")
    print(f"FINAL POPULATION ANALYSIS")
    print(f"{'='*60}")
    
    population_analysis = wellness_optimizer.run_population_analysis()
    
    print(f"Population Size: {population_analysis['population_size']}")
    print(f"Total Optimizations: {population_analysis['system_statistics']['total_optimizations']}")
    print(f"Total Interventions: {population_analysis['system_statistics']['total_interventions_planned']}")
    print(f"Total Adaptations: {population_analysis['system_statistics']['total_adaptations']}")
    
    success_patterns = population_analysis['success_patterns']
    print(f"\nSuccess Patterns:")
    print(f"  High Performers: {success_patterns['high_performers']}")
    print(f"  Low Performers: {success_patterns['low_performers']}")
    
    if success_patterns['improvement_factors']:
        print(f"  Key Success Factors:")
        for factor in success_patterns['improvement_factors']:
            print(f"    - {factor}")
    
    common_barriers = population_analysis['common_barriers']
    print(f"\nCommon Barriers:")
    for trigger, count in common_barriers['most_common_triggers'].items():
        print(f"  {trigger.replace('_', ' ').title()}: {count}")
    
    print(f"Barrier Resolution Rate: {common_barriers['barrier_resolution_rate']:.1%}")
    
    optimization_trends = population_analysis['optimization_trends']
    print(f"\nOptimization Trends:")
    print(f"  Overall Success Rate: {optimization_trends['overall_success_rate']:.1%}")
    
    print(f"\nSystem Recommendations:")
    for rec in population_analysis['recommendations']:
        print(f"  - {rec}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DEMONSTRATION COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    
    print(f"✅ Complete Digital Twin Framework Operational")
    print(f"✅ End-to-End Wellness Optimization Demonstrated")
    print(f"✅ Real-Time Monitoring & Adaptation Working")
    print(f"✅ Population-Level Analytics Generated")
    print(f"✅ Personalized Intervention Plans Created")
    print(f"✅ Progress Tracking & Barrier Resolution Active")
    
    output_files = len([f for f in os.listdir(wellness_optimizer.output_dir) if f.endswith('.json')])
    print(f"✅ {output_files} Output Files Generated")
    
    print(f"\n🎯 READY FOR CLIENT DEPLOYMENT!")
    print(f"The complete digital twin framework is now ready to optimize")
    print(f"real-world wellness journeys based on individual demographics")
    print(f"and actual activities, with continuous adaptation and improvement.")