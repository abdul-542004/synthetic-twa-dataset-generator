"""
Intervention Planning System for Digital Twin Framework
Creates personalized, actionable wellness intervention plans based on optimization results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json


@dataclass
class InterventionAction:
    """Single actionable intervention step"""
    action_id: str
    category: str  # 'behavioral', 'environmental', 'social', 'cognitive'
    description: str
    target_behavior: str
    difficulty_level: int  # 1-5 scale
    time_commitment_minutes: int
    frequency_per_week: int
    expected_behavior_change: float
    start_week: int
    duration_weeks: int
    success_metrics: List[str]
    barriers: List[str]
    enablers: List[str]


@dataclass
class InterventionPlan:
    """Complete personalized intervention plan"""
    person_id: str
    plan_id: str
    creation_date: datetime
    total_duration_weeks: int
    target_outcomes: Dict[str, float]
    phases: List[Dict]  # Weekly progression
    total_actions: int
    estimated_success_probability: float
    weekly_time_commitment: int
    key_focus_areas: List[str]


class InterventionPlanner:
    """
    Creates evidence-based, personalized intervention plans
    Translates digital twin optimization results into actionable programs
    """
    
    def __init__(self):
        """Initialize with intervention knowledge base"""
        
        # Evidence-based intervention library
        self.intervention_library = self._build_intervention_library()
        
        # Behavior change difficulty factors
        self.change_difficulty = {
            'motion_days_week': 3,  # Moderate
            'sleep_hours': 4,  # Hard
            'diet_mediterranean_score': 4,  # Hard
            'meditation_minutes_week': 2,  # Easy-moderate
            'alcohol_drinks_week': 5,  # Very hard
            'smoking_status': 5,  # Very hard
            'social_connections_count': 4,  # Hard
            'purpose_meaning_score': 5,  # Very hard
        }
        
        # Phase progression strategy
        self.phase_strategy = {
            'foundation': {'weeks': 1, 'intensity': 0.3},  # Build habits
            'building': {'weeks': 3, 'intensity': 0.6},   # Increase intensity
            'integration': {'weeks': 4, 'intensity': 0.8}, # Full integration
            'maintenance': {'weeks': 4, 'intensity': 1.0}  # Sustain changes
        }
    
    def _build_intervention_library(self) -> Dict:
        """Build comprehensive intervention library"""
        
        return {
            'motion_days_week': {
                'behavioral': [
                    {
                        'name': 'gradual_exercise_increase',
                        'description': 'Increase exercise by 1 day every 2 weeks',
                        'actions': [
                            'Start with 10-minute walks 3x/week',
                            'Add 5 minutes each week',
                            'Introduce strength training every other week',
                            'Track progress with activity app'
                        ],
                        'difficulty': 2,
                        'time_per_session': 20,
                        'expected_change_per_month': 1.5
                    },
                    {
                        'name': 'activity_substitution',
                        'description': 'Replace sedentary activities with active ones',
                        'actions': [
                            'Take stairs instead of elevator',
                            'Walk during phone calls',
                            'Park farther away from destinations',
                            'Do bodyweight exercises during TV time'
                        ],
                        'difficulty': 1,
                        'time_per_session': 5,
                        'expected_change_per_month': 1.0
                    }
                ],
                'environmental': [
                    {
                        'name': 'exercise_environment_setup',
                        'description': 'Create supportive exercise environment',
                        'actions': [
                            'Set up home exercise space',
                            'Lay out workout clothes night before',
                            'Join local gym or fitness class',
                            'Find exercise buddy or group'
                        ],
                        'difficulty': 2,
                        'time_per_session': 0,
                        'expected_change_per_month': 0.5
                    }
                ]
            },
            
            'diet_mediterranean_score': {
                'behavioral': [
                    {
                        'name': 'meal_planning_prep',
                        'description': 'Weekly Mediterranean meal planning and prep',
                        'actions': [
                            'Plan Mediterranean meals for the week',
                            'Grocery shop with Mediterranean list',
                            'Prep vegetables and grains on Sundays',
                            'Cook extra portions for leftovers'
                        ],
                        'difficulty': 3,
                        'time_per_session': 90,
                        'expected_change_per_month': 1.5
                    },
                    {
                        'name': 'gradual_food_swaps',
                        'description': 'Gradually swap foods for Mediterranean options',
                        'actions': [
                            'Replace refined grains with whole grains',
                            'Add one serving of fish per week',
                            'Use olive oil instead of other fats',
                            'Include nuts as daily snack'
                        ],
                        'difficulty': 2,
                        'time_per_session': 0,
                        'expected_change_per_month': 1.0
                    }
                ],
                'social': [
                    {
                        'name': 'cooking_social_support',
                        'description': 'Build social support for healthy cooking',
                        'actions': [
                            'Cook Mediterranean meals with family/friends',
                            'Join healthy cooking class or group',
                            'Share healthy recipes with others',
                            'Start workplace healthy lunch group'
                        ],
                        'difficulty': 2,
                        'time_per_session': 60,
                        'expected_change_per_month': 0.8
                    }
                ]
            },
            
            'meditation_minutes_week': {
                'behavioral': [
                    {
                        'name': 'progressive_meditation',
                        'description': 'Gradually build meditation practice',
                        'actions': [
                            'Start with 2 minutes daily guided meditation',
                            'Increase by 1 minute each week',
                            'Use meditation app with reminders',
                            'Track meditation streak'
                        ],
                        'difficulty': 1,
                        'time_per_session': 5,
                        'expected_change_per_month': 50
                    },
                    {
                        'name': 'mindfulness_integration',
                        'description': 'Integrate mindfulness into daily activities',
                        'actions': [
                            'Practice mindful eating at one meal daily',
                            'Do breathing exercises during commute',
                            'Take mindful walking breaks',
                            'Use mindfulness cues throughout day'
                        ],
                        'difficulty': 2,
                        'time_per_session': 3,
                        'expected_change_per_month': 30
                    }
                ],
                'environmental': [
                    {
                        'name': 'meditation_space_setup',
                        'description': 'Create dedicated meditation environment',
                        'actions': [
                            'Set up quiet meditation corner',
                            'Remove distractions from space',
                            'Add comfortable cushion or chair',
                            'Set meditation app on phone home screen'
                        ],
                        'difficulty': 1,
                        'time_per_session': 0,
                        'expected_change_per_month': 10
                    }
                ]
            },
            
            'sleep_hours': {
                'behavioral': [
                    {
                        'name': 'sleep_hygiene_routine',
                        'description': 'Establish consistent sleep hygiene practices',
                        'actions': [
                            'Set consistent bedtime and wake time',
                            'Create 30-minute wind-down routine',
                            'Avoid screens 1 hour before bed',
                            'Track sleep with app or journal'
                        ],
                        'difficulty': 3,
                        'time_per_session': 30,
                        'expected_change_per_month': 0.5
                    },
                    {
                        'name': 'gradual_bedtime_adjustment',
                        'description': 'Gradually shift bedtime earlier',
                        'actions': [
                            'Move bedtime 15 minutes earlier each week',
                            'Use bedroom only for sleep',
                            'Create calming bedtime ritual',
                            'Avoid caffeine after 2 PM'
                        ],
                        'difficulty': 4,
                        'time_per_session': 0,
                        'expected_change_per_month': 0.75
                    }
                ],
                'environmental': [
                    {
                        'name': 'sleep_environment_optimization',
                        'description': 'Optimize bedroom for quality sleep',
                        'actions': [
                            'Make room dark with blackout curtains',
                            'Keep bedroom cool (65-68°F)',
                            'Use white noise machine or earplugs',
                            'Remove electronic devices from bedroom'
                        ],
                        'difficulty': 2,
                        'time_per_session': 0,
                        'expected_change_per_month': 0.3
                    }
                ]
            },
            
            'social_connections_count': {
                'social': [
                    {
                        'name': 'reconnection_campaign',
                        'description': 'Systematically reconnect with existing relationships',
                        'actions': [
                            'Reach out to one old friend per week',
                            'Schedule regular check-ins with close friends',
                            'Plan monthly social activities',
                            'Join alumni or professional networks'
                        ],
                        'difficulty': 3,
                        'time_per_session': 30,
                        'expected_change_per_month': 0.5
                    },
                    {
                        'name': 'new_social_activities',
                        'description': 'Join activities to meet new people',
                        'actions': [
                            'Join hobby-based club or group',
                            'Attend community events or workshops',
                            'Volunteer for causes you care about',
                            'Take group classes (fitness, cooking, etc.)'
                        ],
                        'difficulty': 4,
                        'time_per_session': 120,
                        'expected_change_per_month': 1.0
                    }
                ]
            },
            
            'alcohol_drinks_week': {
                'behavioral': [
                    {
                        'name': 'gradual_reduction_strategy',
                        'description': 'Systematically reduce alcohol consumption',
                        'actions': [
                            'Track current drinking patterns',
                            'Reduce by 1 drink per week each month',
                            'Replace alcohol with non-alcoholic alternatives',
                            'Identify and avoid drinking triggers'
                        ],
                        'difficulty': 4,
                        'time_per_session': 0,
                        'expected_change_per_month': -2.0
                    },
                    {
                        'name': 'alcohol_free_days',
                        'description': 'Establish alcohol-free days each week',
                        'actions': [
                            'Designate 2 alcohol-free days per week',
                            'Plan engaging activities for those days',
                            'Track alcohol-free streaks',
                            'Find supportive social activities without alcohol'
                        ],
                        'difficulty': 3,
                        'time_per_session': 0,
                        'expected_change_per_month': -1.5
                    }
                ],
                'social': [
                    {
                        'name': 'social_support_network',
                        'description': 'Build support network for reduced drinking',
                        'actions': [
                            'Find accountability partner',
                            'Join support group or online community',
                            'Inform close friends of reduction goals',
                            'Plan social activities not centered on drinking'
                        ],
                        'difficulty': 3,
                        'time_per_session': 60,
                        'expected_change_per_month': -1.0
                    }
                ]
            }
        }
    
    def create_intervention_plan(self, person_id: str, digital_twin_optimization: Dict,
                               duration_weeks: int = 12, intensity_preference: str = 'moderate') -> InterventionPlan:
        """
        Create comprehensive intervention plan based on optimization results
        
        Args:
            person_id: Individual identifier
            digital_twin_optimization: Results from digital twin optimization
            duration_weeks: Total intervention duration
            intensity_preference: 'gentle', 'moderate', or 'intensive'
            
        Returns:
            Complete intervention plan
        """
        
        print(f"Creating intervention plan for {person_id}...")
        
        # Extract key information from optimization
        behavior_changes = digital_twin_optimization.get('optimal_behavior_changes', {})
        target_outcomes = digital_twin_optimization.get('target_outcomes', {})
        change_feasibility = digital_twin_optimization.get('change_feasibility', {})
        
        # Prioritize behaviors by importance and feasibility
        prioritized_behaviors = self._prioritize_behaviors(behavior_changes, change_feasibility)
        
        # Select appropriate interventions
        selected_interventions = self._select_interventions(
            prioritized_behaviors, intensity_preference
        )
        
        # Create phased plan
        phases = self._create_phased_plan(selected_interventions, duration_weeks)
        
        # Calculate success probability
        success_probability = self._estimate_success_probability(
            prioritized_behaviors, change_feasibility, intensity_preference
        )
        
        # Calculate time commitment
        weekly_time_commitment = self._calculate_time_commitment(selected_interventions)
        
        plan = InterventionPlan(
            person_id=person_id,
            plan_id=f"{person_id}_plan_{datetime.now().strftime('%Y%m%d_%H%M')}",
            creation_date=datetime.now(),
            total_duration_weeks=duration_weeks,
            target_outcomes=target_outcomes,
            phases=phases,
            total_actions=sum(len(phase['actions']) for phase in phases),
            estimated_success_probability=success_probability,
            weekly_time_commitment=weekly_time_commitment,
            key_focus_areas=list(prioritized_behaviors.keys())[:3]
        )
        
        return plan
    
    def _prioritize_behaviors(self, behavior_changes: Dict, change_feasibility: Dict) -> Dict:
        """Prioritize behaviors by impact and feasibility"""
        
        prioritized = {}
        
        for behavior, change in behavior_changes.items():
            if abs(change) < 0.1:  # Skip minimal changes
                continue
            
            # Calculate priority score
            impact_score = abs(change)  # Larger changes = higher impact
            feasibility_score = change_feasibility.get(behavior, {}).get('feasibility_score', 0.5)
            difficulty = self.change_difficulty.get(behavior, 3)
            
            # Priority = impact * feasibility / difficulty
            priority_score = (impact_score * feasibility_score) / difficulty
            
            prioritized[behavior] = {
                'change_needed': change,
                'priority_score': priority_score,
                'feasibility_score': feasibility_score,
                'difficulty': difficulty,
                'impact_score': impact_score
            }
        
        # Sort by priority score
        return dict(sorted(prioritized.items(), key=lambda x: x[1]['priority_score'], reverse=True))
    
    def _select_interventions(self, prioritized_behaviors: Dict, intensity_preference: str) -> Dict:
        """Select appropriate interventions for each behavior"""
        
        intensity_multipliers = {
            'gentle': 0.7,
            'moderate': 1.0,
            'intensive': 1.3
        }
        
        multiplier = intensity_multipliers.get(intensity_preference, 1.0)
        selected = {}
        
        for behavior, info in prioritized_behaviors.items():
            if behavior in self.intervention_library:
                behavior_interventions = self.intervention_library[behavior]
                
                # Select best intervention types based on feasibility and difficulty
                selected_types = []
                
                if info['feasibility_score'] > 0.7:
                    # High feasibility - can handle behavioral interventions
                    if 'behavioral' in behavior_interventions:
                        selected_types.extend(behavior_interventions['behavioral'])
                
                if info['difficulty'] <= 3:
                    # Not too difficult - add environmental support
                    if 'environmental' in behavior_interventions:
                        selected_types.extend(behavior_interventions['environmental'])
                
                if behavior in ['social_connections_count', 'alcohol_drinks_week']:
                    # Social behaviors benefit from social interventions
                    if 'social' in behavior_interventions:
                        selected_types.extend(behavior_interventions['social'])
                
                # Adjust for intensity preference
                adjusted_interventions = []
                for intervention in selected_types:
                    adjusted = intervention.copy()
                    adjusted['time_per_session'] = int(
                        adjusted['time_per_session'] * multiplier
                    )
                    adjusted['expected_change_per_month'] *= multiplier
                    adjusted_interventions.append(adjusted)
                
                selected[behavior] = adjusted_interventions
        
        return selected
    
    def _create_phased_plan(self, selected_interventions: Dict, duration_weeks: int) -> List[Dict]:
        """Create week-by-week phased intervention plan"""
        
        phases = []
        current_week = 1
        
        # Distribute interventions across phases
        behaviors_list = list(selected_interventions.keys())
        
        # Phase 1: Foundation (Weeks 1-2) - Start with easiest behaviors
        foundation_weeks = min(2, duration_weeks // 4)
        for week in range(foundation_weeks):
            week_actions = []
            
            # Start with 1-2 easiest behaviors
            for i, behavior in enumerate(behaviors_list[:2]):
                interventions = selected_interventions[behavior]
                if interventions:
                    # Use first (usually easiest) intervention
                    intervention = interventions[0]
                    actions = intervention['actions'][:2]  # Start with first 2 actions
                    
                    for action in actions:
                        week_actions.append({
                            'behavior': behavior,
                            'action': action,
                            'difficulty': intervention['difficulty'],
                            'time_commitment': intervention['time_per_session'] * 0.5,  # Reduced for foundation
                            'frequency': 3 if week == 0 else 4  # Gradual increase
                        })
            
            phases.append({
                'week': current_week,
                'phase_name': 'Foundation',
                'focus': 'Building basic habits',
                'actions': week_actions,
                'intensity': 0.3
            })
            current_week += 1
        
        # Phase 2: Building (Weeks 3-6) - Add more behaviors and intensity
        building_weeks = min(4, duration_weeks // 3)
        for week in range(building_weeks):
            week_actions = []
            
            # Include more behaviors (up to 3-4)
            num_behaviors = min(3 + week, len(behaviors_list))
            for i, behavior in enumerate(behaviors_list[:num_behaviors]):
                interventions = selected_interventions[behavior]
                if interventions:
                    intervention = interventions[0]
                    # Use more actions as weeks progress
                    num_actions = min(2 + week, len(intervention['actions']))
                    actions = intervention['actions'][:num_actions]
                    
                    for action in actions:
                        week_actions.append({
                            'behavior': behavior,
                            'action': action,
                            'difficulty': intervention['difficulty'],
                            'time_commitment': intervention['time_per_session'] * (0.6 + week * 0.1),
                            'frequency': 4 + week
                        })
            
            phases.append({
                'week': current_week,
                'phase_name': 'Building',
                'focus': 'Increasing intensity and adding behaviors',
                'actions': week_actions,
                'intensity': 0.6 + (week * 0.1)
            })
            current_week += 1
        
        # Phase 3: Integration (Remaining weeks) - Full program
        while current_week <= duration_weeks:
            week_actions = []
            
            # Include all behaviors at full intensity
            for behavior in behaviors_list:
                interventions = selected_interventions[behavior]
                if interventions:
                    # Use multiple intervention types if available
                    for intervention in interventions[:2]:  # Max 2 intervention types per behavior
                        for action in intervention['actions']:
                            week_actions.append({
                                'behavior': behavior,
                                'action': action,
                                'difficulty': intervention['difficulty'],
                                'time_commitment': intervention['time_per_session'],
                                'frequency': 5 + min(week - building_weeks - foundation_weeks, 2)
                            })
            
            phase_name = 'Integration' if current_week <= duration_weeks - 2 else 'Maintenance'
            
            phases.append({
                'week': current_week,
                'phase_name': phase_name,
                'focus': 'Full program implementation' if phase_name == 'Integration' else 'Sustaining changes',
                'actions': week_actions,
                'intensity': 0.9 if phase_name == 'Integration' else 1.0
            })
            current_week += 1
        
        return phases
    
    def _estimate_success_probability(self, prioritized_behaviors: Dict, 
                                    change_feasibility: Dict, intensity_preference: str) -> float:
        """Estimate overall success probability for the intervention plan"""
        
        if not prioritized_behaviors:
            return 0.5
        
        # Base success rates by intensity
        intensity_success_rates = {
            'gentle': 0.75,
            'moderate': 0.65,
            'intensive': 0.55
        }
        
        base_rate = intensity_success_rates.get(intensity_preference, 0.65)
        
        # Adjust based on average feasibility
        feasibility_scores = [
            change_feasibility.get(behavior, {}).get('feasibility_score', 0.5)
            for behavior in prioritized_behaviors.keys()
        ]
        avg_feasibility = np.mean(feasibility_scores)
        
        # Adjust based on number of behaviors (more behaviors = lower success rate)
        num_behaviors = len(prioritized_behaviors)
        complexity_penalty = max(0.1, 1.0 - (num_behaviors - 1) * 0.1)
        
        # Adjust based on difficulty
        avg_difficulty = np.mean([
            info['difficulty'] for info in prioritized_behaviors.values()
        ])
        difficulty_penalty = max(0.1, 1.0 - (avg_difficulty - 1) * 0.15)
        
        final_probability = base_rate * avg_feasibility * complexity_penalty * difficulty_penalty
        return max(0.1, min(0.95, final_probability))
    
    def _calculate_time_commitment(self, selected_interventions: Dict) -> int:
        """Calculate average weekly time commitment in minutes"""
        
        total_time = 0
        total_interventions = 0
        
        for behavior, interventions in selected_interventions.items():
            for intervention in interventions:
                sessions_per_week = 5  # Assume 5 sessions per week on average
                time_per_week = intervention['time_per_session'] * sessions_per_week
                total_time += time_per_week
                total_interventions += 1
        
        return int(total_time) if total_interventions == 0 else int(total_time / len(selected_interventions))
    
    def generate_weekly_checklist(self, plan: InterventionPlan, week_number: int) -> Dict:
        """Generate actionable weekly checklist for specific week"""
        
        if week_number < 1 or week_number > len(plan.phases):
            return {"error": f"Week {week_number} not found in plan"}
        
        phase = plan.phases[week_number - 1]
        
        # Group actions by behavior
        behavior_actions = {}
        for action in phase['actions']:
            behavior = action['behavior']
            if behavior not in behavior_actions:
                behavior_actions[behavior] = []
            behavior_actions[behavior].append(action)
        
        # Create structured checklist
        checklist = {
            'week': week_number,
            'phase': phase['phase_name'],
            'focus': phase['focus'],
            'intensity_level': phase['intensity'],
            'daily_actions': {},
            'weekly_goals': {},
            'tracking_metrics': [],
            'total_time_estimate': sum(action['time_commitment'] for action in phase['actions'])
        }
        
        # Create daily action structure
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days:
            checklist['daily_actions'][day] = []
        
        # Distribute actions across the week
        for behavior, actions in behavior_actions.items():
            for action in actions:
                frequency = action['frequency']
                days_to_schedule = min(frequency, 7)
                
                # Spread actions across the week
                day_indices = np.linspace(0, 6, days_to_schedule, dtype=int)
                for day_idx in day_indices:
                    checklist['daily_actions'][days[day_idx]].append({
                        'behavior': behavior,
                        'action': action['action'],
                        'time_minutes': action['time_commitment'],
                        'difficulty': action['difficulty']
                    })
        
        # Set weekly goals
        for behavior in behavior_actions.keys():
            checklist['weekly_goals'][behavior] = f"Implement {behavior} changes consistently"
            checklist['tracking_metrics'].append(f"{behavior}_compliance_rate")
        
        return checklist
    
    def generate_plan_summary(self, plan: InterventionPlan) -> str:
        """Generate human-readable plan summary"""
        
        summary = f"""
# PERSONALIZED WELLNESS INTERVENTION PLAN

**Person ID:** {plan.person_id}
**Plan ID:** {plan.plan_id}
**Created:** {plan.creation_date.strftime('%Y-%m-%d')}
**Duration:** {plan.total_duration_weeks} weeks
**Estimated Success Rate:** {plan.estimated_success_probability:.1%}
**Weekly Time Commitment:** {plan.weekly_time_commitment} minutes

## TARGET OUTCOMES
"""
        
        for outcome, target in plan.target_outcomes.items():
            summary += f"- **{outcome.replace('_', ' ').title()}:** {target}\n"
        
        summary += f"\n## KEY FOCUS AREAS\n"
        for i, area in enumerate(plan.key_focus_areas, 1):
            summary += f"{i}. {area.replace('_', ' ').title()}\n"
        
        summary += f"\n## PHASE OVERVIEW\n"
        
        phase_names = {}
        for phase in plan.phases:
            phase_name = phase['phase_name']
            if phase_name not in phase_names:
                phase_names[phase_name] = []
            phase_names[phase_name].append(phase['week'])
        
        for phase_name, weeks in phase_names.items():
            week_range = f"Week {min(weeks)}" if len(weeks) == 1 else f"Weeks {min(weeks)}-{max(weeks)}"
            summary += f"**{phase_name}** ({week_range}): "
            
            if phase_name == 'Foundation':
                summary += "Build basic habits with gentle introduction\n"
            elif phase_name == 'Building':
                summary += "Increase intensity and add more behaviors\n"
            elif phase_name == 'Integration':
                summary += "Full program implementation\n"
            elif phase_name == 'Maintenance':
                summary += "Sustain changes and establish long-term habits\n"
        
        summary += f"\n## SUCCESS FACTORS\n"
        summary += f"- **Total Actions:** {plan.total_actions} specific interventions\n"
        summary += f"- **Gradual Progression:** Builds from foundation to full integration\n"
        summary += f"- **Personalized Approach:** Based on individual feasibility assessment\n"
        summary += f"- **Evidence-Based:** Uses proven behavior change techniques\n"
        
        return summary


if __name__ == "__main__":
    # Demonstration of intervention planning
    
    print("="*80)
    print("INTERVENTION PLANNING SYSTEM DEMONSTRATION")
    print("Evidence-Based Personalized Wellness Programs")
    print("="*80)
    
    # Example optimization result from digital twin
    sample_optimization = {
        'person_id': 'person_001',
        'optimization_successful': True,
        'optimal_behavior_changes': {
            'motion_days_week': 2.0,
            'diet_mediterranean_score': 1.5,
            'meditation_minutes_week': 60.0,
            'sleep_hours': 0.5,
            'alcohol_drinks_week': -3.0
        },
        'target_outcomes': {
            'biological_age_acceleration': -1.0,
            'life_satisfaction_score': 8.0
        },
        'change_feasibility': {
            'motion_days_week': {'feasibility_score': 0.8, 'recommendation': 'Highly feasible'},
            'diet_mediterranean_score': {'feasibility_score': 0.6, 'recommendation': 'Moderately feasible'},
            'meditation_minutes_week': {'feasibility_score': 0.9, 'recommendation': 'Highly feasible'},
            'sleep_hours': {'feasibility_score': 0.4, 'recommendation': 'Challenging'},
            'alcohol_drinks_week': {'feasibility_score': 0.5, 'recommendation': 'Challenging'}
        }
    }
    
    # Create intervention planner
    planner = InterventionPlanner()
    
    # Generate 12-week intervention plan
    plan = planner.create_intervention_plan(
        person_id="person_001",
        digital_twin_optimization=sample_optimization,
        duration_weeks=12,
        intensity_preference='moderate'
    )
    
    # Display plan summary
    print(planner.generate_plan_summary(plan))
    
    # Show weekly checklist examples
    print("\n" + "="*50)
    print("SAMPLE WEEKLY CHECKLISTS")
    print("="*50)
    
    # Week 1 (Foundation)
    week1_checklist = planner.generate_weekly_checklist(plan, 1)
    print(f"\n**WEEK 1 - {week1_checklist['phase'].upper()}**")
    print(f"Focus: {week1_checklist['focus']}")
    print(f"Estimated Time: {week1_checklist['total_time_estimate']} minutes")
    
    print(f"\nDaily Actions:")
    for day, actions in week1_checklist['daily_actions'].items():
        if actions:
            print(f"  {day}:")
            for action in actions:
                print(f"    - {action['action']} ({action['time_minutes']} min)")
    
    # Week 6 (Building phase)
    week6_checklist = planner.generate_weekly_checklist(plan, 6)
    print(f"\n**WEEK 6 - {week6_checklist['phase'].upper()}**")
    print(f"Focus: {week6_checklist['focus']}")
    print(f"Estimated Time: {week6_checklist['total_time_estimate']} minutes")
    
    print(f"\nSample Monday Actions:")
    monday_actions = week6_checklist['daily_actions']['Monday']
    for action in monday_actions[:3]:  # Show first 3 actions
        print(f"  - {action['action']} ({action['time_minutes']} min)")
    
    print(f"\n" + "="*50)
    print("INTERVENTION PLANNING DEMONSTRATION COMPLETE")
    print("="*50)
    print(f"✅ Created {plan.total_duration_weeks}-week personalized intervention plan")
    print(f"✅ {plan.total_actions} specific actionable interventions")
    print(f"✅ {plan.estimated_success_probability:.1%} estimated success probability")
    print(f"✅ {plan.weekly_time_commitment} minutes average weekly commitment")
    print(f"✅ Structured progression from foundation to maintenance")
    print(f"\nReady for deployment in digital twin framework!")