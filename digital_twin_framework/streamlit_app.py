#!/usr/bin/env python3
"""
Interactive Digital Twin Wellness Simulator
Streamlit app for demonstrating personalized wellness optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dataclasses import asdict
import sys
import os
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from digital_twin_framework.core.digital_twin_framework import PersonalDigitalTwin, DigitalTwinOrchestrator
from digital_twin_framework.core.intervention_planner import InterventionPlanner
from digital_twin_framework.core.progress_monitor import ProgressMonitor

# Set page configuration
st.set_page_config(
    page_title="Digital Twin Wellness Simulator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2e86c1;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e3f2fd;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .danger-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .feature-box {
        text-align: center;
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .feature-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .comparison-current {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 0.5rem 0;
    }
    .comparison-optimized {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .step-indicator {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
    }
    .step-number {
        background: #2196f3;
        color: white;
        border-radius: 50%;
        width: 2rem;
        height: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        font-weight: bold;
    }
    .progress-bar {
        background: #e0e0e0;
        border-radius: 1rem;
        height: 0.5rem;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .progress-fill {
        background: linear-gradient(90deg, #4caf50, #81c784);
        height: 100%;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""

    # Initialize session state
    if 'digital_twin' not in st.session_state:
        st.session_state.digital_twin = None
    if 'progress_history' not in st.session_state:
        st.session_state.progress_history = []
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'intervention_planner' not in st.session_state:
        st.session_state.intervention_planner = InterventionPlanner()
    if 'intervention_plan' not in st.session_state:
        st.session_state.intervention_plan = None
    if 'plan_summary' not in st.session_state:
        st.session_state.plan_summary = ""
    if 'weekly_checklists' not in st.session_state:
        st.session_state.weekly_checklists = {}
    if 'progress_monitor' not in st.session_state:
        st.session_state.progress_monitor = None



    # Enhanced sidebar with guided navigation
    with st.sidebar:
        st.header("🧭 Your Wellness Journey")
        
        # Progress indicator
        twin_created = st.session_state.digital_twin is not None
        optimization_done = st.session_state.optimization_results is not None
        progress_tracked = len(st.session_state.progress_history) > 0
        
        st.markdown("### 📍 Journey Progress")
        
        # Step indicators
        steps = [
            ("🏠 Learn About Digital Twins", True, "home"),
            ("👤 Create Your Digital Twin", twin_created, "create"),
            ("🎯 Optimize Your Wellness", optimization_done, "optimize"),
            ("📊 Track Your Progress", progress_tracked, "progress"),
            ("📈 Analyze Your Results", progress_tracked, "analytics")
        ]
        
        for i, (step_name, completed, step_id) in enumerate(steps, 1):
            status_icon = "✅" if completed else "⏳"
            color = "#4caf50" if completed else "#e0e0e0"
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 0.5rem 0; padding: 0.5rem; 
                        background: {'#e8f5e8' if completed else '#f5f5f5'}; 
                        border-radius: 0.5rem; border-left: 4px solid {color};">
                <span style="margin-right: 0.5rem; font-size: 1.2rem;">{status_icon}</span>
                <span style="font-size: 0.9rem;">{step_name}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Choose a section:",
            ["🏠 Home", "👤 Create Digital Twin", "🎯 Optimize Wellness", "📊 Progress Tracking", "📈 Analytics Dashboard", "🔬 System Status"]
        )

        st.markdown("---")
        
        # Quick stats if twin exists
        if twin_created:
            st.markdown("### � Quick Stats")
            wellness_score = st.session_state.digital_twin._calculate_wellness_score()
            st.metric("Current Wellness Score", f"{wellness_score:.0f}/100")
            
            if optimization_done:
                success_prob = st.session_state.optimization_results.get('estimated_success_probability', 0)
                st.metric("Success Probability", f"{success_prob:.0%}")
        
        st.markdown("---")
        st.markdown("### 💡 About This Tool")
        st.info("""
        **Digital Twin Wellness Framework**
        
        This AI-powered system creates a virtual model of your health profile to:
        
        • **Identify** what's harming your health
        • **Predict** outcomes of lifestyle changes  
        • **Optimize** your personal wellness plan
        • **Adapt** based on your real progress
        
        Built on research from Blue Zones, TWA studies, and expert consensus biomarkers.
        """)
        


    # Clear the main area and route to selected page
    main_container = st.container()
    
    with main_container:
        if page == "🏠 Home":
            show_home_page()
        elif page == "👤 Create Digital Twin":
            show_create_twin_page()
        elif page == "🎯 Optimize Wellness":
            show_optimization_page()
        elif page == "📊 Progress Tracking":
            show_progress_page()
        elif page == "📈 Analytics Dashboard":
            show_analytics_page()
        elif page == "🔬 System Status":
            show_system_status_page()

def show_home_page():
    """Display the home page with overview"""

    st.markdown('<div class="section-header">Transform Your Health with AI-Powered Wellness Optimization</div>', unsafe_allow_html=True)
    
    # Hero section with value proposition
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; margin-bottom: 1rem;">🎯 Your Personal Digital Twin Awaits</h2>
        <p style="font-size: 1.2rem; margin-bottom: 1rem;">
            Discover what you're doing wrong, what you could do better, and see the transformative results of optimized living.
        </p>
        <p style="font-size: 1rem; opacity: 0.9;">
            Get personalized, science-backed recommendations that adapt to your unique lifestyle and goals.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Problem & Solution Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ❌ The Problem: Wellness Without Direction
        
        Most people struggle with:
        - **Generic advice** that doesn't fit their unique situation
        - **No clear understanding** of what's actually harming their health
        - **Overwhelming choices** without knowing what to prioritize
        - **No way to predict** if lifestyle changes will actually work
        - **Lack of adaptation** as their life circumstances change
        
        ### 🎯 What Makes This Different
        
        Our Digital Twin Framework provides:
        - **Personalized Analysis**: Identifies YOUR specific problem areas
        - **Clear Prioritization**: Shows exactly what to fix first
        - **Predictive Power**: Forecasts your health outcomes
        - **Adaptive Learning**: Adjusts as you progress
        """)
    
    with col2:
        st.markdown("""
        ### ✅ The Solution: Your AI-Powered Health Coach
        
        **🧬 Digital Twin Technology**
        - Creates a virtual model of YOUR unique health profile
        - Analyzes your demographics, behaviors, and goals
        - Identifies hidden health risks and opportunities
        
        **🎯 Personalized Optimization**
        - Compares your current lifestyle to your optimal potential
        - Shows specific changes with predicted impact
        - Prioritizes actions based on your capacity for change
        
        **📊 Real Results Tracking**
        - Monitors your actual progress vs predictions
        - Adapts recommendations based on what works for YOU
        - Provides clear before/after comparisons
        """)

    # Key Features with Icons
    st.markdown("---")
    st.markdown('<div class="section-header">🚀 How Your Digital Twin Transforms Your Wellness</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 0.5rem; margin: 0.5rem 0;">
            <div style="font-size: 3rem;">🔍</div>
            <h4>Analyze Your Current State</h4>
            <p>Identify what's working and what's harming your health</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 0.5rem; margin: 0.5rem 0;">
            <div style="font-size: 3rem;">⚡</div>
            <h4>Optimize Your Plan</h4>
            <p>Get specific actions ranked by impact and feasibility</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 0.5rem; margin: 0.5rem 0;">
            <div style="font-size: 3rem;">📊</div>
            <h4>Predict Your Future</h4>
            <p>See how changes will impact your biological age and wellness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 0.5rem; margin: 0.5rem 0;">
            <div style="font-size: 3rem;">🔄</div>
            <h4>Adapt & Improve</h4>
            <p>Continuously refine based on your real progress</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Interactive Demo Section
    st.markdown('<div class="section-header">🎯 See It In Action: Interactive Demo</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Experience the power of personalized wellness optimization with our sample simulation.**
    
    This demo will show you:
    - How the AI analyzes a person's current lifestyle
    - What specific problems it identifies
    - The personalized recommendations it generates
    - The predicted health improvements
    """)

    st.markdown("---")
    st.markdown('<div class="section-header">🛠️ What Runs Behind the Scenes</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🧬 `digital_twin_framework.py`
        Builds your **Digital Twin**—the smart model that spots what's helping or hurting you right now.
        """)

    with col2:
        st.markdown("""
        #### 🛠️ `intervention_planner.py`
        Turns insights into **step-by-step plans** you can actually follow without feeling overwhelmed.
        """)

    with col3:
        st.markdown("""
        #### 📈 `progress_monitor.py`
        Checks how it’s going, **nudges you when you slip**, and adjusts the plan when life happens.
        """)

    if st.button("🚀 Run Interactive Demo", type="primary", use_container_width=True):
        run_enhanced_demo()

def run_enhanced_demo():
    """Run an enhanced demo showing the full digital twin process"""
    
    with st.spinner("🧬 Creating digital twin and analyzing wellness profile..."):
        # Create sample person with some problematic behaviors
        sample_demographics = {
            'age_numeric': 42,
            'gender': 'Male',
            'ethnicity': 'White',
            'education': 'Bachelor+',
            'income_bracket': '$75-100k',
            'urban_rural': 'Urban',
            'fitness_level': 'Low'
        }

        sample_behaviors = {
            'motion_days_week': 1,  # Very low exercise
            'diet_mediterranean_score': 3.0,  # Poor diet
            'meditation_minutes_week': 0,  # No meditation
            'sleep_hours': 6.0,  # Insufficient sleep
            'sleep_quality_score': 4,  # Poor sleep quality
            'purpose_meaning_score': 5,  # Moderate purpose
            'social_connections_count': 2,  # Limited social connections
            'nature_minutes_week': 15,  # Very little nature time
            'cultural_hours_week': 0.5,  # Minimal cultural activities
            'smoking_status': 'Former',
            'alcohol_drinks_week': 14,  # Excessive drinking
            'processed_food_servings_week': 15,  # High processed food
            'added_sugar_grams_day': 65,  # High sugar intake
            'sodium_grams_day': 6.5  # High sodium
        }

        # Create digital twin
        orchestrator = DigitalTwinOrchestrator(random_seed=42)
        twin = orchestrator.create_digital_twin(
            person_id="demo_user",
            demographics=sample_demographics,
            initial_behaviors=sample_behaviors
        )

        # Run optimization
        targets = {
            'biological_age_acceleration': -2.0,
            'life_satisfaction_score': 8.0
        }

        results = twin.optimize_activities(targets)

        # Store in session state for display
        st.session_state.digital_twin = twin

        plan = handle_post_optimization(
            twin,
            results,
            targets
        )

    st.success("✅ Digital Twin Analysis Complete!")

    # Show comprehensive results
    show_enhanced_demo_results(twin, results, sample_behaviors, plan)

def show_enhanced_demo_results(twin, results, sample_behaviors, plan):
    """Display comprehensive demo results with current vs optimized comparison"""
    
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Digital Twin Analysis Results</div>', unsafe_allow_html=True)
    
    # Current State Analysis
    st.markdown("### ⚠️ Current State: What's Harming Your Health")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="warning-card">
            <h4>🚨 High-Risk Behaviors Identified</h4>
        """, unsafe_allow_html=True)
        
        # Identify problematic behaviors
        problems = []
        if sample_behaviors['motion_days_week'] < 3:
            problems.append(f"• **Exercise**: Only {sample_behaviors['motion_days_week']} day/week (Need: 4-5 days)")
        if sample_behaviors['alcohol_drinks_week'] > 7:
            problems.append(f"• **Alcohol**: {sample_behaviors['alcohol_drinks_week']} drinks/week (Safe limit: ≤7)")
        if sample_behaviors['diet_mediterranean_score'] < 6:
            problems.append(f"• **Diet Quality**: Score {sample_behaviors['diet_mediterranean_score']}/10 (Target: 7+)")
        if sample_behaviors['sleep_hours'] < 7:
            problems.append(f"• **Sleep**: {sample_behaviors['sleep_hours']} hours (Need: 7-9 hours)")
        if sample_behaviors['processed_food_servings_week'] > 10:
            problems.append(f"• **Processed Food**: {sample_behaviors['processed_food_servings_week']} servings/week (Limit: <7)")
        
        for problem in problems:
            st.markdown(problem)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Current wellness metrics
        wellness_score = twin._calculate_wellness_score()
        bio_age = twin.current_outcomes.get('biological_age_acceleration', 0)
        life_sat = twin.current_outcomes.get('life_satisfaction_score', 0)
        
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Current Health Status</h4>
        """, unsafe_allow_html=True)
        
        st.metric("Overall Wellness Score", f"{wellness_score:.1f}/100", help="Based on all behaviors and outcomes")
        st.metric("Biological Age Acceleration", f"+{bio_age:.1f} years", 
                 help="How much older your body is than your chronological age")
        st.metric("Life Satisfaction", f"{life_sat:.1f}/10",
                 help="Current reported life satisfaction level")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Optimization Results
    if results['optimization_successful']:
        st.markdown("---")
        st.markdown("### ✅ AI Optimization: Your Path to Better Health")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="recommendation-card">
                <h4>🎯 Prioritized Action Plan</h4>
                <p>Ranked by impact and feasibility for your profile:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show top recommendations with impact
            recommendations = results['optimal_behavior_changes']
            
            # Rank by importance and feasibility
            ranked_recs = []
            for behavior, change in recommendations.items():
                current = sample_behaviors.get(behavior, 0)
                new_val = max(0, current + change)
                
                if abs(change) > 0.1:  # Only significant changes
                    impact_score = abs(change) * get_behavior_impact(behavior)
                    ranked_recs.append({
                        'behavior': behavior,
                        'current': current,
                        'target': new_val,
                        'change': change,
                        'impact': impact_score
                    })
            
            # Sort by impact
            ranked_recs.sort(key=lambda x: x['impact'], reverse=True)
            
            for i, rec in enumerate(ranked_recs[:5], 1):
                behavior_name = rec['behavior'].replace('_', ' ').title()
                current = rec['current']
                target = rec['target']
                change = rec['change']
                
                if rec['behavior'] in ['alcohol_drinks_week', 'processed_food_servings_week', 
                                     'added_sugar_grams_day', 'sodium_grams_day']:
                    action_type = "🔻 REDUCE"
                    color = "red"
                else:
                    action_type = "📈 INCREASE"
                    color = "green"
                
                st.markdown(f"""
                **#{i}. {action_type} {behavior_name}**  
                Current: {current:.1f} → Target: {target:.1f} (Change: {change:+.1f})
                """)
        
        with col2:
            st.markdown("""
            <div class="recommendation-card">
                <h4>🎯 Predicted Outcomes</h4>
                <p>Expected improvements with this plan:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate predicted improvements
            predicted_outcomes = twin.predict_outcome_changes(recommendations, months_ahead=6)
            
            # Show key improvements
            if 'biological_age_acceleration' in predicted_outcomes:
                bio_improvement = predicted_outcomes['biological_age_acceleration'] - bio_age
                st.metric("Biological Age Improvement", 
                         f"{bio_improvement:.1f} years younger",
                         help="Expected reduction in biological age")
            
            if 'life_satisfaction_score' in predicted_outcomes:
                life_improvement = predicted_outcomes['life_satisfaction_score'] - life_sat
                st.metric("Life Satisfaction Boost", 
                         f"+{life_improvement:.1f} points",
                         help="Expected increase in life satisfaction")
            
            wellness_improvement = 15  # Estimated based on changes
            st.metric("Wellness Score Increase", 
                     f"+{wellness_improvement:.0f} points",
                     help="Expected overall wellness improvement")
            
            success_prob = results.get('estimated_success_probability', 0.5)
            st.metric("Success Probability", f"{success_prob:.1%}",
                     help="Likelihood of achieving these results")

        # Current vs Optimized Comparison
        st.markdown("---")
        st.markdown("### 📊 Your Life: Current vs Optimized")
        
        # Create comparison visualization
        create_current_vs_optimized_chart(sample_behaviors, recommendations)
        
        # Lifestyle Impact Summary
        st.markdown("---")
        st.markdown("### 🌟 What This Means for Your Daily Life")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: #e8f4f8; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                <h4>⏰ Time Investment</h4>
                <p><strong>~45 minutes/day</strong></p>
                <p>Small changes, big impact</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: #e8f4f8; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                <h4>🎯 Focus Areas</h4>
                <p><strong>Exercise, Diet, Sleep</strong></p>
                <p>Priority behaviors first</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: #e8f4f8; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                <h4>📈 Timeline</h4>
                <p><strong>6-12 weeks</strong></p>
                <p>To see significant changes</p>
            </div>
            """, unsafe_allow_html=True)

        # Show simplified plan overview for demo (no interactive widgets)
        if plan:
            st.markdown("---")
            st.markdown("### 🗺️ Your Personalized Plan Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Plan Length", f"{plan.total_duration_weeks} weeks")
            with col2:
                st.metric("Success Probability", f"{plan.estimated_success_probability:.0%}")
            with col3:
                st.metric("Weekly Time", f"~{plan.weekly_time_commitment} min")
            
            st.info("💡 **Next Steps:** Use the 'Optimize Wellness' tab to create your real intervention plan with detailed weekly checklists!")
        else:
            st.info("💡 **Next Steps:** Use the 'Create Digital Twin' tab to input your real data, then optimize your wellness plan!")

def get_behavior_impact(behavior):
    """Get the relative impact score for different behaviors"""
    impact_scores = {
        'motion_days_week': 0.9,
        'diet_mediterranean_score': 0.8,
        'sleep_hours': 0.8,
        'alcohol_drinks_week': 0.7,
        'meditation_minutes_week': 0.6,
        'processed_food_servings_week': 0.6,
        'social_connections_count': 0.5,
        'added_sugar_grams_day': 0.5,
        'sodium_grams_day': 0.4,
        'nature_minutes_week': 0.4,
        'cultural_hours_week': 0.3
    }
    return impact_scores.get(behavior, 0.5)

def create_current_vs_optimized_chart(current_behaviors, recommendations):
    """Create a comparison chart showing current vs optimized behaviors"""
    
    # Select key behaviors for visualization
    key_behaviors = ['motion_days_week', 'diet_mediterranean_score', 'sleep_hours', 
                    'alcohol_drinks_week', 'processed_food_servings_week']
    
    current_values = []
    optimized_values = []
    behavior_names = []
    
    for behavior in key_behaviors:
        if behavior in current_behaviors and behavior in recommendations:
            current = current_behaviors[behavior]
            optimized = max(0, current + recommendations[behavior])
            
            current_values.append(current)
            optimized_values.append(optimized)
            behavior_names.append(behavior.replace('_', ' ').title())
    
    if current_values:
        # Create side-by-side bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(behavior_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, current_values, width, label='Current', color='#ff7f7f', alpha=0.8)
        bars2 = ax.bar(x + width/2, optimized_values, width, label='Optimized', color='#7fbf7f', alpha=0.8)
        
        ax.set_xlabel('Wellness Behaviors')
        ax.set_ylabel('Values')
        ax.set_title('Current vs Optimized Lifestyle Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(behavior_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # Interpretation
        st.markdown("""
        **💡 Chart Interpretation:**
        - **Red bars** show your current behaviors
        - **Green bars** show your optimized targets  
        - Bigger gaps = more room for improvement
        - Focus on the behaviors with the largest positive changes
        """)

def show_create_twin_page():
    """Page for creating a digital twin"""

    st.markdown('<div class="section-header">👤 Create Your Digital Twin</div>', unsafe_allow_html=True)
    st.markdown("Input your personal information to create your wellness digital twin")

    with st.form("twin_creation_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 👤 Demographics")

            age = st.slider("Age", 18, 90, 35)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Hispanic", "Asian", "Other"])
            education = st.selectbox("Education", ["Less than HS", "High School", "Some College", "Bachelor+", "Graduate"])
            income = st.selectbox("Income Bracket", ["<$35k", "$35-50k", "$50-75k", "$75-100k", "$100-150k", ">$150k"])
            location = st.selectbox("Location Type", ["Urban", "Suburban", "Rural"])
            fitness = st.selectbox("Current Fitness Level", ["Low", "Medium", "High"])

        with col2:
            st.markdown("### 🏃 Current Behaviors")

            motion_days = st.slider("Exercise Days/Week", 0, 7, 3)
            diet_score = st.slider("Mediterranean Diet Score (0-10)", 0.0, 10.0, 5.0)
            meditation = st.slider("Meditation Minutes/Week", 0, 300, 0)
            sleep_hours = st.slider("Sleep Hours/Night", 4, 12, 7)
            sleep_quality = st.slider("Sleep Quality (1-10)", 1, 10, 6)
            purpose = st.slider("Purpose/Meaning Score (1-10)", 1, 10, 6)
            social = st.slider("Social Connections Count", 0, 20, 4)
            nature = st.slider("Nature Time Minutes/Week", 0, 500, 60)
            cultural = st.slider("Cultural Activities Hours/Week", 0, 20, 2)

            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            alcohol = st.slider("Alcohol Drinks/Week", 0, 30, 6)
            processed_food = st.slider("Processed Food Servings/Week", 0, 30, 8)
            sugar = st.slider("Added Sugar Grams/Day", 0, 200, 35)
            sodium = st.slider("Sodium Grams/Day", 0.0, 10.0, 4.5)

        submitted = st.form_submit_button("🧬 Create My Digital Twin", type="primary", use_container_width=True)

        if submitted:
            # Create demographics dict
            demographics = {
                'age_numeric': age,
                'gender': gender,
                'ethnicity': ethnicity,
                'education': education,
                'income_bracket': income,
                'urban_rural': location,
                'fitness_level': fitness
            }

            # Create behaviors dict
            behaviors = {
                'motion_days_week': motion_days,
                'diet_mediterranean_score': diet_score,
                'meditation_minutes_week': meditation,
                'sleep_hours': sleep_hours,
                'sleep_quality_score': sleep_quality,
                'purpose_meaning_score': purpose,
                'social_connections_count': social,
                'nature_minutes_week': nature,
                'cultural_hours_week': cultural,
                'smoking_status': smoking,
                'alcohol_drinks_week': alcohol,
                'processed_food_servings_week': processed_food,
                'added_sugar_grams_day': sugar,
                'sodium_grams_day': sodium
            }

            # Create digital twin
            with st.spinner("Creating your personalized digital twin..."):
                try:
                    orchestrator = DigitalTwinOrchestrator(random_seed=42)
                    twin = orchestrator.create_digital_twin(
                        person_id=f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        demographics=demographics,
                        initial_behaviors=behaviors
                    )

                    st.session_state.digital_twin = twin
                    st.session_state.optimization_results = None
                    st.session_state.intervention_plan = None
                    st.session_state.plan_summary = ""
                    st.session_state.weekly_checklists = {}
                    st.session_state.progress_monitor = None
                    st.session_state.progress_history = []
                    st.success("✅ Your digital twin has been created successfully!")

                    # Display initial wellness profile
                    wellness_score = twin._calculate_wellness_score()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Wellness Score", f"{wellness_score:.1f}/100")
                    with col2:
                        st.metric("Biological Age Acceleration",
                                f"{twin.current_outcomes.get('biological_age_acceleration', 0):.1f} years")
                    with col3:
                        st.metric("Life Satisfaction",
                                f"{twin.current_outcomes.get('life_satisfaction_score', 0):.1f}/10")

                except Exception as e:
                    st.error(f"❌ Error creating digital twin: {str(e)}")

def show_optimization_page():
    """Page for wellness optimization"""

    st.markdown('<div class="section-header">🎯 Optimize Your Wellness</div>', unsafe_allow_html=True)

    if st.session_state.digital_twin is None:
        st.warning("⚠️ Please create your digital twin first in the 'Create Digital Twin' section.")
        return

    twin = st.session_state.digital_twin

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🎯 Set Your Wellness Goals")

        bio_age_target = st.slider(
            "Target Biological Age Acceleration (years)",
            -3.0, 3.0, -1.0,
            help="Negative values mean younger biological age than chronological age"
        )

        life_satisfaction_target = st.slider(
            "Target Life Satisfaction Score (1-10)",
            1.0, 10.0, 8.0,
            help="Higher scores indicate greater life satisfaction"
        )

        optimize_button = st.button("🚀 Optimize My Wellness Plan", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📊 Current Status")

        wellness_score = twin._calculate_wellness_score()
        current_bio_age = twin.current_outcomes.get('biological_age_acceleration', 0)
        current_life_sat = twin.current_outcomes.get('life_satisfaction_score', 0)

        st.metric("Current Wellness Score", f"{wellness_score:.1f}/100")
        st.metric("Current Biological Age Acceleration", f"{current_bio_age:.1f} years")
        st.metric("Current Life Satisfaction", f"{current_life_sat:.1f}/10")

        # Show goal gaps
        bio_gap = bio_age_target - current_bio_age
        life_gap = life_satisfaction_target - current_life_sat

        if bio_gap < 0:
            st.success(f"🎯 Biological age goal: {bio_gap:.1f} years improvement needed")
        else:
            st.info(f"📈 Biological age goal: {bio_gap:.1f} years beyond current")

        if life_gap > 0:
            st.success(f"🎯 Life satisfaction goal: {life_gap:.1f} points improvement needed")
        else:
            st.info(f"📈 Life satisfaction goal: {life_gap:.1f} points beyond current")

    if optimize_button:
        with st.spinner("🧬 AI analyzing your profile and creating personalized optimization plan..."):
            targets = {
                'biological_age_acceleration': bio_age_target,
                'life_satisfaction_score': life_satisfaction_target
            }

            try:
                results = twin.optimize_activities(targets)

                if results['optimization_successful']:
                    st.success("✅ Optimization completed! Your personalized wellness transformation plan is ready.")

                    plan = handle_post_optimization(
                        twin,
                        results,
                        targets
                    )

                    # Show comprehensive optimization results
                    show_optimization_results(twin, results, targets, plan)

                else:
                    st.error("❌ Optimization failed. Please try adjusting your goals or contact support.")

            except Exception as e:
                st.error(f"❌ Error during optimization: {str(e)}")

def show_optimization_results(twin, results, targets, plan):
    """Display comprehensive optimization results with current vs optimized comparison"""
    
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Your Personalized Wellness Transformation Plan</div>', unsafe_allow_html=True)
    
    # Current vs Target Analysis
    st.markdown("### 📊 Current vs Your Goals")
    
    col1, col2, col3 = st.columns(3)
    
    current_bio_age = twin.current_outcomes.get('biological_age_acceleration', 0)
    current_life_sat = twin.current_outcomes.get('life_satisfaction_score', 0)
    current_wellness = twin._calculate_wellness_score()
    
    with col1:
        st.markdown("**Current State**")
        st.metric("Wellness Score", f"{current_wellness:.1f}/100")
        st.metric("Bio Age Acceleration", f"{current_bio_age:+.1f} years")
        st.metric("Life Satisfaction", f"{current_life_sat:.1f}/10")
    
    with col2:
        st.markdown("**Your Goals**")
        target_bio_age = targets['biological_age_acceleration']
        target_life_sat = targets['life_satisfaction_score']
        st.metric("Target Bio Age", f"{target_bio_age:+.1f} years")
        st.metric("Target Life Satisfaction", f"{target_life_sat:.1f}/10")
        
        # Calculate gaps
        bio_gap = target_bio_age - current_bio_age
        life_gap = target_life_sat - current_life_sat
        
        if abs(bio_gap) > 0.1:
            st.write(f"Gap: {bio_gap:+.1f} years needed")
        if abs(life_gap) > 0.1:
            st.write(f"Gap: {life_gap:+.1f} points needed")
    
    with col3:
        # Predict outcomes with optimization
        recommendations = results['optimal_behavior_changes']
        predicted_outcomes = twin.predict_outcome_changes(recommendations, months_ahead=6)
        
        st.markdown("**Predicted Results**")
        predicted_wellness = current_wellness + 15  # Estimate improvement
        predicted_bio = predicted_outcomes.get('biological_age_acceleration', current_bio_age)
        predicted_life = predicted_outcomes.get('life_satisfaction_score', current_life_sat)
        
        st.metric("Predicted Wellness", f"{predicted_wellness:.1f}/100", f"+{predicted_wellness-current_wellness:.1f}")
        st.metric("Predicted Bio Age", f"{predicted_bio:+.1f} years", f"{predicted_bio-current_bio_age:+.1f}")
        st.metric("Predicted Life Sat", f"{predicted_life:.1f}/10", f"{predicted_life-current_life_sat:+.1f}")

    # Problem Areas Identification
    st.markdown("---")
    st.markdown("### ⚠️ What's Currently Harming Your Health")
    
    # For demo, use the original problematic behaviors to ensure detection works
    demo_behaviors = {
        'motion_days_week': 1,
        'diet_mediterranean_score': 3.0,
        'meditation_minutes_week': 0,
        'sleep_hours': 6.0,
        'alcohol_drinks_week': 14,
        'processed_food_servings_week': 15,
        'added_sugar_grams_day': 65,
        'sodium_grams_day': 6.5
    }
    
    # Use demo behaviors for demo, otherwise use twin's current behaviors
    behaviors_to_analyze = demo_behaviors if twin.person_id == "demo_user" else twin.current_behaviors
    problem_areas = identify_problem_areas(behaviors_to_analyze)
    
    if problem_areas['high_priority'] or problem_areas['moderate_priority']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🚨 High-Priority Issues**")
            if problem_areas['high_priority']:
                for problem in problem_areas['high_priority']:
                    st.markdown(f"""
                    <div class="warning-card">
                        <strong>{problem['name']}</strong><br>
                        Current: {problem['current']}<br>
                        Risk Level: {problem['risk_level']}<br>
                        <small>{problem['impact']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No high-priority issues identified!")
        
        with col2:
            st.markdown("**⚠️ Areas for Improvement**")
            if problem_areas['moderate_priority']:
                for problem in problem_areas['moderate_priority']:
                    st.markdown(f"""
                    <div style="background: #fff3cd; padding: 0.8rem; border-radius: 0.5rem; margin: 0.3rem 0;">
                        <strong>{problem['name']}</strong><br>
                        Current: {problem['current']}<br>
                        <small>{problem['impact']}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No moderate-priority issues identified!")
    else:
        st.success("🎉 No major health issues identified! You're doing great overall.")

    # Prioritized Action Plan
    st.markdown("---")
    st.markdown("### 🎯 Your Prioritized Action Plan")
    
    recommendations = results['optimal_behavior_changes']
    
    # Rank recommendations by impact and feasibility
    ranked_recommendations = rank_recommendations(recommendations, twin.current_behaviors)
    
    st.markdown("**Actions ranked by impact and feasibility for your profile:**")
    
    for i, rec in enumerate(ranked_recommendations[:6], 1):
        behavior_name = rec['behavior'].replace('_', ' ').title()
        current = rec['current']
        target = rec['target']
        change = rec['change']
        feasibility = rec['feasibility']
        impact = rec['impact']
        
        # Determine action type and color
        if rec['behavior'] in ['alcohol_drinks_week', 'processed_food_servings_week', 
                             'added_sugar_grams_day', 'sodium_grams_day']:
            action_icon = "🔻"
            action_word = "REDUCE"
            color = "#ffebee"
        else:
            action_icon = "📈"
            action_word = "INCREASE"
            color = "#e8f5e8"
        
        st.markdown(f"""
        <div style="background: {color}; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {'#f44336' if action_icon == '🔻' else '#4caf50'}; margin: 0.5rem 0;">
            <strong>#{i}. {action_icon} {action_word} {behavior_name}</strong><br>
            Current: {current:.1f} → Target: {target:.1f} (Change: {change:+.1f})<br>
            <small>Impact: {impact} | Feasibility: {feasibility} | {get_behavior_explanation(rec['behavior'])}</small>
        </div>
        """, unsafe_allow_html=True)

    # Current vs Optimized Lifestyle Comparison
    st.markdown("---")
    st.markdown("### 📊 Your Life: Current vs Optimized")
    
    create_lifestyle_comparison_chart(twin.current_behaviors, recommendations)
    
    # Success Probability and Timeline
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    success_prob = results.get('estimated_success_probability', 0.5)
    
    with col1:
        st.metric("Success Probability", f"{success_prob:.1%}")
        if success_prob >= 0.7:
            st.success("High likelihood of success!")
        elif success_prob >= 0.5:
            st.info("Good chance of success with commitment.")
        else:
            st.warning("Consider starting with fewer changes.")
    
    with col2:
        daily_time = calculate_daily_time_commitment(recommendations)
        st.metric("Daily Time Investment", f"~{daily_time} minutes")
        st.write("Small daily changes for big results")
    
    with col3:
        st.metric("Expected Timeline", "6-12 weeks")
        st.write("To see significant improvements")

    render_life_improvement_blueprint(twin, results, ranked_recommendations, problem_areas)

    planner = st.session_state.intervention_planner
    render_intervention_plan(plan, planner)

def identify_problem_areas(current_behaviors):
    """Identify problematic behaviors that need attention"""
    
    high_priority = []
    moderate_priority = []
    
    # Define thresholds for problematic behaviors
    if current_behaviors.get('motion_days_week', 0) < 3:
        high_priority.append({
            'name': 'Exercise Frequency',
            'current': f"{current_behaviors.get('motion_days_week', 0):.1f} days/week",
            'risk_level': 'HIGH',
            'impact': 'Major cardiovascular and metabolic risks'
        })
    
    if current_behaviors.get('alcohol_drinks_week', 0) >= 14:
        high_priority.append({
            'name': 'Alcohol Consumption',
            'current': f"{current_behaviors.get('alcohol_drinks_week', 0):.1f} drinks/week",
            'risk_level': 'HIGH',
            'impact': 'Liver damage, cancer risk, sleep disruption'
        })
    elif current_behaviors.get('alcohol_drinks_week', 0) > 7:
        moderate_priority.append({
            'name': 'Alcohol Consumption',
            'current': f"{current_behaviors.get('alcohol_drinks_week', 0):.1f} drinks/week",
            'impact': 'Above recommended limits for optimal health'
        })
    
    if current_behaviors.get('sleep_hours', 8) < 6.5:
        high_priority.append({
            'name': 'Sleep Duration',
            'current': f"{current_behaviors.get('sleep_hours', 8):.1f} hours/night",
            'risk_level': 'HIGH',
            'impact': 'Cognitive decline, immune suppression, weight gain'
        })
    elif current_behaviors.get('sleep_hours', 8) < 7:
        moderate_priority.append({
            'name': 'Sleep Duration',
            'current': f"{current_behaviors.get('sleep_hours', 8):.1f} hours/night",
            'impact': 'Suboptimal recovery and performance'
        })
    
    if current_behaviors.get('diet_mediterranean_score', 10) < 5:
        moderate_priority.append({
            'name': 'Diet Quality',
            'current': f"{current_behaviors.get('diet_mediterranean_score', 5):.1f}/10",
            'impact': 'Poor nutrition foundation for health'
        })
    
    if current_behaviors.get('processed_food_servings_week', 5) > 12:
        moderate_priority.append({
            'name': 'Processed Food Intake',
            'current': f"{current_behaviors.get('processed_food_servings_week', 5):.1f} servings/week",
            'impact': 'Inflammation, metabolic dysfunction'
        })
    
    # Additional checks for sugar and sodium
    if current_behaviors.get('added_sugar_grams_day', 25) > 50:
        moderate_priority.append({
            'name': 'Added Sugar Intake',
            'current': f"{current_behaviors.get('added_sugar_grams_day', 25):.1f} grams/day",
            'impact': 'Increased diabetes and aging acceleration risk'
        })
    
    if current_behaviors.get('sodium_grams_day', 2.3) > 6:
        moderate_priority.append({
            'name': 'Sodium Intake',
            'current': f"{current_behaviors.get('sodium_grams_day', 2.3):.1f} grams/day",
            'impact': 'Elevated blood pressure and cardiovascular risk'
        })
    
    # Check for low beneficial behaviors
    if current_behaviors.get('meditation_minutes_week', 60) == 0:
        moderate_priority.append({
            'name': 'Stress Management',
            'current': f"{current_behaviors.get('meditation_minutes_week', 60):.1f} minutes/week",
            'impact': 'Missing stress management and mental health benefits'
        })
    
    return {
        'high_priority': high_priority,
        'moderate_priority': moderate_priority
    }

def rank_recommendations(recommendations, current_behaviors):
    """Rank recommendations by impact and feasibility"""
    
    ranked = []
    
    for behavior, change in recommendations.items():
        if abs(change) > 0.1:  # Only significant changes
            current = current_behaviors.get(behavior, 0)
            target = max(0, current + change)
            
            # Calculate impact score
            impact_score = get_behavior_impact(behavior) * abs(change)
            
            # Determine feasibility
            feasibility = get_behavior_feasibility(behavior, abs(change))
            impact_level = get_impact_level(impact_score)
            
            ranked.append({
                'behavior': behavior,
                'current': current,
                'target': target,
                'change': change,
                'impact_score': impact_score,
                'impact': impact_level,
                'feasibility': feasibility
            })
    
    # Sort by impact score
    ranked.sort(key=lambda x: x['impact_score'], reverse=True)
    
    return ranked

def get_behavior_feasibility(behavior, change_magnitude):
    """Determine feasibility level for behavior changes"""
    
    # Easier behaviors to change
    easy_behaviors = ['nature_minutes_week', 'cultural_hours_week']
    # Moderate difficulty
    moderate_behaviors = ['motion_days_week', 'meditation_minutes_week', 'social_connections_count']
    # Harder behaviors to change
    hard_behaviors = ['alcohol_drinks_week', 'smoking_status', 'sleep_hours']
    
    if behavior in easy_behaviors:
        return "Very Feasible"
    elif behavior in moderate_behaviors:
        if change_magnitude > 2:
            return "Challenging"
        else:
            return "Feasible"
    elif behavior in hard_behaviors:
        if change_magnitude > 3:
            return "Very Challenging"
        else:
            return "Challenging"
    else:
        return "Feasible"

def get_impact_level(impact_score):
    """Convert impact score to readable level"""
    if impact_score > 2:
        return "Very High Impact"
    elif impact_score > 1:
        return "High Impact"
    elif impact_score > 0.5:
        return "Moderate Impact"
    else:
        return "Low Impact"

def get_behavior_explanation(behavior):
    """Get explanation for why this behavior matters"""
    
    explanations = {
        'motion_days_week': 'Regular exercise is the #1 factor for healthy aging',
        'diet_mediterranean_score': 'Mediterranean diet reduces disease risk by 30-40%',
        'sleep_hours': 'Sleep is when your body repairs and consolidates memories',
        'alcohol_drinks_week': 'Excess alcohol accelerates aging and increases cancer risk',
        'meditation_minutes_week': 'Meditation reduces stress and improves brain health',
        'processed_food_servings_week': 'Processed foods drive inflammation and metabolic issues',
        'social_connections_count': 'Strong relationships increase lifespan by 50%',
        'added_sugar_grams_day': 'Added sugar drives diabetes and accelerated aging',
        'sodium_grams_day': 'High sodium increases cardiovascular disease risk'
    }
    
    return explanations.get(behavior, 'Important for overall wellness')

def create_lifestyle_comparison_chart(current_behaviors, recommendations):
    """Create detailed current vs optimized lifestyle comparison"""
    
    # Select behaviors for visualization
    key_behaviors = [
        'motion_days_week', 'diet_mediterranean_score', 'sleep_hours',
        'meditation_minutes_week', 'alcohol_drinks_week', 'processed_food_servings_week'
    ]
    
    current_values = []
    optimized_values = []
    behavior_names = []
    improvements = []
    
    for behavior in key_behaviors:
        if behavior in current_behaviors and behavior in recommendations:
            current = current_behaviors[behavior]
            change = recommendations[behavior]
            optimized = max(0, current + change)
            
            current_values.append(current)
            optimized_values.append(optimized)
            behavior_names.append(behavior.replace('_', ' ').title())
            improvements.append(abs(change))
    
    if current_values:
        # Create the comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Left chart: Current vs Optimized bars
        x = np.arange(len(behavior_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, current_values, width, label='Current', 
                       color='#ff6b6b', alpha=0.7)
        bars2 = ax1.bar(x + width/2, optimized_values, width, label='Optimized', 
                       color='#4ecdc4', alpha=0.7)
        
        ax1.set_xlabel('Wellness Behaviors')
        ax1.set_ylabel('Values')
        ax1.set_title('Current vs Optimized Lifestyle')
        ax1.set_xticks(x)
        ax1.set_xticklabels(behavior_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Right chart: Improvement magnitude
        colors = ['#ff6b6b' if behavior in ['alcohol_drinks_week', 'processed_food_servings_week'] 
                 else '#4ecdc4' for behavior in key_behaviors]
        
        bars3 = ax2.bar(behavior_names, improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Wellness Behaviors')
        ax2.set_ylabel('Magnitude of Change')
        ax2.set_title('Size of Recommended Changes')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, improvement in zip(bars3, improvements):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{improvement:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        
        # Add interpretation
        st.markdown("""
        **📊 Chart Interpretation:**
        - **Left Chart**: Shows your current lifestyle (red) vs optimized targets (teal)
        - **Right Chart**: Shows the magnitude of changes needed (bigger bars = more change required)
        - **Red changes**: Reduce harmful behaviors
        - **Teal changes**: Increase beneficial behaviors
        """)

def calculate_daily_time_commitment(recommendations):
    """Calculate estimated daily time commitment for recommended changes"""
    
    time_estimates = {
        'motion_days_week': 30,  # 30 min per additional exercise day
        'meditation_minutes_week': 1,  # Direct time mapping
        'diet_mediterranean_score': 15,  # Food prep time per point
        'sleep_hours': 0,  # No additional active time
        'social_connections_count': 60,  # 1 hour per week per connection
        'nature_minutes_week': 1,  # Direct time mapping
        'cultural_hours_week': 60,  # Direct time mapping
    }
    
    total_weekly_minutes = 0
    
    for behavior, change in recommendations.items():
        if behavior in time_estimates and change > 0:
            total_weekly_minutes += abs(change) * time_estimates[behavior]
    
    return int(total_weekly_minutes / 7)  # Convert to daily minutes


def get_strength_insights(current_behaviors: Dict[str, float]) -> List[str]:
    """Highlight behaviors that are already supporting the user."""
    strengths = []

    motion = current_behaviors.get('motion_days_week')
    if isinstance(motion, (int, float)) and motion >= 4:
        strengths.append("You're already active most days—keep protecting that energy.")

    diet = current_behaviors.get('diet_mediterranean_score')
    if isinstance(diet, (int, float)) and diet >= 7:
        strengths.append("Your meals are mostly Mediterranean-style, which lowers long-term risk.")

    sleep = current_behaviors.get('sleep_hours')
    if isinstance(sleep, (int, float)) and sleep >= 7:
        strengths.append("Adequate sleep gives you recovery superpowers—nice work.")

    meditation = current_behaviors.get('meditation_minutes_week')
    if isinstance(meditation, (int, float)) and meditation >= 60:
        strengths.append("You already have a stress-buffering mindfulness habit.")

    social = current_behaviors.get('social_connections_count')
    if isinstance(social, (int, float)) and social >= 6:
        strengths.append("Strong social ties are one of the best predictors of long life. You're on it.")

    return strengths


def create_daily_rhythm_plan(recommendations: Dict[str, float]) -> Dict[str, List[str]]:
    """Translate recommendations into a friendly morning/day/evening rhythm."""
    rhythm = {
        'Morning Boost': [],
        'Daytime Momentum': [],
        'Evening Wind-Down': []
    }

    if recommendations.get('motion_days_week', 0) > 0:
        rhythm['Morning Boost'].append("Block 20-30 minutes for movement before the day gets noisy.")

    if recommendations.get('meditation_minutes_week', 0) > 0:
        rhythm['Morning Boost'].append("Start with 5 minutes of guided breathing to set the tone.")

    if recommendations.get('diet_mediterranean_score', 0) > 0:
        rhythm['Daytime Momentum'].append("Prep a Mediterranean-style lunch with plants, whole grains, and healthy fats.")

    if recommendations.get('social_connections_count', 0) > 0:
        rhythm['Daytime Momentum'].append("Schedule a meaningful check-in or shared activity this week.")

    if recommendations.get('processed_food_servings_week', 0) < 0:
        rhythm['Daytime Momentum'].append("Swap packaged snacks for nuts, fruit, or yogurt you enjoy.")

    if recommendations.get('alcohol_drinks_week', 0) < 0:
        rhythm['Evening Wind-Down'].append("Pick alcohol-free evening rituals—herbal tea, short walk, or reading.")

    if recommendations.get('sleep_hours', 0) > 0:
        rhythm['Evening Wind-Down'].append("Choose a consistent lights-out time and build a 30-minute wind-down routine.")

    # Remove empty segments
    return {segment: actions for segment, actions in rhythm.items() if actions}


def generate_reflection_prompts(problem_areas: Dict[str, List[Dict]]) -> List[str]:
    """Produce gentle reflection prompts based on top problem areas."""
    prompts = []

    for area in problem_areas.get('high_priority', [])[:2]:
        prompts.append(
            f"What would change in your week if {area['name'].lower()} shifted closer to the healthy range?"
        )

    if not prompts and problem_areas.get('moderate_priority'):
        prompts.append(
            f"Which of these feels easiest to upgrade first: {', '.join([a['name'] for a in problem_areas['moderate_priority'][:3]])}?"
        )

    prompts.append("What support or accountability would make these changes feel lighter?")
    return prompts


def render_life_improvement_blueprint(
    twin: PersonalDigitalTwin,
    results: Dict,
    ranked_recommendations: List[Dict],
    problem_areas: Dict[str, List[Dict]]
):
    """Show an easy-to-follow life improvement summary."""
    st.markdown("---")
    st.markdown('<div class="section-header">🌈 Life Improvement Blueprint</div>', unsafe_allow_html=True)

    strengths = get_strength_insights(twin.current_behaviors)
    if strengths:
        st.markdown("**✅ Keep doing this**")
        for strength in strengths:
            st.markdown(f"- {strength}")
    else:
        st.info("No major strengths detected yet—perfect time to build new habits.")

    st.markdown("**🎯 Quick wins to focus on first**")
    for rec in ranked_recommendations[:3]:
        behavior_name = rec['behavior'].replace('_', ' ').title()
        st.markdown(
            f"- {behavior_name}: shift by {rec['change']:+.1f} to unlock {rec['impact'].lower()} impact ({rec['feasibility'].lower()})"
        )

    rhythm = create_daily_rhythm_plan(results.get('optimal_behavior_changes', {}))
    if rhythm:
        st.markdown("**🕒 Daily rhythm to try this week**")
        columns = st.columns(len(rhythm))
        for (segment, actions), column in zip(rhythm.items(), columns):
            with column:
                st.markdown(f"**{segment}**")
                for action in actions:
                    st.markdown(f"- {action}")

    prompts = generate_reflection_prompts(problem_areas)
    if prompts:
        st.markdown("**🧠 Reflection prompts**")
        for prompt in prompts:
            st.markdown(f"- {prompt}")


def handle_post_optimization(
    twin: PersonalDigitalTwin,
    results: Dict,
    targets: Dict,
    duration_weeks: int = 12,
    intensity_preference: str = 'moderate'
):
    """Create intervention plan and progress monitor after optimization."""

    if 'target_outcomes' not in results or not results.get('target_outcomes'):
        results['target_outcomes'] = targets

    planner: InterventionPlanner = st.session_state.intervention_planner
    plan = planner.create_intervention_plan(
        person_id=twin.person_id,
        digital_twin_optimization=results,
        duration_weeks=duration_weeks,
        intensity_preference=intensity_preference
    )

    st.session_state.intervention_plan = plan
    st.session_state.plan_summary = planner.generate_plan_summary(plan)
    st.session_state.weekly_checklists = {}

    plan_dict = asdict(plan)
    st.session_state.progress_monitor = ProgressMonitor(twin.person_id, plan_dict)
    st.session_state.optimization_results = results

    return plan


def render_intervention_plan(plan, planner: InterventionPlanner):
    """Display structured intervention plan information."""

    if plan is None:
        return

    st.markdown('<div class="section-header">🗺️ Personalized Intervention Plan</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Plan Length", f"{plan.total_duration_weeks} weeks")
    with col2:
        st.metric("Success Probability", f"{plan.estimated_success_probability:.0%}")
    with col3:
        st.metric("Weekly Time", f"~{plan.weekly_time_commitment} min")

    phase_rows = []
    for phase in plan.phases:
        actions = phase.get('actions', [])
        time_commitment = int(sum(action.get('time_commitment', 0) for action in actions))
        phase_rows.append({
            'Week': phase['week'],
            'Phase': phase['phase_name'],
            'Focus': phase['focus'],
            'Actions': len(actions),
            'Intensity': f"{phase['intensity']*100:.0f}%",
            'Time Commitment (min)': time_commitment
        })

    if phase_rows:
        phase_df = pd.DataFrame(phase_rows)
        st.dataframe(phase_df, use_container_width=True)

    st.info("💡 **Next Steps:** Go to the 'Progress Tracking' tab to view detailed weekly checklists and log your progress!")
    
    st.markdown("**📋 Plan Overview Complete**")
    st.markdown(f"• **{len(plan.phases)} weeks** of structured interventions")
    st.markdown(f"• **{plan.estimated_success_probability:.0%}** estimated success probability") 
    st.markdown(f"• **~{plan.weekly_time_commitment} minutes/week** time commitment")

    if st.session_state.plan_summary:
        with st.expander("📄 Read the full plan summary"):
            st.markdown(st.session_state.plan_summary)

        st.download_button(
            label="Download plan summary",
            data=st.session_state.plan_summary,
            file_name=f"{plan.plan_id}_summary.md"
        )


def show_progress_page():
    """Page for tracking progress"""

    st.markdown('<div class="section-header">📊 Progress Tracking</div>', unsafe_allow_html=True)

    if st.session_state.digital_twin is None:
        st.warning("⚠️ Please create your digital twin first.")
        return

    twin = st.session_state.digital_twin
    plan = st.session_state.intervention_plan

    if plan is None:
        st.warning("⚠️ Run an optimization to generate your intervention plan before tracking progress.")
        return

    planner = st.session_state.intervention_planner

    st.markdown("### � Weekly Checklist Preview")
    total_weeks = len(plan.phases)

    if total_weeks:
        selected_week = st.selectbox(
            "Choose a week from your plan",
            list(range(1, total_weeks + 1)),
            key="progress_week_selector"
        )

        if selected_week not in st.session_state.weekly_checklists:
            checklist = planner.generate_weekly_checklist(plan, selected_week)
            st.session_state.weekly_checklists[selected_week] = checklist
        else:
            checklist = st.session_state.weekly_checklists[selected_week]

        if 'error' in checklist:
            st.warning(checklist['error'])
        else:
            st.markdown(f"**Phase:** {checklist['phase']} | **Focus:** {checklist['focus']}")
            st.markdown(f"**Estimated weekly time:** {checklist['total_time_estimate']} minutes")

            day_columns = st.columns(3)
            for idx, (day, actions) in enumerate(checklist['daily_actions'].items()):
                column = day_columns[idx % 3]
                with column:
                    # Simple day header with weekend indicator
                    weekend_emoji = "🌿" if day in ['Saturday', 'Sunday'] else "💪"
                    st.markdown(f"**{weekend_emoji} {day}**")
                    
                    if actions:
                        for action in actions:
                            difficulty = "●" if action['difficulty'] <= 2 else "●●" if action['difficulty'] <= 3 else "●●●"
                            st.markdown(f"• {action['action']} *({action['time_minutes']}min, {difficulty})*")
                    else:
                        st.markdown("*� Rest & recovery day*")

            st.markdown("**Weekly goals:**")
            for behavior, goal in checklist['weekly_goals'].items():
                st.markdown(f"- {behavior.replace('_', ' ').title()}: {goal}")
    else:
        st.info("Your plan does not include detailed weekly phases yet. Try regenerating with a longer duration.")

    st.markdown("### �📈 Simulate Progress Over Time")

    col1, col2 = st.columns(2)

    with col1:
        weeks_to_simulate = st.slider("Weeks to simulate", 1, 12, 4, key="simulation_weeks")
        compliance_level = st.selectbox(
            "Compliance Level",
            ["High (90%)", "Good (75%)", "Moderate (60%)", "Low (40%)"],
            index=1,
            key="compliance_selector"
        )

        # Map compliance level to actual percentage
        compliance_map = {
            "High (90%)": 0.9,
            "Good (75%)": 0.75,
            "Moderate (60%)": 0.6,
            "Low (40%)": 0.4
        }
        compliance_rate = compliance_map[compliance_level]

    with col2:
        st.markdown("### 🎯 Simulation Parameters")
        st.write(f"**Duration:** {weeks_to_simulate} weeks")
        st.write(f"**Compliance:** {compliance_rate:.0%}")
        st.write(f"**Plan Intensity:** Based on your optimization results")

        simulate_button = st.button("▶️ Run Progress Simulation", type="primary", use_container_width=True)

    if simulate_button:
        with st.spinner(f"Simulating {weeks_to_simulate} weeks of progress..."):
            opt_results = st.session_state.optimization_results
            if opt_results and 'optimal_behavior_changes' in opt_results:
                target_changes = opt_results['optimal_behavior_changes']
            else:
                target_changes = {
                    'motion_days_week': 1,
                    'diet_mediterranean_score': 0.5,
                    'meditation_minutes_week': 15,
                    'social_connections_count': 1
                }

            monitor = ProgressMonitor(twin.person_id, asdict(plan))
            st.session_state.progress_monitor = monitor

            progress_data = []
            current_date = datetime.now()

            for week in range(weeks_to_simulate + 1):
                behaviors = twin.current_behaviors.copy()

                if week > 0:
                    for behavior, target_change in target_changes.items():
                        current_value = behaviors.get(behavior)
                        if isinstance(current_value, (int, float)):
                            random_factor = 0.8 + np.random.random() * 0.4
                            actual_change = target_change * compliance_rate * random_factor
                            behaviors[behavior] = max(0, current_value + actual_change)

                    twin.update_actual_behaviors(behaviors, current_date)

                wellness_score = twin._calculate_wellness_score()
                metrics = {
                    'wellness_score': wellness_score,
                    'biological_age_acceleration': twin.current_outcomes.get('biological_age_acceleration', 0),
                    'life_satisfaction_score': twin.current_outcomes.get('life_satisfaction_score', 0)
                }

                compliance_value = 1.0 if week == 0 else compliance_rate
                monitor.record_progress(
                    metrics,
                    compliance_value,
                    measurement_date=current_date,
                    data_source='simulation'
                )

                progress_data.append({
                    'week': week,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'wellness_score': wellness_score,
                    'behaviors': behaviors.copy(),
                    'metrics': metrics.copy()
                })

                current_date += timedelta(weeks=1)

            st.session_state.progress_history = progress_data

        st.success("✅ Progress simulation completed!")

        # Display results
        st.markdown("### 📊 Progress Results")

        # Create progress dataframe
        df_progress = pd.DataFrame(progress_data)

        col1, col2 = st.columns(2)

        with col1:
            # Wellness score over time
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_progress['week'], df_progress['wellness_score'], marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Week')
            ax.set_ylabel('Wellness Score')
            ax.set_title('Wellness Score Progress')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            st.pyplot(fig, use_container_width=True)

        with col2:
            # Key behavior changes
            st.markdown("#### 🔄 Behavior Changes")

            initial_behaviors = progress_data[0]['behaviors']
            final_behaviors = progress_data[-1]['behaviors']

            key_behaviors = ['motion_days_week', 'diet_mediterranean_score',
                           'meditation_minutes_week', 'social_connections_count']

            for behavior in key_behaviors:
                if behavior in initial_behaviors and behavior in final_behaviors:
                    initial = initial_behaviors[behavior]
                    final = final_behaviors[behavior]
                    
                    # Only process numeric values
                    if isinstance(initial, (int, float)) and isinstance(final, (int, float)):
                        change = final - initial

                        if abs(change) > 0.1:
                            behavior_name = behavior.replace('_', ' ').title()
                            st.metric(
                                f"{behavior_name}",
                                f"{final:.1f}",
                                f"{change:+.1f}",
                                delta_color="normal" if change > 0 else "inverse"
                            )

        # Progress summary with detailed insights
        initial_score = progress_data[0]['wellness_score']
        final_score = progress_data[-1]['wellness_score']
        improvement = final_score - initial_score

        st.markdown("---")
        st.markdown("### 🎯 Progress Summary & Impact Analysis")
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Starting Wellness Score", f"{initial_score:.1f}/100")
        with col2:
            st.metric("Final Wellness Score", f"{final_score:.1f}/100")
        with col3:
            st.metric("Total Improvement", f"{improvement:+.1f} points")
        with col4:
            weekly_rate = improvement / weeks_to_simulate if weeks_to_simulate > 0 else 0
            st.metric("Weekly Progress Rate", f"{weekly_rate:+.1f} pts/week")

        # Detailed progress interpretation
        st.markdown("### 📊 What These Results Mean")
        
        if improvement > 5:
            st.success(f"""
            🎉 **Excellent Progress!** Your wellness score improved by {improvement:.1f} points over {weeks_to_simulate} weeks.
            
            **What this means:**
            - You're on track to achieve your wellness goals
            - Your compliance level of {compliance_rate:.0%} is working well
            - Continue with your current approach
            """)
        elif improvement > 0:
            st.info(f"""
            📈 **Good Progress!** Your wellness score improved by {improvement:.1f} points over {weeks_to_simulate} weeks.
            
            **What this means:**
            - Steady improvement in the right direction
            - Consider increasing compliance for faster results
            - Focus on the highest-impact behaviors
            """)
        elif improvement > -2:
            st.warning(f"""
            📊 **Stable Progress** Your wellness score changed minimally ({improvement:+.1f} points).
            
            **What this means:**
            - You're maintaining your current health level
            - Results may take longer to show with {compliance_rate:.0%} compliance
            - Consider focusing on 1-2 key behaviors first
            """)
        else:
            st.error(f"""
            ⚠️ **Concerning Trend** Your wellness score decreased by {abs(improvement):.1f} points.
            
            **What this means:**
            - Your current approach may need adjustment
            - Consider reducing the intensity of changes
            - Focus on the most feasible behaviors first
            """)

        # Show behavior-specific insights
        st.markdown("---")
        st.markdown("### 🔍 Behavior-Specific Insights")
        
        initial_behaviors = progress_data[0]['behaviors']
        final_behaviors = progress_data[-1]['behaviors']
        
        # Calculate behavior changes and rank by improvement
        behavior_changes = []
        for behavior in initial_behaviors:
            if behavior in final_behaviors:
                initial_val = initial_behaviors[behavior]
                final_val = final_behaviors[behavior]
                
                # Only process numeric values
                if isinstance(initial_val, (int, float)) and isinstance(final_val, (int, float)):
                    change = final_val - initial_val
                    
                    if abs(change) > 0.1:
                        behavior_changes.append({
                            'behavior': behavior,
                            'initial': initial_val,
                            'final': final_val,
                            'change': change,
                            'percent_change': (change / initial_val * 100) if initial_val > 0 else 0
                        })
        
        # Sort by absolute change
        behavior_changes.sort(key=lambda x: abs(x['change']), reverse=True)
        
        if behavior_changes:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**� Biggest Improvements**")
                positive_changes = [b for b in behavior_changes if b['change'] > 0][:3]
                
                for change in positive_changes:
                    behavior_name = change['behavior'].replace('_', ' ').title()
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>{behavior_name}</strong><br>
                        {change['initial']:.1f} → {change['final']:.1f} 
                        (+{change['change']:.1f}, {change['percent_change']:+.0f}%)<br>
                        <small>✅ Great progress on this behavior!</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**⚠️ Areas Needing Attention**")
                negative_changes = [b for b in behavior_changes if b['change'] < 0][:3]
                
                if negative_changes:
                    for change in negative_changes:
                        behavior_name = change['behavior'].replace('_', ' ').title()
                        st.markdown(f"""
                        <div class="warning-card">
                            <strong>{behavior_name}</strong><br>
                            {change['initial']:.1f} → {change['final']:.1f} 
                            ({change['change']:.1f}, {change['percent_change']:.0f}%)<br>
                            <small>⚠️ This behavior moved in wrong direction</small>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("🎉 No behaviors declined - excellent work!")

        # Recommendations for next steps
        st.markdown("---")
        st.markdown("### 🎯 Recommended Next Steps")
        
        if improvement > 3:
            st.markdown("""
            **Continue Current Approach:**
            - ✅ Your compliance level is working well
            - ✅ Consider gradually increasing targets
            - ✅ Add 1-2 new behaviors to optimize
            """)
        elif improvement > 0:
            st.markdown("""
            **Optimize Current Strategy:**
            - 📈 Increase compliance to 80%+ for faster results
            - 🎯 Focus on your top 3 highest-impact behaviors
            - ⏰ Consider extending timeline to 8-12 weeks
            """)
        else:
            st.markdown("""
            **Adjust Strategy:**
            - 🔄 Reduce number of simultaneous changes
            - 📝 Focus on 1-2 most feasible behaviors first
            - 🤝 Consider additional support or accountability
            - ⏰ Allow more time for each change to establish
            """)

def show_analytics_page():
    """Page for analytics and insights"""

    st.markdown('<div class="section-header">📈 Analytics Dashboard</div>', unsafe_allow_html=True)

    if st.session_state.digital_twin is None:
        st.warning("⚠️ Please create your digital twin first.")
        return

    twin = st.session_state.digital_twin
    monitor = st.session_state.progress_monitor
    plan = st.session_state.intervention_plan

    # Manual Progress Logging Section
    st.markdown("### 📝 Log Real Progress")
    
    if plan is None:
        st.info("Complete an optimization first to enable progress tracking.")
    else:
        with st.form("progress_logging"):
            st.markdown("**Record your actual progress for this week:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                compliance_this_week = st.slider("How well did you follow the plan?", 0, 100, 75) / 100
                wellness_feeling = st.slider("Overall wellness feeling (1-10)", 1, 10, 7)
                energy_level = st.slider("Energy level (1-10)", 1, 10, 6)
            
            with col2:
                sleep_quality = st.slider("Sleep quality (1-10)", 1, 10, 7)
                stress_level = st.slider("Stress level (1-10)", 1, 10, 4)
                motivation_level = st.slider("Motivation level (1-10)", 1, 10, 6)
            
            barriers_encountered = st.text_area("Any barriers or challenges this week?", placeholder="e.g., Too busy, lost motivation, weather was bad...")
            
            log_progress = st.form_submit_button("📊 Log This Week's Progress", type="primary")
            
            if log_progress:
                if monitor:
                    # Create metrics from user input
                    user_metrics = {
                        'wellness_feeling': wellness_feeling,
                        'energy_level': energy_level,
                        'sleep_quality': sleep_quality,
                        'stress_level': stress_level,
                        'motivation_level': motivation_level
                    }
                    
                    # Record progress in monitor
                    monitor.record_progress(
                        user_metrics,
                        compliance_this_week,
                        measurement_date=datetime.now(),
                        data_source='user_input'
                    )
                    
                    # Handle barriers if reported
                    if barriers_encountered.strip():
                        barrier_suggestions = monitor.suggest_manual_adaptation(
                            barriers_encountered,
                            barrier_type='general'
                        )
                        
                        st.success("✅ Progress logged successfully!")
                        
                        if barrier_suggestions['adaptation_suggestions']:
                            st.markdown("**🔧 Suggested adaptations based on your barriers:**")
                            for suggestion in barrier_suggestions['adaptation_suggestions']:
                                st.markdown(f"- **{suggestion['type']}**: {suggestion['description']}")
                                st.markdown(f"  *Rationale: {suggestion['rationale']}*")
                    else:
                        st.success("✅ Progress logged successfully!")
                else:
                    st.warning("Progress monitor not initialized. Run a simulation first.")

    # Analytics from Progress Monitor
    st.markdown("---")
    st.markdown("### 📊 Progress Analytics")
    
    if monitor and len(monitor.progress_history) > 0:
        st.markdown("#### 🎯 Adaptation Report")
        
        adaptation_report = monitor.get_adaptation_report()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Adaptations", adaptation_report.get('total_adaptations', 0))
        with col2:
            st.metric("Recent Adaptations", adaptation_report.get('recent_adaptations_30_days', 0))
        with col3:
            st.metric("Current Compliance", f"{adaptation_report.get('current_compliance_rate', 0.0):.1%}")
        with col4:
            st.metric("Plan Version", adaptation_report.get('current_plan_version', 0))
        
        if adaptation_report.get('adaptation_triggers'):
            st.markdown("**Adaptation Triggers:**")
            for trigger, count in adaptation_report['adaptation_triggers'].items():
                st.markdown(f"- {trigger.replace('_', ' ').title()}: {count} times")
        
        if adaptation_report.get('adaptation_timeline'):
            st.markdown("**Recent Adaptations:**")
            for adaptation in adaptation_report['adaptation_timeline']:
                date = datetime.fromisoformat(adaptation['date']).strftime('%Y-%m-%d')
                st.markdown(f"- **{date}**: {adaptation['description']} → {adaptation['adaptation']}")
        
        # Handle case where no adaptations have been made yet
        if adaptation_report.get('message'):
            st.info(f"ℹ️ {adaptation_report['message']}")
        
        effectiveness = adaptation_report.get('adaptation_effectiveness', {})
        if 'interpretation' in effectiveness:
            st.markdown(f"**Adaptation Effectiveness:** {effectiveness['interpretation']}")
            if 'success_rate' in effectiveness:
                st.markdown(f"Success rate: {effectiveness['success_rate']:.1%}")

    # Get progress report from twin
    progress_report = twin.get_progress_report()

    # Handle case where progress report has errors
    if 'error' in progress_report:
        st.info(f"ℹ️ {progress_report['error']}")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Behavior Trends")

        if 'behavior_trends' in progress_report:
            trends_df = pd.DataFrame([
                {
                    'Behavior': k.replace('_', ' ').title(),
                    'Initial': v['initial'],
                    'Current': v['current'],
                    'Change': v['change'],
                    'Trend': v['trajectory']
                }
                for k, v in progress_report['behavior_trends'].items()
                if isinstance(v['change'], (int, float)) and abs(v['change']) > 0.1
            ])

            if not trends_df.empty:
                st.dataframe(trends_df, use_container_width=True)

                # Trend visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                trends_df = trends_df[trends_df['Change'] != 0]
                if not trends_df.empty:
                    bars = ax.barh(trends_df['Behavior'], trends_df['Change'])
                    ax.set_xlabel('Change')
                    ax.set_title('Behavior Changes')
                    ax.grid(True, alpha=0.3)

                    # Color bars based on change direction
                    for bar, change in zip(bars, trends_df['Change']):
                        bar.set_color('green' if change > 0 else 'red')

                    st.pyplot(fig, use_container_width=True)
            else:
                st.info("No significant behavior changes detected yet.")
        else:
            st.info("No progress data available yet.")

    with col2:
        st.markdown("### 🎯 Outcome Changes")

        if 'outcome_trends' in progress_report:
            outcomes_df = pd.DataFrame([
                {
                    'Outcome': k.replace('_', ' ').title(),
                    'Initial': v['initial'],
                    'Current': v['current'],
                    'Change': v['change'],
                    'Percent Change': v['percent_change']
                }
                for k, v in progress_report['outcome_trends'].items()
            ])

            st.dataframe(outcomes_df, use_container_width=True)

            # Key metrics display
            wellness_score = progress_report.get('current_wellness_score', 0)
            st.metric("Current Wellness Score", f"{wellness_score:.1f}/100")

            # Optimization count
            opt_count = progress_report.get('optimization_count', 0)
            st.metric("Optimizations Performed", opt_count)

        else:
            st.info("No outcome data available yet.")

    st.markdown("---")

    # Progress Monitor Integration
    if monitor and len(monitor.progress_history) > 1:
        st.markdown("---")
        st.markdown("### 📈 Progress Trends Analysis")
        
        # Create trends from progress history
        progress_df = pd.DataFrame([
            {
                'date': record['date'],
                'compliance': record['compliance_rate'],
                **record['metrics']
            }
            for record in monitor.progress_history
        ])
        
        if not progress_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Compliance trend
                fig, ax = plt.subplots(figsize=(10, 6))
                dates = pd.to_datetime(progress_df['date'])
                ax.plot(dates, progress_df['compliance'], marker='o', linewidth=2)
                ax.set_xlabel('Date')
                ax.set_ylabel('Compliance Rate')
                ax.set_title('Compliance Over Time')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            
            with col2:
                # Key metrics trend
                if 'wellness_feeling' in progress_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(dates, progress_df['wellness_feeling'], marker='o', linewidth=2, label='Wellness')
                    if 'energy_level' in progress_df.columns:
                        ax.plot(dates, progress_df['energy_level'], marker='s', linewidth=2, label='Energy')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Score (1-10)')
                    ax.set_title('Wellness Metrics Over Time')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(1, 10)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

    # Population comparison (simulated)
    st.markdown("### 🌍 Population Comparison")

    st.info("📊 This feature compares your progress with similar individuals in our research database.")

    # Simulated population data
    population_data = {
        'Your Score': progress_report.get('current_wellness_score', 65),
        'Similar Age Group': np.random.normal(68, 5),
        'Similar Demographics': np.random.normal(70, 4),
        'General Population': np.random.normal(60, 8)
    }

    # Create comparison chart
    fig, ax = plt.subplots(figsize=(12, 7))
    groups = list(population_data.keys())
    scores = list(population_data.values())

    bars = ax.bar(groups, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Wellness Score')
    ax.set_title('Population Comparison')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.1f}', ha='center', va='bottom')

    st.pyplot(fig, use_container_width=True)

    # Insights
    your_score = population_data['Your Score']
    similar_group = population_data['Similar Age Group']

    if your_score > similar_group:
        st.success(f"🎉 You're performing {your_score - similar_group:.1f} points above your similar age group!")
    elif your_score < similar_group:
        st.info(f"📈 You have {similar_group - your_score:.1f} points of potential improvement compared to similar individuals.")
    else:
        st.info("📊 Your performance is on par with similar individuals.")

def show_system_status_page():
    """Page showing system status and integration verification"""
    
    st.markdown('<div class="section-header">🔬 System Status & Integration</div>', unsafe_allow_html=True)
    
    st.markdown("### 🧬 Backend Systems Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Digital Twin Framework**")
        twin_status = "✅ Active" if st.session_state.digital_twin else "⏸️ Inactive"
        st.markdown(f"Status: {twin_status}")
        if st.session_state.digital_twin:
            st.markdown(f"Person ID: `{st.session_state.digital_twin.person_id}`")
            wellness_score = st.session_state.digital_twin._calculate_wellness_score()
            st.markdown(f"Current Wellness: {wellness_score:.1f}/100")
    
    with col2:
        st.markdown("**Intervention Planner**")
        plan_status = "✅ Plan Ready" if st.session_state.intervention_plan else "⏸️ No Plan"
        st.markdown(f"Status: {plan_status}")
        if st.session_state.intervention_plan:
            plan = st.session_state.intervention_plan
            st.markdown(f"Plan ID: `{plan.plan_id}`")
            st.markdown(f"Duration: {plan.total_duration_weeks} weeks")
            st.markdown(f"Success Probability: {plan.estimated_success_probability:.1%}")
    
    with col3:
        st.markdown("**Progress Monitor**")
        monitor_status = "✅ Monitoring" if st.session_state.progress_monitor else "⏸️ Not Monitoring"
        st.markdown(f"Status: {monitor_status}")
        if st.session_state.progress_monitor:
            monitor = st.session_state.progress_monitor
            st.markdown(f"Progress Records: {len(monitor.progress_history)}")
            st.markdown(f"Adaptations Made: {len(monitor.adaptation_events)}")
            compliance = monitor._get_recent_compliance()
            st.markdown(f"Recent Compliance: {compliance:.1%}")
    
    # Integration Flow Status
    st.markdown("---")
    st.markdown("### 🔄 Integration Flow")
    
    flow_steps = [
        ("1. Digital Twin Created", st.session_state.digital_twin is not None),
        ("2. Optimization Completed", st.session_state.optimization_results is not None),
        ("3. Intervention Plan Generated", st.session_state.intervention_plan is not None),
        ("4. Progress Monitor Active", st.session_state.progress_monitor is not None),
        ("5. Progress Data Captured", len(st.session_state.progress_history) > 0)
    ]
    
    for step_name, completed in flow_steps:
        status_icon = "✅" if completed else "⏳"
        st.markdown(f"{status_icon} {step_name}")
    
    # Data Export
    st.markdown("---")
    st.markdown("### 📤 Data Export")
    
    if st.session_state.digital_twin:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Progress Data"):
                if st.session_state.progress_history:
                    progress_df = pd.DataFrame(st.session_state.progress_history)
                    csv_data = progress_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"progress_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No progress data to export")
        
        with col2:
            if st.button("📋 Export System Report"):
                report = {
                    "system_status": {
                        "digital_twin_active": st.session_state.digital_twin is not None,
                        "intervention_plan_ready": st.session_state.intervention_plan is not None,
                        "progress_monitor_active": st.session_state.progress_monitor is not None
                    },
                    "progress_summary": {
                        "total_records": len(st.session_state.progress_history),
                        "recent_compliance": st.session_state.progress_monitor._get_recent_compliance() if st.session_state.progress_monitor else 0,
                        "adaptations_made": len(st.session_state.progress_monitor.adaptation_events) if st.session_state.progress_monitor else 0
                    } if st.session_state.progress_monitor else {},
                    "export_date": datetime.now().isoformat()
                }
                
                report_json = pd.io.json.dumps(report, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=report_json,
                    file_name=f"system_report_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
    
    # Reset Options
    st.markdown("---")
    st.markdown("### 🔄 Reset Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reset All Data", type="secondary"):
            for key in ['digital_twin', 'optimization_results', 'intervention_plan', 
                       'progress_monitor', 'progress_history', 'weekly_checklists']:
                if key in st.session_state:
                    if key == 'progress_history':
                        st.session_state[key] = []
                    elif key == 'weekly_checklists':
                        st.session_state[key] = {}
                    else:
                        st.session_state[key] = None
            st.success("✅ All data reset successfully!")
            st.rerun()
    
    with col2:
        if st.button("🎯 Reset Progress Only", type="secondary"):
            st.session_state.progress_history = []
            st.session_state.progress_monitor = None
            st.success("✅ Progress data reset!")
            st.rerun()
    
    with col3:
        if st.button("📋 Reset Plan Only", type="secondary"):
            st.session_state.intervention_plan = None
            st.session_state.weekly_checklists = {}
            st.session_state.progress_monitor = None
            st.success("✅ Intervention plan reset!")
            st.rerun()


if __name__ == "__main__":
    main()