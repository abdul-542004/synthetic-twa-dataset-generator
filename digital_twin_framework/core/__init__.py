"""
Digital Twin Framework for Personalized Wellness Optimization

This package provides a complete digital twin system for personalized wellness optimization,
built on research-validated synthetic TWA (Tiny Wellness Activities) datasets.

Core Components:
- PersonalDigitalTwin: Individual wellness modeling and optimization
- InterventionPlanner: Evidence-based intervention program creation  
- ProgressMonitor: Real-time monitoring and adaptive plan modification
- DigitalTwinOrchestrator: Population-level management and analytics

Usage:
    from digital_twin_framework.core import (
        PersonalDigitalTwin, 
        DigitalTwinOrchestrator,
        InterventionPlanner,
        ProgressMonitor
    )
"""

from .digital_twin_framework import PersonalDigitalTwin, DigitalTwinOrchestrator
from .intervention_planner import InterventionPlanner, InterventionPlan, InterventionAction
from .progress_monitor import ProgressMonitor, AdaptationTrigger, ProgressMetric, AdaptationEvent

__version__ = "1.0.0"
__author__ = "Digital Twin Framework Team"

__all__ = [
    "PersonalDigitalTwin",
    "DigitalTwinOrchestrator", 
    "InterventionPlanner",
    "InterventionPlan",
    "InterventionAction",
    "ProgressMonitor",
    "AdaptationTrigger",
    "ProgressMetric", 
    "AdaptationEvent"
]