"""
RUL (Remaining Useful Life) estimation for DCRM analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
import joblib
from pathlib import Path


class RULEstimator:
    """RUL estimator for circuit breakers based on DCRM analysis."""
    
    def __init__(self):
        # RUL parameters based on typical circuit breaker maintenance schedules
        self.base_rul_days = {
            "Healthy": 365,  # 1 year for healthy breakers
            "Worn Arcing Contact": 90,  # 3 months for worn contacts
            "Misaligned Mechanism": 30  # 1 month for misaligned mechanisms
        }
        
        # Health score to RUL multiplier
        self.health_multipliers = {
            (80, 100): 1.0,    # Excellent: full RUL
            (60, 79): 0.8,     # Good: 80% of RUL
            (40, 59): 0.5,     # Fair: 50% of RUL
            (20, 39): 0.3,     # Poor: 30% of RUL
            (0, 19): 0.1       # Critical: 10% of RUL
        }
        
        # Component-specific adjustments
        self.component_adjustments = {
            "arcing_contact_critical": 0.5,  # Reduce RUL by 50% if critical
            "main_contact_critical": 0.6,   # Reduce RUL by 40% if critical
            "mechanism_critical": 0.3,      # Reduce RUL by 70% if critical
            "anomaly_detected": 0.7         # Reduce RUL by 30% if anomaly
        }
    
    def estimate_rul(self, 
                    prediction: str, 
                    health_score: int, 
                    component_insights: Dict[str, Any], 
                    anomaly_detected: bool = False,
                    operation_count: int = None) -> Dict[str, Any]:
        """
        Estimate RUL based on analysis results.
        
        Args:
            prediction: Predicted fault class
            health_score: Overall health score (0-100)
            component_insights: Component health insights
            anomaly_detected: Whether anomaly was detected
            operation_count: Number of operations (if available)
        
        Returns:
            Dictionary with RUL estimation results
        """
        # Base RUL from prediction
        base_rul = self.base_rul_days.get(prediction, 180)  # Default 6 months
        
        # Health score multiplier
        health_multiplier = self._get_health_multiplier(health_score)
        
        # Component adjustments
        component_multiplier = self._get_component_multiplier(component_insights)
        
        # Anomaly adjustment
        anomaly_multiplier = self.component_adjustments["anomaly_detected"] if anomaly_detected else 1.0
        
        # Operation count adjustment (if available)
        operation_multiplier = self._get_operation_multiplier(operation_count) if operation_count else 1.0
        
        # Calculate final RUL
        estimated_rul_days = int(base_rul * health_multiplier * component_multiplier * 
                                anomaly_multiplier * operation_multiplier)
        
        # Calculate confidence based on health score and component status
        confidence = self._calculate_confidence(health_score, component_insights, anomaly_detected)
        
        # Generate maintenance recommendations based on RUL
        maintenance_recommendations = self._generate_rul_recommendations(
            estimated_rul_days, prediction, health_score, component_insights
        )
        
        return {
            "estimated_rul_days": estimated_rul_days,
            "estimated_rul_weeks": round(estimated_rul_days / 7, 1),
            "estimated_rul_months": round(estimated_rul_days / 30, 1),
            "confidence_percentage": confidence,
            "next_inspection_date": (datetime.now() + timedelta(days=estimated_rul_days)).strftime("%Y-%m-%d"),
            "urgency_level": self._get_urgency_level(estimated_rul_days),
            "maintenance_recommendations": maintenance_recommendations,
            "risk_factors": self._identify_risk_factors(component_insights, anomaly_detected)
        }
    
    def _get_health_multiplier(self, health_score: int) -> float:
        """Get health score multiplier for RUL calculation."""
        for (min_score, max_score), multiplier in self.health_multipliers.items():
            if min_score <= health_score <= max_score:
                return multiplier
        return 0.1  # Default for very low scores
    
    def _get_component_multiplier(self, component_insights: Dict[str, Any]) -> float:
        """Get component-based multiplier for RUL calculation."""
        multiplier = 1.0
        
        # Check for critical components
        if component_insights.get("arcing_contact_health") == "Critical":
            multiplier *= self.component_adjustments["arcing_contact_critical"]
        
        if component_insights.get("main_contact_health") == "Critical":
            multiplier *= self.component_adjustments["main_contact_critical"]
        
        if component_insights.get("mechanism_health") == "Critical":
            multiplier *= self.component_adjustments["mechanism_critical"]
        
        return multiplier
    
    def _get_operation_multiplier(self, operation_count: int) -> float:
        """Get operation count-based multiplier for RUL calculation."""
        if operation_count < 1000:
            return 1.0  # Low usage
        elif operation_count < 5000:
            return 0.9  # Moderate usage
        elif operation_count < 10000:
            return 0.8  # High usage
        else:
            return 0.7  # Very high usage
    
    def _calculate_confidence(self, health_score: int, component_insights: Dict[str, Any], 
                            anomaly_detected: bool) -> float:
        """Calculate confidence in RUL estimation."""
        base_confidence = min(health_score, 90)  # Cap at 90%
        
        # Reduce confidence for critical components
        critical_components = sum(1 for status in component_insights.values() 
                                if status == "Critical")
        confidence_reduction = critical_components * 10
        
        # Reduce confidence for anomalies
        if anomaly_detected:
            confidence_reduction += 15
        
        return max(base_confidence - confidence_reduction, 20)  # Minimum 20%
    
    def _get_urgency_level(self, rul_days: int) -> str:
        """Get urgency level based on RUL."""
        if rul_days <= 7:
            return "CRITICAL"
        elif rul_days <= 30:
            return "HIGH"
        elif rul_days <= 90:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_rul_recommendations(self, rul_days: int, prediction: str, 
                                    health_score: int, component_insights: Dict[str, Any]) -> list:
        """Generate maintenance recommendations based on RUL."""
        recommendations = []
        
        if rul_days <= 7:
            recommendations.extend([
                "URGENT: Schedule immediate inspection and maintenance",
                "Consider taking breaker out of service if possible",
                "Prepare for emergency maintenance procedures"
            ])
        elif rul_days <= 30:
            recommendations.extend([
                "Schedule maintenance within 2 weeks",
                "Increase monitoring frequency",
                "Prepare maintenance resources and personnel"
            ])
        elif rul_days <= 90:
            recommendations.extend([
                "Schedule maintenance within 1 month",
                "Continue regular monitoring",
                "Plan maintenance during next scheduled outage"
            ])
        else:
            recommendations.extend([
                "Continue normal maintenance schedule",
                "Monitor health trends monthly",
                "Plan maintenance for next scheduled interval"
            ])
        
        # Add component-specific recommendations
        if component_insights.get("arcing_contact_health") == "Critical":
            recommendations.append("Priority: Replace arcing contacts")
        
        if component_insights.get("mechanism_health") == "Critical":
            recommendations.append("Priority: Repair operating mechanism")
        
        if component_insights.get("main_contact_health") == "Critical":
            recommendations.append("Priority: Replace main contacts")
        
        return recommendations
    
    def _identify_risk_factors(self, component_insights: Dict[str, Any], 
                             anomaly_detected: bool) -> list:
        """Identify risk factors affecting RUL."""
        risk_factors = []
        
        if component_insights.get("arcing_contact_health") in ["Degraded", "Critical"]:
            risk_factors.append("Arcing contact wear")
        
        if component_insights.get("main_contact_health") in ["Degraded", "Critical"]:
            risk_factors.append("Main contact degradation")
        
        if component_insights.get("mechanism_health") in ["Degraded", "Critical"]:
            risk_factors.append("Mechanism misalignment")
        
        if anomaly_detected:
            risk_factors.append("Anomalous resistance patterns")
        
        if not risk_factors:
            risk_factors.append("No significant risk factors identified")
        
        return risk_factors


def estimate_rul_from_analysis(analysis_result: Dict[str, Any], 
                             operation_count: int = None) -> Dict[str, Any]:
    """
    Estimate RUL from complete analysis result.
    
    Args:
        analysis_result: Complete analysis result from inference
        operation_count: Number of operations (optional)
    
    Returns:
        RUL estimation results
    """
    estimator = RULEstimator()
    
    # Extract information from analysis result
    prediction = analysis_result.get("prediction", "Healthy")
    health_score = analysis_result.get("health_score", 50)
    component_insights = analysis_result.get("component_insights", {})
    anomaly_detected = analysis_result.get("anomaly", False)
    
    return estimator.estimate_rul(
        prediction=prediction,
        health_score=health_score,
        component_insights=component_insights,
        anomaly_detected=anomaly_detected,
        operation_count=operation_count
    )


if __name__ == "__main__":
    # Example usage
    estimator = RULEstimator()
    
    # Example analysis result
    example_result = {
        "prediction": "Worn Arcing Contact",
        "health_score": 45,
        "component_insights": {
            "arcing_contact_health": "Critical",
            "main_contact_health": "Good",
            "mechanism_health": "Good"
        },
        "anomaly": True
    }
    
    rul_result = estimate_rul_from_analysis(example_result, operation_count=2500)
    print("RUL Estimation Results:")
    for key, value in rul_result.items():
        print(f"{key}: {value}")
