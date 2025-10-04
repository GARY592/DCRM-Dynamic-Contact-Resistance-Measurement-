"""
Component-level feature mapping for DCRM analysis.
Maps extracted features to specific circuit breaker components.
"""

from typing import Dict, List, Tuple
import pandas as pd


def get_component_insights(features: Dict[str, float], prediction: int) -> Dict[str, any]:
    """
    Map features to component-level insights for DCRM analysis.
    
    Args:
        features: Dictionary of extracted features
        prediction: Predicted class (0=Healthy, 1=Worn Arcing Contact, 2=Misaligned Mechanism)
    
    Returns:
        Dictionary with component-specific insights
    """
    insights = {
        "arcing_contact_health": "Unknown",
        "main_contact_health": "Unknown", 
        "mechanism_health": "Unknown",
        "key_indicators": [],
        "component_risks": []
    }
    
    # Feature thresholds based on typical DCRM patterns
    thresholds = {
        "mean": 100.0,      # Normal resistance baseline
        "std": 5.0,         # Normal variation
        "peaks_count": 3,   # Normal peak count
        "slope": 0.0,       # Normal slope (no trend)
        "range": 20.0,      # Normal resistance range
        "area": 100.0       # Normal area under curve
    }
    
    # Arcing Contact Analysis
    arcing_indicators = []
    if features.get("peaks_count", 0) > thresholds["peaks_count"]:
        arcing_indicators.append("High peak count suggests arcing instability")
    if features.get("std", 0) > thresholds["std"] * 1.5:
        arcing_indicators.append("High resistance variation indicates arcing wear")
    if features.get("range", 0) > thresholds["range"] * 1.2:
        arcing_indicators.append("Wide resistance range suggests arcing contact degradation")
    
    if arcing_indicators:
        insights["arcing_contact_health"] = "Degraded"
        insights["component_risks"].extend(arcing_indicators)
    else:
        insights["arcing_contact_health"] = "Good"
    
    # Main Contact Analysis  
    main_indicators = []
    if features.get("mean", 0) > thresholds["mean"] * 1.1:
        main_indicators.append("Elevated mean resistance suggests main contact wear")
    if features.get("slope", 0) > thresholds["slope"] + 2.0:
        main_indicators.append("Positive resistance trend indicates main contact degradation")
    if features.get("area", 0) > thresholds["area"] * 1.3:
        main_indicators.append("High area under curve suggests main contact issues")
    
    if main_indicators:
        insights["main_contact_health"] = "Degraded"
        insights["component_risks"].extend(main_indicators)
    else:
        insights["main_contact_health"] = "Good"
    
    # Operating Mechanism Analysis
    mechanism_indicators = []
    if features.get("slope", 0) < thresholds["slope"] - 2.0:
        mechanism_indicators.append("Negative resistance trend suggests mechanism misalignment")
    if features.get("peaks_count", 0) < 2:
        mechanism_indicators.append("Low peak count suggests mechanism sluggishness")
    if features.get("std", 0) < thresholds["std"] * 0.5:
        mechanism_indicators.append("Low variation suggests mechanism binding")
    
    if mechanism_indicators:
        insights["mechanism_health"] = "Degraded"
        insights["component_risks"].extend(mechanism_indicators)
    else:
        insights["mechanism_health"] = "Good"
    
    # Key indicators based on prediction
    if prediction == 1:  # Worn Arcing Contact
        insights["key_indicators"].append("Primary issue: Arcing contact wear detected")
        insights["arcing_contact_health"] = "Critical"
    elif prediction == 2:  # Misaligned Mechanism
        insights["key_indicators"].append("Primary issue: Operating mechanism misalignment")
        insights["mechanism_health"] = "Critical"
    else:  # Healthy
        insights["key_indicators"].append("All components within normal parameters")
    
    return insights


def get_maintenance_recommendations(insights: Dict[str, any], prediction: int, anomaly: bool) -> List[str]:
    """
    Generate predictive maintenance recommendations based on analysis.
    
    Args:
        insights: Component insights from get_component_insights
        prediction: Predicted class
        anomaly: Whether anomaly was detected
    
    Returns:
        List of maintenance recommendations
    """
    recommendations = []
    
    # Immediate actions based on prediction
    if prediction == 1:  # Worn Arcing Contact
        recommendations.extend([
            "URGENT: Schedule arcing contact inspection within 30 days",
            "Check arcing contact material and replace if necessary",
            "Monitor contact resistance trends closely",
            "Consider reducing switching frequency if possible"
        ])
    elif prediction == 2:  # Misaligned Mechanism
        recommendations.extend([
            "URGENT: Schedule mechanism alignment check within 14 days", 
            "Inspect operating mechanism linkages and guides",
            "Check lubrication and mechanical clearances",
            "Verify proper contact timing and travel"
        ])
    else:  # Healthy
        recommendations.extend([
            "Continue normal maintenance schedule",
            "Monitor resistance trends monthly",
            "Next scheduled inspection: 6 months"
        ])
    
    # Component-specific recommendations
    if insights["arcing_contact_health"] == "Critical":
        recommendations.append("CRITICAL: Arcing contact replacement required")
    elif insights["arcing_contact_health"] == "Degraded":
        recommendations.append("Schedule arcing contact maintenance")
    
    if insights["main_contact_health"] == "Critical":
        recommendations.append("CRITICAL: Main contact replacement required")
    elif insights["main_contact_health"] == "Degraded":
        recommendations.append("Schedule main contact inspection")
    
    if insights["mechanism_health"] == "Critical":
        recommendations.append("CRITICAL: Mechanism overhaul required")
    elif insights["mechanism_health"] == "Degraded":
        recommendations.append("Schedule mechanism maintenance")
    
    # Anomaly-based recommendations
    if anomaly:
        recommendations.extend([
            "ANOMALY DETECTED: Investigate unusual resistance patterns",
            "Compare with historical data for trend analysis",
            "Consider additional diagnostic tests"
        ])
    
    # General recommendations
    recommendations.extend([
        "Document all findings in maintenance log",
        "Update breaker condition assessment",
        "Schedule follow-up analysis in 3 months"
    ])
    
    return recommendations


def get_health_score(insights: Dict[str, any]) -> int:
    """
    Calculate overall health score (0-100) based on component analysis.
    
    Args:
        insights: Component insights from get_component_insights
    
    Returns:
        Health score (0-100, where 100 is perfect health)
    """
    score = 100
    
    # Deduct points for component issues
    if insights["arcing_contact_health"] == "Critical":
        score -= 40
    elif insights["arcing_contact_health"] == "Degraded":
        score -= 20
    
    if insights["main_contact_health"] == "Critical":
        score -= 30
    elif insights["main_contact_health"] == "Degraded":
        score -= 15
    
    if insights["mechanism_health"] == "Critical":
        score -= 30
    elif insights["mechanism_health"] == "Degraded":
        score -= 15
    
    # Ensure score doesn't go below 0
    return max(0, score)

