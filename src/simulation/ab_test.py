"""A/B Test Simulation for Intervention Strategies.

Simulates counterfactual outcomes for different recovery strategies:
- Discount-based: Increases conversion probability via price incentive
- Urgency-based: Increases conversion probability via scarcity messaging
- Value-based: Increases conversion probability via engagement boost

Model outputs P(Revenue=1) = conversion probability directly.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import Config


def simulate_discount_effect(
    X: pd.DataFrame,
    model,
    discount_pct: float,
    discount_sensitivity: float = 0.02,
) -> np.ndarray:
    """Simulate effect of discount by directly boosting conversion probability.
    
    Since price_sensitivity feature was removed (was a simulated proxy),
    we model discount as a direct conversion lift proportional to discount %.
    """
    base_probs = model.predict_proba(X)[:, 1]  # P(conversion)
    # Discount increases conversion; diminishing returns at high probability
    lift = discount_sensitivity * discount_pct * (1 - base_probs)
    return np.minimum(base_probs + lift, 1.0)


def simulate_urgency_effect(
    X: pd.DataFrame,
    model,
    urgency_boost: float = 0.1,
) -> np.ndarray:
    """Simulate urgency intervention by boosting conversion probability."""
    base_probs = model.predict_proba(X)[:, 1]  # P(conversion)
    # Urgency increases conversion; diminishing returns
    lift = urgency_boost * (1 - base_probs)
    return np.minimum(base_probs + lift, 1.0)


def simulate_value_effect(
    X: pd.DataFrame,
    model,
    engagement_boost: float = 0.15,
    engagement_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Simulate value-based intervention by boosting engagement signals."""
    X_sim = X.copy()
    if engagement_cols is None:
        engagement_cols = ["engagement_score", "Total_pages", "total_duration"]
    
    for col in engagement_cols:
        if col in X_sim.columns:
            X_sim[col] = X_sim[col] * (1 + engagement_boost)
    
    return model.predict_proba(X_sim)[:, 1]  # P(conversion)


def estimate_revenue_recovery(
    base_conversion_probs: np.ndarray,
    intervention_conversion_probs: np.ndarray,
    avg_order_value: float = 100.0,
    intervention_cost: float = 0.0,
) -> Dict[str, float]:
    """Estimate revenue impact of intervention.
    
    Model outputs P(conversion) directly. Revenue = P(conversion) * AOV - cost.
    """
    base_expected = base_conversion_probs * avg_order_value
    intervention_expected = intervention_conversion_probs * avg_order_value - intervention_cost
    
    uplift = intervention_expected - base_expected
    
    return {
        "base_expected_revenue_per_session": float(np.mean(base_expected)),
        "intervention_expected_revenue_per_session": float(np.mean(intervention_expected)),
        "uplift_per_session": float(np.mean(uplift)),
        "total_uplift": float(np.sum(uplift)),
        "sessions_improved": int(np.sum(uplift > 0)),
    }


def run_ab_test_simulation(
    X: pd.DataFrame,
    model,
    config: Config,
    n_simulations: int = 1000,
) -> Dict[str, Any]:
    """Run full A/B test simulation across all strategies."""
    sim_config = config.get("simulation.ab_test", {})
    discount_range = sim_config.get("discount_range", [5, 10, 15, 20, 25])
    strategy_types = sim_config.get("strategy_types", ["discount", "urgency", "value"])
    avg_order_value = sim_config.get("avg_order_value", 100.0)
    
    base_conversion_probs = model.predict_proba(X)[:, 1]
    
    results = {
        "base_conversion_rate": float(np.mean(base_conversion_probs)),
        "base_abandonment_rate": float(np.mean(1 - base_conversion_probs)),
        "strategies": {},
    }
    
    for strategy in strategy_types:
        results["strategies"][strategy] = {}
        
        if strategy == "discount":
            for discount in discount_range:
                intervention_probs = simulate_discount_effect(X, model, discount)
                revenue = estimate_revenue_recovery(
                    base_conversion_probs, 
                    intervention_probs, 
                    avg_order_value, 
                    intervention_cost=discount/100 * avg_order_value
                )
                results["strategies"][strategy][f"discount_{discount}pct"] = {
                    "discount_pct": discount,
                    "mean_conversion_after": float(np.mean(intervention_probs)),
                    "mean_abandonment_after": float(np.mean(1 - intervention_probs)),
                    **revenue,
                }
        
        elif strategy == "urgency":
            urgency_levels = [0.05, 0.10, 0.15, 0.20]
            for boost in urgency_levels:
                intervention_probs = simulate_urgency_effect(X, model, boost)
                revenue = estimate_revenue_recovery(base_conversion_probs, intervention_probs, avg_order_value)
                results["strategies"][strategy][f"urgency_boost_{int(boost*100)}pct"] = {
                    "urgency_boost": boost,
                    "mean_conversion_after": float(np.mean(intervention_probs)),
                    "mean_abandonment_after": float(np.mean(1 - intervention_probs)),
                    **revenue,
                }
        
        elif strategy == "value":
            engagement_boosts = [0.10, 0.15, 0.20, 0.25]
            for boost in engagement_boosts:
                intervention_probs = simulate_value_effect(X, model, boost)
                revenue = estimate_revenue_recovery(base_conversion_probs, intervention_probs, avg_order_value)
                results["strategies"][strategy][f"value_boost_{int(boost*100)}pct"] = {
                    "engagement_boost": boost,
                    "mean_conversion_after": float(np.mean(intervention_probs)),
                    "mean_abandonment_after": float(np.mean(1 - intervention_probs)),
                    **revenue,
                }
    
    # Find best strategy
    best_strategy = None
    best_uplift = -np.inf
    
    for strategy, variants in results["strategies"].items():
        for variant_name, metrics in variants.items():
            if metrics["uplift_per_session"] > best_uplift:
                best_uplift = metrics["uplift_per_session"]
                best_strategy = {"strategy": strategy, "variant": variant_name, **metrics}
    
    results["best_strategy"] = best_strategy
    
    return results


def save_ab_test_results(results: Dict[str, Any], output_path: str):
    """Save A/B test results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def print_ab_test_summary(results: Dict[str, Any]):
    """Print formatted A/B test results."""
    print("\n" + "=" * 60)
    print("A/B TEST SIMULATION RESULTS")
    print("=" * 60)
    print(f"Base Conversion Rate:  {results['base_conversion_rate']:.2%}")
    print(f"Base Abandonment Rate: {results['base_abandonment_rate']:.2%}")
    print("-" * 60)
    
    for strategy, variants in results["strategies"].items():
        print(f"\n{strategy.upper()} Strategy:")
        for variant_name, metrics in variants.items():
            conv_delta = metrics['mean_conversion_after'] - results['base_conversion_rate']
            aband_delta = metrics['mean_abandonment_after'] - results['base_abandonment_rate']
            print(f"  {variant_name}:")
            print(f"    Conversion: {metrics['mean_conversion_after']:.2%} "
                  f"(Delta {conv_delta:+.2%})")
            print(f"    Abandonment: {metrics['mean_abandonment_after']:.2%} "
                  f"(Delta {aband_delta:+.2%})")
            print(f"    Uplift/session: ${metrics['uplift_per_session']:.2f}")
            print(f"    Total uplift: ${metrics['total_uplift']:.2f}")
            print(f"    Sessions improved: {metrics['sessions_improved']}/{len(results)}")
    
    best = results.get("best_strategy")
    if best:
        print("\n" + "=" * 60)
        print(f"BEST STRATEGY: {best['strategy'].upper()} - {best['variant']}")
        print(f"Expected Uplift: ${best['uplift_per_session']:.2f} per session")
        print("=" * 60)