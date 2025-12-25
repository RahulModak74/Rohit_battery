#!/usr/bin/env python3
"""
SECOND LIFE BATTERY SCORER V5 ULTIMATE + MODAK DISTANCE
========================================================
Scoring with 5 dimensions including physics compliance via Modak distance.

SCORING DIMENSIONS (5 total):
1. SOH Change (35%) - Model's degradation prediction
2. SOH Absolute (25%) - Current health state
3. Modak Distance (20%) - Physics compliance score [NEW!]
4. Resistance (12%) - Power capability
5. Usage/C-Rate (8%) - Operating stress

MODAK DISTANCE INTERPRETATION:
- D_M < 1.5: Excellent physics compliance (100%)
- D_M 1.5-2.0: Good compliance (75-100%)
- D_M 2.0-2.5: Acceptable (50-75%)
- D_M 2.5-3.0: Monitor (25-50%)
- D_M 3.0-4.0: Anomaly (0-25%)
- D_M > 4.0: Severe anomaly (0%)

TIER CUTOFFS (5-dimension calibrated):
- PREMIUM:  ≥78 (Physics-compliant, low degradation)
- STANDARD: 70-77 (Good second-life candidates)
- ECONOMY:  62-69 (Budget second-life)
- MARGINAL: 50-61 (Limited value)
- RECYCLE:  <50 (Not suitable)

Usage: python3 scorer_v5_modak.py inference_results_v5_modak.csv
"""

import pandas as pd
import sys
import numpy as np

# =============================================================================
# V5 + MODAK THRESHOLDS
# =============================================================================

# Component thresholds
THRESHOLDS = {
    'soh_change': {'worst': -0.0005, 'best': -0.00005},  # Per-cycle degradation rate
    'soh': {'worst': 0.60, 'best': 0.95},
    'resistance': {'worst': 0.150, 'best': 0.020},  # Ohms
    'c_rate': {'worst': 2.0, 'best': 0.3},
    # NEW: Modak distance thresholds (typical range 1.0 - 4.0)
    'modak_distance': {'worst': 4.0, 'best': 1.0}
}

# Hard limits for automatic tier capping/rejection
HARD_LIMITS = {
    # Resistance limits
    'resistance_economy': 0.080,    # >80mΩ caps at ECONOMY
    'resistance_marginal': 0.100,   # >100mΩ caps at MARGINAL
    'resistance_recycle': 0.120,    # >120mΩ → RECYCLE
    
    # SOH change limits (per cycle)
    'soh_change_fast': -0.0003,     # Fast degradation → cap at STANDARD
    'soh_change_severe': -0.0005,   # Severe degradation → cap at MARGINAL
    
    # SOH absolute limits
    'soh_economy': 0.72,            # <72% caps at ECONOMY
    'soh_marginal': 0.65,           # <65% caps at MARGINAL
    'soh_recycle': 0.55,            # <55% → RECYCLE
    
    # NEW: Modak distance limits
    'modak_monitor': 2.5,           # D_M > 2.5 → cap at STANDARD + flag
    'modak_anomaly': 3.0,           # D_M > 3.0 → cap at ECONOMY + flag
    'modak_severe': 3.5,            # D_M > 3.5 → cap at MARGINAL
    'modak_reject': 4.5,            # D_M > 4.5 → RECYCLE (physics failure)
}

# Tier score cutoffs
TIER_CUTOFFS = {
    'PREMIUM': 78.0,
    'STANDARD': 70.0,
    'ECONOMY': 62.0,
    'MARGINAL': 50.0,
    'RECYCLE': 0.0
}

# 5-DIMENSION WEIGHTS (must sum to 1.0)
WEIGHTS = {
    'soh_change': 0.35,      # Degradation rate (model's key contribution)
    'soh': 0.25,             # Current health
    'modak_distance': 0.20,  # NEW: Physics compliance
    'resistance': 0.12,      # Power capability
    'usage': 0.08            # Operating stress (C-rate)
}

# Verify weights sum to 1.0
assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1.0"


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def linear_score(value, best, worst, max_score=100, min_score=0):
    """
    Linear interpolation between best and worst values.
    
    Args:
        value: The actual value to score
        best: The value that gets max_score
        worst: The value that gets min_score
        max_score: Score for best value (default 100)
        min_score: Score for worst value (default 0)
    
    Returns:
        Score between min_score and max_score
    """
    if best < worst:
        # Lower is better (e.g., degradation rate, resistance)
        if value <= best: 
            return max_score
        if value >= worst: 
            return min_score
        return max_score - (value - best) / (worst - best) * (max_score - min_score)
    else:
        # Higher is better (e.g., SOH)
        if value >= best: 
            return max_score
        if value <= worst: 
            return min_score
        return min_score + (value - worst) / (best - worst) * (max_score - min_score)


def score_modak_distance(modak_distance):
    """
    Score Modak distance on 0-100 scale.
    
    Typical range: 1.0 - 4.0
    - D_M < 1.0: Perfect (score 100) - rare, excellent physics match
    - D_M = 1.5: Excellent (score 90)
    - D_M = 2.0: Good (score 75)
    - D_M = 2.5: Acceptable (score 50)
    - D_M = 3.0: Concerning (score 25)
    - D_M = 3.5: Poor (score 10)
    - D_M >= 4.0: Failure (score 0)
    """
    return linear_score(
        modak_distance,
        best=THRESHOLDS['modak_distance']['best'],      # 1.0 → 100
        worst=THRESHOLDS['modak_distance']['worst'],    # 4.0 → 0
        max_score=100,
        min_score=0
    )


def apply_hard_limits(raw_score, resistance, soh_change, soh, modak_distance=None):
    """
    Apply hard limits based on critical failure conditions.
    
    Returns:
        (adjusted_score, flags_list, override_tier_or_none)
    """
    adjusted = raw_score
    flags = []
    override_tier = None
    
    # =========================
    # RESISTANCE LIMITS
    # =========================
    if resistance >= HARD_LIMITS['resistance_recycle']:
        override_tier = "RECYCLE"
        flags.append(f"R≥{HARD_LIMITS['resistance_recycle']*1000:.0f}mΩ")
    elif resistance >= HARD_LIMITS['resistance_marginal']:
        adjusted = min(adjusted, TIER_CUTOFFS['MARGINAL'] + 8)
        flags.append(f"R>{HARD_LIMITS['resistance_marginal']*1000:.0f}mΩ")
    elif resistance >= HARD_LIMITS['resistance_economy']:
        adjusted = min(adjusted, TIER_CUTOFFS['ECONOMY'] + 8)
        flags.append(f"R>{HARD_LIMITS['resistance_economy']*1000:.0f}mΩ")
    
    # =========================
    # SOH CHANGE LIMITS
    # =========================
    if soh_change <= HARD_LIMITS['soh_change_severe']:
        adjusted = min(adjusted, TIER_CUTOFFS['MARGINAL'] + 8)
        flags.append("Severe degradation")
    elif soh_change <= HARD_LIMITS['soh_change_fast']:
        adjusted = min(adjusted, TIER_CUTOFFS['STANDARD'] + 5)
        flags.append("Fast degradation")
    
    # =========================
    # SOH ABSOLUTE LIMITS
    # =========================
    if soh <= HARD_LIMITS['soh_recycle']:
        override_tier = "RECYCLE"
        flags.append(f"SOH<{HARD_LIMITS['soh_recycle']*100:.0f}%")
    elif soh <= HARD_LIMITS['soh_marginal']:
        adjusted = min(adjusted, TIER_CUTOFFS['MARGINAL'] + 8)
        flags.append(f"SOH<{HARD_LIMITS['soh_marginal']*100:.0f}%")
    elif soh <= HARD_LIMITS['soh_economy']:
        adjusted = min(adjusted, TIER_CUTOFFS['ECONOMY'] + 8)
        flags.append(f"SOH<{HARD_LIMITS['soh_economy']*100:.0f}%")
    
    # =========================
    # MODAK DISTANCE LIMITS (NEW!)
    # =========================
    if modak_distance is not None:
        if modak_distance >= HARD_LIMITS['modak_reject']:
            override_tier = "RECYCLE"
            flags.append(f"D_M≥{HARD_LIMITS['modak_reject']:.1f} (physics failure)")
        elif modak_distance >= HARD_LIMITS['modak_severe']:
            adjusted = min(adjusted, TIER_CUTOFFS['MARGINAL'] + 8)
            flags.append(f"D_M>{HARD_LIMITS['modak_severe']:.1f} (severe anomaly)")
        elif modak_distance >= HARD_LIMITS['modak_anomaly']:
            adjusted = min(adjusted, TIER_CUTOFFS['ECONOMY'] + 8)
            flags.append(f"D_M>{HARD_LIMITS['modak_anomaly']:.1f} (anomaly)")
        elif modak_distance >= HARD_LIMITS['modak_monitor']:
            adjusted = min(adjusted, TIER_CUTOFFS['STANDARD'] + 5)
            flags.append(f"D_M>{HARD_LIMITS['modak_monitor']:.1f} (monitor)")
    
    # =========================
    # RUL SANITY CHECK
    # =========================
    # If RUL is 0 but trying to be PREMIUM, cap at ECONOMY
    # (This catches the Battery 2 issue you mentioned)
    
    return adjusted, flags, override_tier


def apply_rul_sanity_check(score, tier, rul, soh_change, flags):
    """
    Additional sanity check: batteries with RUL=0 cannot be PREMIUM.
    """
    adjusted_score = score
    adjusted_tier = tier
    
    if rul <= 0:
        if tier == "PREMIUM":
            adjusted_tier = "ECONOMY"
            adjusted_score = min(score, TIER_CUTOFFS['ECONOMY'] + 5)
            flags.append("RUL=0 (tier capped)")
        elif tier == "STANDARD":
            adjusted_tier = "ECONOMY"
            adjusted_score = min(score, TIER_CUTOFFS['ECONOMY'] + 5)
            flags.append("RUL=0 (tier capped)")
    
    # Also check if degradation is impossibly high (>1% per cycle is suspicious)
    if soh_change <= -0.01:  # -1% per cycle
        if tier in ["PREMIUM", "STANDARD"]:
            adjusted_tier = "MARGINAL"
            adjusted_score = min(score, TIER_CUTOFFS['MARGINAL'] + 5)
            flags.append("Extreme degradation rate")
    
    return adjusted_score, adjusted_tier, flags


def get_tier(score, override=None):
    """Determine tier from score using calibrated cutoffs"""
    if override:
        return override
    if score >= TIER_CUTOFFS['PREMIUM']:
        return "PREMIUM"
    if score >= TIER_CUTOFFS['STANDARD']:
        return "STANDARD"
    if score >= TIER_CUTOFFS['ECONOMY']:
        return "ECONOMY"
    if score >= TIER_CUTOFFS['MARGINAL']:
        return "MARGINAL"
    return "RECYCLE"


def get_price(tier):
    """Get price range for tier"""
    prices = {
        "PREMIUM": "₹8000-10000/kWh",
        "STANDARD": "₹5000-7000/kWh",
        "ECONOMY": "₹3000-5000/kWh",
        "MARGINAL": "₹1000-3000/kWh",
        "RECYCLE": "<₹1000/kWh"
    }
    return prices.get(tier, "N/A")


def get_anomaly_status(modak_distance):
    """Get anomaly status string from Modak distance"""
    if modak_distance is None:
        return "N/A"
    if modak_distance < 2.0:
        return "NORMAL"
    elif modak_distance < 3.0:
        return "MONITOR"
    else:
        return "ANOMALY"


def get_conclusion(tier_with, tier_without, model_impact, flags, modak_distance=None):
    """Generate conclusion message with Modak status"""
    
    # Build flag text
    if flags:
        flag_text = f" ({'; '.join(flags)})"
    else:
        flag_text = ""
    
    # Add Modak status to conclusion
    if modak_distance is not None:
        anomaly_status = get_anomaly_status(modak_distance)
        modak_text = f" [D_M={modak_distance:.2f}, {anomaly_status}]"
    else:
        modak_text = ""
    
    if tier_with == "RECYCLE":
        return f"❌ RECYCLE - Not suitable for second-life{flag_text}{modak_text}"
    elif tier_with == "MARGINAL":
        return f"⚠️ MARGINAL - Limited value, consider recycling{flag_text}{modak_text}"
    elif tier_with == tier_without:
        return f"✅ {tier_with} - Model confirms assessment{flag_text}{modak_text}"
    elif model_impact > 0:
        return f"✅ {tier_with} - Model UPGRADES from {tier_without} (+{model_impact:.1f} pts){flag_text}{modak_text}"
    else:
        return f"⚠️ {tier_with} - Model DOWNGRADES from {tier_without} ({model_impact:.1f} pts){flag_text}{modak_text}"


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process(df):
    """Process battery data and generate 5-dimensional scores"""
    results = []
    
    # Check if Modak distance columns exist
    has_modak = 'modak_distance_mean' in df.columns
    
    print(f"📋 Processing {len(df)} batteries with V5 + MODAK scorer...")
    print(f"   Scoring dimensions: 5 (SOH_change, SOH, Modak, Resistance, Usage)")
    print(f"   Modak distance available: {'✅ YES' if has_modak else '❌ NO (using 4-dim scoring)'}")
    print(f"   Tier cutoffs: PREMIUM≥{TIER_CUTOFFS['PREMIUM']}, "
          f"STANDARD≥{TIER_CUTOFFS['STANDARD']}, "
          f"ECONOMY≥{TIER_CUTOFFS['ECONOMY']}\n")
    
    for _, r in df.iterrows():
        # Extract values from inference results
        soh = r['soh_mean']
        soh_change = r['soh_change']
        rul = r['rul_mean']
        
        # Resistance (handle different column names)
        resistance = r.get('resistance_mean', 
                          r.get('internal_resistance_mean', 
                          r.get('current_resistance', 0.025)))
        
        # Modak distance (if available)
        modak_distance = r.get('modak_distance_mean', None) if has_modak else None
        
        # C-rate estimation
        current = abs(r.get('current_current', 0))
        charge_cap = r.get('charge_capacity_mean', 50)
        c_rate = current / charge_cap if charge_cap > 0 else 0.5
        
        # =========================
        # COMPONENT SCORES (0-100)
        # =========================
        
        # 1. SOH Change score (35% weight)
        soh_change_score = linear_score(
            soh_change,
            best=THRESHOLDS['soh_change']['best'],
            worst=THRESHOLDS['soh_change']['worst']
        )
        
        # 2. SOH Absolute score (25% weight)
        soh_score = linear_score(
            soh,
            best=THRESHOLDS['soh']['best'],
            worst=THRESHOLDS['soh']['worst']
        )
        
        # 3. Modak Distance score (20% weight) - NEW!
        if modak_distance is not None:
            modak_score = score_modak_distance(modak_distance)
        else:
            modak_score = 75.0  # Default if not available
        
        # 4. Resistance score (12% weight)
        resistance_score = linear_score(
            resistance,
            best=THRESHOLDS['resistance']['best'],
            worst=THRESHOLDS['resistance']['worst']
        )
        
        # 5. Usage/C-rate score (8% weight)
        usage_score = linear_score(
            c_rate,
            best=THRESHOLDS['c_rate']['best'],
            worst=THRESHOLDS['c_rate']['worst']
        )
        
        # =========================
        # WEIGHTED COMPOSITE SCORES
        # =========================
        
        # Score WITH all 5 dimensions (including model's soh_change)
        raw_with = (
            soh_change_score * WEIGHTS['soh_change'] +
            soh_score * WEIGHTS['soh'] +
            modak_score * WEIGHTS['modak_distance'] +
            resistance_score * WEIGHTS['resistance'] +
            usage_score * WEIGHTS['usage']
        )
        
        # Score WITHOUT model (4 dimensions, no soh_change, redistributed weights)
        # Redistribute soh_change weight to other components
        raw_without = (
            soh_score * 0.40 +           # 25% + 15% from soh_change
            modak_score * 0.25 +         # 20% + 5% from soh_change
            resistance_score * 0.20 +    # 12% + 8% from soh_change
            usage_score * 0.15           # 8% + 7% from soh_change
        )
        
        # =========================
        # APPLY HARD LIMITS
        # =========================
        
        adj_with, flags_with, override_with = apply_hard_limits(
            raw_with, resistance, soh_change, soh, modak_distance
        )
        
        adj_without, flags_without, override_without = apply_hard_limits(
            raw_without, resistance, soh_change, soh, modak_distance
        )
        
        # Determine initial tiers
        tier_with = get_tier(adj_with, override_with)
        tier_without = get_tier(adj_without, override_without)
        
        # =========================
        # RUL SANITY CHECK
        # =========================
        adj_with, tier_with, flags_with = apply_rul_sanity_check(
            adj_with, tier_with, rul, soh_change, flags_with
        )
        
        # Calculate model impact
        model_impact = adj_with - adj_without
        
        # Generate conclusion
        conclusion = get_conclusion(
            tier_with, tier_without, model_impact, flags_with, modak_distance
        )
        
        # =========================
        # BUILD RESULT ROW
        # =========================
        result_row = {
            'Battery': int(r['sample_id']),
            'SOH_%': round(soh * 100, 1),
            'SOH_Change_%_per_cycle': round(soh_change * 100, 4),  # Now in % per cycle
            'Resistance_mOhm': round(resistance * 1000, 1),
            'RUL_cycles': int(rul),
            'C_Rate': round(c_rate, 2),
            
            # Component scores
            'Score_SOH_Change': round(soh_change_score, 1),
            'Score_SOH': round(soh_score, 1),
            'Score_Modak': round(modak_score, 1),
            'Score_Resistance': round(resistance_score, 1),
            'Score_Usage': round(usage_score, 1),
            
            # Composite scores
            'Score_WITH_Model': round(adj_with, 1),
            'Score_WITHOUT_Model': round(adj_without, 1),
            'Model_Impact': round(model_impact, 1),
            
            # Tier and pricing
            'Tier': tier_with,
            'Price': get_price(tier_with),
            
            # Modak distance details
            'Modak_Distance': round(modak_distance, 3) if modak_distance else None,
            'Anomaly_Status': get_anomaly_status(modak_distance),
            'Physics_Compliance_%': round(modak_score, 1),
            
            # Conclusion
            'Conclusion': conclusion
        }
        
        results.append(result_row)
    
    return pd.DataFrame(results)


def print_summary(results_df):
    """Print summary statistics"""
    
    print("=" * 100)
    print("SUMMARY - 5-DIMENSIONAL SCORING")
    print("=" * 100)
    
    # Tier distribution
    print("\n📊 TIER DISTRIBUTION:")
    for tier in ['PREMIUM', 'STANDARD', 'ECONOMY', 'MARGINAL', 'RECYCLE']:
        count = sum(results_df['Tier'] == tier)
        if count > 0:
            batts = results_df[results_df['Tier'] == tier]['Battery'].tolist()
            pct = count / len(results_df) * 100
            print(f"   {tier:10s}: {count:3d} batteries ({pct:5.1f}%) - "
                  f"IDs: {batts[:10]}{'...' if len(batts) > 10 else ''}")
    
    # Score statistics
    print("\n📈 SCORE STATISTICS:")
    print(f"   Average Score WITH Model:     {results_df['Score_WITH_Model'].mean():.1f}")
    print(f"   Average Score WITHOUT Model:  {results_df['Score_WITHOUT_Model'].mean():.1f}")
    print(f"   Average Model Impact:         {results_df['Model_Impact'].mean():+.1f} points")
    
    # Component score averages
    print("\n🎯 COMPONENT SCORE AVERAGES:")
    print(f"   SOH Change:  {results_df['Score_SOH_Change'].mean():.1f}/100 (weight: {WEIGHTS['soh_change']*100:.0f}%)")
    print(f"   SOH:         {results_df['Score_SOH'].mean():.1f}/100 (weight: {WEIGHTS['soh']*100:.0f}%)")
    print(f"   Modak:       {results_df['Score_Modak'].mean():.1f}/100 (weight: {WEIGHTS['modak_distance']*100:.0f}%)")
    print(f"   Resistance:  {results_df['Score_Resistance'].mean():.1f}/100 (weight: {WEIGHTS['resistance']*100:.0f}%)")
    print(f"   Usage:       {results_df['Score_Usage'].mean():.1f}/100 (weight: {WEIGHTS['usage']*100:.0f}%)")
    
    # Modak distance analysis
    if results_df['Modak_Distance'].notna().any():
        print("\n📏 MODAK DISTANCE ANALYSIS:")
        modak_values = results_df['Modak_Distance'].dropna()
        print(f"   Mean:   {modak_values.mean():.3f}")
        print(f"   Std:    {modak_values.std():.3f}")
        print(f"   Min:    {modak_values.min():.3f}")
        print(f"   Max:    {modak_values.max():.3f}")
        
        print("\n🚨 ANOMALY STATUS DISTRIBUTION:")
        for status in ['NORMAL', 'MONITOR', 'ANOMALY']:
            count = sum(results_df['Anomaly_Status'] == status)
            pct = count / len(results_df) * 100
            print(f"   {status:10s}: {count:3d} ({pct:5.1f}%)")
        
        # Flag anomalies
        anomalies = results_df[results_df['Anomaly_Status'] == 'ANOMALY']
        if len(anomalies) > 0:
            print("\n⚠️  ANOMALOUS BATTERIES (require investigation):")
            for _, row in anomalies.iterrows():
                print(f"   Battery {row['Battery']}: D_M={row['Modak_Distance']:.2f}, "
                      f"SOH={row['SOH_%']}%, Tier={row['Tier']}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scorer_v5_modak.py <inference_results_v5_modak.csv>")
        print("\nExpected columns:")
        print("  - sample_id, soh_mean, soh_change, rul_mean")
        print("  - resistance_mean (or internal_resistance_mean)")
        print("  - modak_distance_mean (optional, enables 5th dimension)")
        sys.exit(1)
    
    # Load data
    df = pd.read_csv(sys.argv[1])
    print(f"✅ Loaded {len(df)} batteries from {sys.argv[1]}\n")
    
    # Process
    results = process(df)
    
    # Display individual results
    print("=" * 100)
    print("SECOND LIFE SCORING V5 + MODAK DISTANCE")
    print("=" * 100)
    print(f"\nSCORING WEIGHTS:")
    print(f"  • SOH Change (degradation rate):  {WEIGHTS['soh_change']*100:.0f}%")
    print(f"  • SOH (current health):           {WEIGHTS['soh']*100:.0f}%")
    print(f"  • Modak Distance (physics):       {WEIGHTS['modak_distance']*100:.0f}%  ← NEW!")
    print(f"  • Resistance (power capability):  {WEIGHTS['resistance']*100:.0f}%")
    print(f"  • Usage (C-rate stress):          {WEIGHTS['usage']*100:.0f}%")
    
    print(f"\nMODAK DISTANCE THRESHOLDS:")
    print(f"  • D_M < 2.0: NORMAL (physics-compliant)")
    print(f"  • D_M 2.0-3.0: MONITOR (slight deviation)")
    print(f"  • D_M > 3.0: ANOMALY (investigate!)")
    print(f"  • D_M > {HARD_LIMITS['modak_reject']}: RECYCLE (physics failure)\n")
    
    print("=" * 100)
    print("INDIVIDUAL BATTERY RESULTS")
    print("=" * 100)
    
    for _, r in results.iterrows():
        print(f"\n📋 Battery {r['Battery']}:")
        print(f"   Health:  SOH={r['SOH_%']}%, ΔSOH={r['SOH_Change_%_per_cycle']}%/cycle, RUL={r['RUL_cycles']} cycles")
        print(f"   Physics: R={r['Resistance_mOhm']}mΩ, C-rate={r['C_Rate']}")
        
        if r['Modak_Distance'] is not None:
            print(f"   Modak:   D_M={r['Modak_Distance']:.3f} → {r['Anomaly_Status']} "
                  f"(Physics Compliance: {r['Physics_Compliance_%']:.0f}%)")
        
        print(f"   Scores:  SOH_Chg={r['Score_SOH_Change']:.0f}, SOH={r['Score_SOH']:.0f}, "
              f"Modak={r['Score_Modak']:.0f}, R={r['Score_Resistance']:.0f}, Usage={r['Score_Usage']:.0f}")
        print(f"   Final:   WITH={r['Score_WITH_Model']:.1f}, WITHOUT={r['Score_WITHOUT_Model']:.1f}, "
              f"Impact={r['Model_Impact']:+.1f}")
        print(f"   → {r['Conclusion']}")
        print(f"   → Price: {r['Price']}")
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_file = "scores_v5_modak.csv"
    results.to_csv(output_file, index=False)
    print(f"\n💾 Saved to {output_file}")
    
    # Also save a compact version for quick review
    compact_cols = ['Battery', 'SOH_%', 'SOH_Change_%_per_cycle', 'Modak_Distance', 
                    'Anomaly_Status', 'Score_WITH_Model', 'Tier', 'Price', 'Conclusion']
    compact_df = results[[c for c in compact_cols if c in results.columns]]
    compact_file = "scores_v5_modak_compact.csv"
    compact_df.to_csv(compact_file, index=False)
    print(f"💾 Compact version saved to {compact_file}")


if __name__ == "__main__":
    main()
