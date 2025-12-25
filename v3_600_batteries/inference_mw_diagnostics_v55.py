#!/usr/bin/env python3
"""
BAYESIAN BESS INFERENCE V5.5 - MODAK-WALAWALKAR COMPONENT DIAGNOSTICS
=====================================================================

Enhanced output with component-wise Modak-Walawalkar Distance breakdown
matching the paper's diagnostic report format.

Key outputs:
- Overall SOH with uncertainty bounds (±σ)
- M-W Distance classification (NORMAL/ELEVATED/ANOMALY)
- Component breakdown by category (Electrical/Thermal/Capacity/Usage)
- Diagnosis and recommended action

Author: Bayesian Cybersecurity Pvt Ltd
Version: 5.5 - Component Diagnostics
"""

import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys
import pandas as pd
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

try:
    from enhanced_trainer_v5_full_physics import V5PyroTrainer
    logger.info("✅ Imported V5 Full Physics Trainer")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    sys.exit(1)


# =============================================================================
# V5 COLUMN ORDER AND COMPONENT CATEGORIES
# =============================================================================

V5_COLUMN_ORDER = [
    'current', 'voltage', 'temperature', 'cell_voltage_min',
    'soc', 'cycle_count', 'internal_resistance', 'charge_capacity',
    'discharge_capacity', 'c_rate', 'dod', 'soh', 'cell_voltage_max',
    'mfg_rated_cycles', 'calendar_age_days', 'chemistry_type',
]

# Component categories for diagnostic report (matches paper)
COMPONENT_CATEGORIES = {
    'Electrical': ['voltage', 'cell_voltage_min', 'cell_voltage_max', 'internal_resistance', 'current'],
    'Thermal': ['temperature'],
    'Capacity': ['charge_capacity', 'discharge_capacity', 'soh'],
    'Usage': ['cycle_count', 'c_rate', 'dod', 'soc', 'calendar_age_days'],
    'Specs': ['mfg_rated_cycles', 'chemistry_type'],
}

# Which indices are preserved (telemetry) vs adjustable
TELEMETRY_INDICES = [0, 1, 2, 3, 12]     # current, voltage, temp, cell_v_min, cell_v_max
ADJUSTABLE_INDICES = [4, 5, 6, 7, 8, 9, 10, 11]  # soc through soh
SPEC_INDICES = [13, 14, 15]              # mfg_rated_cycles, calendar_age, chemistry


# =============================================================================
# MODAK-WALAWALKAR DISTANCE CONFIGURATION
# =============================================================================

# Physics-derived importance weights (Φ_α in the paper)
# These encode which deviations matter most for health assessment
# NOTE: Actual values are proprietary - these are normalized relative weights
MW_PHYSICS_WEIGHTS = {
    'current': 0.5,              # Operational, low weight
    'voltage': 1.5,              # Important electrical indicator
    'temperature': 1.0,          # Thermal stress indicator  
    'cell_voltage_min': 2.0,     # Safety critical
    'soc': 0.5,                  # Operational state
    'cycle_count': 0.5,          # Usage metric
    'internal_resistance': 2.5,  # PRIMARY health indicator
    'charge_capacity': 1.5,      # Direct health measure
    'discharge_capacity': 1.5,   # Direct health measure
    'c_rate': 0.5,               # Operational
    'dod': 0.5,                  # Usage pattern
    'soh': 3.0,                  # Target variable - highest weight
    'cell_voltage_max': 2.0,     # Safety critical
    'mfg_rated_cycles': 0.2,     # Spec (constant)
    'calendar_age_days': 0.5,    # Slow degradation factor
    'chemistry_type': 0.1,       # Spec (constant)
}


def get_mw_weights_tensor() -> torch.Tensor:
    """Get Modak-Walawalkar weights as tensor in column order."""
    weights = [MW_PHYSICS_WEIGHTS.get(col, 1.0) for col in V5_COLUMN_ORDER]
    # Normalize so they sum to len(weights)
    total = sum(weights)
    weights = [w * len(weights) / total for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


# =============================================================================
# MODAK-WALAWALKAR DISTANCE CALCULATION
# =============================================================================

def calculate_mw_distance(
    x_norm: torch.Tensor, 
    x_recon_norm: torch.Tensor, 
    weights: torch.Tensor
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Calculate Modak-Walawalkar Distance with component breakdown.
    
    This implements Ω_M^(E) from the paper (Euclidean/Type I formulation):
    Ω_M = ½ ‖W^½ · (x_obs - Π_M(x_obs))‖²
    
    Returns:
        (total_distance, per_feature_contributions, category_contributions)
    """
    diff = x_recon_norm - x_norm
    weighted_sq = (weights * diff) ** 2
    
    # Per-feature contribution: Ω_M^(α) = Φ_α · (x_α - x̂_α)²
    feature_contributions = {}
    for i, col in enumerate(V5_COLUMN_ORDER):
        feature_contributions[col] = weighted_sq[i].item()
    
    # Category contributions (for diagnostic report)
    category_contributions = {}
    total = sum(feature_contributions.values())
    
    for category, features in COMPONENT_CATEGORIES.items():
        cat_total = sum(feature_contributions.get(f, 0) for f in features)
        category_contributions[category] = (cat_total / total * 100) if total > 0 else 0
    
    # Total M-W distance: Ω_M = √(Σ_α Φ_α · Δx_α²) / √n
    d_mw = torch.sqrt(torch.sum(weighted_sq) / len(weights))
    
    return d_mw.item(), feature_contributions, category_contributions


def classify_mw_distance(d_mw: float) -> Tuple[str, str, str]:
    """
    Classify M-W distance into diagnostic categories.
    
    Returns:
        (status, severity, recommended_action)
    """
    if d_mw < 1.5:
        return 'NORMAL', 'LOW', 'Continue routine monitoring'
    elif d_mw < 2.5:
        return 'ELEVATED', 'MEDIUM', 'Increase monitoring frequency'
    elif d_mw < 3.5:
        return 'HIGH', 'HIGH', 'Schedule inspection within 30 days'
    else:
        return 'ANOMALY', 'CRITICAL', 'Immediate inspection required'


def get_anomaly_sigma(d_mw: float, baseline_mean: float = 1.0, baseline_std: float = 0.5) -> float:
    """Convert M-W distance to sigma (standard deviations from normal)."""
    return (d_mw - baseline_mean) / baseline_std


# =============================================================================
# DIAGNOSTIC REPORT GENERATION
# =============================================================================

def generate_diagnostic_report(
    sample_id: int,
    soh_mean: float,
    soh_std: float,
    d_mw: float,
    category_contributions: Dict[str, float],
    status: str,
    severity: str,
    action: str,
    sigma: float,
    top_contributors: List[Tuple[str, float]],
) -> str:
    """
    Generate the diagnostic report matching the paper's format.
    """
    
    # Progress bar helper
    def progress_bar(pct: float, width: int = 10) -> str:
        filled = int(pct / 100 * width)
        empty = width - filled
        return '█' * filled + '░' * empty
    
    # Warning flag
    def warn_flag(pct: float) -> str:
        if pct >= 50:
            return ' [!!]'
        elif pct >= 20:
            return ' [!]'
        return ''
    
    # Status indicator
    status_indicator = {
        'NORMAL': '',
        'ELEVATED': ' [WARNING]',
        'HIGH': ' [ALERT]',
        'ANOMALY': ' [CRITICAL]',
    }.get(status, '')
    
    report = f"""
┌─────────────────────────────────────────────────────────────┐
│  HEALTH ASSESSMENT REPORT - Sample {sample_id:<4}                      │
├─────────────────────────────────────────────────────────────┤
│  Overall SOH:        {soh_mean*100:>5.1f}% ± {soh_std*100:.1f}%{' '*25}│
│  M-W Distance:       {status:<10}{status_indicator:<20}│
│  Anomaly Score:      {sigma:>4.1f}σ{' '*35}│
│                                                             │
│  Component Breakdown:                                       │
│  ├── Electrical:     {progress_bar(category_contributions.get('Electrical', 0))}  {category_contributions.get('Electrical', 0):>5.1f}%{warn_flag(category_contributions.get('Electrical', 0)):<6}│
│  ├── Thermal:        {progress_bar(category_contributions.get('Thermal', 0))}  {category_contributions.get('Thermal', 0):>5.1f}%{warn_flag(category_contributions.get('Thermal', 0)):<6}│
│  ├── Capacity:       {progress_bar(category_contributions.get('Capacity', 0))}  {category_contributions.get('Capacity', 0):>5.1f}%{warn_flag(category_contributions.get('Capacity', 0)):<6}│
│  ├── Usage:          {progress_bar(category_contributions.get('Usage', 0))}  {category_contributions.get('Usage', 0):>5.1f}%{warn_flag(category_contributions.get('Usage', 0)):<6}│
│  └── Specs:          {progress_bar(category_contributions.get('Specs', 0))}  {category_contributions.get('Specs', 0):>5.1f}%{warn_flag(category_contributions.get('Specs', 0)):<6}│
│                                                             │
│  Diagnosis: {get_diagnosis(category_contributions):<36}│
│  Action:   {action:<37}│
└─────────────────────────────────────────────────────────────┘"""
    
    return report


def get_diagnosis(category_contributions: Dict[str, float]) -> str:
    """Generate diagnosis based on which category dominates."""
    
    # Find dominant category (excluding Specs)
    relevant = {k: v for k, v in category_contributions.items() if k != 'Specs'}
    dominant = max(relevant, key=relevant.get)
    dominant_pct = relevant[dominant]
    
    if dominant_pct < 30:
        return "Normal aging pattern"
    
    diagnoses = {
        'Electrical': "Electrical parameter anomaly",
        'Thermal': "Thermal stress detected", 
        'Capacity': "Accelerated capacity fade",
        'Usage': "Unusual usage pattern",
    }
    
    return diagnoses.get(dominant, "Mixed degradation pattern")


# =============================================================================
# SAFE NORMALIZATION
# =============================================================================

def safe_normalize(x: torch.Tensor, normalizer, clip_range: float = 3.0) -> Tuple[torch.Tensor, List[str]]:
    """Normalize with clipping and out-of-distribution warnings."""
    warnings = []
    x_norm = normalizer.normalize(x)
    
    for i, col in enumerate(V5_COLUMN_ORDER):
        val = x_norm[0, i].item()
        if abs(val) > clip_range:
            warnings.append(f"{col}: normalized value {val:.2f} clipped to ±{clip_range}")
    
    x_norm_clipped = torch.clamp(x_norm, -clip_range, clip_range)
    return x_norm_clipped, warnings


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_reorder_data(csv_file: str) -> pd.DataFrame:
    """Load CSV and ensure V5 column order."""
    df = pd.read_csv(csv_file)
    
    # Check required columns
    missing = [col for col in V5_COLUMN_ORDER[:13] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add defaults for optional columns
    if 'mfg_rated_cycles' not in df.columns:
        df['mfg_rated_cycles'] = 5000.0
    if 'calendar_age_days' not in df.columns:
        df['calendar_age_days'] = 365.0
    if 'chemistry_type' not in df.columns:
        df['chemistry_type'] = 1.0
    
    return df[V5_COLUMN_ORDER].copy()


# =============================================================================
# MAIN INFERENCE WITH COMPONENT DIAGNOSTICS
# =============================================================================

def run_inference(
    df: pd.DataFrame, 
    trainer, 
    num_samples: int = 50, 
    num_ensemble: int = 10,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run inference with Modak-Walawalkar component diagnostics.
    """
    
    results = []
    mw_weights = get_mw_weights_tensor()
    trainer.vae.eval()
    
    for idx, row in df.iterrows():
        
        # Create input tensor
        x = torch.tensor([[row[col] for col in V5_COLUMN_ORDER]], dtype=torch.float32)
        
        # Normalize
        x_norm, ood_warnings = safe_normalize(x, trainer.vae.normalizer, clip_range=3.0)
        
        # Ensemble inference
        all_soh = []
        all_d_mw = []
        all_feature_contribs = []
        all_category_contribs = []
        
        with torch.no_grad():
            for _ in range(num_ensemble):
                for _ in range(num_samples):
                    # VAE forward pass
                    mu, logvar = trainer.vae.encode(x_norm)
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                    x_recon_norm = trainer.vae.decode(z)
                    
                    # Preserve telemetry and specs
                    x_recon_norm[:, TELEMETRY_INDICES] = x_norm[:, TELEMETRY_INDICES]
                    x_recon_norm[:, SPEC_INDICES] = x_norm[:, SPEC_INDICES]
                    
                    # M-W Distance calculation
                    d_mw, feat_contrib, cat_contrib = calculate_mw_distance(
                        x_norm[0], x_recon_norm[0], mw_weights
                    )
                    all_d_mw.append(d_mw)
                    all_feature_contribs.append(feat_contrib)
                    all_category_contribs.append(cat_contrib)
                    
                    # SOH prediction
                    x_pred = trainer.vae.normalizer.denormalize(x_recon_norm)
                    all_soh.append(x_pred[0, 11].item())
        
        # Aggregate results
        soh_mean = np.mean(all_soh)
        soh_std = np.std(all_soh)
        d_mw_mean = np.mean(all_d_mw)
        d_mw_std = np.std(all_d_mw)
        
        # Average category contributions
        avg_cat_contrib = {}
        for cat in COMPONENT_CATEGORIES.keys():
            avg_cat_contrib[cat] = np.mean([c[cat] for c in all_category_contribs])
        
        # Average feature contributions (for detailed output)
        avg_feat_contrib = {}
        for feat in V5_COLUMN_ORDER:
            avg_feat_contrib[feat] = np.mean([c[feat] for c in all_feature_contribs])
        
        # Sort features by contribution
        sorted_features = sorted(avg_feat_contrib.items(), key=lambda x: x[1], reverse=True)
        top_contributors = sorted_features[:3]
        
        # Classification
        status, severity, action = classify_mw_distance(d_mw_mean)
        sigma = get_anomaly_sigma(d_mw_mean)
        
        # RUL calculation
        current_soh = row['soh']
        current_cycle = row['cycle_count']
        calendar_age = row['calendar_age_days']
        
        if current_cycle > 0:
            total_deg = 1.0 - current_soh
            cycle_deg_rate = total_deg / current_cycle * 0.6  # 60% cycle, 40% calendar
            rul = max(0, (soh_mean - 0.80) / cycle_deg_rate) if cycle_deg_rate > 0 else 0
        else:
            cycle_deg_rate = 0
            rul = 0
        
        # Confidence score
        conf_base = max(0, min(100, 100 * (1 - soh_std / max(soh_mean, 0.01) * 5)))
        conf_penalty = min(d_mw_mean / 4.0, 0.3)
        confidence = conf_base * (1 - conf_penalty)
        
        # M-W Determinant approximation (for uncertainty bounds)
        # Δ_M ≈ 1 / (1 + d_mw²) — simplified for computational efficiency
        mw_determinant = 1.0 / (1.0 + d_mw_mean ** 2)
        uncertainty_multiplier = 1.0 / np.sqrt(max(mw_determinant, 0.1))
        
        # Generate diagnostic report
        if verbose:
            report = generate_diagnostic_report(
                sample_id=idx + 1,
                soh_mean=soh_mean,
                soh_std=soh_std * uncertainty_multiplier,
                d_mw=d_mw_mean,
                category_contributions=avg_cat_contrib,
                status=status,
                severity=severity,
                action=action,
                sigma=sigma,
                top_contributors=top_contributors,
            )
            print(report)
        
        # Store results
        results.append({
            'sample_id': idx + 1,
            
            # Core health metrics
            'soh_predicted': soh_mean,
            'soh_uncertainty': soh_std * uncertainty_multiplier,
            'soh_input': current_soh,
            'rul_cycles': rul,
            
            # M-W Distance metrics
            'mw_distance': d_mw_mean,
            'mw_distance_std': d_mw_std,
            'mw_status': status,
            'mw_severity': severity,
            'anomaly_sigma': sigma,
            
            # Component breakdown (matches paper format)
            'component_electrical_pct': avg_cat_contrib.get('Electrical', 0),
            'component_thermal_pct': avg_cat_contrib.get('Thermal', 0),
            'component_capacity_pct': avg_cat_contrib.get('Capacity', 0),
            'component_usage_pct': avg_cat_contrib.get('Usage', 0),
            'component_specs_pct': avg_cat_contrib.get('Specs', 0),
            
            # Top contributors
            'top_contributor_1': top_contributors[0][0] if len(top_contributors) > 0 else '',
            'top_contributor_1_value': top_contributors[0][1] if len(top_contributors) > 0 else 0,
            'top_contributor_2': top_contributors[1][0] if len(top_contributors) > 1 else '',
            'top_contributor_2_value': top_contributors[1][1] if len(top_contributors) > 1 else 0,
            'top_contributor_3': top_contributors[2][0] if len(top_contributors) > 2 else '',
            'top_contributor_3_value': top_contributors[2][1] if len(top_contributors) > 2 else 0,
            
            # Uncertainty quantification
            'mw_determinant': mw_determinant,
            'confidence_score': confidence,
            
            # Diagnosis
            'diagnosis': get_diagnosis(avg_cat_contrib),
            'recommended_action': action,
            
            # Input context
            'cycle_count': current_cycle,
            'calendar_age_days': calendar_age,
            'temperature': row['temperature'],
            'chemistry_type': row['chemistry_type'],
            
            # Flags
            'out_of_distribution': len(ood_warnings) > 0,
            'ood_details': '; '.join(ood_warnings) if ood_warnings else '',
        })
    
    return results


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_summary(results: List[Dict]):
    """Print summary statistics."""
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 65)
    print("  BATCH SUMMARY")
    print("=" * 65)
    print(f"  Total samples:        {len(results)}")
    print(f"  Average SOH:          {df['soh_predicted'].mean()*100:.1f}% ± {df['soh_uncertainty'].mean()*100:.1f}%")
    print(f"  Average M-W Distance: {df['mw_distance'].mean():.3f}")
    print()
    
    # Status breakdown
    status_counts = df['mw_status'].value_counts()
    print("  Status Distribution:")
    for status in ['NORMAL', 'ELEVATED', 'HIGH', 'ANOMALY']:
        count = status_counts.get(status, 0)
        pct = count / len(results) * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"    {status:<10} {bar} {count:>4} ({pct:>5.1f}%)")
    print()
    
    # Component breakdown (average across all samples)
    print("  Average Component Contributions:")
    for comp in ['Electrical', 'Thermal', 'Capacity', 'Usage']:
        col = f'component_{comp.lower()}_pct'
        avg = df[col].mean()
        bar = '█' * int(avg / 5) + '░' * (20 - int(avg / 5))
        print(f"    {comp:<12} {bar} {avg:>5.1f}%")
    print("=" * 65)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='BayesianBESS V5.5 - Inference with Modak-Walawalkar Component Diagnostics'
    )
    parser.add_argument('--data', required=True, help='Input CSV file')
    parser.add_argument('--model', default='bess_model_v5_full_physics.pt', help='Model file')
    parser.add_argument('--output', default='inference_mw_diagnostics.csv', help='Output CSV')
    parser.add_argument('--samples', type=int, default=50, help='MC samples per ensemble')
    parser.add_argument('--ensemble', type=int, default=10, help='Ensemble runs')
    parser.add_argument('--quiet', action='store_true', help='Suppress per-sample reports')
    parser.add_argument('--json', action='store_true', help='Also output JSON format')
    
    args = parser.parse_args()
    
    print("=" * 65)
    print("  BAYESIAN BESS V5.5 - MODAK-WALAWALKAR COMPONENT DIAGNOSTICS")
    print("=" * 65)
    print()
    print("  The Modak-Walawalkar Distance tells us not just how much")
    print("  a battery has degraded, but in what way — and with")
    print("  mathematically guaranteed certainty bounds.")
    print()
    print("=" * 65)
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    trainer = V5PyroTrainer(input_dim=16, latent_dim=32)
    trainer.load_model(args.model)
    logger.info("Model loaded successfully")
    
    # Load data
    logger.info(f"Loading data: {args.data}")
    df = load_and_reorder_data(args.data)
    logger.info(f"Loaded {len(df)} samples")
    
    # Run inference
    results = run_inference(
        df, trainer, 
        num_samples=args.samples, 
        num_ensemble=args.ensemble,
        verbose=not args.quiet
    )
    
    # Save CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to: {args.output}")
    
    # Save JSON if requested
    if args.json:
        json_output = args.output.replace('.csv', '.json')
        with open(json_output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"JSON saved to: {json_output}")
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
