#!/usr/bin/env python3
"""
BAYESIAN BESS INFERENCE V5 - WITH MODAK DISTANCE DEBUGGING
===========================================================

Added detailed per-feature Modak distance breakdown to identify
which features contribute most to reconstruction error.

Author: Bayesian Cybersecurity
Version: 5.4 - With Modak Debugging
"""

import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

try:
    from enhanced_trainer_v5_full_physics import V5PyroTrainer
    logger.info("✅ Imported V5 Full Physics Trainer")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    sys.exit(1)


# =============================================================================
# V5 COLUMN ORDER
# =============================================================================

V5_COLUMN_ORDER = [
    'current', 'voltage', 'temperature', 'cell_voltage_min',
    'soc', 'cycle_count', 'internal_resistance', 'charge_capacity',
    'discharge_capacity', 'c_rate', 'dod', 'soh', 'cell_voltage_max',
    'mfg_rated_cycles', 'calendar_age_days', 'chemistry_type',
]

# Feature categories for debugging
TELEMETRY_INDICES = [0, 1, 2, 3, 12]     # current, voltage, temp, cell_v_min, cell_v_max
ADJUSTABLE_INDICES = [4, 5, 6, 7, 8, 9, 10, 11]  # soc through soh (NOT cell_v_max!)
SPEC_INDICES = [13, 14, 15]              # mfg_rated_cycles, calendar_age, chemistry

# =============================================================================
# MODAK DISTANCE CONFIG
# =============================================================================

MODAK_SCALES = {
    'current': 0.1, 'voltage': 0.1, 'temperature': 0.1, 'cell_voltage_min': 0.1,
    'soc': 0.25, 'cycle_count': 0.25, 'internal_resistance': 0.25,
    'charge_capacity': 0.25, 'discharge_capacity': 0.25, 'c_rate': 0.25,
    'dod': 0.25, 'soh': 0.2, 'cell_voltage_max': 0.1,
    'mfg_rated_cycles': 0.1, 'calendar_age_days': 0.1, 'chemistry_type': 0.1,
}


def get_modak_weights() -> torch.Tensor:
    weights = [1.0 / MODAK_SCALES.get(col, 0.25) for col in V5_COLUMN_ORDER]
    return torch.tensor(weights, dtype=torch.float32)


def calculate_modak_distance_with_breakdown(
    x_norm: torch.Tensor, 
    x_recon_norm: torch.Tensor, 
    weights: torch.Tensor
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate Modak distance with per-feature breakdown.
    
    Returns:
        (total_distance, {feature_name: contribution})
    """
    diff = x_recon_norm - x_norm
    weighted_sq = (weights * diff) ** 2
    
    # Per-feature contribution
    contributions = {}
    for i, col in enumerate(V5_COLUMN_ORDER):
        contributions[col] = weighted_sq[i].item()
    
    # Total distance
    d_m = torch.sqrt(torch.sum(weighted_sq) / len(weights))
    
    return d_m.item(), contributions


def classify_anomaly(d_m: float) -> Tuple[str, int, float]:
    if d_m < 1.5:
        return 'NORMAL', 0, max(0, 100 * (1 - d_m / 4))
    elif d_m < 2.5:
        return 'MONITOR', 1, max(0, 100 * (1 - d_m / 4))
    else:
        return 'ANOMALY', 2, max(0, 100 * (1 - d_m / 4))


# =============================================================================
# SAFE NORMALIZATION WITH CLIPPING
# =============================================================================

def safe_normalize(x: torch.Tensor, normalizer, clip_range: float = 3.0) -> Tuple[torch.Tensor, List[str]]:
    """Normalize and clip to prevent out-of-distribution explosions."""
    warnings = []
    x_norm = normalizer.normalize(x)
    
    for i, col in enumerate(V5_COLUMN_ORDER):
        val = x_norm[0, i].item()
        if abs(val) > clip_range:
            warnings.append(f"⚠️ {col}={x[0,i].item():.3f} → normalized={val:.2f} (CLIPPED to ±{clip_range})")
    
    x_norm_clipped = torch.clamp(x_norm, -clip_range, clip_range)
    return x_norm_clipped, warnings


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_reorder_data(csv_file: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load CSV and reorder columns to V5 order."""
    df = pd.read_csv(csv_file)
    warnings = []
    
    missing = [col for col in V5_COLUMN_ORDER[:13] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    if 'mfg_rated_cycles' not in df.columns:
        df['mfg_rated_cycles'] = 5000.0
    if 'calendar_age_days' not in df.columns:
        df['calendar_age_days'] = 365.0
    if 'chemistry_type' not in df.columns:
        df['chemistry_type'] = 1.0
    
    df_reordered = df[V5_COLUMN_ORDER].copy()
    return df_reordered, warnings


# =============================================================================
# INFERENCE WITH MODAK DEBUGGING
# =============================================================================

def run_inference_with_debug(df: pd.DataFrame, trainer, num_samples: int = 50, num_ensemble: int = 10) -> List[Dict]:
    """Run inference with detailed Modak distance debugging."""
    
    results = []
    modak_weights = get_modak_weights()
    trainer.vae.eval()
    
    for idx, row in df.iterrows():
        logger.info(f"\n{'='*70}")
        logger.info(f"SAMPLE {idx + 1}: SOH={row['soh']*100:.1f}%, Cycles={row['cycle_count']:.0f}")
        logger.info(f"{'='*70}")
        
        # Create input tensor
        x = torch.tensor([[
            row['current'], row['voltage'], row['temperature'], row['cell_voltage_min'],
            row['soc'], row['cycle_count'], row['internal_resistance'], row['charge_capacity'],
            row['discharge_capacity'], row['c_rate'], row['dod'], row['soh'], row['cell_voltage_max'],
            row['mfg_rated_cycles'], row['calendar_age_days'], row['chemistry_type'],
        ]], dtype=torch.float32)
        
        # Safe normalization
        x_norm, norm_warnings = safe_normalize(x, trainer.vae.normalizer, clip_range=3.0)
        
        if norm_warnings:
            logger.warning("   Normalization warnings:")
            for w in norm_warnings:
                logger.warning(f"      {w}")
        
        # =====================================================================
        # DETAILED INPUT vs NORMALIZED COMPARISON
        # =====================================================================
        logger.info(f"\n   📊 INPUT VALUES (Raw → Normalized):")
        logger.info(f"   {'Feature':<22} {'Raw':>12} {'Normalized':>12}")
        logger.info(f"   {'-'*50}")
        for i, col in enumerate(V5_COLUMN_ORDER):
            raw_val = x[0, i].item()
            norm_val = x_norm[0, i].item()
            flag = " ⚠️" if abs(norm_val) > 2.5 else ""
            logger.info(f"   {col:<22} {raw_val:>12.4f} {norm_val:>12.4f}{flag}")
        
        # =====================================================================
        # ENSEMBLE INFERENCE WITH RECONSTRUCTION TRACKING
        # =====================================================================
        all_soh = []
        all_modak = []
        all_contributions = []
        all_recon_diffs = []
        
        with torch.no_grad():
            for ensemble_idx in range(num_ensemble):
                for sample_idx in range(num_samples):
                    # Encode
                    mu, logvar = trainer.vae.encode(x_norm)
                    std = torch.exp(0.5 * logvar)
                    z = mu + std * torch.randn_like(std)
                    
                    # Decode
                    x_recon_norm = trainer.vae.decode(z)
                    
                    # PRESERVE telemetry (0-3), cell_voltage_max (12), and specs (13-15)
                    x_recon_norm[:, 0:4] = x_norm[:, 0:4]      # current, voltage, temp, cell_v_min
                    x_recon_norm[:, 12] = x_norm[:, 12]        # cell_voltage_max (TELEMETRY!)
                    x_recon_norm[:, 13:16] = x_norm[:, 13:16]  # specs
                    
                    # Modak distance with breakdown
                    d_m, contributions = calculate_modak_distance_with_breakdown(
                        x_norm[0], x_recon_norm[0], modak_weights
                    )
                    all_modak.append(d_m)
                    all_contributions.append(contributions)
                    
                    # Track reconstruction differences for adjustable features
                    recon_diff = {}
                    for i in ADJUSTABLE_INDICES:
                        col = V5_COLUMN_ORDER[i]
                        recon_diff[col] = (x_recon_norm[0, i] - x_norm[0, i]).item()
                    all_recon_diffs.append(recon_diff)
                    
                    # SOH prediction
                    x_pred = trainer.vae.normalizer.denormalize(x_recon_norm)
                    all_soh.append(x_pred[0, 11].item())
        
        # =====================================================================
        # MODAK DISTANCE BREAKDOWN ANALYSIS
        # =====================================================================
        
        # Average contributions across all samples
        avg_contributions = {}
        for col in V5_COLUMN_ORDER:
            avg_contributions[col] = np.mean([c[col] for c in all_contributions])
        
        # Average reconstruction differences
        avg_recon_diffs = {}
        for col in [V5_COLUMN_ORDER[i] for i in ADJUSTABLE_INDICES]:
            avg_recon_diffs[col] = np.mean([d[col] for d in all_recon_diffs])
        
        # Sort by contribution (highest first)
        sorted_contributions = sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"\n   🔍 MODAK DISTANCE BREAKDOWN (Per-Feature Contribution):")
        logger.info(f"   {'Feature':<22} {'Contribution':>12} {'% of Total':>12} {'Recon Diff':>12}")
        logger.info(f"   {'-'*60}")
        
        total_contribution = sum(avg_contributions.values())
        top_contributors = []
        
        for col, contrib in sorted_contributions:
            pct = (contrib / total_contribution * 100) if total_contribution > 0 else 0
            recon_diff = avg_recon_diffs.get(col, 0)
            
            # Flag high contributors
            flag = ""
            if pct > 20:
                flag = " 🔴 HIGH"
                top_contributors.append((col, contrib, pct))
            elif pct > 10:
                flag = " 🟡 MED"
            
            logger.info(f"   {col:<22} {contrib:>12.4f} {pct:>11.1f}% {recon_diff:>+12.4f}{flag}")
        
        # =====================================================================
        # DIAGNOSIS
        # =====================================================================
        
        modak_mean = np.mean(all_modak)
        modak_std = np.std(all_modak)
        soh_mean = np.mean(all_soh)
        soh_std = np.std(all_soh)
        
        logger.info(f"\n   📏 TOTAL MODAK DISTANCE: {modak_mean:.3f} ± {modak_std:.3f}")
        
        if top_contributors:
            logger.info(f"\n   🚨 TOP CONTRIBUTORS TO HIGH MODAK DISTANCE:")
            for col, contrib, pct in top_contributors:
                input_val = x[0, V5_COLUMN_ORDER.index(col)].item()
                norm_val = x_norm[0, V5_COLUMN_ORDER.index(col)].item()
                recon_diff = avg_recon_diffs.get(col, 0)
                
                logger.info(f"      • {col}: {pct:.1f}% of distance")
                logger.info(f"        Input: {input_val:.4f} | Normalized: {norm_val:.4f}")
                logger.info(f"        Reconstruction diff: {recon_diff:+.4f} (in normalized space)")
                
                # Suggest fix
                if col == 'charge_capacity' or col == 'discharge_capacity':
                    logger.info(f"        💡 FIX: Check if capacity values match training data range")
                elif col == 'cycle_count':
                    logger.info(f"        💡 FIX: Cycle count may be outside training distribution")
                elif col == 'c_rate':
                    logger.info(f"        💡 FIX: C-rate value may need adjustment")
                elif col == 'dod':
                    logger.info(f"        💡 FIX: DOD value may be unusual for training data")
        
        # =====================================================================
        # RUL CALCULATION (with calendar/cycle split)
        # =====================================================================
        
        current_soh = row['soh']
        current_cycle = row['cycle_count']
        calendar_age_days = row['calendar_age_days']
        total_degradation = 1.0 - current_soh
        
        if current_cycle > 0 and calendar_age_days > 0:
            cycles_per_day = current_cycle / calendar_age_days
            
            if cycles_per_day < 0.3:
                calendar_fraction = 0.70
            elif cycles_per_day < 0.7:
                calendar_fraction = 0.55
            elif cycles_per_day < 1.2:
                calendar_fraction = 0.40
            else:
                calendar_fraction = 0.25
            
            cycle_degradation = total_degradation * (1 - calendar_fraction)
            cycle_deg_rate = cycle_degradation / current_cycle
            
            logger.info(f"\n   📊 Degradation split: {calendar_fraction*100:.0f}% calendar, {(1-calendar_fraction)*100:.0f}% cycle")
            logger.info(f"      Cycle deg rate: {cycle_deg_rate*100:.4f}%/cycle")
        else:
            cycle_deg_rate = 2e-5
            calendar_fraction = 0.5
        
        soh_change = -cycle_deg_rate
        
        eol = 0.80
        if soh_mean > eol and cycle_deg_rate > 0:
            rul = (soh_mean - eol) / cycle_deg_rate
        else:
            rul = 0
        
        # Classification
        anomaly_status, anomaly_score, physics_compliance = classify_anomaly(modak_mean)
        
        # Confidence
        conf_raw = max(0, min(100, 100 * (1 - soh_std / max(soh_mean, 0.01) * 5)))
        modak_penalty = min(modak_mean / 3.0, 0.3)
        conf_adjusted = conf_raw * (1 - modak_penalty)
        
        # Final summary
        logger.info(f"\n   {'='*50}")
        logger.info(f"   FINAL RESULTS:")
        logger.info(f"   {'='*50}")
        logger.info(f"   SOH predicted: {soh_mean*100:.2f}% ± {soh_std*100:.2f}%")
        logger.info(f"   RUL: {rul:.0f} cycles")
        logger.info(f"   Modak Distance: {modak_mean:.3f} → {anomaly_status}")
        logger.info(f"   Physics Compliance: {physics_compliance:.1f}%")
        
        results.append({
            'sample_id': idx + 1,
            'operation_mode': 'charging' if row['current'] > 0 else 'discharging',
            'current_soh': current_soh,
            'soh_mean': soh_mean,
            'soh_std': soh_std,
            'soh_change': soh_change,
            'rul_mean': rul,
            'modak_distance_mean': modak_mean,
            'modak_distance_std': modak_std,
            'anomaly_status': anomaly_status,
            'anomaly_score': anomaly_score,
            'physics_compliance_score': physics_compliance,
            'confidence_score': conf_adjusted,
            'confidence_score_raw': conf_raw,
            'confidence_penalty_modak': modak_penalty * 100,
            'resistance_mean': row['internal_resistance'],
            'current_cycle': current_cycle,
            'current_current': row['current'],
            'current_voltage': row['voltage'],
            'current_temperature': row['temperature'],
            'chemistry_type': row['chemistry_type'],
            'mfg_rated_cycles': row['mfg_rated_cycles'],
            'calendar_age_days': calendar_age_days,
            'cycles_per_day': current_cycle / calendar_age_days if calendar_age_days > 0 else 0,
            'calendar_aging_fraction': calendar_fraction,
            'cycle_deg_rate_per_cycle': cycle_deg_rate,
            # Add top contributors for debugging
            'top_modak_contributor_1': sorted_contributions[0][0] if sorted_contributions else '',
            'top_modak_contributor_1_pct': (sorted_contributions[0][1] / total_contribution * 100) if sorted_contributions and total_contribution > 0 else 0,
            'top_modak_contributor_2': sorted_contributions[1][0] if len(sorted_contributions) > 1 else '',
            'top_modak_contributor_2_pct': (sorted_contributions[1][1] / total_contribution * 100) if len(sorted_contributions) > 1 and total_contribution > 0 else 0,
            'out_of_distribution': len(norm_warnings) > 0,
            'ood_warnings': '; '.join(norm_warnings) if norm_warnings else '',
        })
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BayesianBESS V5 - Inference with Modak Debugging')
    parser.add_argument('--data', required=True, help='Input CSV')
    parser.add_argument('--model', default='bess_model_v5_full_physics.pt')
    parser.add_argument('--output', default='inference_results_v5_modak.csv')
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--ensemble', type=int, default=10)
    parser.add_argument('--clip', type=float, default=3.0)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("   BAYESIAN BESS V5 - INFERENCE WITH MODAK DEBUGGING")
    print("=" * 70)
    
    # Load model
    logger.info(f"📂 Loading model: {args.model}")
    trainer = V5PyroTrainer(input_dim=16, latent_dim=32)
    trainer.load_model(args.model)
    logger.info("✅ Model loaded")
    
    # Show training data statistics
    logger.info(f"\n📊 TRAINING DATA STATISTICS (from normalizer):")
    logger.info(f"   {'Feature':<22} {'Mean':>12} {'Std':>12}")
    logger.info(f"   {'-'*50}")
    for i, col in enumerate(V5_COLUMN_ORDER):
        mean = trainer.vae.normalizer.means[i].item()
        std = trainer.vae.normalizer.stds[i].item()
        logger.info(f"   {col:<22} {mean:>12.4f} {std:>12.4f}")
    
    # Load data
    logger.info(f"\n📂 Loading data: {args.data}")
    df, _ = load_and_reorder_data(args.data)
    logger.info(f"✅ Loaded {len(df)} samples")
    
    # Run inference with debugging
    results = run_inference_with_debug(df, trainer, args.samples, args.ensemble)
    
    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    logger.info(f"\n💾 Results saved to: {args.output}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"   Samples processed: {len(results)}")
    print(f"   Avg SOH: {results_df['soh_mean'].mean()*100:.1f}%")
    print(f"   Avg Modak Distance: {results_df['modak_distance_mean'].mean():.3f}")


if __name__ == "__main__":
    main()
