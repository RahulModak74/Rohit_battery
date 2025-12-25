#!/usr/bin/env python3
"""
Realistic Li-ion Battery Data Generator - FAWKES-ALIGNED VERSION
=================================================================

FIXES FOR FAWKES DATA COMPATIBILITY:
1. Larger capacity range (40-60 Ah) to match Fawkes ~51 Ah
2. Proper SOC/DOD relationship (DOD = 1 - SOC when applicable)
3. Charge/discharge capacity = nominal capacity * SOH (not current-based)
4. More diverse calendar age (up to 1500+ days)
5. Internal resistance range aligned with Fawkes (0.025-0.040 Ω)

Author: Bayesian Cybersecurity
Version: 6.0 - Fawkes-Aligned
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FawkesAlignedBatterySimulator:
    """
    Battery simulator aligned with Fawkes real-world data characteristics.
    """
    
    def __init__(self, 
                 nominal_capacity_ah: float = 50.0,
                 num_cells_series: int = 14,
                 mfg_rated_cycles: int = 2000,
                 chemistry_type: str = "NMC",
                 mfg_date: datetime = None):
        
        self.nominal_capacity_ah = nominal_capacity_ah
        self.num_cells_series = num_cells_series
        self.chemistry_type = chemistry_type
        self.mfg_rated_cycles = mfg_rated_cycles
        self.mfg_date = mfg_date if mfg_date else datetime(2022, 1, 1)
        
        # Chemistry-specific parameters
        if chemistry_type == "LFP":
            self.base_calendar_rate = 0.0008
            self.base_cycle_rate = 1.5e-5
            self.activation_energy = 0.55
            self.cell_voltage_range = (2.8, 3.65)
            self.initial_resistance = 0.025
        else:  # NMC
            self.base_calendar_rate = 0.0012
            self.base_cycle_rate = 2.5e-5
            self.activation_energy = 0.65
            self.cell_voltage_range = (3.0, 4.2)
            self.initial_resistance = 0.028  # Fawkes has 0.0315
        
        # Scale by manufacturer rating
        rating_scale = 2000.0 / mfg_rated_cycles
        self.calendar_aging_rate = self.base_calendar_rate * rating_scale
        self.cycle_aging_rate = self.base_cycle_rate * rating_scale
        
        logger.debug(f"Battery: {chemistry_type}, {mfg_rated_cycles} cycles, {nominal_capacity_ah}Ah")
    
    def arrhenius_factor(self, temperature_celsius: float) -> float:
        T_ref = 25.0
        T_kelvin = temperature_celsius + 273.15
        T_ref_kelvin = T_ref + 273.15
        Ea = self.activation_energy
        kB = 8.617e-5
        factor = np.exp(-Ea / kB * (1/T_kelvin - 1/T_ref_kelvin))
        return np.clip(factor, 0.3, 3.0)
    
    def stress_factor(self, soc: float, c_rate: float, dod: float, temperature: float) -> float:
        soc_stress = 1.0 + max(0, (soc - 0.85)) * 3.0
        c_rate_stress = 1.0 + max(0, (c_rate - 0.5)) * 0.5
        dod_stress = 1.0 + max(0, (dod - 0.80)) * 1.5
        
        if temperature > 35:
            temp_stress = 1.0 + (temperature - 35) * 0.05
        elif temperature < 10:
            temp_stress = 1.0 + (10 - temperature) * 0.03
        else:
            temp_stress = 1.0
        
        return np.clip(soc_stress * c_rate_stress * dod_stress * temp_stress, 1.0, 5.0)
    
    def simulate_battery_life(self,
                             battery_id: int,
                             total_cycles: int = 500,
                             mean_temperature: float = 28.0,
                             mean_c_rate: float = 0.5,
                             mean_dod: float = 0.70,
                             samples_per_battery: int = 100) -> pd.DataFrame:
        """
        Simulate battery with Fawkes-aligned parameters.
        """
        
        records = []
        commissioning_date = self.mfg_date + timedelta(days=np.random.randint(30, 180))
        cycles_per_day = np.random.uniform(0.2, 1.5)  # Include low-usage like Fawkes (0.29)
        
        for sample_idx in range(samples_per_battery):
            progress = sample_idx / samples_per_battery
            cycle_count = int(progress * total_cycles)
            
            days_elapsed = cycle_count / max(cycles_per_day, 0.1)
            current_date = commissioning_date + timedelta(days=days_elapsed)
            calendar_age_days = (current_date - self.mfg_date).days
            
            # Temperature
            temperature = np.clip(np.random.normal(mean_temperature, 8), 5, 45)
            
            # ================================================================
            # SOC and DOD with PROPER RELATIONSHIP
            # ================================================================
            # SOC = current charge level (0.1 to 0.98)
            # DOD = how much was discharged in last cycle (can be independent of current SOC)
            
            # Generate SOC with realistic distribution
            soc = np.clip(np.random.beta(3, 2) * 0.85 + 0.1, 0.1, 0.98)
            
            # DOD options:
            # 1. Correlated with SOC: if SOC is high, DOD in last cycle was low
            # 2. Independent: DOD represents typical cycle depth, not current state
            
            # Use mixture approach:
            if np.random.rand() < 0.4:
                # 40% of time: DOD inversely related to current SOC
                # High SOC (0.9+) → low DOD (just charged, didn't discharge much)
                dod = np.clip(1.0 - soc + np.random.normal(0, 0.1), 0.05, 0.95)
            else:
                # 60% of time: DOD represents typical usage pattern
                dod = np.clip(np.random.beta(2, 2) * 0.7 + 0.2, 0.1, 0.95)
            
            # Special case: high SOC often means low DOD (just charged)
            if soc > 0.9:
                # 50% chance of very low DOD when highly charged
                if np.random.rand() < 0.5:
                    dod = np.random.uniform(0.05, 0.15)  # Like Fawkes: 0.073
            
            # C-rate
            c_rate_mode = np.random.choice(['slow', 'normal', 'fast', 'regen'], p=[0.3, 0.4, 0.1, 0.2])
            if c_rate_mode == 'slow':
                c_rate = np.random.uniform(0.1, 0.3)
            elif c_rate_mode == 'normal':
                c_rate = np.random.uniform(0.3, 0.8)
            elif c_rate_mode == 'fast':
                c_rate = np.random.uniform(0.8, 2.0)
            else:
                c_rate = np.random.uniform(0.2, 0.6)
            
            # Current (EV-scale)
            current = c_rate * self.nominal_capacity_ah
            if np.random.rand() > 0.45:
                current = -current
            current = current * (1 + np.random.normal(0, 0.05))
            
            # Cell voltage based on SOC
            v_min, v_max = self.cell_voltage_range
            cell_voltage = v_min + (v_max - v_min) * soc
            cell_voltage += np.random.normal(0, 0.015)
            cell_voltage = np.clip(cell_voltage, v_min, v_max)
            
            # Cell voltage variation
            imbalance = np.random.exponential(0.012)
            cell_voltage_min = cell_voltage - imbalance
            cell_voltage_max = cell_voltage + imbalance
            
            # ================================================================
            # SOH DEGRADATION
            # ================================================================
            calendar_degradation = self.calendar_aging_rate * np.sqrt(calendar_age_days + 1)
            temp_factor = self.arrhenius_factor(temperature)
            stress = self.stress_factor(soc, c_rate, dod, temperature)
            cycle_degradation = self.cycle_aging_rate * cycle_count * temp_factor * stress
            
            soh = 1.0 - (calendar_degradation + cycle_degradation)
            soh = np.clip(soh, 0.60, 1.0)
            soh_measured = np.clip(soh + np.random.normal(0, 0.005), 0.60, 1.0)
            
            # ================================================================
            # CAPACITY: Based on SOH * nominal (like Fawkes: 51.11 Ah)
            # ================================================================
            current_capacity = self.nominal_capacity_ah * soh
            
            # Charge and discharge capacity represent TOTAL available capacity
            # NOT the instantaneous current flow
            charge_capacity = current_capacity * (1 + np.random.normal(0, 0.02))
            discharge_capacity = current_capacity * (1 + np.random.normal(0, 0.02))
            
            # Ensure realistic bounds
            charge_capacity = np.clip(charge_capacity, 
                                     self.nominal_capacity_ah * 0.5, 
                                     self.nominal_capacity_ah * 1.05)
            discharge_capacity = np.clip(discharge_capacity, 
                                        self.nominal_capacity_ah * 0.5, 
                                        self.nominal_capacity_ah * 1.05)
            
            # ================================================================
            # INTERNAL RESISTANCE (Fawkes: 0.0315 Ω)
            # ================================================================
            resistance_growth = (1.0 - soh) * 2.5  # Grows with degradation
            internal_resistance = self.initial_resistance * (1 + resistance_growth)
            internal_resistance = np.clip(internal_resistance, 0.018, 0.080)
            # Add some noise
            internal_resistance += np.random.normal(0, 0.002)
            internal_resistance = np.clip(internal_resistance, 0.015, 0.100)
            
            records.append({
                'battery_id': battery_id,
                'current': round(current, 2),
                'voltage': round(cell_voltage, 4),
                'temperature': round(temperature, 1),
                'cell_voltage_min': round(cell_voltage_min, 4),
                'soc': round(soc, 4),
                'cycle_count': cycle_count,
                'internal_resistance': round(internal_resistance, 6),
                'charge_capacity': round(charge_capacity, 2),
                'discharge_capacity': round(discharge_capacity, 2),
                'c_rate': round(c_rate, 4),
                'dod': round(dod, 4),
                'soh': round(soh_measured, 4),
                'cell_voltage_max': round(cell_voltage_max, 4),
                'mfg_rated_cycles': self.mfg_rated_cycles,
                'calendar_age_days': calendar_age_days,
                'chemistry_type': 1.0 if self.chemistry_type == "LFP" else 0.0,
            })
        
        return pd.DataFrame(records)


def generate_fawkes_aligned_fleet(
    num_batteries: int = 1000,
    samples_per_battery: int = 100,
    output_file: str = 'battery_training_data_v6_fawkes_1k_batteries.csv'
) -> pd.DataFrame:
    """
    Generate fleet data aligned with Fawkes real-world characteristics.
    """
    
    logger.info("=" * 80)
    logger.info("🔋 FAWKES-ALIGNED BATTERY DATA GENERATOR v6.0")
    logger.info("=" * 80)
    logger.info(f"Generating {num_batteries} batteries × {samples_per_battery} samples")
    logger.info(f"\nKEY ALIGNMENTS WITH FAWKES DATA:")
    logger.info(f"  • Capacity: 45-60 Ah (Fawkes: ~51 Ah)")
    logger.info(f"  • Resistance: 0.018-0.080 Ω (Fawkes: 0.0315 Ω)")
    logger.info(f"  • SOC/DOD: Proper inverse relationship when high SOC")
    logger.info(f"  • Calendar age: up to 1500+ days (Fawkes: 1309 days)")
    logger.info(f"  • mfg_rated_cycles: 800-3000 (Fawkes: 1000)")
    logger.info("=" * 80)
    
    all_data = []
    
    # Archetypes weighted toward Fawkes-like batteries
    archetypes = [
        # Fawkes-like: low mfg_rated_cycles, NMC, ~50Ah
        {'name': 'fawkes_like', 'mfg_range': (800, 1200), 'chemistry': 'NMC', 
         'capacity_range': (45, 55), 'weight': 0.30},
        
        # Standard EV NMC
        {'name': 'standard_ev', 'mfg_range': (1200, 2000), 'chemistry': 'NMC', 
         'capacity_range': (40, 60), 'weight': 0.25},
        
        # Premium EV NMC
        {'name': 'premium_ev', 'mfg_range': (2000, 3000), 'chemistry': 'NMC', 
         'capacity_range': (50, 70), 'weight': 0.15},
        
        # LFP variants
        {'name': 'lfp_standard', 'mfg_range': (2000, 4000), 'chemistry': 'LFP', 
         'capacity_range': (50, 80), 'weight': 0.15},
        
        {'name': 'lfp_commercial', 'mfg_range': (4000, 6000), 'chemistry': 'LFP', 
         'capacity_range': (80, 120), 'weight': 0.15},
    ]
    
    weights = [a['weight'] for a in archetypes]
    
    for battery_id in range(1, num_batteries + 1):
        archetype = np.random.choice(archetypes, p=weights)
        
        mfg_rated_cycles = int(np.random.uniform(*archetype['mfg_range']))
        chemistry = archetype['chemistry']
        nominal_capacity = np.random.uniform(*archetype['capacity_range'])
        
        # Calendar age: include long-aged batteries like Fawkes (1309 days)
        days_ago = np.random.randint(100, 1800)
        mfg_date = datetime.now() - timedelta(days=days_ago)
        
        # Usage intensity (cycles per day)
        # Fawkes: 381 cycles / 1309 days = 0.29 cycles/day
        cycles_per_day = np.random.choice(
            [0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
            p=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05]  # Bias toward low usage
        )
        
        # Total cycles based on age and usage
        max_possible_cycles = int(days_ago * cycles_per_day)
        total_cycles = min(max_possible_cycles, int(mfg_rated_cycles * 0.8))
        total_cycles = max(50, total_cycles)
        
        # Operating conditions
        mean_temp = np.random.uniform(22, 35)
        mean_c_rate = np.random.uniform(0.2, 0.6)
        mean_dod = np.random.uniform(0.4, 0.75)
        
        simulator = FawkesAlignedBatterySimulator(
            nominal_capacity_ah=nominal_capacity,
            mfg_rated_cycles=mfg_rated_cycles,
            chemistry_type=chemistry,
            mfg_date=mfg_date
        )
        
        battery_data = simulator.simulate_battery_life(
            battery_id=battery_id,
            total_cycles=total_cycles,
            mean_temperature=mean_temp,
            mean_c_rate=mean_c_rate,
            mean_dod=mean_dod,
            samples_per_battery=samples_per_battery
        )
        
        all_data.append(battery_data)
        
        if battery_id % 30 == 0:
            final_soh = battery_data.iloc[-1]['soh']
            cal_age = battery_data.iloc[-1]['calendar_age_days']
            logger.info(f"  ✓ Battery {battery_id}/{num_batteries}: "
                       f"{archetype['name']}, {chemistry}, "
                       f"Cap={nominal_capacity:.0f}Ah, "
                       f"Age={cal_age}d, SOH={final_soh*100:.1f}%")
    
    # Combine all data
    fleet_data = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns to V5 format
    v5_columns = [
        'current', 'voltage', 'temperature', 'cell_voltage_min',
        'soc', 'cycle_count', 'internal_resistance', 'charge_capacity',
        'discharge_capacity', 'c_rate', 'dod', 'soh', 'cell_voltage_max',
        'mfg_rated_cycles', 'calendar_age_days', 'chemistry_type'
    ]
    fleet_data = fleet_data[v5_columns]
    
    # Save
    fleet_data.to_csv(output_file, index=False)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ DATA GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total samples: {len(fleet_data):,}")
    logger.info(f"Output file: {output_file}")
    
    logger.info("\n📊 DATA DISTRIBUTION vs FAWKES:")
    logger.info(f"{'Parameter':<25} {'Generated Range':<25} {'Fawkes Value':<15}")
    logger.info("-" * 65)
    logger.info(f"{'current':<25} {fleet_data['current'].min():.1f} to {fleet_data['current'].max():.1f} A{'':<5} -6.51 A")
    logger.info(f"{'voltage':<25} {fleet_data['voltage'].min():.2f} to {fleet_data['voltage'].max():.2f} V{'':<5} 4.04 V")
    logger.info(f"{'soc':<25} {fleet_data['soc'].min():.2f} to {fleet_data['soc'].max():.2f}{'':<10} 0.927")
    logger.info(f"{'dod':<25} {fleet_data['dod'].min():.2f} to {fleet_data['dod'].max():.2f}{'':<10} 0.073")
    logger.info(f"{'charge_capacity':<25} {fleet_data['charge_capacity'].min():.1f} to {fleet_data['charge_capacity'].max():.1f} Ah{'':<3} 51.11 Ah")
    logger.info(f"{'discharge_capacity':<25} {fleet_data['discharge_capacity'].min():.1f} to {fleet_data['discharge_capacity'].max():.1f} Ah{'':<3} 51.79 Ah")
    logger.info(f"{'internal_resistance':<25} {fleet_data['internal_resistance'].min():.4f} to {fleet_data['internal_resistance'].max():.4f} Ω{'':<1} 0.0315 Ω")
    logger.info(f"{'mfg_rated_cycles':<25} {fleet_data['mfg_rated_cycles'].min():.0f} to {fleet_data['mfg_rated_cycles'].max():.0f}{'':<8} 1000")
    logger.info(f"{'calendar_age_days':<25} {fleet_data['calendar_age_days'].min():.0f} to {fleet_data['calendar_age_days'].max():.0f}{'':<6} 1309")
    
    # Check SOC/DOD correlation for high SOC cases
    high_soc_data = fleet_data[fleet_data['soc'] > 0.9]
    if len(high_soc_data) > 0:
        logger.info(f"\n📈 HIGH SOC (>0.9) SAMPLES: {len(high_soc_data)}")
        logger.info(f"   DOD range when SOC>0.9: {high_soc_data['dod'].min():.3f} to {high_soc_data['dod'].max():.3f}")
        logger.info(f"   DOD mean when SOC>0.9: {high_soc_data['dod'].mean():.3f}")
        low_dod_high_soc = len(high_soc_data[high_soc_data['dod'] < 0.15])
        logger.info(f"   Samples with SOC>0.9 AND DOD<0.15: {low_dod_high_soc} ({100*low_dod_high_soc/len(high_soc_data):.1f}%)")
    
    logger.info("=" * 80)
    
    return fleet_data


def main():
    """Generate Fawkes-aligned training data"""
    
    fleet_data = generate_fawkes_aligned_fleet(
        num_batteries=1000,
        samples_per_battery=100,
        output_file='battery_training_data_v6_fawkes_1k_batteries.csv'
    )
    
    print("\n" + "=" * 80)
    print("SAMPLE DATA (first 5 rows):")
    print("=" * 80)
    print(fleet_data.head())
    print("=" * 80)


if __name__ == "__main__":
    main()
