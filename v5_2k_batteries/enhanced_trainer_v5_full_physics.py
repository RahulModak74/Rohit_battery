#!/usr/bin/env python3
"""
Physics-Informed Deep BESS - V5 ULTIMATE COMBINED PHYSICS
==========================================================

COMBINES BEST OF BOTH V3 VERSIONS:
1. SEI growth physics (sqrt-t law, thickness, swelling)
2. Mechanical degradation (stress accumulation, particle cracking)
3. Charge/discharge differentiation (separate aging coefficients)
4. Chemistry-dependent parameters (LFP vs NMC)
5. Arrhenius temperature acceleration
6. Calendar aging (sqrt-time law)
7. Lithium plating risk (low temp + high C-rate charging)

V5 BAYESIAN PRIORS (15 total - Most Advanced!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  chemistry_prob              - Chemistry probability (Beta distribution)
2.  chemistry_type              - LFP vs NMC (Bernoulli)
3.  log_activation_energy       - Arrhenius temperature activation energy
4.  calendar_aging_coefficient  - Calendar aging rate (sqrt-time)
5.  sei_growth_coefficient      - SEI layer growth rate (sqrt-cycles)
6.  cycle_aging_coefficient_charge      - Cycle aging during charging
7.  cycle_aging_coefficient_discharge   - Cycle aging during discharging
8.  c_rate_stress_charge        - C-rate stress for charging
9.  c_rate_stress_discharge     - C-rate stress for discharging
10. soc_stress                  - High SOC voltage stress
11. dod_stress                  - Deep discharge stress
12. plating_risk_coefficient    - Lithium plating (charge-specific)
13. swelling_coefficient        - SEI volume expansion
14. stress_accumulation_rate    - Mechanical stress buildup
15. particle_cracking_threshold - Critical stress for particle cracking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

V5 TELEMETRY STRUCTURE (16 features):
┌─────────────────────────────────────────────────────────────────────┐
│ TELEMETRY (σ=0.01, preserved):     Indices 0-3                     │
│   [0] current            - Direct measurement (A)                   │
│   [1] voltage            - Direct measurement (V)                   │
│   [2] temperature        - Direct measurement (°C)                  │
│   [3] cell_voltage_min   - Direct measurement (V)                   │
├─────────────────────────────────────────────────────────────────────┤
│ ADJUSTABLE (σ=0.15, inferred):     Indices 4-12                    │
│   [4] soc                - State of Charge                          │
│   [5] cycle_count        - Total cycles                             │
│   [6] internal_resistance - Ohms                                    │
│   [7] charge_capacity    - Ah                                       │
│   [8] discharge_capacity - Ah                                       │
│   [9] c_rate             - C-rate                                   │
│   [10] dod               - Depth of Discharge                       │
│   [11] soh               - State of Health (TARGET)                 │
│   [12] cell_voltage_max  - Maximum cell voltage                     │
├─────────────────────────────────────────────────────────────────────┤
│ SPECS (σ=0.001, exact):            Indices 13-15                   │
│   [13] mfg_rated_cycles  - Manufacturer rating                      │
│   [14] calendar_age_days - Days since manufacturing                 │
│   [15] chemistry_type    - 1.0=LFP, 0.0=NMC                        │
└─────────────────────────────────────────────────────────────────────┘

Author: Bayesian Cybersecurity
Version: 5.0 - Ultimate Combined Physics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import argparse
import time
from typing import Dict, List, Tuple, Optional

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import ClippedAdam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU setup
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    logger.info("🚀 Using CUDA GPU")
else:
    device = torch.device('cpu')
    logger.info("💻 Using CPU")


# ============================================================================
# PHYSICS CONSTANTS (Combined from both V3 versions)
# ============================================================================

class PhysicsConstants:
    """Battery degradation physics constants - COMBINED"""
    
    # Reference conditions
    REFERENCE_TEMP = 298.15  # K (25°C)
    REFERENCE_MFG_RATING = 5000.0  # cycles
    
    # Boltzmann constant (eV/K)
    BOLTZMANN_CONSTANT = 8.617333e-5
    
    # Chemistry-specific parameters (LFP vs NMC)
    # LFP (Lithium Iron Phosphate)
    LFP_ACTIVATION_ENERGY = 0.45  # eV
    LFP_CALENDAR_BASE = 2e-5
    LFP_CYCLE_BASE = 5e-6
    
    # NMC (Nickel Manganese Cobalt)
    NMC_ACTIVATION_ENERGY = 0.55  # eV
    NMC_CALENDAR_BASE = 3e-5
    NMC_CYCLE_BASE = 7e-6
    
    # Safe operating ranges
    SAFE_C_RATE_CHARGE = 0.7
    SAFE_C_RATE_DISCHARGE = 1.0
    
    # SEI growth parameters (from train_battery_model_v3)
    SEI_GROWTH_RATE = 1.5  # nm per sqrt(cycles)
    SEI_VOLUME_EXPANSION = 0.035  # 3.5% volume increase per 10 nm SEI
    MAX_SWELLING_STRAIN = 0.15  # 15% maximum volume expansion
    
    # Mechanical stress parameters (from train_battery_model_v3)
    CYCLE_STRESS_COEFFICIENT = 0.02  # Stress per cycle
    CRITICAL_STRESS = 1.5  # GPa - particle cracking threshold
    
    # Lithium plating threshold
    LITHIUM_PLATING_THRESHOLD_TEMP = 15  # °C


# ============================================================================
# DATA NORMALIZER
# ============================================================================

class DataNormalizer:
    """Normalize data with feature-specific handling"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.means = None
        self.stds = None
    
    def fit(self, data: torch.Tensor):
        """Fit normalizer to data"""
        self.means = data.mean(dim=0)
        self.stds = data.std(dim=0) + 1e-6
        return self
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data"""
        return (data - self.means.to(data.device)) / self.stds.to(data.device)
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data"""
        return data * self.stds.to(data.device) + self.means.to(data.device)


# ============================================================================
# V5 ULTIMATE BAYESIAN PHYSICS (15 PRIORS - MOST ADVANCED!)
# ============================================================================

class UltimateBayesianPhysics:
    """
    V5 Ultimate Physics - Combines ALL mechanisms:
    - SEI growth and swelling
    - Mechanical stress and particle cracking
    - Charge/discharge differentiation
    - Chemistry-dependent parameters
    - Arrhenius temperature acceleration
    - Calendar and cycle aging
    - Lithium plating risk
    """
    
    def __init__(self):
        self.constants = PhysicsConstants()
    
    def sample_physics_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Sample ALL 15 physics parameters - V5 ULTIMATE
        
        Combines:
        - 11 priors from enhanced_trainer_v3 (charge/discharge, chemistry)
        - 7 priors from train_battery_model_v3 (SEI, mechanical)
        Total unique: 15 priors (some overlap handled intelligently)
        """
        
        # 1. Chemistry probability and type
        chemistry_prob = pyro.sample(
            "chemistry_prob",
            dist.Beta(torch.tensor(7.0), torch.tensor(3.0))
        )
        chemistry_type = pyro.sample(
            "chemistry_type",
            dist.Bernoulli(chemistry_prob)
        )
        
        # Chemistry-dependent base parameters
        if chemistry_type > 0.5:  # LFP
            calendar_base = self.constants.LFP_CALENDAR_BASE
            cycle_base = self.constants.LFP_CYCLE_BASE
            activation_mean = np.log(self.constants.LFP_ACTIVATION_ENERGY)
        else:  # NMC
            calendar_base = self.constants.NMC_CALENDAR_BASE  
            cycle_base = self.constants.NMC_CYCLE_BASE
            activation_mean = np.log(self.constants.NMC_ACTIVATION_ENERGY)
        
        # 2. Activation energy (Arrhenius)
        log_activation_energy = pyro.sample(
            "log_activation_energy",
            dist.Normal(torch.tensor(activation_mean), torch.tensor(0.1))
        )
        
        # 3. Calendar aging coefficient
        calendar_aging_coefficient = pyro.sample(
            "calendar_aging_coefficient", 
            dist.LogNormal(torch.tensor(np.log(calendar_base)), torch.tensor(0.3))
        )
        
        # 4-5. SEI growth coefficient (from train_battery_model_v3)
        sei_growth_coefficient_raw = pyro.sample(
            "sei_growth_coefficient_raw",
            dist.Normal(torch.tensor(0.0), torch.tensor(0.5))
        )
        sei_growth_coefficient = F.softplus(sei_growth_coefficient_raw)
        
        # 6-7. Cycle aging coefficients - SEPARATE FOR CHARGE/DISCHARGE
        cycle_aging_coefficient_charge = pyro.sample(
            "cycle_aging_coefficient_charge",
            dist.LogNormal(torch.tensor(np.log(cycle_base)), torch.tensor(0.3))
        )
        
        cycle_aging_coefficient_discharge = pyro.sample(
            "cycle_aging_coefficient_discharge", 
            dist.LogNormal(torch.tensor(np.log(cycle_base * 0.8)), torch.tensor(0.3))
        )
        
        # 8-9. C-rate stress - SEPARATE FOR CHARGE/DISCHARGE
        c_rate_stress_charge = pyro.sample(
            "c_rate_stress_charge",
            dist.LogNormal(torch.tensor(np.log(0.4)), torch.tensor(0.2))
        )
        
        c_rate_stress_discharge = pyro.sample(
            "c_rate_stress_discharge",
            dist.LogNormal(torch.tensor(np.log(0.25)), torch.tensor(0.2))
        )
        
        # 10. SOC stress
        soc_stress = pyro.sample(
            "soc_stress", 
            dist.LogNormal(torch.tensor(np.log(2.0)), torch.tensor(0.3))
        )
        
        # 11. DOD stress
        dod_stress = pyro.sample(
            "dod_stress",
            dist.LogNormal(torch.tensor(np.log(1.0)), torch.tensor(0.2))
        )
        
        # 12. Lithium plating risk (charging-specific)
        plating_risk_coefficient = pyro.sample(
            "plating_risk_coefficient",
            dist.LogNormal(torch.tensor(np.log(0.3)), torch.tensor(0.2))
        )
        
        # 13. Swelling coefficient (from train_battery_model_v3)
        swelling_coefficient_raw = pyro.sample(
            "swelling_coefficient_raw",
            dist.Normal(torch.tensor(0.035), torch.tensor(0.01))
        )
        swelling_coefficient = torch.clamp(swelling_coefficient_raw, min=0.01, max=0.06)
        
        # 14. Stress accumulation rate (from train_battery_model_v3)
        stress_accumulation_rate_raw = pyro.sample(
            "stress_accumulation_rate_raw",
            dist.Normal(torch.tensor(0.02), torch.tensor(0.005))
        )
        stress_accumulation_rate = torch.clamp(stress_accumulation_rate_raw, min=0.005, max=0.04)
        
        # 15. Particle cracking threshold (from train_battery_model_v3)
        particle_cracking_threshold = pyro.sample(
            "particle_cracking_threshold",
            dist.Normal(torch.tensor(1.2), torch.tensor(0.3))
        )
        
        return {
            'chemistry_type': chemistry_type,
            'chemistry_prob': chemistry_prob,
            'log_activation_energy': log_activation_energy,
            'calendar_aging_coefficient': calendar_aging_coefficient,
            'sei_growth_coefficient': sei_growth_coefficient,
            'cycle_aging_coefficient_charge': cycle_aging_coefficient_charge,
            'cycle_aging_coefficient_discharge': cycle_aging_coefficient_discharge,
            'c_rate_stress_charge': c_rate_stress_charge,
            'c_rate_stress_discharge': c_rate_stress_discharge,
            'soc_stress': soc_stress,
            'dod_stress': dod_stress,
            'plating_risk_coefficient': plating_risk_coefficient,
            'swelling_coefficient': swelling_coefficient,
            'stress_accumulation_rate': stress_accumulation_rate,
            'particle_cracking_threshold': particle_cracking_threshold
        }
    
    def arrhenius_factor(self, temperature_celsius: torch.Tensor,
                        log_activation_energy: torch.Tensor) -> torch.Tensor:
        """Arrhenius temperature dependency - BOTH V3 versions use this"""
        temperature_celsius = torch.clamp(temperature_celsius, 0.0, 60.0)
        T_kelvin = temperature_celsius + 273.15
        T_ref = self.constants.REFERENCE_TEMP
        
        Ea = torch.exp(log_activation_energy).clamp(0.3, 1.5)
        kB = self.constants.BOLTZMANN_CONSTANT
        
        exponent = -Ea / kB * (1/T_kelvin - 1/T_ref)
        exponent = torch.clamp(exponent, -10.0, 10.0)
        
        return torch.exp(exponent)
    
    def sei_growth_factor(self, cycle_number: torch.Tensor,
                         sei_growth_coefficient: torch.Tensor) -> torch.Tensor:
        """
        SEI growth following sqrt(cycles) law
        From train_battery_model_v3.py
        """
        cycle_number = torch.clamp(cycle_number, 0.0, 10000.0)
        eps = 1e-6
        
        # SEI thickness grows as sqrt(cycles)
        sei_thickness_nm = self.constants.SEI_GROWTH_RATE * torch.sqrt(cycle_number + eps)
        
        # Growth factor (normalized)
        sqrt_factor = torch.sqrt(cycle_number + 1 + eps) / torch.sqrt(cycle_number + 2 + eps)
        growth_factor = sei_growth_coefficient * sqrt_factor
        
        return growth_factor
    
    def mechanical_swelling_factor(self, cycle_number: torch.Tensor,
                                   swelling_coefficient: torch.Tensor) -> torch.Tensor:
        """
        Mechanical swelling from SEI growth
        From train_battery_model_v3.py
        """
        cycle_number = torch.clamp(cycle_number, 0.0, 10000.0)
        eps = 1e-6
        
        # SEI thickness in nm
        sei_thickness_nm = self.constants.SEI_GROWTH_RATE * torch.sqrt(cycle_number + eps)
        
        # Volume expansion proportional to SEI thickness
        volume_expansion = swelling_coefficient * (sei_thickness_nm / 10.0)
        volume_expansion = torch.clamp(volume_expansion, 0.0, self.constants.MAX_SWELLING_STRAIN)
        
        swelling_factor = 1.0 + volume_expansion
        
        return swelling_factor
    
    def mechanical_stress_factor(self, cycle_number: torch.Tensor, 
                                c_rate: torch.Tensor,
                                stress_accumulation_rate: torch.Tensor,
                                particle_cracking_threshold: torch.Tensor) -> torch.Tensor:
        """
        Mechanical stress accumulation and particle cracking
        From train_battery_model_v3.py
        """
        cycle_number = torch.clamp(cycle_number, 0.0, 10000.0)
        c_rate = torch.clamp(torch.abs(c_rate), 0.0, 5.0)
        
        # Base stress accumulates with cycles
        base_stress = stress_accumulation_rate * cycle_number
        
        # High C-rate amplifies stress
        c_rate_amplification = 1.0 + 0.5 * c_rate
        
        accumulated_stress = base_stress * c_rate_amplification
        accumulated_stress = torch.clamp(accumulated_stress, 0.0, 2.0)
        
        # Sigmoid function for particle cracking onset
        cracking_factor = F.sigmoid(
            (accumulated_stress - particle_cracking_threshold) * 5.0
        )
        
        stress_factor = 1.0 + cracking_factor
        
        return stress_factor
    
    def compute_charge_discharge_factors(self, current: torch.Tensor,
                                       c_rate: torch.Tensor,
                                       temperature: torch.Tensor,
                                       soc: torch.Tensor,
                                       params: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute separate stress factors for charge and discharge
        From enhanced_trainer_v3.py
        """
        # Determine charge/discharge from current sign
        is_charging = current > 0
        is_discharging = current < 0
        
        # Current magnitude
        current_magnitude = torch.abs(c_rate)
        
        # Initialize stress factors
        charge_stress = torch.ones_like(current)
        discharge_stress = torch.ones_like(current)
        
        # Charging-specific stress (lithium plating risk)
        if is_charging.any():
            # High C-rate charging stress
            excess_c_rate_charge = F.relu(current_magnitude - self.constants.SAFE_C_RATE_CHARGE)
            c_rate_stress_charge = 1.0 + params['c_rate_stress_charge'] * excess_c_rate_charge
            
            # Lithium plating risk (low temperature + high C-rate)
            plating_risk = F.relu(self.constants.LITHIUM_PLATING_THRESHOLD_TEMP - temperature) / 15.0
            plating_stress = 1.0 + params['plating_risk_coefficient'] * plating_risk * current_magnitude
            
            charge_stress = c_rate_stress_charge * plating_stress
        
        # Discharging-specific stress
        if is_discharging.any():
            # High C-rate discharge stress
            excess_c_rate_discharge = F.relu(current_magnitude - self.constants.SAFE_C_RATE_DISCHARGE)
            c_rate_stress_discharge = 1.0 + params['c_rate_stress_discharge'] * excess_c_rate_discharge
            
            # Mechanical stress at low SOC (particle cracking risk)
            mech_stress = 1.0 + 0.5 * F.relu(0.2 - soc)
            
            discharge_stress = c_rate_stress_discharge * mech_stress
        
        return charge_stress, discharge_stress
    
    def calendar_aging(self, calendar_age_days: torch.Tensor,
                      calendar_aging_coefficient: torch.Tensor,
                      temperature_celsius: torch.Tensor,
                      soc: torch.Tensor) -> torch.Tensor:
        """
        Calendar aging: sqrt-time law
        BOTH V3 versions use this
        """
        calendar_age_days = torch.clamp(calendar_age_days, 0.0, 10000.0)
        soc = torch.clamp(soc, 0.0, 1.0)
        
        eps = 1e-6
        sqrt_time = torch.sqrt(calendar_age_days + eps)
        base_degradation = calendar_aging_coefficient * sqrt_time
        
        # SOC stress on calendar aging (high SOC accelerates)
        soc_stress = 1.0 + F.relu(soc - 0.85) * 2.0 / 0.15
        
        return base_degradation * soc_stress
    
    def cycle_aging(self, cycle_count: torch.Tensor,
                   current: torch.Tensor,
                   c_rate: torch.Tensor,
                   temperature_celsius: torch.Tensor,
                   soc: torch.Tensor,
                   dod: torch.Tensor,
                   params: Dict) -> torch.Tensor:
        """
        V5 ULTIMATE cycle aging - Combines EVERYTHING:
        - Charge/discharge differentiation
        - SEI growth
        - Mechanical stress
        - Temperature acceleration (Arrhenius)
        - SOC/DOD stress
        """
        cycle_count = torch.clamp(cycle_count, 0.0, 20000.0)
        c_rate = torch.clamp(c_rate, 0.0, 5.0)
        soc = torch.clamp(soc, 0.0, 1.0)
        dod = torch.clamp(dod, 0.0, 1.0)
        
        # 1. SEI growth (fundamental cycle aging mechanism)
        sei_growth = self.sei_growth_factor(cycle_count, params['sei_growth_coefficient'])
        
        # 2. Mechanical swelling from SEI
        swelling = self.mechanical_swelling_factor(cycle_count, params['swelling_coefficient'])
        
        # 3. Mechanical stress and particle cracking
        mechanical_stress = self.mechanical_stress_factor(
            cycle_count, c_rate,
            params['stress_accumulation_rate'],
            params['particle_cracking_threshold']
        )
        
        # 4. Charge/discharge specific factors
        charge_stress, discharge_stress = self.compute_charge_discharge_factors(
            current, c_rate, temperature_celsius, soc, params
        )
        
        # 5. Temperature acceleration (Arrhenius)
        temp_factor = self.arrhenius_factor(temperature_celsius, params['log_activation_energy'])
        
        # 6. SOC stress
        soc_stress_factor = 1.0 + F.relu(soc - 0.85) * params['soc_stress'] / 0.15
        
        # 7. DOD stress
        dod_stress_factor = 1.0 + F.relu(dod - 0.80) * params['dod_stress'] / 0.20
        
        # 8. Select charge/discharge coefficient
        is_charging = current > 0
        cycle_aging_coefficient = torch.where(
            is_charging,
            params['cycle_aging_coefficient_charge'],
            params['cycle_aging_coefficient_discharge']
        )
        
        # Select charge/discharge stress
        current_stress = torch.where(
            is_charging,
            charge_stress,
            discharge_stress
        )
        
        # COMBINED degradation (multiplicative - all mechanisms interact)
        base_degradation = cycle_aging_coefficient * cycle_count
        
        stress_multiplier = (
            temp_factor *           # Arrhenius
            sei_growth *            # SEI growth
            swelling *              # Mechanical swelling
            mechanical_stress *     # Particle cracking
            current_stress *        # Charge/discharge specific
            soc_stress_factor *     # SOC stress
            dod_stress_factor       # DOD stress
        )
        
        stress_multiplier = torch.clamp(stress_multiplier, 0.5, 10.0)
        
        return base_degradation * stress_multiplier
    
    def manufacturer_scaling(self, mfg_rated_cycles: torch.Tensor) -> torch.Tensor:
        """Manufacturer rating scaling - BOTH V3 versions use this"""
        mfg_rated_cycles = torch.clamp(mfg_rated_cycles, 2000.0, 10000.0)
        return self.constants.REFERENCE_MFG_RATING / mfg_rated_cycles
    
    def compute_ultimate_degradation(self, x: torch.Tensor, 
                                    params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        V5 ULTIMATE degradation computation
        
        V5 Input features (16 total):
        [0] current (TELEMETRY)
        [1] voltage (TELEMETRY)
        [2] temperature (TELEMETRY)
        [3] cell_voltage_min (TELEMETRY)
        [4] soc (ADJUSTABLE)
        [5] cycle_count (ADJUSTABLE)
        [6] internal_resistance (ADJUSTABLE)
        [7] charge_capacity (ADJUSTABLE)
        [8] discharge_capacity (ADJUSTABLE)
        [9] c_rate (ADJUSTABLE)
        [10] dod (ADJUSTABLE)
        [11] soh (TARGET)
        [12] cell_voltage_max (ADJUSTABLE)
        [13] mfg_rated_cycles (SPEC)
        [14] calendar_age_days (SPEC)
        [15] chemistry_type (SPEC)
        """
        # Extract features
        current = x[:, 0]
        temperature = x[:, 2]
        soc = x[:, 4]
        cycle_count = x[:, 5]
        c_rate = x[:, 9]
        dod = x[:, 10]
        mfg_rated_cycles = x[:, 13]
        calendar_age_days = x[:, 14]
        
        # Manufacturer scaling
        mfg_scaling = self.manufacturer_scaling(mfg_rated_cycles)
        
        # Calendar aging (additive component)
        calendar_loss = self.calendar_aging(
            calendar_age_days,
            params['calendar_aging_coefficient'],
            temperature,
            soc
        )
        
        # Cycle aging (multiplicative component - all mechanisms)
        cycle_loss = self.cycle_aging(
            cycle_count,
            current,
            c_rate,
            temperature,
            soc,
            dod,
            params
        )
        
        # Total degradation
        # Calendar aging is additive (independent of cycling)
        # Cycle aging is multiplicative (all factors interact)
        # Both scaled by manufacturer rating
        total_degradation = (calendar_loss + cycle_loss) * mfg_scaling
        
        return total_degradation


# ============================================================================
# V5 PYRO VAE MODEL
# ============================================================================

class V5PyroPhysicsVAE(PyroModule):
    """V5 Ultimate VAE with all 15 physics priors"""
    
    def __init__(self, input_dim: int = 16, latent_dim: int = 32, hidden_dims: List[int] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [128, 64]
        
        # Normalizer
        self.normalizer = DataNormalizer(input_dim)
        
        # V5 Ultimate physics
        self.physics = UltimateBayesianPhysics()
        
        # Encoder network
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder network
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder_layers = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution"""
        h = self.encoder_layers(x)
        z_loc = self.fc_mu(h)
        z_logvar = self.fc_logvar(h)
        z_logvar = torch.clamp(z_logvar, -10, 10)
        return z_loc, z_logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction"""
        return self.decoder_layers(z)
    
    def reconstruct_conditional(self, x: torch.Tensor) -> torch.Tensor:
        """
        Conditional reconstruction: Preserve telemetry, adjust parameters
        
        V5: Indices 0-3 (telemetry) and 13-15 (specs) preserved
        """
        with torch.no_grad():
            z_mu, z_logvar = self.encode(x)
            z_scale = torch.exp(0.5 * z_logvar)
            z = z_mu + z_scale * torch.randn_like(z_scale)
            
            x_recon = self.decode(z)
            
            # V5 PRESERVE TELEMETRY: Indices 0-3
            x_recon[:, 0:4] = x[:, 0:4]
            
            # PRESERVE SPECS: Indices 13-15
            x_recon[:, 13:16] = x[:, 13:16]
            
            return x_recon
    
    def model(self, x: torch.Tensor, beta_kl: float = 1.0, beta_physics: float = 0.01):
        """
        V5 Pyro model with weighted observation noise
        """
        pyro.module("decoder", self.decoder_layers)
        
        batch_size = x.shape[0]
        
        # Sample ALL 15 physics parameters
        physics_params = self.physics.sample_physics_parameters()
        
        # Compute physics-based degradation
        expected_degradation = self.physics.compute_ultimate_degradation(x, physics_params)
        expected_degradation = torch.clamp(expected_degradation, 0.0, 0.5)
        
        # Extract SOH (V5: index 11)
        observed_soh = x[:, 11]
        expected_soh = 1.0 - expected_degradation
        expected_soh = torch.clamp(expected_soh, 0.5, 1.0)
        
        # Physics constraint
        physics_loss = torch.mean((observed_soh - expected_soh) ** 2)
        pyro.factor("physics_constraint", -beta_physics * physics_loss)
        
        with pyro.plate("data", batch_size):
            z_loc, z_logvar = self.encode(x)
            z_scale = torch.exp(0.5 * z_logvar)
            
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
            
            x_recon = self.decode(z)
            
            # V5 WEIGHTED OBSERVATION NOISE
            obs_scale = torch.ones(self.input_dim, device=x.device)
            obs_scale[0:4] = 0.01    # Telemetry
            obs_scale[4:13] = 0.15   # Adjustable
            obs_scale[13:16] = 0.001 # Specs
            
            pyro.sample(
                "obs",
                dist.Normal(x_recon, obs_scale).to_event(1),
                obs=x
            )
    
    def guide(self, x: torch.Tensor, beta_kl: float = 1.0, beta_physics: float = 0.01):
        """V5 variational guide - ALL 15 parameters"""
        pyro.module("encoder", self.encoder_layers)
        pyro.module("fc_mu", self.fc_mu)
        pyro.module("fc_logvar", self.fc_logvar)
        
        batch_size = x.shape[0]
        
        # Guide for ALL 15 physics parameters
        physics_param_specs = [
            ('chemistry_prob', 'beta', (7.0, 3.0)),
            ('log_activation_energy', 'normal', (np.log(0.5), 0.1)),
            ('calendar_aging_coefficient', 'lognormal', (np.log(2.5e-5), 0.3)),
            ('sei_growth_coefficient_raw', 'normal', (0.0, 0.5)),
            ('cycle_aging_coefficient_charge', 'lognormal', (np.log(6e-6), 0.3)),
            ('cycle_aging_coefficient_discharge', 'lognormal', (np.log(4.8e-6), 0.3)),
            ('c_rate_stress_charge', 'lognormal', (np.log(0.4), 0.2)),
            ('c_rate_stress_discharge', 'lognormal', (np.log(0.25), 0.2)),
            ('soc_stress', 'lognormal', (np.log(2.0), 0.3)),
            ('dod_stress', 'lognormal', (np.log(1.0), 0.2)),
            ('plating_risk_coefficient', 'lognormal', (np.log(0.3), 0.2)),
            ('swelling_coefficient_raw', 'normal', (0.035, 0.01)),
            ('stress_accumulation_rate_raw', 'normal', (0.02, 0.005)),
            ('particle_cracking_threshold', 'normal', (1.2, 0.3))
        ]
        
        for name, dist_type, (init_loc, init_scale) in physics_param_specs:
            if dist_type == 'beta':
                alpha_loc = pyro.param(f"{name}_alpha_loc", torch.tensor(init_loc),
                                      constraint=dist.constraints.positive)
                beta_loc = pyro.param(f"{name}_beta_loc", torch.tensor(init_scale),
                                     constraint=dist.constraints.positive)
                chemistry_prob = pyro.sample(name, dist.Beta(alpha_loc, beta_loc))
                pyro.sample("chemistry_type", dist.Bernoulli(chemistry_prob))
            else:
                loc = pyro.param(f"{name}_loc", torch.tensor(init_loc))
                scale = pyro.param(f"{name}_scale", torch.tensor(init_scale),
                                  constraint=dist.constraints.positive)
                if dist_type == 'lognormal':
                    pyro.sample(name, dist.LogNormal(loc, scale))
                else:  # normal
                    pyro.sample(name, dist.Normal(loc, scale))
        
        with pyro.plate("data", batch_size):
            z_loc, z_logvar = self.encode(x)
            z_scale = torch.exp(0.5 * z_logvar)
            
            pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))


# ============================================================================
# V5 TRAINER
# ============================================================================

class V5PyroTrainer:
    """V5 Ultimate trainer with 15 physics priors"""
    
    def __init__(self, input_dim: int = 16, latent_dim: int = 32, 
                 learning_rate: float = 1e-4):
        
        self.vae = V5PyroPhysicsVAE(
            input_dim=input_dim,
            latent_dim=latent_dim
        ).to(device)
        
        self.optimizer = ClippedAdam({
            "lr": learning_rate,
            "clip_norm": 5.0
        })
        
        self.svi = SVI(
            self.vae.model,
            self.vae.guide,
            self.optimizer,
            loss=Trace_ELBO()
        )
        
        logger.info(f"🔥 V5 ULTIMATE Pyro VAE Initialized")
        logger.info(f"   15 Bayesian physics priors (MOST ADVANCED!)")
        logger.info(f"   ✅ SEI growth & mechanical degradation")
        logger.info(f"   ✅ Charge/discharge differentiation")
        logger.info(f"   ✅ Chemistry-dependent (LFP/NMC)")
        logger.info(f"   ✅ Arrhenius temperature acceleration")
        logger.info(f"   ✅ Telemetry preservation (indices 0-3)")
    
    def train(self, train_data: torch.Tensor, num_epochs: int = 50, 
              batch_size: int = 128, beta_kl: float = 0.5, beta_physics: float = 0.01):
        """Train V5 ultimate model"""
        
        logger.info(f"\n🚀 Starting V5 ULTIMATE training...")
        logger.info(f"   Data: {train_data.shape}")
        logger.info(f"   Epochs: {num_epochs}")
        logger.info(f"   15 Bayesian priors learning...")
        
        num_batches = len(train_data) // batch_size
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            indices = torch.randperm(len(train_data))
            train_data_shuffled = train_data[indices]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch = train_data_shuffled[start_idx:end_idx].to(device)
                
                try:
                    batch_norm = self.vae.normalizer.normalize(batch)
                    loss = self.svi.step(batch_norm, beta_kl, beta_physics)
                    epoch_loss += loss
                except Exception as e:
                    logger.warning(f"Batch {batch_idx} failed: {e}")
                    continue
            
            if num_batches > 0:
                epoch_loss /= num_batches
            
            if (epoch + 1) % 5 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}s): Loss={epoch_loss:.4f}")
        
        logger.info(f"✅ V5 ULTIMATE training complete!")
    
    def predict_rul(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[float, float]:
        """RUL prediction with V5 ultimate physics"""
        self.vae.eval()
        
        with torch.no_grad():
            x_norm = self.vae.normalizer.normalize(x.to(device))
            
            rul_samples = []
            for _ in range(num_samples):
                z_mu, z_logvar = self.vae.encode(x_norm)
                z_scale = torch.exp(0.5 * z_logvar)
                z = z_mu + z_scale * torch.randn_like(z_scale)
                
                x_recon = self.vae.decode(z)
                x_recon[:, 0:4] = x_norm[:, 0:4]
                x_recon[:, 13:16] = x_norm[:, 13:16]
                
                x_pred = self.vae.normalizer.denormalize(x_recon)
                
                predicted_soh = x_pred[0, 11].item()
                current_cycle = x_pred[0, 5].item()
                
                eol = 0.80
                if predicted_soh > eol and current_cycle > 0:
                    deg_rate = (1.0 - predicted_soh) / current_cycle
                    rul = (predicted_soh - eol) / deg_rate
                else:
                    rul = 0
                
                rul_samples.append(rul)
            
            return np.mean(rul_samples), np.std(rul_samples)
    
    def get_physics_insights(self) -> Dict:
        """Extract learned physics parameters"""
        insights = {}
        param_store = pyro.get_param_store()
        
        param_names = [
            'calendar_aging_coefficient', 'sei_growth_coefficient_raw',
            'cycle_aging_coefficient_charge', 'cycle_aging_coefficient_discharge',
            'c_rate_stress_charge', 'c_rate_stress_discharge',
            'soc_stress', 'dod_stress', 'plating_risk_coefficient',
            'swelling_coefficient_raw', 'stress_accumulation_rate_raw',
            'particle_cracking_threshold', 'log_activation_energy'
        ]
        
        for name in param_names:
            loc_key = f"{name}_loc"
            if loc_key in param_store:
                insights[name] = param_store[loc_key].detach().cpu().item()
        
        return insights
    
    def save_model(self, filepath: str):
        """Save V5 model"""
        checkpoint = {
            'vae_state_dict': self.vae.state_dict(),
            'pyro_params': pyro.get_param_store().get_state(),
            'normalizer_means': self.vae.normalizer.means.cpu(),
            'normalizer_stds': self.vae.normalizer.stds.cpu(),
            'input_dim': self.vae.input_dim,
            'latent_dim': self.vae.latent_dim,
            'version': 'V5_ULTIMATE'
        }
        torch.save(checkpoint, filepath)
        logger.info(f"💾 V5 ULTIMATE model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load V5 model"""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        self.vae.load_state_dict(checkpoint['vae_state_dict'])
        pyro.get_param_store().set_state(checkpoint['pyro_params'])
        self.vae.normalizer.means = checkpoint['normalizer_means'].to(device)
        self.vae.normalizer.stds = checkpoint['normalizer_stds'].to(device)
        logger.info(f"📂 V5 ULTIMATE model loaded from {filepath}")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_v5_training_data(csv_file: str) -> torch.Tensor:
    """
    Load V5 training data
    
    Expected columns (in order):
    current, voltage, temperature, cell_voltage_min,
    soc, cycle_count, internal_resistance, charge_capacity, discharge_capacity,
    c_rate, dod, soh, cell_voltage_max,
    mfg_rated_cycles, calendar_age_days, chemistry_type
    """
    df = pd.read_csv(csv_file)
    
    expected_cols = [
        'current', 'voltage', 'temperature', 'cell_voltage_min',
        'soc', 'cycle_count', 'internal_resistance',
        'charge_capacity', 'discharge_capacity',
        'c_rate', 'dod', 'soh', 'cell_voltage_max',
        'mfg_rated_cycles', 'calendar_age_days', 'chemistry_type'
    ]
    
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df_ordered = df[expected_cols]
    data_tensor = torch.tensor(df_ordered.values, dtype=torch.float32)
    
    logger.info(f"✅ Loaded {len(data_tensor)} samples (V5 structure)")
    return data_tensor


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train V5 ULTIMATE BayesianBESS')
    parser.add_argument('--data', required=True, help='Training CSV file')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta_kl', type=float, default=0.5)
    parser.add_argument('--beta_physics', type=float, default=0.01)
    parser.add_argument('--output', default='bess_model_v5_ultimate.pt')
    parser.add_argument('--latent_dim', type=int, default=32)
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("  V5 ULTIMATE - 15 BAYESIAN PHYSICS PRIORS")
    logger.info("="*70)
    
    # Load data
    train_data = load_v5_training_data(args.data)
    
    # Create trainer
    trainer = V5PyroTrainer(
        input_dim=16,
        latent_dim=args.latent_dim,
        learning_rate=args.lr
    )
    
    # Fit normalizer
    trainer.vae.normalizer.fit(train_data)
    
    # Train
    trainer.train(
        train_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        beta_kl=args.beta_kl,
        beta_physics=args.beta_physics
    )
    
    # Save
    trainer.save_model(args.output)
    
    # Show learned physics
    insights = trainer.get_physics_insights()
    logger.info("\n📊 Learned Physics Parameters:")
    for name, value in insights.items():
        logger.info(f"   {name}: {value:.6f}")


if __name__ == '__main__':
    main()
