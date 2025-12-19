#!/usr/bin/env python
# coding: utf-8
"""
Centralized deprivation cost model utilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GLOBAL_SEED = 0


@dataclass
class CentralizedConfig:
    """Community-level configuration for centralized model."""

    name: str
    survey_path: str
    residents_path: str

    resource_column: str = "4_water"
    lockdown_days: float = 8.0
    arrival_day: float = 3.0
    mm_resource_factor: float = 1.1
    non_arrival_rate: float = 0.0

    averaging_runs: int = 100

    base_arrival_rate: float = 2.0
    base_service_rate: float = 1.0


@dataclass
class CentralizedEnvironment:
    """Centralized environment built from a community config."""

    config: CentralizedConfig
    averaged_residents_df: pd.DataFrame


def _legacy_need(x: float, t_free_days: float, t_arrival_days: float) -> float:
    """Compute resource need: 0 if x > t_free, else t_free - max(x, t_arrival)."""
    if x > t_free_days:
        return 0.0
    return float(t_free_days - max(x, t_arrival_days))


# ---------------------------------------------------------------------------
# Resident water demand generation
# ---------------------------------------------------------------------------

def generate_resident_water_demand(config: CentralizedConfig) -> pd.DataFrame:
    """Generate resident water preparedness and need for a single run."""
    survey_df = pd.read_csv(config.survey_path)
    residents_df = pd.read_csv(config.residents_path)

    col = config.resource_column
    col_new = f"{col}_new"
    col_need = f"{col}_new_need"

    column_data = survey_df[col]
    full_series = pd.Series(np.nan, index=np.arange(len(residents_df)))
    full_series.iloc[: len(column_data)] = column_data

    sorted_data = np.sort(column_data.dropna().values)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    for i in range(len(column_data), len(residents_df)):
        u = np.random.rand()
        interpolated_value = np.interp(u, cdf_values, sorted_data)
        full_series.iat[i] = int(round(interpolated_value))

    nan_indices = full_series[full_series.isna()].index
    if len(nan_indices) > 0:
        fill_choices = sorted_data
        for idx in nan_indices:
            full_series.at[idx] = int(round(np.random.choice(fill_choices)))

    def remap_preparedness(value: int) -> int:
        if value == 0:
            return 0
        if value == 1:
            return int(np.random.choice([1, 2, 3]))
        if value == 2:
            return int(np.random.choice([4, 5, 6]))
        if value == 3:
            return 7
        if value == 4:
            return int(np.random.choice(list(range(8))))
        return int(value)

    residents_df[col_new] = full_series.astype(int).apply(remap_preparedness)

    def compute_need(x: float) -> float:
        if x > config.lockdown_days:
            return 0.0
        return float(config.lockdown_days - max(x, config.arrival_day))

    residents_df[col_need] = residents_df[col_new].apply(compute_need)

    return residents_df


def build_average_residents(config: CentralizedConfig) -> pd.DataFrame:
    """Build averaged resident data for centralized model."""
    np.random.seed(GLOBAL_SEED)

    one_df = generate_resident_water_demand(config)
    col_new = f"{config.resource_column}_new"
    col_need = f"{config.resource_column}_new_need"

    base_df = one_df.drop(columns=[c for c in [col_new, col_need] if c in one_df.columns]).reset_index(drop=True)

    prepared = one_df[col_new].astype(float).values
    need = np.array(
        [_legacy_need(x, config.lockdown_days, config.arrival_day) for x in prepared],
        dtype=float,
    )

    base_df["average_resource_have"] = prepared
    base_df["resource_need"] = need

    return base_df


def build_environment(config: CentralizedConfig) -> CentralizedEnvironment:
    """Build centralized environment for a community."""
    averaged_residents_df = build_average_residents(config)
    return CentralizedEnvironment(config=config, averaged_residents_df=averaged_residents_df)


# ---------------------------------------------------------------------------
# Queue simulation and deprivation cost
# ---------------------------------------------------------------------------

def simulate_queue_times(
    num_residents: int,
    arrival_rate: float,
    service_rate: float,
    non_arrival_rate: float,
) -> Tuple[Dict[int, float], Set[int]]:
    """Simulate queue waiting times for residents at relief center."""
    no_arrival_count = int(non_arrival_rate * num_residents)
    if no_arrival_count > 0:
        no_arrival_indices = set(np.random.choice(num_residents, no_arrival_count, replace=False))
    else:
        no_arrival_indices = set()

    depart_times: Dict[int, float] = {}
    last_depart_time = 0.0

    for idx in range(num_residents):
        if idx in no_arrival_indices:
            depart_times[idx] = None
            continue

        arrival_interval = np.random.exponential(1.0 / arrival_rate)
        current_time = last_depart_time + arrival_interval
        service_time = np.random.normal(1.0 / service_rate, (1.0 / service_rate) / 2.0)
        service_time = max(service_time, 0.0)

        depart_times[idx] = current_time + service_time
        last_depart_time = depart_times[idx]

    return depart_times, no_arrival_indices


def calculate_deprivation_cost_single(
    t_received_min: float,
    t_free_days: float,
    t_arrival_days: float,
    resource_received_days: float,
    resource_have_days: float,
    a: float,
    b: float,
) -> float:
    """Two-phase deprivation cost calculation."""
    e = np.exp(1.0)

    # Phase 1: before external resource arrival
    tt_hours = max(0.0, t_arrival_days * 24.0 + t_received_min / 60.0 - resource_have_days * 24.0)
    deprivation_cost_1 = a * (e ** (b * tt_hours))

    # Phase 2: after receiving external resources
    if resource_received_days > 0.0:
        if resource_have_days >= t_arrival_days:
            remaining_days = max(0.0, t_free_days - (resource_received_days + resource_have_days))
        else:
            remaining_days = max(0.0, t_free_days - (t_arrival_days + resource_received_days))

        tt2_hours = remaining_days * 24.0
        deprivation_cost_2 = a * (e ** (b * tt2_hours))
    else:
        deprivation_cost_2 = 0.0

    return deprivation_cost_1 + deprivation_cost_2


def compute_single_run_centralized(
    env: CentralizedEnvironment,
    t_free_days: float | None = None,
    t_arrival_days: float | None = None,
    mm: float | None = None,
    queue_speed_factor: float = 1.0,
    a: float = 0.2869,
    b: float = 0.0998,
    seed: int = 0,
) -> float:
    """Run one centralized queue simulation and return average deprivation cost."""
    np.random.seed(seed)

    config = env.config
    t_free_days = t_free_days if t_free_days is not None else config.lockdown_days
    t_arrival_days = t_arrival_days if t_arrival_days is not None else config.arrival_day
    mm = mm if mm is not None else config.mm_resource_factor

    df = env.averaged_residents_df.copy().reset_index(drop=True)
    n_residents = len(df)

    if "average_resource_have" not in df.columns or "resource_need" not in df.columns:
        raise ValueError("averaged_residents_df must contain 'average_resource_have' and 'resource_need' columns.")

    arrival_rate = config.base_arrival_rate * queue_speed_factor
    service_rate = config.base_service_rate * queue_speed_factor

    depart_times, no_arrival_indices = simulate_queue_times(
        num_residents=n_residents,
        arrival_rate=arrival_rate,
        service_rate=service_rate,
        non_arrival_rate=config.non_arrival_rate,
    )

    deprivation_costs: list[float] = []

    for idx in range(n_residents):
        have_days = float(df.loc[idx, "average_resource_have"])
        need_days = float(df.loc[idx, "resource_need"])

        if idx in no_arrival_indices:
            t_received_min = (t_free_days - t_arrival_days) * 24.0 * 60.0
            resource_received_days = 0.0
        else:
            t_received_min = float(depart_times[idx])
            resource_received_days = need_days * mm

        cost = calculate_deprivation_cost_single(
            t_received_min=t_received_min,
            t_free_days=t_free_days,
            t_arrival_days=t_arrival_days,
            resource_received_days=resource_received_days,
            resource_have_days=have_days,
            a=a,
            b=b,
        )
        deprivation_costs.append(cost)

    return float(np.mean(deprivation_costs))


def compute_centralized_deprivation_cost_average(
    env: CentralizedEnvironment,
    t_free_days: float | None = None,
    t_arrival_days: float | None = None,
    mm: float | None = None,
    queue_speed_factor: float = 1.0,
    a: float = 0.2869,
    b: float = 0.0998,
    num_simulations: int = 100,
) -> float:
    """Run multiple queue simulations and return average deprivation cost."""
    config = env.config
    t_free_days = t_free_days if t_free_days is not None else config.lockdown_days
    t_arrival_days = t_arrival_days if t_arrival_days is not None else config.arrival_day
    mm = mm if mm is not None else config.mm_resource_factor

    costs: list[float] = []
    for s in range(num_simulations):
        cost = compute_single_run_centralized(
            env=env,
            t_free_days=t_free_days,
            t_arrival_days=t_arrival_days,
            mm=mm,
            queue_speed_factor=queue_speed_factor,
            a=a,
            b=b,
            seed=GLOBAL_SEED + s,
        )
        costs.append(cost)

    return float(np.mean(costs))


def save_centralized_result(
    avg_cost: float,
    env: CentralizedEnvironment,
    filename: str = "centralized_deprivation_cost.csv",
) -> None:
    """Save centralized result to CSV."""
    output_dir = Path("figures") / "centralized" / env.config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(
        {
            "community": [env.config.name],
            "avg_deprivation_cost": [avg_cost],
            "log10_avg_deprivation_cost": [np.log10(avg_cost)],
        }
    )
    results_df.to_csv(output_dir / filename, index=False)


# ---------------------------------------------------------------------------
# Community configurations
# ---------------------------------------------------------------------------

LAURELHURST_CEN_CONFIG = CentralizedConfig(
    name="Laurelhurst",
    survey_path="data/laurelhurst_survey_with_address.csv",
    residents_path="data/laurelhurst_address.csv",
    resource_column="4_water",
    lockdown_days=8.0,
    arrival_day=3.0,
    mm_resource_factor=1.1,
    non_arrival_rate=0.0,
    averaging_runs=100,
    base_arrival_rate=2.0,
    base_service_rate=1.0,
)

SOUTHPARK_CEN_CONFIG = CentralizedConfig(
    name="SouthPark",
    survey_path="data/southpark_survey_with_address.csv",
    residents_path="data/southpark_address.csv",
    resource_column="4_water",
    lockdown_days=8.0,
    arrival_day=3.0,
    mm_resource_factor=1.1,
    non_arrival_rate=0.0,
    averaging_runs=100,
    base_arrival_rate=2.0,
    base_service_rate=1.0,
)

