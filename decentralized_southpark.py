#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------

@dataclass
class DecentralizedConfig:
    name: str
    survey_path: str
    residents_path: str
    distance_matrix_path: str
    resource_column: str = "4_water"
    lockdown_days: float = 8.0
    arrival_day: float = 3.0
    mm_resource_factor: float = 1.1
    averaging_runs: int = 100
    degree_para: Tuple[float, float] = (1.0, 0.5)
    dist_decay_alpha: float = -1.0
    prop_fri: float = 0.4
    min_prob_strong_tie: float = 0.5
    min_prob_social_tie: float = 0.5
    arrival_rate_center: float = 2.0
    service_rate_center: float = 1.0
    omega_captain: float = 1.0 / 15.0
    u_lognormal: float = 1.2
    sigma_lognormal: float = 0.4


@dataclass
class DecentralizedEnvironment:
    config: DecentralizedConfig
    averaged_residents_df: pd.DataFrame
    strong_tie_matrix_avg: np.ndarray
    weak_tie_matrix_avg: np.ndarray
    stranger_tie_matrix_avg: np.ndarray
    strong_tie_count_avg: np.ndarray
    weak_tie_count_avg: np.ndarray


@dataclass
class SharingSolution:
    df: pd.DataFrame
    captain_set: set[int]
    share_sets: Dict[int, set[int]]
    Q_per_captain: float
    total_resources_received: Dict[int, float]

SOUTHPARK_CONFIG = DecentralizedConfig(
    name="SouthPark",
    survey_path="data/southpark_survey_with_address.csv",
    residents_path="data/southpark_address.csv",
    distance_matrix_path="data/southpark_distance_matrix.npy",
    resource_column="4_water",
    lockdown_days=8.0,
    arrival_day=3.0,
    mm_resource_factor=1.1,
    averaging_runs=100,
    degree_para=(0.8266, 0.0669),
    dist_decay_alpha=-1.43,
    prop_fri=0.2984610391613645,
    min_prob_strong_tie=0.5,
    min_prob_social_tie=0.5,
    arrival_rate_center=2.0,
    service_rate_center=1.0,
    omega_captain=1.0 / 15.0,
    u_lognormal=1.2,
    sigma_lognormal=0.4,
)

# ------------------------------------------------------------

def _legacy_need(x: float, t_free_days: float, t_arrival_days: float) -> float:
    if x > t_free_days:
        return 0.0
    return float(t_free_days - max(x, t_arrival_days))


def generate_resident_water_demand(config: DecentralizedConfig) -> pd.DataFrame:
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
        for idx in nan_indices:
            full_series.at[idx] = int(round(np.random.choice(sorted_data)))

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


def build_average_residents(config: DecentralizedConfig) -> pd.DataFrame:
    np.random.seed(GLOBAL_SEED)
    one_df = generate_resident_water_demand(config)
    col_new = f"{config.resource_column}_new"
    col_need = f"{config.resource_column}_new_need"

    base_df = one_df.drop(columns=[c for c in [col_new, col_need] if c in one_df.columns]).reset_index(drop=True)
    prepared = one_df[col_new].astype(float).values
    need = np.array([_legacy_need(x, config.lockdown_days, config.arrival_day) for x in prepared], dtype=float)

    base_df["average_resource_have"] = prepared
    base_df["resource_need"] = need
    return base_df

# --------------------------------------------------------

def _get_degree_list(n: int, para: Tuple[float, float], seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    return np.random.negative_binomial(n=para[0], p=para[1], size=n)


def _distance_decay_function(distance: np.ndarray, alpha: float) -> np.ndarray:
    eps = 1e-12
    return (distance + eps) ** alpha


def _get_social_tie_matrix(probability_np: np.ndarray, degree_list: np.ndarray, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    if degree_list.sum() % 2 != 0:
        degree_list[0] += 1

    n = probability_np.shape[0]
    social_tie_matrix = np.zeros_like(probability_np, dtype=int)
    accumulated_degree_list = np.zeros(n, dtype=int)

    for i in range(n):
        remain_degree = degree_list - accumulated_degree_list
        if remain_degree[i] <= 0:
            continue
        reachable = np.where(remain_degree > 0)[0]
        select_prob = probability_np[i, reachable] / np.sum(probability_np[i, reachable])
        selected = np.random.choice(reachable, size=remain_degree[i], p=select_prob, replace=True)
        social_tie_matrix[i, selected] = 1
        social_tie_matrix[selected, i] = 1
        accumulated_degree_list = np.sum(social_tie_matrix, axis=1)

    return social_tie_matrix


def _split_social_tie_matrix(social_tie_matrix: np.ndarray, p: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    n = social_tie_matrix.shape[0]
    strong_tie_matrix = np.zeros_like(social_tie_matrix, dtype=int)
    upper_triangle = np.triu(social_tie_matrix, k=1)

    for i in range(n):
        for j in range(i + 1, n):
            if upper_triangle[i, j] == 1 and np.random.uniform() < p:
                strong_tie_matrix[i, j] = 1
                strong_tie_matrix[j, i] = 1

    weak_tie_matrix = social_tie_matrix - strong_tie_matrix
    stranger_tie_matrix = np.ones_like(social_tie_matrix, dtype=int) - social_tie_matrix
    np.fill_diagonal(stranger_tie_matrix, 0)
    return strong_tie_matrix, weak_tie_matrix, stranger_tie_matrix


def get_social_network(distance_matrix: np.ndarray, config: DecentralizedConfig, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree_list = _get_degree_list(distance_matrix.shape[0], config.degree_para, seed=seed)
    probability_np = _distance_decay_function(distance_matrix, config.dist_decay_alpha)
    social_tie_matrix = _get_social_tie_matrix(probability_np, degree_list, seed=seed)
    return _split_social_tie_matrix(social_tie_matrix, p=config.prop_fri, seed=seed)


def build_average_network(config: DecentralizedConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance_matrix = np.load(config.distance_matrix_path)
    n = distance_matrix.shape[0]

    strong_acc = np.zeros((n, n), dtype=float)
    weak_acc = np.zeros((n, n), dtype=float)
    stranger_acc = np.zeros((n, n), dtype=float)

    for run in range(config.averaging_runs):
        seed = GLOBAL_SEED + run
        strong, weak, stranger = get_social_network(distance_matrix, config, seed)
        strong_acc += strong
        weak_acc += weak
        stranger_acc += stranger

    strong_avg = strong_acc / float(config.averaging_runs)
    weak_avg = weak_acc / float(config.averaging_runs)
    stranger_avg = stranger_acc / float(config.averaging_runs)

    return strong_avg, weak_avg, stranger_avg, strong_avg.sum(axis=1), weak_avg.sum(axis=1)

# -------------------------------------------------------------

def build_environment(config: DecentralizedConfig) -> DecentralizedEnvironment:
    averaged_residents_df = build_average_residents(config)
    strong_avg, weak_avg, stranger_avg, strong_count, weak_count = build_average_network(config)

    return DecentralizedEnvironment(
        config=config,
        averaged_residents_df=averaged_residents_df,
        strong_tie_matrix_avg=strong_avg,
        weak_tie_matrix_avg=weak_avg,
        stranger_tie_matrix_avg=stranger_avg,
        strong_tie_count_avg=strong_count,
        weak_tie_count_avg=weak_count,
    )


def get_expected_social_tie_weight(env: DecentralizedEnvironment, a: int, b: int) -> float:
    return 3.0 * env.strong_tie_matrix_avg[a, b] + 2.0 * env.weak_tie_matrix_avg[a, b] + 1.0 * env.stranger_tie_matrix_avg[a, b]


def has_effective_strong_tie(env: DecentralizedEnvironment, a: int, b: int) -> bool:
    return float(env.strong_tie_matrix_avg[a, b]) >= env.config.min_prob_strong_tie


def has_effective_social_tie(env: DecentralizedEnvironment, a: int, b: int) -> bool:
    return (env.strong_tie_matrix_avg[a, b] + env.weak_tie_matrix_avg[a, b]) >= env.config.min_prob_social_tie


WEIGHTS = {"sharing_preference_new": 1/3, "social_ties_new": 1/3, "participation_new": 1/3}


def build_captain_candidate_features(env: DecentralizedEnvironment, residents_df: pd.DataFrame) -> pd.DataFrame:
    survey_df = pd.read_csv(env.config.survey_path)
    n_households = len(residents_df)
    features_df = residents_df.copy().reset_index(drop=True)

    features_df["strong_tie"] = env.strong_tie_count_avg
    features_df["weak_tie"] = env.weak_tie_count_avg

    survey_df["sharing_preference"] = survey_df[["7_water"]].sum(axis=1)
    column_data = survey_df["sharing_preference"]
    sorted_data = np.sort(column_data.dropna().values)
    cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    full_series = pd.Series(0.0, index=np.arange(n_households))
    for i in range(n_households):
        full_series.iat[i] = np.interp(np.random.rand(), cdf_values, sorted_data)
    features_df["sharing_preference_all"] = full_series
    col = features_df["sharing_preference_all"]
    features_df["sharing_preference_new"] = (col - col.min()) / (col.max() - col.min())

    features_df["social_ties"] = features_df[["strong_tie", "weak_tie"]].sum(axis=1)
    col_st = features_df["social_ties"]
    col_st_norm = (col_st - col_st.min()) / (col_st.max() - col_st.min())
    sorted_st = np.sort(col_st_norm.values)
    cdf_st = np.arange(1, len(sorted_st) + 1) / len(sorted_st)
    full_series_st = pd.Series([np.interp(np.random.rand(), cdf_st, sorted_st) for _ in range(n_households)])
    features_df["social_ties_new"] = full_series_st

    col_part = survey_df["11_hours"].dropna()
    sorted_part = np.sort(col_part.values)
    cdf_part = np.arange(1, len(sorted_part) + 1) / len(sorted_part)
    full_series_part = pd.Series([np.interp(np.random.rand(), cdf_part, sorted_part) for _ in range(n_households)])
    col_p = full_series_part
    features_df["participation_new"] = (col_p - col_p.min()) / (col_p.max() - col_p.min())

    features_df["captain_score"] = (
        features_df["sharing_preference_new"] * WEIGHTS["sharing_preference_new"]
        + features_df["social_ties_new"] * WEIGHTS["social_ties_new"]
        + features_df["participation_new"] * WEIGHTS["participation_new"]
    )
    return features_df


def select_captains(features_df: pd.DataFrame, n_captains: int) -> pd.Index:
    return features_df.sort_values(by="captain_score", ascending=False).head(n_captains).index


def solve_sharing_model(env: DecentralizedEnvironment, n_captains: int, t_free_days: float, t_arrival_days: float, mm: float, a: float, b: float, seed: int) -> SharingSolution | None:
    np.random.seed(seed)
    n_households = len(env.averaged_residents_df)
    df = env.averaged_residents_df.copy().reset_index(drop=True)

    features_df = build_captain_candidate_features(env, df)
    captain_indices = select_captains(features_df, n_captains)
    if len(captain_indices) == 0:
        return None
    captain_set = set(int(i) for i in captain_indices)

    share_sets: Dict[int, set[int]] = {}
    for idx in captain_indices:
        current_sharing_pre = features_df.loc[idx, "sharing_preference_all"]
        share_sets[int(idx)] = set()
        for j in range(n_households):
            if j == idx or j in captain_set:
                continue
            if current_sharing_pre == 3:
                share_sets[int(idx)].add(j)
            elif current_sharing_pre == 2 and has_effective_social_tie(env, int(idx), int(j)):
                share_sets[int(idx)].add(j)
            elif current_sharing_pre == 1 and has_effective_strong_tie(env, int(idx), int(j)):
                share_sets[int(idx)].add(j)

    g_values = {i: {j: get_expected_social_tie_weight(env, i, j) for j in range(n_households)} for i in range(n_households)}

    total_resource_need = float(df["resource_need"].sum())
    Q_per_captain = math.ceil(total_resource_need / len(captain_indices)) * mm
    S = {int(i): max(Q_per_captain - float(df.loc[i, "resource_need"]), 0.0) for i in captain_indices}

    m = gp.Model(f"Resource_Sharing_{env.config.name}")
    m.setParam(gp.GRB.Param.TimeLimit, 3000)
    m.setParam("OutputFlag", 0)
    m.ModelSense = GRB.MAXIMIZE

    x = {}
    for i in share_sets.keys():
        for j in range(n_households):
            if j in share_sets[i]:
                x[i, j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"x_{i}_{j}")
            else:
                x[i, j] = m.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=0.0, name=f"x_{i}_{j}")

    y = {j: m.addVar(vtype=GRB.BINARY, name=f"y_{j}") for j in range(n_households)}

    for i in share_sets.keys():
        m.addConstr(gp.quicksum(x[i, j] for j in share_sets[i]) <= S[i])

    for j in range(n_households):
        if j not in captain_set:
            m.addConstr(gp.quicksum(x[i, j] for i in share_sets.keys()) <= float(df.loc[j, "resource_need"]) * y[j])

    primary_obj = gp.quicksum(x[i, j] * g_values[i][j] for i in share_sets.keys() for j in share_sets[i])
    secondary_obj = gp.quicksum(y[j] for j in range(n_households))
    m.setObjectiveN(primary_obj, index=0, priority=1)
    m.setObjectiveN(secondary_obj, index=1, priority=0)

    m.optimize()
    if m.SolCount == 0:
        return None

    total_resources_received = {j: sum(x[i, j].X for i in share_sets.keys()) for j in range(n_households)}
    return SharingSolution(df=df, captain_set=captain_set, share_sets=share_sets, Q_per_captain=Q_per_captain, total_resources_received=total_resources_received)


def simulate_queue_and_deprivation(env: DecentralizedEnvironment, sol: SharingSolution, t_free_days: float, t_arrival_days: float, a: float, b: float, seed: int) -> float:
    np.random.seed(seed)
    df, captain_set, share_sets, Q_per_captain, total_resources_received = sol.df, sol.captain_set, sol.share_sets, sol.Q_per_captain, sol.total_resources_received
    n_households = len(df)

    hub_arrival_rate = env.config.omega_captain * env.config.arrival_rate_center
    hub_service_rate = env.config.omega_captain * env.config.service_rate_center

    current_time = 0.0
    final_receive_time: Dict[int, float] = {}
    for i in share_sets.keys():
        current_time += np.random.exponential(1.0 / hub_arrival_rate)
        service_time = max(np.random.normal(1.0 / hub_service_rate, (1.0 / hub_service_rate) / 2.0), 0.0)
        final_receive_time[i] = current_time + service_time
        current_time = final_receive_time[i]

    receive_time: Dict[int, list[float]] = {}
    rng = np.random.default_rng(seed)
    for i in share_sets.keys():
        served = [j for j in share_sets[i] if total_resources_received.get(j, 0.0) > 0]
        if not served:
            continue
        rng.shuffle(served)
        t = final_receive_time[i]
        for j in served:
            t += np.random.lognormal(env.config.u_lognormal, env.config.sigma_lognormal)
            receive_time.setdefault(j, []).append(t)

    for j, times in receive_time.items():
        final_receive_time[j] = min(times)

    def calc_cost(t_recv, t_free, t_arr, res_recv, res_have, a, b):
        e = np.exp(1.0)
        tt = max(0.0, t_arr * 24 + t_recv / 60 - res_have * 24)
        c1 = a * (e ** (b * tt))
        c2 = 0.0
        if res_recv > 0:
            rem = max(0.0, t_free - (res_recv + res_have if res_have >= t_arr else t_arr + res_recv))
            c2 = a * (e ** (b * rem * 24))
        return c1 + c2

    costs = []
    for j in range(n_households):
        have = float(df.loc[j, "average_resource_have"])
        need = float(df.loc[j, "resource_need"])
        if j in captain_set:
            recv = min(Q_per_captain, need)
            t_recv = final_receive_time.get(j, (t_free_days - t_arrival_days) * 24 * 60)
        else:
            recv = float(total_resources_received.get(j, 0.0))
            t_recv = final_receive_time.get(j, (t_free_days - t_arrival_days) * 24 * 60) if recv > 0 else (t_free_days - t_arrival_days) * 24 * 60
            if recv <= 0:
                recv = 0.0
        costs.append(calc_cost(t_recv, t_free_days, t_arrival_days, recv, have, a, b))
    return float(np.mean(costs))


def compute_decentralized_deprivation_cost_average(env: DecentralizedEnvironment, n_captains: int, t_free_days: float, t_arrival_days: float, mm: float, a: float = 0.2869, b: float = 0.0998, num_simulations: int = 50, fix_sharing_solution: bool = True) -> float:
    sol = solve_sharing_model(env, n_captains, t_free_days, t_arrival_days, mm, a, b, GLOBAL_SEED)
    if sol is None:
        return 1e9
    costs = [simulate_queue_and_deprivation(env, sol, t_free_days, t_arrival_days, a, b, GLOBAL_SEED + s) for s in range(num_simulations)]
    return float(np.mean(costs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SouthPark decentralized sharing model.")
    parser.add_argument("--num-simulations", type=int, default=50, help="Number of simulations per captain setting (default: 50).")
    parser.add_argument("--output-root", type=str, default="figures", help="Root directory to save figures (default: figures).")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    config = SOUTHPARK_CONFIG

    print(f"\nBuilding averaged environment for community: {config.name} ...")
    env = build_environment(config)

    n_values = list(range(10, 160, 10))
    results = []

    print("Running decentralized model over captain counts...")
    for n in n_values:
        print(f"  - n_captains={n}")
        avg_cost = compute_decentralized_deprivation_cost_average(
            env=env,
            n_captains=n,
            t_free_days=config.lockdown_days,
            t_arrival_days=config.arrival_day,
            mm=config.mm_resource_factor,
            num_simulations=args.num_simulations,
            fix_sharing_solution=True,
        )
        results.append({"n_captains": n, "avg_deprivation_cost": avg_cost})

    results_df = pd.DataFrame(results)
    results_df["log10_avg_deprivation_cost"] = np.log10(results_df["avg_deprivation_cost"])




if __name__ == "__main__":
    main()
