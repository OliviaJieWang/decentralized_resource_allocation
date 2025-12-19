#!/usr/bin/env python
# coding: utf-8


from __future__ import annotations

import argparse

import numpy as np

from centralized import (
    LAURELHURST_CEN_CONFIG,
    SOUTHPARK_CEN_CONFIG,
    build_environment,
    compute_centralized_deprivation_cost_average,
    save_centralized_result,
)


COMMUNITY_CONFIGS = {
    "laurelhurst": LAURELHURST_CEN_CONFIG,
    "southpark": SOUTHPARK_CEN_CONFIG,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run centralized deprivation cost model for a community.")
    parser.add_argument(
        "--community",
        type=str,
        default="laurelhurst",
        choices=list(COMMUNITY_CONFIGS.keys()),
        help="Community to run: laurelhurst or southpark (default: laurelhurst).",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=100,
        help="Number of queue simulations to average over (default: 100).",
    )
    parser.add_argument(
        "--queue-speed-factor",
        type=float,
        default=1.0,
        help="Queue speed factor (>1 means faster arrival/service, default: 1.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = COMMUNITY_CONFIGS[args.community]

    print(f"Building averaged centralized environment for community: {config.name} ...")
    env = build_environment(config)

    print(
        f"Running centralized model for {config.name} with "
        f"num_simulations={args.num_simulations}, queue_speed_factor={args.queue_speed_factor} ..."
    )

    avg_cost = compute_centralized_deprivation_cost_average(
        env=env,
        t_free_days=config.lockdown_days,
        t_arrival_days=config.arrival_day,
        mm=config.mm_resource_factor,
        queue_speed_factor=args.queue_speed_factor,
        num_simulations=args.num_simulations,
    )

    print("\nCentralized result:")
    print(f"  Community                : {config.name}")
    print(f"  Avg deprivation cost     : {avg_cost:.6f}")
    print(f"  log10(avg deprivation)   : {np.log10(avg_cost):.6f}")

    save_centralized_result(avg_cost=avg_cost, env=env)


if __name__ == "__main__":
    main()
