import os
import argparse
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from critical_demand import run_simulation, RESULTS_COLUMN_NAMES
from utils import read_input_file
import numpy as np
import pandas as pd


def sentitivity_analysis(filename):

    df_costs, df_timeseries, df_settings, df_sensitivity = read_input_file(filename)

    system_sa_results = None
    assets_sa_results = None

    for i, row in df_sensitivity.iterrows():
        name = f"{row.category}-{row.variable_name}"
        param_values = np.linspace(
            row.min_val,
            row.max_val,
            int(np.ceil((row.max_val - row.min_val) / row.step)) + 1,
        )

        if row.category == "settings":
            initial_val = df_settings.loc[row.variable_name]
            for val in param_values:
                df_settings.loc[row.variable_name] = val
                (
                    results,
                    asset_results,
                    energy_system,
                    result_div,
                    system_results,
                    date_time_index,
                    non_critical_demand,
                    critical_demand,
                ) = run_simulation(df_costs, df_timeseries, df_settings)
                system_results["sa_input_variable_name"] = row.variable_name
                system_results["sa_input_variable_value"] = val
                system_results = system_results.reset_index()
                if system_sa_results is None:
                    system_sa_results = system_results
                else:
                    system_sa_results = pd.concat(
                        [system_sa_results, system_results], ignore_index=True
                    )
            df_settings.loc[row.variable_name] = initial_val
        else:
            initial_val = df_costs.loc[row.category, row.variable_name]
            for val in param_values:
                df_costs.loc[row.category, row.variable_name] = val
                (
                    results,
                    asset_results,
                    energy_system,
                    result_div,
                    system_results,
                    date_time_index,
                    non_critical_demand,
                    critical_demand,
                ) = run_simulation(df_costs, df_timeseries, df_settings)
                asset_results["sa_input_variable_name"] = row.variable_name
                asset_results["sa_input_variable_value"] = val
                asset_results = asset_results.reset_index()
                if assets_sa_results is None:
                    assets_sa_results = asset_results
                else:
                    assets_sa_results = pd.concat(
                        [assets_sa_results, asset_results], ignore_index=True
                    )
            df_costs.loc[row.category, row.variable_name] = initial_val

    system_sa_results.to_csv("system_sa_results.csv")
    assets_sa_results.to_csv("assets_sa_results.csv")
    return system_sa_results, assets_sa_results
