from critical_demand import run_simulation, RESULTS_COLUMN_NAMES
from utils import read_input_file
import numpy as np
import pandas as pd


df_costs, df_timeseries, df_settings, df_sensitivity = read_input_file(
    "input_case.xlsx"
)

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
                df_results,
                energy_system,
                result_div,
                df_scalars,
                date_time_index,
                non_critical_demand,
                critical_demand,
            ) = run_simulation(df_costs, df_timeseries, df_settings)
            df_scalars["sa_input_variable_name"] = row.variable_name
            df_scalars["sa_input_variable_value"] = val
            df_scalars = df_scalars.reset_index()
            if system_sa_results is None:
                system_sa_results = df_scalars
            else:
                system_sa_results = pd.concat(
                    [system_sa_results, df_scalars], ignore_index=True
                )
        df_settings.loc[row.variable_name] = initial_val
    else:
        initial_val = df_costs.loc[row.category, row.variable_name]
        for val in param_values:
            df_costs.loc[row.category, row.variable_name] = val
            (
                results,
                df_results,
                energy_system,
                result_div,
                df_scalars,
                date_time_index,
                non_critical_demand,
                critical_demand,
            ) = run_simulation(df_costs, df_timeseries, df_settings)
            df_results["sa_input_variable_name"] = row.variable_name
            df_results["sa_input_variable_value"] = val
            df_results = df_results.reset_index()
            if assets_sa_results is None:
                assets_sa_results = df_results
            else:
                assets_sa_results = pd.concat(
                    [assets_sa_results, df_results], ignore_index=True
                )
        df_costs.loc[row.category, row.variable_name] = initial_val

system_sa_results.to_csv("system_sa_results.csv")
assets_sa_results.to_csv("assets_sa_results.csv")
