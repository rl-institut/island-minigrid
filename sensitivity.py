from critical_demand import run_simulation
from utils import read_input_file
import numpy as np


df_costs, df_timeseries, df_settings, df_sensitivity = read_input_file(
    "input_case.xlsx"
)

for i, row in df_sensitivity.iterrows():
    name = f"{row.category}-{row.variable_name}"
    param_values = np.arange(row.min_val, row.max_val, row.step)
    if row.category == "settings":
        initial_val = df_settings.loc[row.variable_name]
        for val in param_values:
            df_settings.loc[row.variable_name] = val
            (
                results,
                df_results,
                energy_system,
                result_div,
                date_time_index,
                non_critical_demand,
                critical_demand,
            ) = run_simulation(df_costs, df_timeseries, df_settings)
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
                date_time_index,
                non_critical_demand,
                critical_demand,
            ) = run_simulation(df_costs, df_timeseries, df_settings)
        df_costs.loc[row.category, row.variable_name] = initial_val
