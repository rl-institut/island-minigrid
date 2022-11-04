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


if __name__ == "__main__":
    # Import data.
    current_directory = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        prog="python sensitivity.py",
        description="Build a simple model with non critical demand",
    )
    parser.add_argument(
        "-i",
        dest="input_file",
        nargs="?",
        type=str,
        help="path to the input file",
        default=os.path.join(current_directory, "input_case.xlsx"),
    )

    args = vars(parser.parse_args())

    filename = args.get("input_file")

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"The file {f} was not found, make sure you you did not make a typo in its name or that the file is accessible from where you executed this code"
        )

    if not os.path.exists("system_sa_results.csv"):
        system_sa_results, assets_sa_results = sensitivity_analysis()
    else:
        system_sa_results = pd.read_csv("system_sa_results.csv")
        assets_sa_results = pd.read_csv("assets_sa_results.csv")

    # loading external resources
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    options = dict(
        # external_stylesheets=external_stylesheets
    )

    demo_app = dash.Dash(__name__, **options)

    demo_app.layout = html.Div(
        children=[
            html.H3("Sensitivity analysis outputs"),
            html.Div(
                children=dash_table.DataTable(
                    system_sa_results.to_dict("records"),
                    [{"name": i, "id": i} for i in system_sa_results.columns],
                )
            ),
            html.Div(
                children=dash_table.DataTable(
                    assets_sa_results.to_dict("records"),
                    [{"name": i, "id": i} for i in assets_sa_results.columns],
                )
            ),
        ]
        # + [dcc.Graph(id="sankey_aggregate", figure=sankey(energy_system, results))]
    )

    demo_app.run_server(debug=True, port=8050)
