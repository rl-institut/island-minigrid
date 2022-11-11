import os
import argparse
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px

from critical_demand import run_simulation, RESULTS_COLUMN_NAMES
from utils import read_input_file
import numpy as np
import pandas as pd


def update_sa_results(results, sa_results, var_name, value, category=None):
    results["sa_input_variable_category"] = category
    results["sa_input_variable_name"] = var_name
    results["sa_input_variable_value"] = value

    results = results.reset_index()
    if sa_results is None:
        answer = results
    else:
        answer = pd.concat([sa_results, results], ignore_index=True)
    return answer


def sensitivity_analysis(filename):

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
                system_sa_results = update_sa_results(
                    system_results,
                    system_sa_results,
                    var_name=row.variable_name,
                    value=val,
                    category="settings",
                )
                assets_sa_results = update_sa_results(
                    asset_results,
                    assets_sa_results,
                    var_name=row.variable_name,
                    value=val,
                    category="settings",
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
                system_sa_results = update_sa_results(
                    system_results,
                    system_sa_results,
                    var_name=row.variable_name,
                    value=val,
                    category=row.category,
                )
                assets_sa_results = update_sa_results(
                    asset_results,
                    assets_sa_results,
                    var_name=row.variable_name,
                    value=val,
                    category=row.category,
                )

            df_costs.loc[row.category, row.variable_name] = initial_val

    system_sa_results.to_csv("system_sa_results.csv", index=False)
    assets_sa_results.to_csv("assets_sa_results.csv", index=False)
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
        system_sa_results, assets_sa_results = sensitivity_analysis(filename)
    else:
        system_sa_results = pd.read_csv("system_sa_results.csv")
        assets_sa_results = pd.read_csv("assets_sa_results.csv")

    sa_columns = [
        "sa_input_variable_category",
        "sa_input_variable_name",
        "sa_input_variable_value",
    ]

    categories = system_sa_results.sa_input_variable_category.unique().tolist()

    system_variables = system_sa_results.sa_input_variable_name.unique().tolist()
    system_output_variables = system_sa_results.param.unique().tolist()

    initial_category = categories[0]
    assets_variables = (
        assets_sa_results.loc[
            assets_sa_results.sa_input_variable_category == initial_category
        ]
        .sa_input_variable_name.unique()
        .tolist()
    )

    assets_output_categories = assets_sa_results.asset.unique().tolist()
    assets_output_variables = assets_sa_results.columns.difference(
        sa_columns + ["asset"]
    ).tolist()
    # loading external resources
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    options = dict(
        # external_stylesheets=external_stylesheets
    )

    demo_app = dash.Dash(__name__, **options)

    demo_app.layout = html.Div(
        children=[
            html.H1("Sensitivity analysis results"),
            html.P(
                "Select a category and parameter from inputs of sensitivity analysis (x-axis) and a category and variable from outputs of sensitivity analysis to update the graph below"
            ),
            html.H3("Input parameter (x axis)"),
            html.Label(htmlFor="input_sa_category_dropdown", children="Category"),
            dcc.Dropdown(
                id="input_sa_category_dropdown",
                options=[{"label": v, "value": v} for v in categories],
                value=categories[0],
            ),
            html.Label(htmlFor="input_sa_variable_dropdown", children="Parameter name"),
            dcc.Dropdown(
                id="input_sa_variable_dropdown",
                options=[{"label": v, "value": v} for v in assets_variables],
                value=assets_variables[0],
            ),
            html.H3("Output variable (y axis)"),
            html.Label(htmlFor="output_sa_category_dropdown", children="Category"),
            dcc.Dropdown(
                id="output_sa_category_dropdown",
                options=[
                    {"label": v, "value": v} for v in ["kpi"] + assets_output_categories
                ],
                value=assets_output_categories[0],
            ),
            html.Label(htmlFor="output_sa_variable_dropdown", children="Variable name"),
            dcc.Dropdown(
                id="output_sa_variable_dropdown",
                options=[{"label": v, "value": v} for v in assets_output_variables],
                value=assets_output_variables[0],
            ),
            html.Div(id="graph-area"),
            # html.Div(
            #     children=dash_table.DataTable(
            #         system_sa_results.to_dict("records"),
            #         [{"name": i, "id": i} for i in system_sa_results.columns],
            #     )
            # ),
            # html.Div(
            #     children=dash_table.DataTable(
            #         assets_sa_results.to_dict("records"),
            #         [{"name": i, "id": i} for i in assets_sa_results.columns],
            #     )
            # ),
        ]
    )

    @demo_app.callback(
        # The value of these components of the layout will be changed by this callback
        [
            Output(
                component_id="input_sa_variable_dropdown", component_property="value"
            ),
            Output(
                component_id="input_sa_variable_dropdown", component_property="options"
            ),
        ],
        # Triggers the callback when the value of one of these components of the layout is changed
        Input(component_id="input_sa_category_dropdown", component_property="value"),
    )
    def change_category_value(category):
        variables = (
            assets_sa_results.loc[
                assets_sa_results.sa_input_variable_category == category
            ]
            .sa_input_variable_name.unique()
            .tolist()
        )
        return variables[0], [{"label": v, "value": v} for v in variables]

    @demo_app.callback(
        # The value of these components of the layout will be changed by this callback
        [
            Output(
                component_id="output_sa_variable_dropdown", component_property="value"
            ),
            Output(
                component_id="output_sa_variable_dropdown", component_property="options"
            ),
        ],
        # Triggers the callback when the value of one of these components of the layout is changed
        Input(component_id="output_sa_category_dropdown", component_property="value"),
    )
    def change_output_category_value(output_category):
        if output_category == "kpi":
            variables = system_output_variables
        else:
            variables = assets_output_variables
        return variables[0], [{"label": v, "value": v} for v in variables]

    @demo_app.callback(
        # The value of these components of the layout will be changed by this callback
        Output(component_id="graph-area", component_property="children"),
        # Triggers the callback when the value of one of these components of the layout is changed
        [
            Input(
                component_id="input_sa_variable_dropdown", component_property="value"
            ),
            Input(
                component_id="output_sa_variable_dropdown",
                component_property="value",
            ),
            Input(
                component_id="output_sa_category_dropdown",
                component_property="value",
            ),
        ],
        [
            State(
                component_id="input_sa_category_dropdown", component_property="value"
            ),
        ],
    )
    def change_asset_variable_value(
        var_name, output_var_name, output_category, category
    ):

        x_label = f"{var_name.title()} of {category.title()}".replace("_", " ")
        y_label = f"{output_var_name.title()} of {output_category.title()}".replace(
            "_", " "
        )

        if output_category == "kpi":
            df = system_sa_results
            df = df.loc[
                (df.sa_input_variable_category == category)
                & (df.sa_input_variable_name == var_name)
                & (df.param == output_var_name)
            ]
            df = df.rename(
                columns={"value": y_label, "sa_input_variable_value": x_label},
                # inplace=True,
            )
        else:
            df = assets_sa_results
            df = df.loc[
                (df.asset == output_category)
                & (df.sa_input_variable_name == var_name)
                # [output_var_name] + sa_columns,
            ]

            df = df.rename(
                columns={output_var_name: y_label, "sa_input_variable_value": x_label},
                # inplace=True,
            )
        return dcc.Graph(figure=px.scatter(df, x=x_label, y=y_label))

    demo_app.run_server(debug=True, port=8050)
