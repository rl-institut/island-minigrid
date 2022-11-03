# -*- coding: utf-8 -*-

"""
General description
-------------------
This example illustrates the combination of Investment and NonConvex options
applied to a diesel generator in a hybrid mini-grid system.

There are the following components:

    - pv: solar potential to generate electricity
    - diesel_source: input diesel for the diesel genset
    - diesel_genset: generates ac electricity
    - rectifier: converts generated ac electricity from the diesel genset
                 to dc electricity
    - inverter: converts generated dc electricity from the pv to ac electricity
    - battery: stores the generated dc electricity
    - demand_el: ac electricity demand (given as a separate *.csv file)
    - excess_el: allows for some electricity overproduction



Installation requirements
-------------------------
This example requires the version v0.5.x of oemof.solph. Install by:

    pip install 'oemof.solph>=0.5,<0.6'

"""

__copyright__ = "oemof developer group"
__license__ = "MIT"

import numpy as np
import os
import pandas as pd
import time
from datetime import datetime, timedelta
from oemof import solph

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from oemof_visio import ESGraphRenderer

    ES_GRAPH = True
except ModuleNotFoundError:
    ES_GRAPH = False


import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from utils import read_input_file, capex_from_investment, encode_image_file

##########################################################################
# Initialize the energy system and calculate necessary parameters
##########################################################################


def other_costs():
    variable_cost_diesel_genset = 0.045  # currency/kWh #ADN: how caculated, doese included opex costs per kWh/a in ??
    diesel_cost = 0.65  # currency/l
    diesel_density = 0.846  # kg/l
    diesel_lhv = 11.83  # kWh/kg
    return variable_cost_diesel_genset, diesel_cost, diesel_density, diesel_lhv


# ENERGY_SYSTEM_GRAPH = encode_image_file(results_json[PATHS_TO_PLOTS][PLOTS_ES])

# TODO use the same column names "Demand","SolarGen", also add "CriticalDemand"
filename = args.get("input_file")

if not os.path.exists(filename):
    raise FileNotFoundError(
        f"The file {f} was not found, make sure you you did not make a typo in its name or that the file is accessible from where you executed this code"
    )
df_costs, data, settings = read_input_file(filename)

start_date_obj = settings.start

# The maximum number of days depends on the given *.csv file.
n_days = settings.n_days
n_days_in_year = 365

case_D = "D"
case_DBPV = "DBPV"
case_BPV = "BPV"

case = settings.case
demand_reduction_factor = settings.maximum_demand_reduction

epc = df_costs["epc"]

# Change the index of data to be able to select data based on the time range.
data.index = pd.date_range(start=start_date_obj, periods=len(data), freq="H")


# Create date and time objects.
start_date = start_date_obj.date()
start_time = start_date_obj.time()
start_datetime = datetime.combine(start_date_obj.date(), start_date_obj.time())
end_datetime = start_datetime + timedelta(days=n_days)

# Create the energy system.
date_time_index = pd.date_range(start=start_date, periods=n_days * 24, freq="H")

# Choose the range of the solar potential and demand
# based on the selected simulation period.
solar_potential = data.SolarGen.loc[start_datetime:end_datetime]
hourly_demand = data.Demand.loc[start_datetime:end_datetime]
non_critical_demand = hourly_demand
critical_demand = data.CriticalDemand.loc[start_datetime:end_datetime]
peak_solar_potential = solar_potential.max()
peak_demand = hourly_demand.max()


def run_simulation(n_days=n_days, case=case):
    variable_cost_diesel_genset, diesel_cost, diesel_density, diesel_lhv = other_costs()
    # Start time for calculating the total elapsed time.
    start_simulation_time = time.time()

    energy_system = solph.EnergySystem(timeindex=date_time_index)

    # -------------------- BUSES --------------------
    # Create electricity and diesel buses.
    b_el_ac = solph.Bus(label="electricity_ac")
    b_el_dc = solph.Bus(label="electricity_dc")
    if case in (case_D, case_DBPV):
        b_diesel = solph.Bus(label="diesel")

    # -------------------- SOURCES --------------------
    if case in (case_D, case_DBPV):
        diesel_source = solph.Source(
            label="diesel_source",
            outputs={
                b_diesel: solph.Flow(
                    variable_costs=diesel_cost / diesel_density / diesel_lhv
                )
            },
        )

    if case in (case_BPV, case_DBPV):
        # EPC stands for the equivalent periodical costs.
        pv = solph.Source(
            label="pv",
            outputs={
                b_el_dc: solph.Flow(
                    fix=solar_potential / peak_solar_potential,
                    investment=solph.Investment(
                        ep_costs=epc.pv
                        * n_days
                        / n_days_in_year  # ADN:why not just put ep_costs=epc_PV??
                    ),
                    variable_costs=0,
                )
            },
        )

    # -------------------- TRANSFORMERS --------------------
    # The diesel genset assumed to have a fixed efficiency of 33%.

    # The output power of the diesel genset can only vary between
    # the given minimum and maximum loads, which represent the fraction
    # of the optimal capacity obtained from the optimization.

    diesel_genset_efficiency = 0.33
    if case in (case_D, case_DBPV):
        min_load = 0
        max_load = 1
        diesel_genset = solph.Transformer(
            label="diesel_genset",
            inputs={b_diesel: solph.Flow()},
            outputs={
                b_el_ac: solph.Flow(
                    nominal_value=None,
                    variable_costs=variable_cost_diesel_genset,
                    # min=min_load,
                    # max=max_load,
                    investment=solph.Investment(
                        ep_costs=epc.diesel_genset * n_days / n_days_in_year,
                        maximum=2 * peak_demand,
                    ),
                    # nonconvex=solph.NonConvex(),
                )
            },
            conversion_factors={b_el_ac: diesel_genset_efficiency},
        )

    # The rectifier assumed to have a fixed efficiency of 98%.
    # its cost already included in the PV cost investment
    rectifier = solph.Transformer(
        label="rectifier",
        inputs={
            b_el_ac: solph.Flow(
                # nominal_value=None,
                investment=solph.Investment(
                    ep_costs=epc.rectifier * n_days / n_days_in_year
                ),
                variable_costs=0,
            )
        },
        outputs={b_el_dc: solph.Flow()},
        conversion_factor={b_el_dc: 0.98,},
    )

    # The inverter assumed to have a fixed efficiency of 98%.
    # its cost already included in the PV cost investment
    inverter = solph.Transformer(
        label="inverter",
        inputs={
            b_el_dc: solph.Flow(
                # nominal_value=None,
                investment=solph.Investment(
                    ep_costs=epc.inverter * n_days / n_days_in_year
                ),
                variable_costs=0,
            )
        },
        outputs={b_el_ac: solph.Flow()},
        conversion_factor={b_el_ac: 0.98,},
    )

    # -------------------- STORAGE --------------------

    if case in (case_BPV, case_DBPV):
        battery = solph.GenericStorage(
            label="battery",
            nominal_storage_capacity=None,
            investment=solph.Investment(ep_costs=epc.battery * n_days / n_days_in_year),
            inputs={b_el_dc: solph.Flow(variable_costs=0.01)},
            outputs={b_el_dc: solph.Flow(investment=solph.Investment(ep_costs=0))},
            initial_storage_level=0.0,
            min_storage_level=0.0,
            max_storage_level=1,
            balanced=True,
            inflow_conversion_factor=0.9,
            outflow_conversion_factor=0.9,
            invest_relation_input_capacity=1,
            invest_relation_output_capacity=0.5,  # fixes the input flow investment to the output flow investment #ADN:why 0.5?
        )

    # -------------------- SINKS (or DEMAND) --------------------
    demand_el = solph.Sink(
        label="electricity_demand",
        inputs={
            b_el_ac: solph.Flow(
                min=(1 - demand_reduction_factor)
                * (non_critical_demand / non_critical_demand.max()),
                max=(non_critical_demand / non_critical_demand.max()),
                nominal_value=non_critical_demand.max(),
            )
        },
    )
    max_allowed_shortage = 0.3
    critical_demand_el = solph.Sink(
        label="electricity_critical_demand",
        inputs={
            b_el_ac: solph.Flow(
                fix=critical_demand,  # / critical_demand.max(),
                # min=0.4,
                # max=1, # non_critical_demand / non_critical_demand.max(),
                nominal_value=1,  # critical_demand.max()
            )
        },
    )

    excess_el = solph.Sink(
        label="excess_el", inputs={b_el_dc: solph.Flow(variable_costs=1e9)},
    )

    energy_system.add(
        b_el_dc, b_el_ac, inverter, rectifier, demand_el, critical_demand_el, excess_el,
    )

    # Add all objects to the energy system.
    if case == case_BPV:
        energy_system.add(
            pv, battery,
        )

    if case == case_DBPV:
        energy_system.add(
            pv, battery, diesel_source, diesel_genset, b_diesel,
        )

    # TODO set the if case
    if case == case_D:
        energy_system.add(
            diesel_source, diesel_genset, b_diesel,
        )
    ##########################################################################
    # Optimise the energy system
    ##########################################################################

    # The higher the MipGap or ratioGap, the faster the solver would converge,
    # but the less accurate the results would be.
    solver_option = {"gurobi": {"MipGap": "0.02"}, "cbc": {"ratioGap": "0.02"}}
    solver = "cbc"

    # TODO command to show the graph, might not work on windows, one could comment those lines

    energy_system_graph = f"case_{case}.png"
    if ES_GRAPH is True:
        es = ESGraphRenderer(
            energy_system, legend=True, filepath=energy_system_graph, img_format="png"
        )
        es.render()

    model = solph.Model(energy_system)
    model.solve(
        solver=solver,
        solve_kwargs={"tee": True},
        cmdline_options=solver_option[solver],
    )

    # End of the calculation time.
    end_simulation_time = time.time()

    print("\n" + 50 * "*")
    print(f"Simulation Time:\t {end_simulation_time-start_simulation_time:.2f} s")

    results = solph.processing.results(model)

    results_pv = solph.views.node(results=results, node="pv")
    if case in (case_D, case_DBPV):
        results_diesel_source = solph.views.node(results=results, node="diesel_source")
        results_diesel_genset = solph.views.node(results=results, node="diesel_genset")

    results_inverter = solph.views.node(results=results, node="inverter")
    results_rectifier = solph.views.node(results=results, node="rectifier")
    if case in (case_BPV, case_DBPV):
        results_battery = solph.views.node(results=results, node="battery")

    results_demand_el = solph.views.node(results=results, node="electricity_demand")
    results_critical_demand_el = solph.views.node(
        results=results, node="electricity_critical_demand"
    )
    results_excess_el = solph.views.node(results=results, node="excess_el")

    # -------------------- SEQUENCES (DYNAMIC) --------------------
    # Hourly demand profile.
    sequences_demand = results_demand_el["sequences"][
        (("electricity_ac", "electricity_demand"), "flow")
    ]

    sequences_critical_demand = results_critical_demand_el["sequences"][
        (("electricity_ac", "electricity_critical_demand"), "flow")
    ]

    if case in (case_BPV, case_DBPV):
        # Hourly profiles for solar potential and pv production.
        sequences_pv = results_pv["sequences"][(("pv", "electricity_dc"), "flow")]

    if case in (case_D, case_DBPV):
        # Hourly profiles for diesel consumption and electricity production
        # in the diesel genset.
        # The 'flow' from oemof is in kWh and must be converted to
        # kg by dividing it by the lower heating value and then to
        # liter by dividing it by the diesel density.#ADN:??
        sequences_diesel_consumption = (
            results_diesel_source["sequences"][(("diesel_source", "diesel"), "flow")]
            / diesel_lhv
            / diesel_density
        )

        # Hourly profiles for electricity production in the diesel genset.
        sequences_diesel_genset = results_diesel_genset["sequences"][
            (("diesel_genset", "electricity_ac"), "flow")
        ]

    # Hourly profiles for excess ac and dc electricity production.
    sequences_excess = results_excess_el["sequences"][
        (("electricity_dc", "excess_el"), "flow")
    ]

    if case in (case_D, case_DBPV):
        # -------------------- SCALARS (STATIC) --------------------
        capacity_diesel_genset = results_diesel_genset["scalars"][
            (("diesel_genset", "electricity_ac"), "invest")
        ]

        # Define a tolerance to force 'too close' numbers to the `min_load`
        # and to 0 to be the same as the `min_load` and 0.
        tol = 1e-8  # ADN ??
        load_diesel_genset = sequences_diesel_genset / capacity_diesel_genset
        sequences_diesel_genset[np.abs(load_diesel_genset) < tol] = 0
    else:
        capacity_diesel_genset = 0

    if case in (case_BPV, case_DBPV):
        capacity_pv = results_pv["scalars"][(("pv", "electricity_dc"), "invest")]

        capacity_battery = results_battery["scalars"][
            (("electricity_dc", "battery"), "invest")
        ]
    else:
        capacity_pv = 0
        capacity_battery = 0

    if "scalars" in results_inverter:
        capacity_inverter = results_inverter["scalars"][
            (("electricity_dc", "inverter"), "invest")
        ]
    else:
        capacity_inverter = 0

    if "scalars" in results_rectifier:
        capacity_rectifier = results_rectifier["scalars"][
            (("electricity_ac", "rectifier"), "invest")
        ]
    else:
        capacity_rectifier = 0

    total_cost_component = (
        (
            epc.diesel_genset * capacity_diesel_genset
            + epc.pv * capacity_pv
            + epc.rectifier * capacity_rectifier
            + epc.inverter * capacity_inverter
            + epc.battery * capacity_battery
        )
        * n_days
        / n_days_in_year
    )

    if case in (case_D, case_DBPV):
        # The only component with the variable cost is the diesel genset
        total_cost_variable = variable_cost_diesel_genset * sequences_diesel_genset.sum(
            axis=0
        )
        total_cost_diesel = diesel_cost * sequences_diesel_consumption.sum(axis=0)
    else:
        total_cost_variable = 0
        total_cost_diesel = 0

    total_revenue = total_cost_component + total_cost_variable + total_cost_diesel
    total_demand = sequences_demand.sum(axis=0) + sequences_critical_demand.sum(axis=0)

    # Levelized cost of electricity in the system in currency's Cent per kWh.
    lcoe = 100 * total_revenue / total_demand

    if case == case_DBPV:
        # The share of renewable energy source used to cover the demand.
        res = (
            100
            * sequences_pv.sum(axis=0)
            / (sequences_diesel_genset.sum(axis=0) + sequences_pv.sum(axis=0))
        )
    elif case == case_D:
        res = 0
    else:
        res = 100

    # The amount of excess electricity (which must probably be dumped).
    excess_rate = (
        100
        * sequences_excess.sum(axis=0)
        / (
            sequences_excess.sum(axis=0)
            + sequences_demand.sum(axis=0)
            + sequences_critical_demand.sum(axis=0)
        )
    )

    critical_demand_fulfilled = 100 * (
        sequences_critical_demand.sum(axis=0)
        / critical_demand[sequences_critical_demand.index].sum(axis=0)
    )
    demand_fulfilled = 100 * (
        sequences_demand.sum(axis=0)
        / non_critical_demand[sequences_demand.index].sum(axis=0)
    )

    ##########################################################################
    # Print the results in the terminal
    ##########################################################################

    print(50 * "*")
    print(f"Peak Demand:\t {sequences_demand.max():.0f} kW")
    print(f"LCOE:\t\t {lcoe:.2f} cent/kWh")
    print(f"RES:\t\t {res:.0f}%")
    print(f"Excess:\t\t {excess_rate:.1f}% of the total production")
    print(
        f"Share of critical demand fulfilled :\t\t {critical_demand_fulfilled:.1f}% of the total critical demand"
    )
    print(
        f"Share of non-critical demand fulfilled :\t\t {demand_fulfilled:.1f}% of the total non critical demand"
    )
    print(50 * "*")
    print("Optimal Capacities:")
    print("-------------------")
    print(f"Diesel Genset:\t {capacity_diesel_genset:.0f} kW")
    print(f"PV:\t\t {capacity_pv:.0f} kW")
    print(f"Battery:\t {capacity_battery:.0f} kW")
    print(f"Inverter:\t {capacity_inverter:.0f} kW")
    print(f"Rectifier:\t {capacity_rectifier:.0f} kW")
    print(50 * "*")

    result_div = html.Div(
        children=[
            html.Div(
                children=[
                    html.P(f"Peak Demand:\t {sequences_demand.max():.0f} kW"),
                    html.P(f"LCOE:\t\t {lcoe:.2f} cent/kWh"),
                    html.P(f"RES:\t\t {res:.0f}%"),
                    html.P(f"Excess:\t\t {excess_rate:.1f}% of the total production"),
                    html.P(
                        f"Share of critical demand fulfilled :\t\t {critical_demand_fulfilled:.1f}%"
                    ),
                    html.P(
                        f"Share of non-critical demand fulfilled :\t\t {demand_fulfilled:.1f}%"
                    ),
                ],
                style={"display": "flex", "justify-content": "space-between"},
            ),
            html.H3("Optimal Capacities:"),
            html.Div(
                children=[
                    html.P(f"Diesel Genset:\t {capacity_diesel_genset:.0f} kW"),
                    html.P(f"PV:\t\t {capacity_pv:.0f} kW"),
                    html.P(f"Battery:\t {capacity_battery:.0f} kW"),
                    html.P(f"Inverter:\t {capacity_inverter:.0f} kW"),
                    html.P(f"Rectifier:\t {capacity_rectifier:.0f} kW"),
                ],
                style={"display": "flex", "justify-content": "space-between"},
            ),
        ]
    )

    return results, energy_system, result_div


def reduced_demand_fig(results):

    results_demand_el = solph.views.node(results=results, node="electricity_demand")

    sequences_demand = results_demand_el["sequences"][
        (("electricity_ac", "electricity_demand"), "flow")
    ]

    nc_demand = non_critical_demand[sequences_demand.index].values

    fig = go.Figure(
        data=[
            go.Scatter(
                x=sequences_demand.index,
                y=sequences_demand.values,
                name="supplied non-critical demand",
                stackgroup="d",
                line_color="#DC267F",
            ),
            go.Scatter(
                x=sequences_demand.index,
                y=nc_demand,
                name="non-critical demand",
                line_color="#648FFF",
            ),
            go.Scatter(
                x=sequences_demand.index,
                y=nc_demand - sequences_demand.values,
                name="demand reduction",
                stackgroup="d",
                line_color="#FE6100",
            ),
            go.Scatter(
                x=sequences_demand.index,
                y=nc_demand * (1 - demand_reduction_factor),
                name="max demand reduction",
                line_color="#FE6100",
                line_dash="dash",
            ),
        ]
    )
    return fig


def sankey(energy_system, results, ts=None):
    """Return a dict to a plotly sankey diagram"""
    busses = []

    labels = []
    sources = []
    targets = []
    values = []

    # draw a node for each of the network's component. The shape depends on the component's type
    for nd in energy_system.nodes:
        if isinstance(nd, solph.Bus):

            # keep the bus reference for drawing edges later
            bus = nd
            busses.append(bus)

            bus_label = bus.label

            labels.append(nd.label)

            flows = solph.views.node(results, bus_label)["sequences"]

            # draw an arrow from the component to the bus
            for component in bus.inputs:
                if component.label not in labels:
                    labels.append(component.label)

                sources.append(labels.index(component.label))
                targets.append(labels.index(bus_label))

                val = flows[((component.label, bus_label), "flow")].sum()
                if ts is not None:
                    val = flows[((component.label, bus_label), "flow")][ts]
                # if val == 0:
                #     val = 1
                values.append(val)

            for component in bus.outputs:
                # draw an arrow from the bus to the component
                if component.label not in labels:
                    labels.append(component.label)

                sources.append(labels.index(bus_label))
                targets.append(labels.index(component.label))

                val = flows[((bus_label, component.label), "flow")].sum()
                if ts is not None:
                    val = flows[((bus_label, component.label), "flow")][ts]
                # if val == 0:
                #     val = 1
                values.append(val)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    hovertemplate="Node has total value %{value}<extra></extra>",
                    color="blue",
                ),
                link=dict(
                    source=sources,  # indices correspond to labels, eg A1, A2, A2, B1, ...
                    target=targets,
                    value=values,
                    hovertemplate="Link from node %{source.label}<br />"
                    + "to node%{target.label}<br />has value %{value}"
                    + "<br />and data <extra></extra>",
                ),
            )
        ]
    )

    fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
    return fig.to_dict()


results, energy_system, result_div = run_simulation(n_days=n_days, case=case)
energy_system_graph = encode_image_file(f"case_{case}.png")

bus_figures = []
if case == case_D:
    busses = ["electricity_ac"]
else:
    busses = ["electricity_ac", "electricity_dc"]
for bus in busses:
    fig = go.Figure(layout=dict(title=f"{bus} bus node"))
    for t, g in solph.views.node(results, node=bus)["sequences"].items():
        idx_asset = abs(t[0].index(bus) - 1)

        fig.add_trace(
            go.Scatter(x=g.index, y=g.values * pow(-1, idx_asset), name=t[0][idx_asset])
        )
    bus_figures.append(fig)

# loading external resources
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
options = dict(
    # external_stylesheets=external_stylesheets
)

demo_app = dash.Dash(__name__, **options)

demo_app.layout = html.Div(
    children=[
        html.H3("Model inputs"),
        html.Div(
            children=[
                html.P(f"Case: {case}"),
                html.P(f"Max demand reduction share: {demand_reduction_factor}"),
                html.P(f"Number of days: {n_days}"),
                html.P(f"Start date: {start_date}"),
            ],
            style={"display": "flex", "justify-content": "space-evenly"},
        ),
        html.Div(children=[html.H3("Results in numbers"), result_div]),
        html.Div(
            children=[
                html.H3("Non critical demand reduction overview"),
                dcc.Graph(id="nc_demand_supply", figure=reduced_demand_fig(results)),
            ]
        ),
        html.H3("Dynamic results"),
        html.P(
            children=[
                "You can adjust the slider to get the energy flow at a single timestep, "
                "or look for a specific timestep in the dropdown menu below ",
                html.Span(
                    "Note if you change the slider "
                    "it will show the value in the dropdown menu, but it you change the dropdown menu directly "
                    "it will not update the slider)"
                ),
            ]
        ),
        dcc.Slider(
            id="ts_slice_slider",
            value=1,
            min=0,
            max=n_days * 24,
            # marks={k: v for k, v in enumerate(date_time_index)},
        ),
        dcc.Dropdown(
            id="ts_slice_select",
            options={k: v for k, v in enumerate(date_time_index)},
            value=None,
        ),
        dcc.Graph(id="sankey", figure=sankey(energy_system, results)),
    ]
    + [dcc.Graph(id=f"{bus}-id", figure=fig,) for bus, fig in zip(busses, bus_figures)]
    + [dcc.Graph(id="sankey_aggregate", figure=sankey(energy_system, results))]
    + [
        html.H4(["Energy system"]),
        html.Img(
            src="data:image/png;base64,{}".format(energy_system_graph.decode()),
            alt="Energy System Graph, if you do not see this image it is because pygraphviz is not installed. "
                "If you are a windows user it might be complicated to install pygraphviz.",
            style={"maxWidth": "100%"},
        ),
    ]
)


@demo_app.callback(
    # The value of these components of the layout will be changed by this callback
    [
        Output(component_id="sankey", component_property="figure"),
        Output(component_id="nc_demand_supply", component_property="figure"),
    ]
    + [Output(component_id=f"{bus}-id", component_property="figure") for bus in busses],
    # Triggers the callback when the value of one of these components of the layout is changed
    Input(component_id="ts_slice_select", component_property="value"),
)
def update_figures(ts):
    ts = int(ts)
    # see if case changes, otherwise do not rerun this
    date_time_index = energy_system.timeindex

    demand_fig = reduced_demand_fig(results)
    max_y = non_critical_demand.max()
    demand_fig.add_trace(
        go.Scatter(
            x=[date_time_index[ts], date_time_index[ts]],
            y=[0, max_y],
            name="none",
            line_color="black",
        )
    )

    bus_figures = []
    for bus in busses:
        fig = go.Figure(layout=dict(title=f"{bus} bus node"))
        max_y = 0
        for t, g in solph.views.node(results, node=bus)["sequences"].items():
            idx_asset = abs(t[0].index(bus) - 1)
            asset_name = t[0][idx_asset]
            if t[0][idx_asset] == "battery":
                if idx_asset == 0:
                    asset_name += " discharge"
                else:
                    asset_name += " charge"
            opts = {}
            negative_sign = pow(-1, idx_asset)
            opts["stackgroup"] = (
                "negative_sign" if negative_sign < 0 else "positive_sign"
            )

            fig.add_trace(
                go.Scatter(
                    x=g.index, y=g.values * negative_sign, name=asset_name, **opts
                )
            )
            if g.max() > max_y:
                max_y = g.max()
        fig.add_trace(
            go.Scatter(
                x=[date_time_index[ts], date_time_index[ts]],
                y=[0, max_y],
                name="none",
                line_color="black",
            )
        )
        bus_figures.append(fig)

    return [
        sankey(energy_system, results, date_time_index[ts]),
        demand_fig,
    ] + bus_figures


@demo_app.callback(
    # The value of these components of the layout will be changed by this callback
    Output(component_id="ts_slice_select", component_property="value"),
    # Triggers the callback when the value of one of these components of the layout is changed
    Input(component_id="ts_slice_slider", component_property="value"),
)
def change_ts_value(val):
    return val


if __name__ == "__main__":
    demo_app.run_server(debug=True, port=settings.port)
