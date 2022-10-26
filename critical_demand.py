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

import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

##########################################################################
# Initialize the energy system and calculate necessary parameters
##########################################################################


start = "2022-01-01"

# The maximum number of days depends on the given *.csv file.
n_days = 10
n_days_in_year = 365

case_D = "D"
case_DBPV = "DBPV"
case_BPV = "BPV"

case = case_D


def epc_costs():
    epc_pv = 152.62  # currency/kW/year
    epc_diesel_genset = 84.8  # currency/kW/year
    epc_rectifier = 62.35  # currency/kW/year
    epc_inverter = 62.35  # currency/kW/year
    epc_battery = 101.00  # currency/kWh/year #ADN: defult was 10137.49 why too high?

    return (epc_pv, epc_diesel_genset, epc_rectifier, epc_inverter, epc_battery)


def other_costs():
    variable_cost_diesel_genset = 0.045  # currency/kWh #ADN: how caculated, doese included opex costs per kWh/a in ??
    diesel_cost = 0.65  # currency/l
    diesel_density = 0.846  # kg/l
    diesel_lhv = 11.83  # kWh/kg
    return variable_cost_diesel_genset, diesel_cost, diesel_density, diesel_lhv


# Import data.
current_directory = os.path.dirname(os.path.abspath(__file__))

# TODO use the same column names "Demand","SolarGen", also add "CriticalDemand"
filename = os.path.join(current_directory, "diesel_genset_data.csv")
data = pd.read_csv(filepath_or_buffer=filename)

# Change the index of data to be able to select data based on the time range.
data.index = pd.date_range(start="2022-01-01", periods=len(data), freq="H")


percent_energy_provider = 0.4  # reduces the energy input in the system
share_demand_critical = 0.5  # share of the demand which is critical #ADN:why??
demand_reduction_factor = 0.15

# Create date and time objects.
start_date_obj = datetime.strptime(start, "%Y-%m-%d")
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


def run_simulation(start=start, n_days=n_days, case=case):
    epc_pv, epc_diesel_genset, epc_rectifier, epc_inverter, epc_battery = epc_costs()
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
                        ep_costs=epc_pv
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
                        ep_costs=epc_diesel_genset * n_days / n_days_in_year,
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
                    ep_costs=epc_rectifier * n_days / n_days_in_year
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
                    ep_costs=epc_inverter * n_days / n_days_in_year
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
            investment=solph.Investment(ep_costs=epc_battery * n_days / n_days_in_year),
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
    from oemof_visio import ESGraphRenderer

    es = ESGraphRenderer(energy_system, legend=True, filepath=f"case_{case}.pdf")
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

    return results, energy_system


def scalar_result_presentation(results, case=case):

    epc_pv, epc_diesel_genset, epc_rectifier, epc_inverter, epc_battery = epc_costs()
    variable_cost_diesel_genset, diesel_cost, diesel_density, diesel_lhv = other_costs()

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
            epc_diesel_genset * capacity_diesel_genset
            + epc_pv * capacity_pv
            + epc_rectifier * capacity_rectifier
            + epc_inverter * capacity_inverter
            + epc_battery * capacity_battery
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

    # print(f"Simulation Time:\t {end_simulation_time-start_simulation_time:.2f} s")
    return html.Div(
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


##########################################################################
# Plot the duration curve for the diesel genset
##########################################################################

# if plt is not None and case in (case_D, case_DBPV):
#
#     # Create the duration curve for the diesel genset.
#     fig, ax = plt.subplots(figsize=(10, 5))
#
#     # Sort the power generated by the diesel genset in a descending order.
#     diesel_genset_duration_curve = np.sort(sequences_diesel_genset)[::-1]
#
#     diesel_genset_duration_curve_in_percentage = (
#         diesel_genset_duration_curve / capacity_diesel_genset * 100
#     )
#
#     percentage = (
#         100
#         * np.arange(1, len(diesel_genset_duration_curve) + 1)
#         / len(diesel_genset_duration_curve)
#     )
#
#     ax.scatter(
#         percentage,
#         diesel_genset_duration_curve,
#         color="dodgerblue",
#         marker="+",
#     )
#
#     # Plot a horizontal line representing the minimum load
#     ax.axhline(
#         y=min_load * capacity_diesel_genset,
#         color="crimson",
#         linestyle="--",
#     )
#     min_load_annotation_text = (
#         f"minimum load: {min_load * capacity_diesel_genset:0.2f} kW"
#     )
#     ax.annotate(
#         min_load_annotation_text,
#         xy=(100, min_load * capacity_diesel_genset),
#         xytext=(0, 5),
#         textcoords="offset pixels",
#         horizontalalignment="right",
#         verticalalignment="bottom",
#     )
#
#     # Plot a horizontal line representing the maximum load
#     ax.axhline(
#         y=max_load * capacity_diesel_genset,
#         color="crimson",
#         linestyle="--",
#     )
#     max_load_annotation_text = (
#         f"maximum load: {max_load * capacity_diesel_genset:0.2f} kW"
#     )
#     ax.annotate(
#         max_load_annotation_text,
#         xy=(100, max_load * capacity_diesel_genset),
#         xytext=(0, -5),
#         textcoords="offset pixels",
#         horizontalalignment="right",
#         verticalalignment="top",
#     )
#
#     ax.set_title(
#         "Duration Curve for the Diesel Genset Electricity Production",
#         fontweight="bold",
#     )
#     ax.set_ylabel("diesel genset production [kW]")
#     ax.set_xlabel("percentage of annual operation [%]")
#
#     # Create the second axis on the right side of the diagram
#     # representing the operation load of the diesel genset.
#     second_ax = ax.secondary_yaxis(
#         "right",
#         functions=(
#             lambda x: x / capacity_diesel_genset * 100,
#             lambda x: x * capacity_diesel_genset / 100,
#         ),
#     )
#     second_ax.set_ylabel("diesel genset operation load [%]")
#
#     plt.show()


import dash
from dash import dcc
from dash import html
import pandas as pd
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go


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


results, energy_system = run_simulation(start=start, n_days=n_days, case=case)

result_div = scalar_result_presentation(results, case=case)

bus_figures = []
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
from oemof_visio import ESGraphRenderer

er = ESGraphRenderer(energy_system, filepath="energy_system.pdf")


demo_app = dash.Dash(__name__, **options)

demo_app.layout = html.Div(
    children=[
        html.H1(children="Hello Dash", id="title"),
        dcc.Slider(
            id="ts_slice",
            value=1,
            min=0,
            max=n_days * 24,
            #marks={k: v for k, v in enumerate(date_time_index)},
        ),
        html.Div(children="""Hello World !""", id="paragraph"),
        dcc.Graph(id="sankey", figure=sankey(energy_system, results)),
    ]
    + [dcc.Graph(id=f"{bus}-id", figure=fig,) for bus, fig in zip(busses, bus_figures)]
    + [dcc.Graph(id="sankey_aggregate", figure=sankey(energy_system, results))]
)


@demo_app.callback(
    # The value of these components of the layout will be changed by this callback
    [Output(component_id="sankey", component_property="figure")]
    + [Output(component_id=f"{bus}-id", component_property="figure") for bus in busses],
    # Triggers the callback when the value of one of these components of the layout is changed
    Input(component_id="ts_slice", component_property="value"),
)
def update_table(ts):
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
                marker=dict(color="black"),
            )
        )
        bus_figures.append(fig)

    return [sankey(energy_system, results, date_time_index[ts])] + bus_figures


if __name__ == "__main__":
    demo_app.run_server(debug=True)
