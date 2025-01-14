# -*- coding: utf-8 -*-

"""
General description
-------------------
This example illustrates the concept of critical vs non-critical demand within a hybrid mini-grid system.

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
import argparse
from datetime import datetime, timedelta
from oemof import solph
from oemof.tools.economics import annuity

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from oemof_visio import ESGraphRenderer

    ES_GRAPH = True
except ModuleNotFoundError:
    ES_GRAPH = False
z_version = 1

if solph.__version__[:3] != "0.5" or (
    solph.__version__[:3] == "0.5" and int(solph.__version__.split(".")[2]) < z_version
):

    raise Exception(
        f"Oemof solph version should be 0.5.{z_version} (current version {solph.__version__}) , please update oemof.solph with for example `pip install oemof.solph==0.5.{z_version}`"
    )

import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from utils import read_input_file, capex_from_investment, encode_image_file

RESULTS_COLUMN_NAMES = [
    "annuity",
    "annual_costs",
    "total_flow",
    "capacity",
    "cash_flow",  # AA: could be named fuel_expenditure_cost
    "total_opex_costs",
    "first_investment",
]
##########################################################################
# Initialize the energy system and calculate necessary parameters
##########################################################################



def diesel_cost(vol_cost, dens, energy_dens):
    return vol_cost / dens / energy_dens

def run_simulation(df_costs, data, settings):

    start_date_obj = settings.start

    # The maximum number of days depends on the given *.csv file.
    n_days = settings.n_days
    n_days_in_year = 365

    case = settings.case
    demand_reduction_factor = settings.maximum_demand_reduction

    epc = df_costs["annuity"]
    opex_var = df_costs["opex_variable"].fillna(0)
    diesel_lhv = df_costs["energy_density"].diesel_genset
    diesel_density = df_costs["density"].diesel_genset
    soc_min = df_costs["soc_min"]
    soc_max = df_costs["soc_max"]
    project_planning_cost = df_costs["capex_fix"].project

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

    # Start time for calculating the total elapsed time.
    start_simulation_time = time.time()

    # -------------- Set up energy system ------------ #
    # First the diesel sinks will be added, then the
    # diesel and RE components if defined in the input sheet

    energy_system = solph.EnergySystem(timeindex=date_time_index)

    # -------------------- BUSES --------------------
    # Create electricity and diesel buses.
    b_el_ac = solph.Bus(label="electricity_ac")

    # -------------------- SINKS (or DEMAND) --------
    demand_el = solph.components.Sink(
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
    critical_demand_el = solph.components.Sink(
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

    energy_system.add(b_el_ac, demand_el, critical_demand_el)

    # -------------------- DIESEL SYSTEM --------------
    if "D" in case:
        diesel_genset_efficiency = 0.33
        min_load = 0.20
        max_load = 1
        b_diesel = solph.Bus(label="diesel")
        diesel_genset = solph.components.Converter(
            label="diesel_genset",
            inputs={b_diesel: solph.Flow()},
            outputs={
                b_el_ac: solph.Flow(
                    variable_costs=opex_var.diesel_genset,
                    min=min_load,
                    max=max_load,
                    nominal_value=solph.Investment(
                        ep_costs=epc.diesel_genset * n_days / n_days_in_year,
                        # maximum=2 * peak_demand,
                        # minimum= 1.2*peak_demand,
                    ),
                    # nonconvex=solph.NonConvex(),
                )
            },
            conversion_factors={b_el_ac: diesel_genset_efficiency},
        )

        diesel_source = solph.components.Source(
            label="diesel_source",
            outputs={b_diesel: solph.Flow(variable_costs=opex_var.diesel_genset)},
        )

        energy_system.add(b_diesel, diesel_genset, diesel_source)


    # -------------------- RE COMPONENTS --------------------
    if "PV" in case:
        b_el_dc = solph.Bus(label="electricity_dc")

        # EPC stands for the equivalent periodical costs.
        pv = solph.components.Source(
            label="pv",
            outputs={
                b_el_dc: solph.Flow(
                    fix=solar_potential / peak_solar_potential,
                    nominal_value=solph.Investment(
                        ep_costs=epc.pv
                                 * n_days
                                 / n_days_in_year  # ADN:why not just put ep_costs=epc_PV??
                    ),
                    variable_costs=opex_var.pv,
                )
            },
        )
        # The rectifier assumed to have a fixed efficiency of 98%.
        # its cost already included in the PV cost investment
        rectifier = solph.components.Converter(
            label="rectifier",
            inputs={
                b_el_ac: solph.Flow(
                    nominal_value=solph.Investment(
                        ep_costs=epc.rectifier * n_days / n_days_in_year
                    ),
                    variable_costs=opex_var.rectifier,
                )
            },
            outputs={b_el_dc: solph.Flow()},
            conversion_factors={
                b_el_dc: 0.98,
            },
        )

        # The inverter assumed to have a fixed efficiency of 98%.
        # its cost already included in the PV cost investment
        inverter = solph.components.Converter(
            label="inverter",
            inputs={
                b_el_dc: solph.Flow(
                    nominal_value=solph.Investment(
                        ep_costs=epc.inverter * n_days / n_days_in_year
                    ),
                    variable_costs=opex_var.inverter,  # has to be fits input sheet
                )
            },
            outputs={b_el_ac: solph.Flow()},
            conversion_factors={
                b_el_ac: 0.98,
            },
        )

        excess_el = solph.components.Sink(
            label="excess_el",
            inputs={b_el_dc: solph.Flow(variable_costs=0.01)},
        )

        energy_system.add(b_el_dc, pv, rectifier, inverter, excess_el)

        # ------------- HYDROGEN COMPONENTS (only if PV) -----------------
        if "H" in case:
            b_h2 = solph.Bus(label="hydrogen")

            fuel_cell_efficiency = 0.99
            fuel_cell = solph.components.Converter(
                label="fuel_cell",
                inputs={b_h2: solph.Flow()},
                outputs={
                    b_el_ac: solph.Flow(
                        variable_costs=opex_var.fuel_cell,
                        min=0.1,
                        max=1,
                        nominal_value=solph.Investment(
                            ep_costs=epc.fuel_cell * n_days / n_days_in_year,
                            maximum=2 * peak_demand,
                            # minimum= 1.2*peak_demand,
                        ),
                        # nonconvex=solph.NonConvex(),
                    )
                },
                conversion_factors={b_el_ac: fuel_cell_efficiency},
            )

            electrolyser = solph.components.Converter(
                label="electrolyser",
                inputs={
                    b_el_dc: solph.Flow(
                        nominal_value=solph.Investment(
                            ep_costs=epc.electrolyser * n_days / n_days_in_year
                        ),
                        variable_costs=opex_var.electrolyser,  # has to be fits input sheet
                    )
                },
                outputs={b_h2: solph.Flow()},
                conversion_factors={
                    b_el_ac: 0.98,
                },
            )

            h2_storage = solph.components.GenericStorage(
                label="h2_storage",
                investment=solph.Investment(ep_costs=epc.h2_storage * n_days / n_days_in_year),
                inputs={b_h2: solph.Flow(variable_costs=0.01)},  # AA: might be replaced by user input's opex_fixed
                outputs={b_h2: solph.Flow(nominal_value=solph.Investment(ep_costs=0))},
                min_storage_level=soc_min.h2_storage,
                max_storage_level=soc_max.h2_storage,
                loss_rate=0.01,
                inflow_conversion_factor=0.9,
                outflow_conversion_factor=0.9,
                invest_relation_input_capacity=1,
                invest_relation_output_capacity=0.5,  # fixes the input flow investment to the output flow investment
            )

            energy_system.add(b_h2, electrolyser, fuel_cell, h2_storage)

        # ------------- BATTERY (only if PV) -----------------
        if "B" in case:
            battery = solph.components.GenericStorage(
                label="battery",
                investment=solph.Investment(ep_costs=epc.battery * n_days / n_days_in_year),
                inputs={
                    b_el_dc: solph.Flow(variable_costs=0.01)
                },
                outputs={b_el_dc: solph.Flow(nominal_value=solph.Investment(ep_costs=0))},
                min_storage_level=soc_min.battery,
                max_storage_level=soc_max.battery,
                loss_rate=0.01,
                inflow_conversion_factor=0.9,
                outflow_conversion_factor=0.9,
                invest_relation_input_capacity=1,
                invest_relation_output_capacity=0.5,  # fixes the input flow investment to the output flow investment
            )

            energy_system.add(battery)


    ##########################################################################
    # ------------------ Optimise the energy system ------------------------ #
    ##########################################################################

    # The higher the MipGap or ratioGap, the faster the solver would converge,
    # but the less accurate the results would be.
    solver_option = {"gurobi": {"MipGap": "0.02"}, "cbc": {"ratioGap": "0.02"}}
    solver = "cbc"

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

    ### ---------- RESULTS PROCESSING ---------- ###
    results = solph.processing.results(model)

    asset_results = df_costs.copy()
    asset_results["capacity"] = 0
    asset_results["total_flow"] = 0
    asset_results["cash_flow"] = 0

    project_lifetime = 25
    wacc = settings.wacc
    CRF = annuity(1, project_lifetime, wacc)

    results_demand_el = solph.views.node(results=results, node="electricity_demand")
    results_critical_demand_el = solph.views.node(
        results=results, node="electricity_critical_demand"
    )

    # Hourly demand profile.
    sequences_demand = results_demand_el["sequences"][
        (("electricity_ac", "electricity_demand"), "flow")
    ]

    sequences_critical_demand = results_critical_demand_el["sequences"][
        (("electricity_ac", "electricity_critical_demand"), "flow")
    ]

    # Set all capacity defaults to 0
    capacities = dict.fromkeys(["diesel_genset", "pv", "inverter",
                                        "rectifier", "battery", "fuel_cell",
                                        "electrolyser", "h2_storage"], 0)

    if "D" in case:
        results_diesel_source = solph.views.node(results=results, node="diesel_source")
        results_diesel_genset = solph.views.node(results=results, node="diesel_genset")
        sequences_diesel_consumption = (
            results_diesel_source["sequences"][(("diesel_source", "diesel"), "flow")]
            / diesel_lhv
            / diesel_density
        )

        asset_results.loc["diesel_genset", "cash_flow"] = (
            opex_var.diesel_genset * sequences_diesel_consumption.sum()
        )

        # Hourly profiles for electricity production in the diesel genset.
        sequences_diesel_genset = results_diesel_genset["sequences"][
            (("diesel_genset", "electricity_ac"), "flow")
        ]
        # -------------------- SCALARS (STATIC) --------------------
        capacities["diesel_genset"] = results_diesel_genset["scalars"][
            (("diesel_genset", "electricity_ac"), "invest")
        ]

        # Define a tolerance to force 'too close' numbers to the `min_load`
        # and to 0 to be the same as the `min_load` and 0.
        tol = 1e-8
        load_diesel_genset = sequences_diesel_genset / capacities["diesel_genset"]
        sequences_diesel_genset[np.abs(load_diesel_genset) < tol] = 0
        asset_results.loc["diesel_genset", "total_flow"] = sequences_diesel_genset.sum()

    if "PV" in case:
        results_pv = solph.views.node(results=results, node="pv")
        results_inverter = solph.views.node(results=results, node="inverter")
        results_rectifier = solph.views.node(results=results, node="rectifier")
        results_excess_el = solph.views.node(results=results, node="excess_el")
        # Hourly profiles for solar potential and pv production.
        sequences_pv = results_pv["sequences"][(("pv", "electricity_dc"), "flow")]
        asset_results.loc["pv", "total_flow"] = sequences_pv.sum()

        sequences_excess = results_excess_el["sequences"][
            (("electricity_dc", "excess_el"), "flow")
        ]

        sequences_inverter = results_inverter["sequences"][
            (("inverter", "electricity_ac"), "flow")
        ]

        sequences_rectifier = results_rectifier["sequences"][
            (("rectifier", "electricity_dc"), "flow")
        ]

        asset_results.loc["inverter", "total_flow"] = sequences_inverter.sum()
        asset_results.loc["rectifier", "total_flow"] = sequences_rectifier.sum()
        capacities["pv"] = results_pv["scalars"][(("pv", "electricity_dc"), "invest")]
        capacities["inverter"] = results_inverter["scalars"][
            (("electricity_dc", "inverter"), "invest")
        ]
        capacities["rectifier"] = results_rectifier["scalars"][
            (("electricity_ac", "rectifier"), "invest")
        ]

        if "B" in case:
            results_battery = solph.views.node(results=results, node="battery")
            capacities["battery"] = results_battery["scalars"][
                (("electricity_dc", "battery"), "invest")
            ]

        if "H" in case:
            results_electrolyser = solph.views.node(results=results, node="electrolyser")
            results_fuel_cell = solph.views.node(results=results, node="fuel_cell")
            results_h2_storage = solph.views.node(results=results, node="h2_storage")
            capacities["electrolyser"] = results_electrolyser["scalars"][
                (("electricity_dc", "electrolyser"), "invest")
            ]
            capacities["fuel_cell"] = results_fuel_cell["scalars"][
                (("fuel_cell", "electricity_ac"), "invest")
            ]
            capacities["h2_storage"] = results_h2_storage["scalars"][
                (("hydrogen", "h2_storage"), "invest")
            ]

    for asset, capacity in capacities.items():
        asset_results.loc[asset, "capacity"] = capacity

    # Scaling annuity to timeframe
    year_fraction = n_days / n_days_in_year

    asset_results["first_investment"] = asset_results.apply(
        lambda x: (x.capex_variable * x.capacity) * year_fraction,
        axis=1,
    )
    # Compute annual costs for each components
    asset_results["annual_costs"] = asset_results.apply(
        lambda x: (x.annuity * x.capacity) * year_fraction
        + x.total_flow * x.opex_variable,
        axis=1,
    )

    asset_results["total_opex_costs"] = asset_results.apply(
        lambda x: (x.opex_fix * x.capacity) * year_fraction
        + x.total_flow * x.opex_variable
        + x.cash_flow,
        axis=1,
    )

    # Save the results
    asset_results = asset_results[RESULTS_COLUMN_NAMES]
    asset_results.to_csv(f"results_{case}.csv")

    NPV = (
        (asset_results.annual_costs.sum() + asset_results.cash_flow.sum()) / CRF
    ) + project_planning_cost

    # supplied demand
    total_demand = sequences_demand.sum(axis=0) + sequences_critical_demand.sum(axis=0)
    supplied_critical_demand = sequences_critical_demand.sum(axis=0)
    supplied_non_critical_demand = sequences_demand.sum(axis=0)

    # Levelized cost of electricity in the system in currency's Cent per kWh.
    lcoe = 100 * (NPV * CRF) / total_demand

    if case == "D":
        res = 0
    else:
        res = 100 * (total_demand - sequences_diesel_genset.sum(axis=0)) / total_demand

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

    original_demand = critical_demand[sequences_critical_demand.index].sum(
        axis=0
    ) + non_critical_demand[sequences_demand.index].sum(axis=0)

    total_opex_costs = asset_results.total_opex_costs.sum()
    first_investment = asset_results.first_investment.sum() + project_planning_cost
    overall_peak_demand = sequences_demand.max() + sequences_critical_demand.max()

    ##########################################################################
    # Print the results in the terminal
    ##########################################################################

    scalars = dict(
        lcoe=lcoe,
        npv=NPV,
        first_investment=first_investment,
        critical_demand_fulfilled=critical_demand_fulfilled,
        demand_fulfilled=demand_fulfilled,
        excess_rate=excess_rate,
        supplied_demand=total_demand,
        original_demand=total_demand,
        total_opex_costs=total_opex_costs,
        res=res,
    )
    system_results = pd.DataFrame.from_records(
        [i for i in scalars.items()], columns=["param", "value"]
    ).set_index("param")

    help_lcoe = """The lcoe is calculated as : (NPV * CRF) / (total critical demand supplied + total non critical demand supplied)
    NPV = sum_i{ annual_costs_i + cash_flow_i)} / CRF (where the sum is over each asset)

    NPV = sum_i{ annual_costs_i + fuel_cost_per_liter_i * consumed_liters_i)} / CRF 

    NPV = sum_i{ annual_costs_i + fuel_cost_per_liter_i * consumed_liters_i)} / CRF ( the consumed_liter_i is equal to the total_flow_i / fuel_energy_density_i / fuel_density_i

    NPV = sum_i{annuity_i * optimized_capacity_i) * year_fraction + total_flow_i * opex_variable_i + fuel_cost_per_liter_i * consumed_liters_i} / CRF

    Note: the CRF factors do cancel each other out in the lcoe calculation

    Note: CRF = (wacc * (1 + wacc) ** n) / ((1 + wacc) ** n - 1), where n ( project_lifetime in years) and wacc are user inputs
    
    Note the annuity of an asset is either a user input (if provided under the column "annuity" under "costs" tab of input file) or calculated from the capex_variable and opex_fix provided by the user
    
    If the asset lifetime is greater or equal to the project lifetime, there is no need to change the asset during the 
    project and the annuity of one asset is calculated the following way:
    
    annuity_i = capex_variable_i * (wacc * (1 + wacc) ** n) / ((1 + wacc) ** n - 1)  + opex_fix_i (1)
    
    If the asset lifetime is smaller than the project lifetime, there are replacement costs
    
    The quantity capex_variable_i * (wacc * (1 + wacc) ** n) / ((1 + wacc) ** n - 1) in equation (1) above 
    is equal to the first_time_investment, in case of replacement of assets this has to be adapted like:
    
    sum_j{first_time_investment/(1 + wacc) ** (j * asset_lifetime)} (where j goes from 0 up to the number of 
    replacements of the asset)
    
    And the annuity becomes:
    annuity_i = sum_j{first_time_investment/(1 + wacc) ** (j * asset_lifetime)}  + opex_fix_i
    
    annuity_i = sum_j{capex_variable_i * (1 + 1/ * (wacc * (1 + wacc) ** n) / (((1 + wacc) ** n - 1) * (1 + wacc) ** (j * asset_lifetime))}  + opex_fix_i
    Reminder: annuity is only calculated that way if NOT provided explicitly by the user

    """

    print(50 * "*")
    print(f"Overall Peak Demand:\t {overall_peak_demand:.0f} kW")
    print(f"LCOE:\t\t {lcoe:.2f} cent/kWh")
    print(f"NPV:\t\t {NPV:.2f} USD")
    print(f"Total opex costs :\t\t {total_opex_costs:.2f} USD/{settings.n_days} days")
    print(f"First investment :\t\t {first_investment:.2f} USD")
    print(
        f"Fuel expenditure :\t\t {asset_results.cash_flow.sum()*CRF:.2f} USD/{settings.n_days} days"
    )
    print(f"RES:\t\t {res:.0f}%")
    print(f"Excess:\t\t {excess_rate:.1f}% of the total production")
    print(f"Supplied demand:\t\t {total_demand:.1f} kWh")
    print(f"Supplied critical demand:\t\t {supplied_critical_demand:.1f} kWh")
    print(f"Supplied non critical demand:\t\t {supplied_non_critical_demand:.1f} kWh")
    print(f"Original demand:\t\t {original_demand:.1f} kWh")
    print(
        f"Share of critical demand fulfilled :\t\t {critical_demand_fulfilled:.0f}% of the total critical demand"
    )
    print(
        f"Share of non-critical demand fulfilled :\t\t {demand_fulfilled:.0f}% of the total non critical demand"
    )
    print(50 * "*")
    print("Optimal Capacities:")
    print("-------------------")
    for asset, capacity in capacities.items():
        unit = "kWh" if asset in ["battery", "h2_storage"] else "kW"
        print(f"{asset.title()}: {capacity:.1f} {unit}")
    print(50 * "*")

    ##############################################
    # ------------ Set up DashApp -------------- #
    ##############################################

    result_div = html.Div(
        children=[
            html.Div(
                children=[
                    html.P(f"Overall peak Demand:\t {overall_peak_demand:.1f} kW"),
                    html.P(f"LCOE:\t\t {lcoe:.2f} cent/kWh", title=help_lcoe),
                    html.P(
                        f"First investment :\t\t {asset_results.first_investment.sum():.2f} USD",
                        title="It is the sum of the product of optimized capacity and annualized costs of each asset",
                    ),
                    html.P(
                        f"Fuel expenditure :\t\t {asset_results.cash_flow.sum()*CRF:.2f} USD/{settings.n_days} days"
                    ),
                    html.P(f"RES:\t\t {res:.0f}%"),
                    html.P(f"Excess:\t\t {excess_rate:.1f}% of the total production"),
                    html.P(
                        f"Share of critical demand fulfilled :\t\t {critical_demand_fulfilled:.0f}%"
                    ),
                    html.P(
                        f"Share of non-critical demand fulfilled :\t\t {demand_fulfilled:.0f}%"
                    ),
                ],
                style={"display": "flex", "justify-content": "space-between"},
            ),
            html.H3("Optimal Capacities:"),
            html.Div(
                children=[
                    html.P(f"{asset.title()}:\t {capacity:.1f} kW") for asset, capacity in capacities.items()
                ],
                style={"display": "flex", "justify-content": "space-between"},
            ),
        ]
    )

    return (
        results,
        asset_results,
        energy_system,
        result_div,
        system_results,
        date_time_index,
        non_critical_demand,
        critical_demand,
    )


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
                y=nc_demand * (1 - settings.maximum_demand_reduction),
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


def plot_bus_flows(busses, results):
    bus_figures = []

    for bus in busses:
        if bus != "battery":
            fig = go.Figure(layout=dict(title=f"{bus} bus node"))
            for t, g in solph.views.node(results, node=bus)["sequences"].items():
                idx_asset = abs(t[0].index(bus) - 1)

                fig.add_trace(
                    go.Scatter(
                        x=g.index, y=g.values * pow(-1, idx_asset), name=t[0][idx_asset]
                    )
                )
        else:
            capacity_battery = asset_results.capacity.battery
            if capacity_battery != 0:
                soc_battery = (
                    solph.views.node(results, node=bus)["sequences"][
                        (("battery", "None"), "storage_content")
                    ]
                    / capacity_battery
                )
            else:
                soc_battery = solph.views.node(results, node=bus)["sequences"][
                    (("battery", "None"), "storage_content")
                ]

            fig = go.Figure(layout=dict(title=f"{bus} node"))

            fig.add_trace(
                go.Scatter(
                    x=soc_battery.index, y=soc_battery.values, name="soc battery"
                )
            )

        bus_figures.append(fig)
    return bus_figures


if __name__ == "__main__":
    filename = "input_case.xlsx"

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"The file was not found, make sure you you did not make a typo in its name or that the file is accessible from where you executed this code"
        )
    df_costs, data, settings, _ = read_input_file(filename)
    (
        results,
        asset_results,
        energy_system,
        result_div,
        system_results,
        date_time_index,
        non_critical_demand,
        critical_demand,
    ) = run_simulation(df_costs, data, settings)
    case = settings.case
    energy_system_graph = encode_image_file(f"case_{case}.png")
    soc_min = df_costs["soc_min"]
    soc_max = df_costs["soc_max"]

    if case == "D":
        busses = ["electricity_ac"]
    else:
        busses = ["electricity_ac", "electricity_dc", "battery", "hydrogen"]

    # plot bus flows
    bus_figures = plot_bus_flows(busses, results)

    # loading external resources
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    options = dict(
        # external_stylesheets=external_stylesheets
    )

    ######### --------- Dash app layout ---------- ###

    demo_app = dash.Dash(__name__, **options)

    demo_app.layout = html.Div(
        children=[
            html.H3("Model inputs"),
            html.Div(
                children=[
                    html.P(f"{param.title()}: {settings[param]}")
                    for param in settings.index
                    if param != "port"
                ],
                style={"display": "flex", "justify-content": "space-evenly"},
            ),
            html.Div(
                children=dash_table.DataTable(
                    df_costs.reset_index().to_dict("records"),
                    [{"name": i, "id": i} for i in df_costs.reset_index().columns],
                )
            ),
            html.Div(children=[html.H3("Results in numbers"), result_div]),
            html.Div(
                children=[
                    html.H3("Non critical demand reduction overview"),
                    dcc.Graph(
                        id="nc_demand_supply", figure=reduced_demand_fig(results)
                    ),
                ]
            ),
            html.Div(
                children=dash_table.DataTable(
                    asset_results.reset_index().to_dict("records"),
                    [{"name": i, "id": i} for i in asset_results.reset_index().columns],
                )
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
                max=len(date_time_index),
                # marks={k: v for k, v in enumerate(date_time_index)},
            ),
            dcc.Dropdown(
                id="ts_slice_select",
                options={k: v for k, v in enumerate(date_time_index)},
                value=None,
            ),
            dcc.Graph(id="sankey", figure=sankey(energy_system, results)),
        ]
        + [
            dcc.Graph(
                id=f"{bus}-id",
                figure=fig,
            )
            for bus, fig in zip(busses, bus_figures)
        ]
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
        + [
            Output(component_id=f"{bus}-id", component_property="figure")
            for bus in busses
        ],
        # Triggers the callback when the value of one of these components of the layout is changed
        Input(component_id="ts_slice_select", component_property="value"),
    )
    def update_figures(ts):
        ts = int(ts) if ts is not None else 0
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
            if bus != "battery":
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
                            x=g.index,
                            y=g.values * negative_sign,
                            name=asset_name,
                            **opts,
                        )
                    )
                    if g.max() > max_y:
                        max_y = g.max()
            else:
                capacity_battery = asset_results.capacity.battery
                if capacity_battery != 0:
                    soc_battery = (
                        solph.views.node(results, node=bus)["sequences"][
                            (("battery", "None"), "storage_content")
                        ]
                        / capacity_battery
                    )
                else:
                    soc_battery = solph.views.node(results, node=bus)["sequences"][
                        (("battery", "None"), "storage_content")
                    ]

                fig = go.Figure(layout=dict(title=f"{bus} node", yaxis_range=[0, 1]))

                fig.add_trace(
                    go.Scatter(
                        x=soc_battery.index, y=soc_battery.values, name="soc battery"
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=soc_battery.index,
                        y=np.ones(len(soc_battery.index)) * soc_min.battery,
                        name="min soc battery",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=soc_battery.index,
                        y=np.ones(len(soc_battery.index)) * soc_max.battery,
                        name="max soc battery",
                    )
                )
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

    demo_app.run_server(debug=True, port=settings.port, use_reloader=False)
