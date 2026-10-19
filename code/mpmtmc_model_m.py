"""
mpmtmc_model_m.py

Model M (anticipatory) for the Multi-Period, Multi-Tour, Multi-Compartment
Vehicle Routing Problem with endogenous cleaning decisions and
compartment-state inheritance (MPMTMC-VRP).

Companion code for:
  Ghodhbani, I. & Ben Aissa, M. (2026). A Multi-Period, Multi-Tour,
  Multi-Compartment Vehicle Routing Problem with Endogenous Cleaning:
  Hierarchical Framework and Structural Analysis.

Model M allows a compartment to be cleaned without an immediate load
(anticipatory cleaning, variable delta), creating a persistent CLEAN state
that is inherited across tours and periods. Model R (see
mpmtmc_model_r.py) is obtained from this file by removing delta and
adding the anti-anticipation constraint H <= comp_used.
"""

import os
from copy import deepcopy
from math import ceil
from itertools import combinations

import pandas as pd
from pyomo.environ import (
    Binary, ConcreteModel, Constraint, ConstraintList, NonNegativeReals,
    Objective, Param, Set, SolverFactory, Var, Expression, minimize, value
)
from pyomo.opt import TerminationCondition


# =============================================================================
# 1. DATA LOADING
# =============================================================================
def load_instance_data(excel_path: str) -> dict:
    xl = pd.ExcelFile(excel_path)

    params_df   = pd.read_excel(xl, sheet_name="Params")
    suppliers_df= pd.read_excel(xl, sheet_name="Suppliers")
    fleet_df    = pd.read_excel(xl, sheet_name="Fleet")
    init_df     = pd.read_excel(xl, sheet_name="Initial_States")
    gamma_df    = pd.read_excel(xl, sheet_name="Incompatibility")
    nodes_df    = pd.read_excel(xl, sheet_name="Nodes")
    dist_df     = pd.read_excel(xl, sheet_name="Distances")

    # --- Quality list: derived from actual supplier data, ordered EV>V>O>L ---
    quality_order = ["EV", "V", "O", "L"]
    raw_quals = suppliers_df["Quality"].dropna().unique().tolist()
    transport_qualities_ordered = [q for q in quality_order if q in raw_quals]
    qualities = ["CLEAN"] + transport_qualities_ordered
    print(f"Qualities detected from instance: {qualities}")

    strict_qualities = ["EV"]
    params = dict(zip(params_df["Parameter"].str.strip(), params_df["Value"]))

    loading_time_per_ton  = float(params.get("LoadingTimePerTon", 1.0))
    fixed_cost_per_vehicle= float(params.get("FixedCostPerVehicle", 15.0))

    node_ids  = nodes_df["Node_ID"].astype(str).tolist()
    depot     = nodes_df.loc[nodes_df["Node_Type"] == "Depot",
                             "Node_ID"].astype(str).iloc[0]

    num_periods = int(params["Num_Periods"])
    periods     = list(range(1, num_periods + 1))

    # --- Demand ---
    demand = {}
    for _, row in suppliers_df.iterrows():
        i = str(row["Supplier_ID"])
        q = str(row["Quality"])
        for p in periods:
            col = f"P{p}_Demand_t"
            val = row[col] if pd.notna(row[col]) else 0.0
            if val > 0:
                demand[(i, q, p)] = float(val)

    all_supplier_ids = suppliers_df["Supplier_ID"].astype(str).unique().tolist()
    suppliers = [sid for sid in all_supplier_ids if sid in node_ids]

    # --- Fleet ---
    vehicles = sorted(fleet_df["Vehicle_ID"].astype(str).unique().tolist())
    compartments_by_vehicle = {}
    capacity     = {}
    availability = {}
    clean_cost   = {}

    for _, row in fleet_df.iterrows():
        v = str(row["Vehicle_ID"])
        c = str(row["Compartment_ID"])
        compartments_by_vehicle.setdefault(v, []).append(c)
        capacity[(v, c)] = float(row["Capacity_t"])
        for p in periods:
            availability[(v, c, p)] = int(row[f"P{p}_Available"])
            clean_cost[(v, c, p)]   = float(row[f"CleanCost_P{p}"])

    # --- Initial states: period 1 only from Excel ---
    default_quality = "L"
    p1 = periods[0]
    init_quality = {}
    # Cells actually left empty in the Excel are tracked separately (used
    # only by Model R, which treats a missing initial state as CLEAN rather
    # than "L"); Model M keeps its usual default of "L" for these cells.
    init_was_missing = set()

    for _, row in init_df.iterrows():
        v = str(row["Vehicle_ID"])
        c = str(row["Compartment_ID"])
        val = row.get("P1_Init_Quality", None)
        if pd.isna(val):
            q0 = default_quality
            init_was_missing.add((v, c))
            print(f"  ⚠ P1_Init_Quality missing for {v}/{c}, "
                  f"using default '{default_quality}'")
        else:
            q0 = str(val).strip()
        if q0 not in qualities:
            raise ValueError(f"Initial quality '{q0}' not allowed ({v}/{c})")
        init_quality[(v, c, p1)] = q0

    for v in vehicles:
        for c in compartments_by_vehicle[v]:
            if (v, c, p1) not in init_quality:
                print(f"  ⚠ No initial state for {v}/{c} — "
                      f"defaulting to '{default_quality}'")
                init_quality[(v, c, p1)] = default_quality
                init_was_missing.add((v, c))
            print(f"  Init {v}/{c} P1: {init_quality[(v, c, p1)]}")

    # Build s_init: p1 uses actual quality, p>=2 uses zeros
    s_init = {}
    for v in vehicles:
        for c in compartments_by_vehicle[v]:
            q0 = init_quality.get((v, c, p1), default_quality)
            for p in periods:
                for q in qualities:
                    if p == p1:
                        s_init[(v, c, p, q)] = 1 if q == q0 else 0
                    else:
                        s_init[(v, c, p, q)] = 0

    # --- Compatibility matrix Gamma ---
    gamma_df = gamma_df.dropna(axis=1, how="all")
    gamma_df.columns = [str(col).strip() for col in gamma_df.columns]
    gamma_df = gamma_df.loc[
        :, ~gamma_df.columns.str.contains(r"^Unnamed", case=False)]
    if gamma_df.columns[0] != "prev_quality":
        gamma_df = gamma_df.rename(
            columns={gamma_df.columns[0]: "prev_quality"})
    gamma_df = gamma_df.dropna(how="all")
    gamma_df["prev_quality"] = gamma_df["prev_quality"].astype(str).str.strip()

    gamma = {}
    for _, row in gamma_df.iterrows():
        q_prev = row["prev_quality"]
        if q_prev not in qualities:
            continue
        for q_next in qualities:
            if q_next not in gamma_df.columns:
                raise ValueError(f"Column '{q_next}' missing in Gamma sheet")
            val = row[q_next]
            if pd.isna(val):
                raise ValueError(f"Missing Gamma value {q_prev}->{q_next}")
            gamma[(q_prev, q_next)] = int(val)

    # --- Distances ---
    dist_df = dist_df.rename(columns={dist_df.columns[0]: "From"})
    distances = {}
    for _, row in dist_df.iterrows():
        i = str(row["From"])
        for j in node_ids:
            distances[(i, j)] = float(row[j])

    travel_time   = distances.copy()
    cleaning_time = 15.0
    t_max_global = float(params.get("T_max", 54000))
    time_budget = {}
    for p in periods:
        key = f"T_max_P{p}"
        time_budget[p] = float(params.get(key, t_max_global))

    return {
        "qualities":               qualities,
        "strict_qualities":        strict_qualities,
        "nodes":                   node_ids,
        "suppliers":               suppliers,
        "depot":                   depot,
        "periods":                 periods,
        "vehicles":                vehicles,
        "compartments_by_vehicle": compartments_by_vehicle,
        "demand":                  demand,
        "capacity":                capacity,
        "availability":            availability,
        "clean_cost":              clean_cost,
        "s_init":                  s_init,
        "init_was_missing":        init_was_missing,
        "gamma":                   gamma,
        "distances":               distances,
        "travel_time":             travel_time,
        "cleaning_time":           cleaning_time,
        "time_budget":             time_budget,
        "loading_time_per_ton":    loading_time_per_ton,
        "fixed_cost_per_vehicle":  fixed_cost_per_vehicle,
    }


# =============================================================================
# 2. MODEL CONSTRUCTION
# =============================================================================
def build_mpmttmc_model(data: dict, max_tours_per_period: int = 1,
                        add_cuts: bool = True,
                        dfj_max_size: int = 3,
                        M_big: int = 5) -> ConcreteModel:
    """
    dfj_max_size : 0 = disabled, 2 = pairs, 3 = pairs + triplets.
    M_big : big-M coefficient for the traceability constraint.
    """
    m = ConcreteModel()

    # ---- Sets ----
    m.Q  = Set(initialize=data["qualities"])
    m.N  = Set(initialize=data["nodes"])
    m.I  = Set(initialize=data["suppliers"])
    m.P  = Set(initialize=data["periods"])
    m.V  = Set(initialize=data["vehicles"])
    m.R  = Set(initialize=list(range(1, max_tours_per_period + 1)))
    vc_pairs = [(v, c) for v in data["vehicles"]
                for c in data["compartments_by_vehicle"][v]]
    m.VC = Set(dimen=2, initialize=vc_pairs)

    # ---- Derived index lists ----
    transport_qualities = [q for q in data["qualities"] if q != "CLEAN"]
    m.QT   = Set(initialize=transport_qualities)
    P_list = sorted(data["periods"])
    R_list = sorted(range(1, max_tours_per_period + 1))
    p1     = P_list[0]
    r_last = R_list[-1]

    # ---- Parameters ----
    m.d          = Param(m.N, m.N,
                         initialize=lambda m, i, j:
                         data["distances"].get((i, j), 0.0))
    m.travel_time= Param(m.N, m.N,
                         initialize=lambda m, i, j:
                         data["travel_time"].get((i, j), 0.0))
    m.Cap        = Param(m.VC,
                         initialize=lambda m, v, c: data["capacity"][(v, c)])
    m.Gamma      = Param(m.Q, m.Q,
                         initialize=lambda m, q, qp: data["gamma"][(q, qp)])
    m.Cclean     = Param(m.VC, m.P,
                         initialize=lambda m, v, c, p:
                         data["clean_cost"][(v, c, p)])
    m.Avail      = Param(m.VC, m.P,
                         initialize=lambda m, v, c, p:
                         data["availability"][(v, c, p)])
    m.Tclean     = Param(initialize=data["cleaning_time"])
    m.theta      = Param(initialize=data["loading_time_per_ton"])
    m.Tmax       = Param(m.P,
                         initialize=lambda m, p: data["time_budget"][p])
    m.fixed_cost = Param(initialize=data["fixed_cost_per_vehicle"])
    m.D          = Param(m.I, m.Q, m.P,
                         initialize=lambda m, i, q, p:
                         data["demand"].get((i, q, p), 0.0),
                         default=0.0)
    m.Sinit      = Param(m.VC, m.P, m.Q,
                         initialize=lambda m, v, c, p, q:
                         data["s_init"][(v, c, p, q)])

    # ---- Variables ----
    m.x          = Var(m.V, m.R, m.P, m.N, m.N,        domain=Binary)
    m.y          = Var(m.VC, m.R, m.P, m.Q,             domain=Binary)
    m.z          = Var(m.VC, m.R, m.P, m.I, m.Q,        domain=Binary)
    m.loaded_qty = Var(m.VC, m.R, m.P, m.I, m.Q,        domain=NonNegativeReals)
    m.H          = Var(m.VC, m.R, m.P,                  domain=Binary)
    m.S          = Var(m.VC, m.R, m.P, m.Q,             domain=Binary)
    m.comp_used  = Var(m.VC, m.R, m.P,                  domain=Binary)
    m.delta      = Var(m.VC, m.R, m.P,                  domain=Binary)
    m.Send       = Var(m.VC, m.P, m.Q,                  domain=Binary)
    m.visit      = Var(m.V, m.R, m.P, m.I,              domain=Binary)
    m.tour_used  = Var(m.V, m.R, m.P,                   domain=Binary)
    m.vehicle_used=Var(m.V, m.P,                         domain=Binary)
    m.u          = Var(m.V, m.R, m.P, m.I,              domain=NonNegativeReals)
    m.w_strict   = Var(m.VC, m.R, m.P,                  domain=Binary)

    # Start-of-period state as an Expression (not a variable): anchored at
    # p1 by Sinit, inherited thereafter from the previous period's Send.
    # This keeps the LP relaxation exact on the inheritance equalities
    # without adding |VC|x|P|x|Q| extra binaries.
    def Sstart_rule(model, v, c, p, q):
        if p == p1:
            return model.Sinit[v, c, p, q]
        else:
            return model.Send[v, c, p - 1, q]
    m.Sstart_expr = Expression(m.VC, m.P, m.Q, rule=Sstart_rule)

    # ---- Objective ----
    m.obj = Objective(
        expr=(
            sum(m.d[i, j] * m.x[v, r, p, i, j]
                for v in m.V for r in m.R for p in m.P
                for i in m.N for j in m.N if i != j)
            + sum(m.Cclean[v, c, p] * m.H[v, c, r, p]
                  for (v, c) in m.VC for r in m.R for p in m.P)
            + sum(m.fixed_cost * m.vehicle_used[v, p]
                  for v in m.V for p in m.P)
        ),
        sense=minimize
    )

    # =========================================================================
    # SYMMETRY BREAKING — identical vehicles/compartments, first tour only
    # =========================================================================
    p1 = data["periods"][0]
    r1 = 1

    def vehicle_fingerprint(v):
        comps = sorted(data["compartments_by_vehicle"][v])
        caps = tuple(data["capacity"][(v, c)] for c in comps)
        init_states = tuple(
            next(q for q in data["qualities"] if data["s_init"][(v, c, p1, q)] == 1)
            for c in comps
        )
        return (len(comps), caps, init_states)

    groups = {}
    for v in data["vehicles"]:
        groups.setdefault(vehicle_fingerprint(v), []).append(v)

    m.sym_vehicle_first_tour = ConstraintList()
    for fp, veh_list in groups.items():
        if len(veh_list) >= 2:
            sorted_veh = sorted(veh_list)
            for v1, v2 in combinations(sorted_veh, 2):
                dist1 = sum(m.d[i, j] * m.x[v1, r1, p1, i, j] for i in m.N for j in m.N if i != j)
                dist2 = sum(m.d[i, j] * m.x[v2, r1, p1, i, j] for i in m.N for j in m.N if i != j)
                m.sym_vehicle_first_tour.add(dist1 >= dist2)

    m.sym_comp_first_tour = ConstraintList()
    for v in data["vehicles"]:
        comps_by_cap = {}
        for c in data["compartments_by_vehicle"][v]:
            cap = data["capacity"][(v, c)]
            init_q = next(q for q in data["qualities"] if data["s_init"][(v, c, p1, q)] == 1)
            comps_by_cap.setdefault((cap, init_q), []).append(c)
        for (cap, init_q), comp_list in comps_by_cap.items():
            if len(comp_list) >= 2:
                sorted_comps = sorted(comp_list)
                for c1, c2 in combinations(sorted_comps, 2):
                    qty1 = sum(m.loaded_qty[v, c1, r1, p1, i, q] for i in m.I for q in m.Q)
                    qty2 = sum(m.loaded_qty[v, c2, r1, p1, i, q] for i in m.I for q in m.Q)
                    m.sym_comp_first_tour.add(qty1 >= qty2)

    # =========================================================================
    # ROUTING CONSTRAINTS
    # =========================================================================
    m.no_self_loop = ConstraintList()
    for v in m.V:
        for r in m.R:
            for p in m.P:
                for i in m.N:
                    m.no_self_loop.add(m.x[v, r, p, i, i] == 0)

    m.tour_depart = ConstraintList()
    m.tour_return = ConstraintList()
    for v in m.V:
        for r in m.R:
            for p in m.P:
                m.tour_depart.add(
                    sum(m.x[v, r, p, data["depot"], j] for j in m.I)
                    == m.tour_used[v, r, p])
                m.tour_return.add(
                    sum(m.x[v, r, p, i, data["depot"]] for i in m.I)
                    == m.tour_used[v, r, p])

    m.flow_conservation = ConstraintList()
    for v in m.V:
        for r in m.R:
            for p in m.P:
                for i in m.I:
                    m.flow_conservation.add(
                        sum(m.x[v, r, p, j, i] for j in m.N if j != i)
                        == sum(m.x[v, r, p, i, j] for j in m.N if j != i))

    m.visit_in  = ConstraintList()
    m.visit_out = ConstraintList()
    for v in m.V:
        for r in m.R:
            for p in m.P:
                for i in m.I:
                    m.visit_in.add(
                        sum(m.x[v, r, p, j, i] for j in m.N if j != i)
                        == m.visit[v, r, p, i])
                    m.visit_out.add(
                        sum(m.x[v, r, p, i, j] for j in m.N if j != i)
                        == m.visit[v, r, p, i])

    m.tour_order = ConstraintList()
    for v in m.V:
        for p in m.P:
            for idx in range(len(R_list) - 1):
                m.tour_order.add(
                    m.tour_used[v, R_list[idx + 1], p]
                    <= m.tour_used[v, R_list[idx], p])

    # Complete subtour elimination (MTZ). DFJ cuts on small subsets are
    # added below (add_valid_inequalities) as reinforcing valid
    # inequalities only — MTZ remains the structural guarantee.
    nI = len(data["suppliers"])
    m.mtz_bounds = ConstraintList()
    m.mtz_ct     = ConstraintList()
    for v in m.V:
        for r in m.R:
            for p in m.P:
                for i in m.I:
                    m.mtz_bounds.add(m.u[v, r, p, i] >= m.visit[v, r, p, i])
                    m.mtz_bounds.add(m.u[v, r, p, i] <= nI * m.visit[v, r, p, i])
                for i in m.I:
                    for j in m.I:
                        if i != j:
                            m.mtz_ct.add(
                                m.u[v, r, p, i] - m.u[v, r, p, j]
                                + nI * m.x[v, r, p, i, j] <= nI - 1)

    # =========================================================================
    # LOADING, ASSIGNMENT AND CAPACITY CONSTRAINTS
    # =========================================================================
    m.no_illegal_loading = ConstraintList()
    for i in m.I:
        for q in m.Q:
            for p in m.P:
                if data["demand"].get((i, q, p), 0) == 0:
                    for (v, c) in m.VC:
                        for r in m.R:
                            m.no_illegal_loading.add(
                                m.loaded_qty[v, c, r, p, i, q] == 0)

    m.no_clean_transport = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                m.no_clean_transport.add(m.y[v, c, r, p, "CLEAN"] == 0)

    m.one_quality = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                m.one_quality.add(
                    sum(m.y[v, c, r, p, q] for q in m.Q) <= 1)

    m.compartment_availability = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                m.compartment_availability.add(
                    sum(m.y[v, c, r, p, q] for q in m.Q) <= m.Avail[v, c, p])

    m.compartment_tour_link = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                m.compartment_tour_link.add(
                    sum(m.y[v, c, r, p, q] for q in m.Q)
                    <= m.tour_used[v, r, p])

    m.capacity_ct = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                m.capacity_ct.add(
                    sum(m.loaded_qty[v, c, r, p, i, q]
                        for i in m.I for q in m.Q)
                    <= m.Cap[v, c])

    m.load_link = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                for i in m.I:
                    for q in m.Q:
                        m.load_link.add(
                            m.loaded_qty[v, c, r, p, i, q]
                            <= m.Cap[v, c] * m.z[v, c, r, p, i, q])

    m.assign_link = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                for i in m.I:
                    for q in m.Q:
                        m.assign_link.add(
                            m.z[v, c, r, p, i, q] <= m.y[v, c, r, p, q])

    m.qty_visit_link = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                for i in m.I:
                    for q in m.Q:
                        m.qty_visit_link.add(
                            m.loaded_qty[v, c, r, p, i, q]
                            <= m.Cap[v, c] * m.visit[v, r, p, i])

    m.service_visit_link = ConstraintList()
    for v in m.V:
        for r in m.R:
            for p in m.P:
                for i in m.I:
                    m.service_visit_link.add(
                        sum(m.z[v, c, r, p, i, q]
                            for (vv, c) in m.VC if vv == v
                            for q in m.Q)
                        <= len(data["compartments_by_vehicle"][v])
                        * m.visit[v, r, p, i])

    # y is bounded above by the binary assignment count (sum of z), not by
    # the continuous loaded quantity: a sub-unit delivery would otherwise
    # wrongly force y=0 even when a supplier is genuinely assigned.
    m.y_requires_loading = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                for q in transport_qualities:
                    m.y_requires_loading.add(
                        m.y[v, c, r, p, q]
                        <= sum(m.z[v, c, r, p, i, q] for i in m.I))

    # =========================================================================
    # DEMAND SATISFACTION — indivisibility (as in Lahyani et al., 2015)
    # =========================================================================
    m.demand_sat = ConstraintList()
    for i in m.I:
        for q in transport_qualities:
            for p in m.P:
                D_val = data["demand"].get((i, q, p), 0.0)
                if D_val > 0:
                    m.demand_sat.add(
                        sum(m.z[v, c, r, p, i, q]
                            for (v, c) in m.VC for r in m.R) == 1
                    )
                else:
                    for (v, c) in m.VC:
                        for r in m.R:
                            m.demand_sat.add(
                                m.z[v, c, r, p, i, q] == 0
                            )

    m.load_indivisibility = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                for i in m.I:
                    for q in transport_qualities:
                        D_val = data["demand"].get((i, q, p), 0.0)
                        m.load_indivisibility.add(
                            m.loaded_qty[v, c, r, p, i, q]
                            == D_val * m.z[v, c, r, p, i, q]
                        )

    # =========================================================================
    # TOUR AND VEHICLE ACTIVATION
    # =========================================================================
    # Tour activation is expressed through the binary assignment z, not the
    # continuous quantity L, since L can legitimately be below one unit.
    m.tour_used_only_if_loading = ConstraintList()
    for v in m.V:
        for r in m.R:
            for p in m.P:
                m.tour_used_only_if_loading.add(
                    sum(m.z[v, c, r, p, i, q]
                        for (vv, c) in m.VC if vv == v
                        for i in m.I for q in m.Q)
                    >= m.tour_used[v, r, p])

    m.link_vehicle_used = ConstraintList()
    for v in m.V:
        for p in m.P:
            for r in m.R:
                m.link_vehicle_used.add(
                    m.vehicle_used[v, p] >= m.tour_used[v, r, p])

    # =========================================================================
    # COMPARTMENT-STATE DYNAMICS
    # =========================================================================
    m.comp_used_def = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                m.comp_used_def.add(
                    m.comp_used[v, c, r, p]
                    == sum(m.y[v, c, r, p, q] for q in transport_qualities))

    # McCormick linearisation of delta = H AND (NOT comp_used): delta=1 iff
    # the compartment is cleaned without loading in the same tour.
    m.delta_ct = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                m.delta_ct.add(
                    m.delta[v, c, r, p] <= m.H[v, c, r, p])
                m.delta_ct.add(
                    m.delta[v, c, r, p] <= 1 - m.comp_used[v, c, r, p])
                m.delta_ct.add(
                    m.delta[v, c, r, p]
                    >= m.H[v, c, r, p] - m.comp_used[v, c, r, p])

    # State update, unified over all q in Q (transport grades + CLEAN):
    # `created` is y[...,q] for a transported grade, delta for CLEAN, and
    # `occ` (= comp_used + delta) marks the tour as "occupied" either by a
    # load or by an anticipatory cleaning. This single formulation avoids
    # a contradiction that a separate CLEAN-only equality would otherwise
    # create against the generic idle-tour inertia bounds.
    def _created(m, v, c, r, p, q):
        if q == "CLEAN":
            return m.delta[v, c, r, p]
        return m.y[v, c, r, p, q]

    m.state_update_r1 = ConstraintList()
    for (v, c) in m.VC:
        for p in m.P:
            occ = m.comp_used[v, c, 1, p] + m.delta[v, c, 1, p]
            for q in m.Q:
                created = _created(m, v, c, 1, p, q)
                m.state_update_r1.add(m.S[v, c, 1, p, q] >= created)
                m.state_update_r1.add(
                    m.S[v, c, 1, p, q] <= created + (1 - occ))
                m.state_update_r1.add(
                    m.S[v, c, 1, p, q]
                    <= created + m.Sstart_expr[v, c, p, q])

    m.state_update_rge2 = ConstraintList()
    for (v, c) in m.VC:
        for p in m.P:
            for r in range(2, max_tours_per_period + 1):
                occ = m.comp_used[v, c, r, p] + m.delta[v, c, r, p]
                for q in m.Q:
                    created = _created(m, v, c, r, p, q)
                    m.state_update_rge2.add(m.S[v, c, r, p, q] >= created)
                    m.state_update_rge2.add(
                        m.S[v, c, r, p, q] <= created + (1 - occ))
                    m.state_update_rge2.add(
                        m.S[v, c, r, p, q]
                        <= created + m.S[v, c, r - 1, p, q])

    m.state_onehot = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                m.state_onehot.add(
                    sum(m.S[v, c, r, p, q] for q in m.Q) == 1)

    m.period_end_state = ConstraintList()
    for (v, c) in m.VC:
        for p in m.P:
            for q in m.Q:
                m.period_end_state.add(
                    m.Send[v, c, p, q] == m.S[v, c, r_last, p, q])

    m.period_end_onehot = ConstraintList()
    for (v, c) in m.VC:
        for p in m.P:
            m.period_end_onehot.add(
                sum(m.Send[v, c, p, q] for q in m.Q) == 1)

    # A CLEAN state created by a paid cleaning operation (delta=1 at some
    # point of the horizon) must be consumed by a load before the horizon
    # ends, since it has a cost. The initial CLEAN state (Sinit), which is
    # free, may remain unused without penalty.
    m.has_been_cleaned = Var(m.VC, domain=Binary)

    m.has_been_cleaned_link = ConstraintList()
    for (v, c) in m.VC:
        for r in m.R:
            for p in m.P:
                m.has_been_cleaned_link.add(
                    m.has_been_cleaned[v, c] >= m.delta[v, c, r, p])

    p_last = P_list[-1]
    m.no_wasted_clean = ConstraintList()
    for (v, c) in m.VC:
        m.no_wasted_clean.add(
            m.Send[v, c, p_last, "CLEAN"] <= 1 - m.has_been_cleaned[v, c])

    # =========================================================================
    # COMPATIBILITY (big-M direct, no auxiliary T variable)
    # =========================================================================
    m.first_tour_clean = ConstraintList()
    for (v, c) in m.VC:
        for p in m.P:
            for qp in transport_qualities:
                m.first_tour_clean.add(
                    sum(m.Gamma[q, qp] * m.Sstart_expr[v, c, p, q] for q in m.Q)
                    + m.H[v, c, 1, p]
                    >= m.y[v, c, 1, p, qp])

    m.clean_feas = ConstraintList()
    for (v, c) in m.VC:
        for p in m.P:
            for r in range(2, max_tours_per_period + 1):
                for qp in transport_qualities:
                    m.clean_feas.add(
                        sum(m.Gamma[q, qp] * m.S[v, c, r - 1, p, q] for q in m.Q)
                        + m.H[v, c, r, p]
                        >= m.y[v, c, r, p, qp])

    m.max_one_cleaning = ConstraintList()
    for (v, c) in m.VC:
        for p in m.P:
            m.max_one_cleaning.add(
                sum(m.H[v, c, r, p] for r in m.R)
                <= 1 + sum(m.comp_used[v, c, r, p]
                           for r in range(1, max_tours_per_period)))

    m.no_redundant_cleaning = ConstraintList()
    for (v, c) in m.VC:
        for p in m.P:
            m.no_redundant_cleaning.add(
                m.H[v, c, 1, p]
                <= 1 - m.Sstart_expr[v, c, p, "CLEAN"]
                + m.comp_used[v, c, 1, p])
            for r in range(2, max_tours_per_period + 1):
                m.no_redundant_cleaning.add(
                    m.H[v, c, r, p]
                    <= 1 - m.S[v, c, r-1, p, "CLEAN"]
                    + m.comp_used[v, c, r, p])

    # =========================================================================
    # TIME BUDGET
    # =========================================================================
    m.time_budget_ct = ConstraintList()
    for v in m.V:
        for p in m.P:
            m.time_budget_ct.add(
                sum(m.travel_time[i, j] * m.x[v, r, p, i, j]
                    for r in m.R for i in m.N for j in m.N if i != j)
                + sum(m.Tclean * m.H[v, c, r, p]
                      for (vv, c) in m.VC if vv == v for r in m.R)
                + sum(m.theta * m.loaded_qty[v, c, r, p, i, q]
                      for (vv, c) in m.VC if vv == v
                      for r in m.R for i in m.I for q in m.Q)
                <= m.Tmax[p])

    # =========================================================================
    # VALID INEQUALITIES
    # =========================================================================
    m.min_tours_cuts = ConstraintList()
    for p in m.P:
        total_demand = sum(m.D[i, q, p]
                           for i in m.I for q in transport_qualities)
        total_cap    = sum(m.Cap[v, c] for (v, c) in m.VC)
        if total_cap > 0:
            m.min_tours_cuts.add(
                sum(m.tour_used[v, r, p] for v in m.V for r in m.R)
                >= ceil(total_demand / total_cap))

    if add_cuts:
        add_valid_inequalities(m, data, transport_qualities, dfj_max_size)

    # =========================================================================
    # TRACEABILITY
    # =========================================================================
    strict_qualities = data.get("strict_qualities", [])

    def strict_link_rule(m, v, c, r, p):
        if not strict_qualities:
            return m.w_strict[v, c, r, p] == 0
        return m.w_strict[v, c, r, p] == sum(
            m.y[v, c, r, p, q] for q in strict_qualities)

    m.strict_link = Constraint(m.VC, m.R, m.P, rule=strict_link_rule)

    def traceability_rule(m, v, c, r, p):
        return (sum(m.z[v, c, r, p, i, q] for i in m.I for q in m.Q)
                <= 1 + M_big * (1 - m.w_strict[v, c, r, p]))

    m.traceability_ct = Constraint(m.VC, m.R, m.P, rule=traceability_rule)

    return m


# =============================================================================
# ADD_VALID_INEQUALITIES (with DFJ cuts)
# =============================================================================
def add_valid_inequalities(m, data, transport_qualities, dfj_max_size=2):
    """
    Strengthening cuts: vehicle, routing, visit-assignment, tour-assignment,
    and DFJ cuts (pairs and, optionally, triplets if dfj_max_size >= 2/3).
    """
    m.vehicle_upper = ConstraintList()
    for v in m.V:
        for p in m.P:
            m.vehicle_upper.add(
                m.vehicle_used[v, p] <= sum(m.tour_used[v, r, p] for r in m.R))

    m.routing_cuts = ConstraintList()
    for v in m.V:
        for r in m.R:
            for p in m.P:
                for i in m.I:
                    for j in m.I:
                        if i != j:
                            m.routing_cuts.add(
                                m.x[v, r, p, i, j] <= m.visit[v, r, p, i])
                            m.routing_cuts.add(
                                m.x[v, r, p, i, j] <= m.visit[v, r, p, j])

    for v in m.V:
        for r in m.R:
            for p in m.P:
                for i in m.I:
                    m.routing_cuts.add(
                        m.visit[v, r, p, i]
                        <= sum(m.z[v, c, r, p, i, q]
                               for (vv, c) in m.VC if vv == v
                               for q in transport_qualities))
                m.routing_cuts.add(
                    m.tour_used[v, r, p]
                    <= sum(m.z[v, c, r, p, i, q]
                           for (vv, c) in m.VC if vv == v
                           for i in m.I for q in transport_qualities))

    # DFJ cuts (reinforcing valid inequalities only — MTZ above remains the
    # structural subtour-elimination guarantee).
    if dfj_max_size >= 2:
        I_list = list(m.I)
        nI = len(I_list)
        if nI >= 2:
            for S in combinations(I_list, 2):
                for v in m.V:
                    for r in m.R:
                        for p in m.P:
                            expr = sum(m.x[v, r, p, i, j] for i in S for j in S if i != j)
                            m.routing_cuts.add(expr <= 1)
        if dfj_max_size >= 3 and nI >= 3:
            for S in combinations(I_list, 3):
                for v in m.V:
                    for r in m.R:
                        for p in m.P:
                            expr = sum(m.x[v, r, p, i, j] for i in S for j in S if i != j)
                            m.routing_cuts.add(expr <= 2)


# =============================================================================
# 3. EXTRACTION FUNCTIONS
# =============================================================================
def extract_tour_details(model) -> pd.DataFrame:
    rows = []
    for v in model.V:
        for r in model.R:
            for p in model.P:
                for i in model.N:
                    for j in model.N:
                        if i != j:
                            try:
                                if value(model.x[v, r, p, i, j]) > 0.5:
                                    rows.append({
                                        "Véhicule": v, "Tour": r,
                                        "Période": p, "Départ": i,
                                        "Arrivée": j,
                                        "Distance (km)": value(model.d[i, j]),
                                        "Temps (min)": value(model.travel_time[i, j])
                                    })
                            except Exception:
                                continue
    return pd.DataFrame(rows)


def extract_loading_details(model) -> pd.DataFrame:
    rows = []
    for (v, c) in model.VC:
        for r in model.R:
            for p in model.P:
                for i in model.I:
                    for q in model.Q:
                        try:
                            qty = value(model.loaded_qty[v, c, r, p, i, q])
                            if qty > 1e-6:
                                rows.append({
                                    "Véhicule": v, "Compartiment": c,
                                    "Tour": r, "Période": p,
                                    "Fournisseur": i, "Qualité": q,
                                    "Quantité (t)": qty
                                })
                        except Exception:
                            continue
    return pd.DataFrame(rows)


def extract_state_details(model) -> pd.DataFrame:
    rows = []
    for (v, c) in model.VC:
        for r in model.R:
            for p in model.P:
                for q in model.Q:
                    try:
                        if value(model.S[v, c, r, p, q]) > 0.5:
                            rows.append({
                                "Véhicule": v, "Compartiment": c,
                                "Tour": r, "Période": p, "État": q
                            })
                    except Exception:
                        continue
    return pd.DataFrame(rows)


def extract_cleaning_details(model) -> pd.DataFrame:
    rows = []
    for (v, c) in model.VC:
        for r in model.R:
            for p in model.P:
                try:
                    if value(model.H[v, c, r, p]) > 0.5:
                        rows.append({
                            "Véhicule": v, "Compartiment": c,
                            "Tour": r, "Période": p,
                            "Coût": value(model.Cclean[v, c, p])
                        })
                except Exception:
                    continue
    return pd.DataFrame(rows)


# =============================================================================
# 4. DIAGNOSTIC
# =============================================================================
def run_diagnostic(model):
    periods_sorted = sorted(list(model.P))

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: State chain per compartment")
    print("=" * 60)
    for (v, c) in model.VC:
        print(f"\n--- {v}/{c} ---")
        for p in periods_sorted:
            for q in model.Q:
                try:
                    ss = value(model.Sstart_expr[v, c, p, q])
                    se = value(model.Send[v, c, p, q])
                    if ss > 0.5 or se > 0.5:
                        print(f"  P{p} | Sstart[{q}]={ss:.0f}  Send[{q}]={se:.0f}")
                except Exception:
                    pass

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: H and y values")
    print("=" * 60)
    for (v, c) in model.VC:
        for p in periods_sorted:
            for r in sorted(list(model.R)):
                try:
                    h = value(model.H[v, c, r, p])
                    if h > 0.5:
                        print(f"  H[{v},{c},r={r},P{p}]=1 "
                              f"(cost={value(model.Cclean[v,c,p])})")
                except Exception:
                    pass
                for q in model.Q:
                    try:
                        if value(model.y[v, c, r, p, q]) > 0.5:
                            print(f"  y[{v},{c},r={r},P{p},{q}]=1")
                    except Exception:
                        pass

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Compatibility check — first tour each period")
    print("=" * 60)
    for (v, c) in model.VC:
        for p in periods_sorted:
            sstart_q = next(
                (q for q in model.Q
                 if value(model.Sstart_expr[v, c, p, q]) > 0.5), None)
            y_q = next(
                (q for q in model.Q
                 if value(model.y[v, c, 1, p, q]) > 0.5), None)
            if sstart_q and y_q:
                try:
                    gamma_val = value(model.Gamma[sstart_q, y_q])
                    h_val     = value(model.H[v, c, 1, p])
                    if gamma_val == 1:
                        compat = "OK"
                    elif h_val > 0.5:
                        compat = "CLEAN_APPLIED"
                    else:
                        compat = "*** VIOLATION ***"
                    print(f"  {v}/{c} P{p}: Sstart={sstart_q} -> y={y_q} "
                          f"Gamma={gamma_val:.0f} H={h_val:.0f} [{compat}]")
                except Exception:
                    pass

    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Sstart vs Send mismatch check")
    print("=" * 60)
    for (v, c) in model.VC:
        for idx in range(1, len(periods_sorted)):
            p_prev = periods_sorted[idx - 1]
            p_curr = periods_sorted[idx]
            for q in model.Q:
                try:
                    ss = value(model.Sstart_expr[v, c, p_curr, q])
                    se = value(model.Send[v, c, p_prev, q])
                    if abs(ss - se) > 0.01:
                        print(f"  *** MISMATCH {v}/{c}: "
                              f"Sstart[P{p_curr},{q}]={ss:.2f} "
                              f"!= Send[P{p_prev},{q}]={se:.2f} ***")
                except Exception:
                    pass
    print("=" * 60)


# =============================================================================
# 5. SCENARIO RUNNER
# =============================================================================
def run_scenario(model_builder, data, cost_pattern,
                 solver_name="scip", time_limit=3600,
                 dfj_max_size=3, M_big=5):
    import time

    start = time.time()
    data_scenario = deepcopy(data)
    for (v, c, p), new_cost in cost_pattern.items():
        data_scenario["clean_cost"][(v, c, p)] = new_cost

    model = model_builder(data_scenario, max_tours_per_period=1,
                          add_cuts=True,
                          dfj_max_size=dfj_max_size,
                          M_big=M_big)

    opt = SolverFactory(solver_name)
    if str(solver_name).lower() == "scip":
        opt.options["limits/time"] = time_limit
    else:
        opt.options["timelimit"] = time_limit

    print(f"Lancement de {solver_name} (DFJ={dfj_max_size}, M_big={M_big})...")
    results = opt.solve(model, tee=True)

    solve_time = time.time() - start
    print(f"→ Temps de résolution : {solve_time:.2f} secondes")

    status   = results.solver.termination_condition
    feasible = False
    try:
        obj_val  = value(model.obj)
        feasible = True
    except Exception:
        obj_val = None

    is_optimal = (status == TerminationCondition.optimal) and feasible

    if feasible:
        run_diagnostic(model)

    stats = {
        "status":     status,
        "is_optimal": is_optimal,
        "feasible":   feasible,
        "solve_time": solve_time,
    }

    if feasible:
        total_clean_cost  = 0.0
        cleaning_by_period= {p: 0   for p in model.P}
        clean_cost_by_p   = {p: 0.0 for p in model.P}

        for (v, c) in model.VC:
            for r in model.R:
                for p in model.P:
                    try:
                        if value(model.H[v, c, r, p]) > 0.5:
                            cost = value(model.Cclean[v, c, p])
                            total_clean_cost        += cost
                            cleaning_by_period[p]   += 1
                            clean_cost_by_p[p]      += cost
                    except Exception:
                        continue

        fixed_cost_total = 0.0
        for v in model.V:
            for p in model.P:
                try:
                    if value(model.vehicle_used[v, p]) > 0.5:
                        fixed_cost_total += value(model.fixed_cost)
                except Exception:
                    continue

        routing_cost = obj_val - total_clean_cost - fixed_cost_total

        # Time used is computed PER VEHICLE, since the budget constraint
        # (eq. time_budget_ct) is enforced per vehicle.
        loading_time = {p: 0.0 for p in model.P}
        time_used_by_vehicle = {(v, p): 0.0
                                for v in model.V for p in model.P}

        for v in model.V:
            for r in model.R:
                for p in model.P:
                    for i in model.N:
                        for j in model.N:
                            if i != j:
                                try:
                                    if value(model.x[v, r, p, i, j]) > 0.5:
                                        time_used_by_vehicle[v, p] += value(
                                            model.travel_time[i, j])
                                except Exception:
                                    continue

        for (v, c) in model.VC:
            for r in model.R:
                for p in model.P:
                    try:
                        if value(model.H[v, c, r, p]) > 0.5:
                            time_used_by_vehicle[v, p] += value(model.Tclean)
                    except Exception:
                        continue
                    for i in model.I:
                        for q in model.Q:
                            try:
                                lq = value(model.loaded_qty[v, c, r, p, i, q])
                                if lq > 1e-6:
                                    t = value(model.theta) * lq
                                    time_used_by_vehicle[v, p] += t
                                    loading_time[p] += t
                            except Exception:
                                continue

        # Displayed total is informative only; the enforced quantity is the
        # per-vehicle maximum (matches the time-budget constraint).
        time_used = {p: sum(time_used_by_vehicle[v, p] for v in model.V)
                     for p in model.P}

        budget    = {p: value(model.Tmax[p]) for p in model.P}
        violation = {p: max(0, max(time_used_by_vehicle[v, p] - budget[p]
                                   for v in model.V))
                     for p in model.P}

        stats.update({
            "objective":               obj_val,
            "routing_cost":            routing_cost,
            "fixed_cost":              fixed_cost_total,
            "total_clean_cost":        total_clean_cost,
            "cleaning_count_by_period":cleaning_by_period,
            "clean_cost_by_period":    clean_cost_by_p,
            "time_used_per_period":    time_used,
            "loading_time_per_period": loading_time,
            "time_budget":             budget,
            "time_violation":          violation,
            "time_budget_respected":   all(v <= 1e-6 for v in violation.values()),
            "details": {
                "arcs":        extract_tour_details(model),
                "chargements": extract_loading_details(model),
                "etats":       extract_state_details(model),
                "nettoyages":  extract_cleaning_details(model),
            },
        })
    else:
        stats.update({
            "objective": None, "routing_cost": None,
            "fixed_cost": None, "total_clean_cost": None,
            "cleaning_count_by_period": {p: 0   for p in model.P},
            "clean_cost_by_period":     {p: 0.0 for p in model.P},
            "time_used_per_period":     {p: 0.0 for p in model.P},
            "loading_time_per_period":  {p: 0.0 for p in model.P},
            "time_budget": {p: value(model.Tmax[p]) for p in model.P},
            "time_violation":           {p: 0.0 for p in model.P},
            "time_budget_respected":    False,
            "details": {},
        })

    return stats, model


# =============================================================================
# 6. SUMMARY PRINT
# =============================================================================
def print_scenario_summary(scenario_name, stats, data):
    print(f"\n{'=' * 60}")
    print(f"Scénario : {scenario_name}")
    print(f"{'=' * 60}")
    print(f"Statut      : {stats['status']} "
          f"(optimal: {stats['is_optimal']}, faisable: {stats['feasible']})")
    if stats["feasible"]:
        print(f"Objectif    : {stats['objective']:.2f}")
        print(f"  Routage   : {stats['routing_cost']:.2f}")
        print(f"  Fixe      : {stats['fixed_cost']:.2f}")
        print(f"  Nettoyage : {stats['total_clean_cost']:.2f}")
        print(f"\n{'Pér':<5} {'Lav':<5} {'CostLav':<10} "
              f"{'Budget':<10} {'Utilisé':<10} {'Charge':<10} {'Viol':<8}")
        for p in sorted(stats["cleaning_count_by_period"]):
            print(f"{p:<5} "
                  f"{stats['cleaning_count_by_period'][p]:<5} "
                  f"{stats['clean_cost_by_period'][p]:<10.2f} "
                  f"{stats['time_budget'][p]:<10.1f} "
                  f"{stats['time_used_per_period'][p]:<10.1f} "
                  f"{stats['loading_time_per_period'][p]:<10.1f} "
                  f"{stats['time_violation'][p]:<8.1f}")
        print(f"Budget respecté : "
              f"{'Oui' if stats['time_budget_respected'] else 'Non'}")
    else:
        print("Aucune solution réalisable.")
    print(f"Temps résolution : {stats['solve_time']:.2f} s")


# =============================================================================
# 7. BENCHMARK RUNNER
# =============================================================================
def run_benchmark(excel_path, scenarios, solver="scip", time_limit=3600,
                  dfj_max_size=3, M_big=5):
    data = load_instance_data(excel_path)
    results = []
    for name, pattern in scenarios:
        stats, _ = run_scenario(
            build_mpmttmc_model, data, pattern, solver, time_limit,
            dfj_max_size=dfj_max_size,
            M_big=M_big
        )
        stats["name"] = name
        results.append(stats)
        print_scenario_summary(name, stats, data)
    return results, data


# =============================================================================
# 8. EXPORT
# =============================================================================
def export_results_m(results, data,
                     output_csv=None, output_excel=None, output_txt=None,
                     output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    output_csv   = output_csv   or os.path.join(output_dir, "benchmark_results.csv")
    output_excel = output_excel or os.path.join(output_dir, "benchmark_results.xlsx")
    output_txt   = output_txt   or os.path.join(output_dir, "benchmark_summary.txt")

    rows = []
    for res in results:
        row = {
            "scenario":            res["name"],
            "feasible":            res["feasible"],
            "is_optimal":          res["is_optimal"],
            "objective":           res["objective"] if res["feasible"] else None,
            "routing_cost":        res.get("routing_cost"),
            "fixed_cost":          res.get("fixed_cost"),
            "total_clean_cost":    res.get("total_clean_cost"),
            "solve_time":          res["solve_time"],
            "time_budget_respected": res.get("time_budget_respected", False),
            "status":              str(res["status"]),
            "num_cleanings_total": sum(
                res["cleaning_count_by_period"].values())
                if res["feasible"] else 0,
        }
        for p in data["periods"]:
            row[f"cleanings_P{p}"] = res["cleaning_count_by_period"].get(p, 0)
            row[f"clean_cost_P{p}"] = (
                res["clean_cost_by_period"].get(p, 0.0)
                if res["feasible"] else None)
            row[f"time_used_P{p}"] = (
                res["time_used_per_period"].get(p, 0.0)
                if res["feasible"] else None)
            row[f"loading_time_P{p}"] = (
                res["loading_time_per_period"].get(p, 0.0)
                if res["feasible"] else None)
            row[f"time_budget_P{p}"]    = res["time_budget"].get(p, 0.0)
            row[f"time_violation_P{p}"] = res["time_violation"].get(p, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Synthèse", index=False)
        for res in results:
            if res["feasible"] and res.get("details"):
                name = res["name"].replace("/", "_")[:30]
                for sheet_key, sheet_name in [
                    ("arcs",        f"{name}_arcs"),
                    ("chargements", f"{name}_chargements"),
                    ("etats",       f"{name}_etats"),
                    ("nettoyages",  f"{name}_nettoyages"),
                ]:
                    df_detail = res["details"][sheet_key]
                    if not df_detail.empty:
                        df_detail.to_excel(writer, sheet_name=sheet_name,
                                           index=False)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("BENCHMARK RESULTS — MPMTMC-VRP (Model M)\n" + "=" * 80 + "\n\n")
        for _, row in df.iterrows():
            f.write(f"Scénario : {row['scenario']}\n")
            f.write(f"  Faisable  : {row['feasible']}\n")
            f.write(f"  Optimal   : {row['is_optimal']}\n")
            if row["feasible"] and pd.notna(row["objective"]):
                f.write(f"  Objectif  : {row['objective']:.2f}\n")
                f.write(f"  Routage   : {row['routing_cost']:.2f}\n")
                f.write(f"  Fixe      : {row['fixed_cost']:.2f}\n")
                f.write(f"  Nettoyage : {row['total_clean_cost']:.2f}\n")
            else:
                f.write("  Aucune solution réalisable.\n")
            f.write(f"  Temps     : {row['solve_time']:.2f} s\n\n")

    print(f"\nExported to:\n  {output_csv}\n  {output_excel}\n  {output_txt}")


# =============================================================================
# 9. EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Replace with the path to your instance workbook (see
    # instances/MPPC-VRP_files_formats.pdf for the expected sheet layout).
    excel_path = "instances/kair_45_2_1.xlsx"

    temp_data   = load_instance_data(excel_path)
    vehicles    = temp_data["vehicles"]
    periods     = temp_data["periods"]
    comp_by_veh = temp_data["compartments_by_vehicle"]

    def variable_pattern(*args):
        if len(args) != len(periods):
            raise ValueError(
                f"Expected {len(periods)} args, got {len(args)}")
        return {(v, c, periods[idx]): args[idx]
                for v in vehicles
                for c in comp_by_veh[v]
                for idx in range(len(periods))}

    scenarios = [
        ("Variable (1,10,12,3,10,10)", variable_pattern(1, 10, 12, 3, 10, 10)),
    ]

    results, data = run_benchmark(
        excel_path, scenarios, solver="scip", time_limit=3600,
        dfj_max_size=3, M_big=5
    )
    export_results_m(results, data)
