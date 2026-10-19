# Instance file format (.xlsx)

Each instance is a single Excel workbook read by `load_instance_data()`
(identical across the three model files) with exactly these sheets.
Sheet and column names are matched literally — case and spelling must
match.

## `Params`
Two columns, `Parameter` / `Value`, one row per parameter (long format,
not one column per parameter). Keys read by the code:

| Parameter          | Meaning                                              | Default if absent |
|---------------------|-------------------------------------------------------|--------------------|
| `Num_Periods`       | Number of planning periods                             | required           |
| `LoadingTimePerTon` | Loading time per ton (minutes)                          | 1.0                |
| `FixedCostPerVehicle` | Fixed cost per vehicle-period activation (EUR)        | 15.0               |
| `T_max`             | Global per-vehicle time budget per period (seconds)     | 54000              |
| `T_max_P{p}`        | Period-specific override of `T_max` (seconds)            | falls back to `T_max` |

## `Suppliers`
One row per (supplier, quality grade) pair actually produced there.
Columns: `Supplier_ID`, `Quality` (one of `EV`, `V`, `O`, `L` — order
fixed, `CLEAN` is added internally as the zeroth state), and one
`P{p}_Demand_t` column per period (tons demanded in that period; a
blank or zero cell means no demand in that period).

## `Fleet`
One row per (vehicle, compartment). Columns: `Vehicle_ID`,
`Compartment_ID`, `Capacity_t`, and, per period `p`: `P{p}_Available`
(0/1) and `CleanCost_P{p}` (EUR — the cost used throughout the paper's
experiments to parametrise period-dependent cleaning costs).

## `Initial_States`
One row per (vehicle, compartment): `Vehicle_ID`, `Compartment_ID`,
`P1_Init_Quality` — the compartment's state at the start of period 1
(must be a valid quality label or `CLEAN`). A blank cell defaults to
`L` in Model~M/the simplified model and is treated as `CLEAN` by
Model~R (see in-code comment in `load_instance_data`). States from
period 2 onward are never read from Excel — they are determined
endogenously by the model itself (state-inheritance mechanism).

## `Incompatibility` (the $\Gamma$ matrix)
A square matrix sheet: first column `prev_quality`, remaining columns
named after each possible next quality (including `CLEAN`). Cell
value is 1 if that transition is allowed without cleaning, 0
otherwise. Must be fully populated (no blank cells) for every quality
actually present in the instance.

## `Nodes`
Columns: `Node_ID`, `Node_Type` (exactly one row must have
`Node_Type = "Depot"`; all others are supplier nodes referenced from
`Suppliers.Supplier_ID`).

## `Distances`
A square matrix sheet: first column holds the "from" node ID, each
subsequent column is named after a `Node_ID`, values are
origin-destination distances (km) used directly as travel times
(`travel_time = distances`, no separate speed conversion).

## Notes
- `mpmtmc_model_simplified.py` reads the same sheets but ignores
  `Incompatibility`/state-tracking columns (used for the Sfax
  instances, which have no quality incompatibility).
- `cleaning_time` (15 minutes per operation) is a hard-coded constant,
  not read from the workbook.
- The synthetic-instance generators (`make_synthetic_instances.py` and
  the `make_synth_*`/`make_fleet_variants_*` scripts referenced in the
  manuscript's experiments) write workbooks in exactly this format and
  can be used as a template for new instances.
