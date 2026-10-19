# MPMTMC-VRP — Code and Data

Companion repository for:

> Ghodhbani, I. & Ben Aissa, M. (2026). *A Multi-Period, Multi-Tour,
> Multi-Compartment Vehicle Routing Problem with Endogenous Cleaning:
> Hierarchical Framework and Structural Analysis.

This repository provides everything needed to reproduce the numerical
results reported in the manuscript: the three MILP models (Pyomo/SCIP),
every problem instance used in the experiments, and the exact file
format each instance follows.

## Repository structure

```
code/
  mpmtmc_model_m.py            Model M (anticipatory cleaning, variable delta)
  mpmtmc_model_r.py            Model R (repair-based cleaning, H <= comp_used)
  mpmtmc_model_simplified.py   Simplified model (no cleaning-state tracking;
                                mutually compatible-quality instances, e.g. Sfax)
  requirements.txt
  FORMAT.md                    Exact .xlsx schema read by load_instance_data()
instances/
  MPPC-VRP_files_formats.pdf   Human-readable description of the instance format
  sfax_39_2_2.xlsx, sfax_39_3_1.xlsx, sfax_39_4_0.xlsx
                                Sfax region (Lahyani et al., 2015) — used with
                                the simplified model (mutually compatible qualities)
  kair_33_1_2.xlsx, kair_33_2_1.xlsx, kair_33_3_0.xlsx, kair_45_1_2.xlsx
                                Kairouan industrial instances (Lahyani et al., 2015)
  kair_27_2_1.xlsx, kair_34_1_2.xlsx, kair_34_2_1.xlsx
                                Additional Kairouan instances used in the
                                structural-regime illustrations
  template5M1.xlsx             Controlled instance: pure economic anticipation
  template_persist3P.xlsx      Controlled instance: cross-period persistence of
                                the anticipatory CLEAN state across an idle period
                                (see note below)
  synth_060.xlsx, synth_090.xlsx, synth_120.xlsx, synth_150.xlsx
                                Synthetic instances for the scalability experiment
  synth_090_fleet{A_mono,B_double,C_triple,D_hetero}.xlsx
                                Fleet-composition variants built on the synth_090
                                demand pattern
LICENSE                        MIT
```

No numerical result files (CSV/Excel outputs) are included. Every
table and figure in the manuscript is reproducible from the code and
instances above; shipping pre-computed outputs alongside them would
only risk getting out of sync with what the solver actually produces.

## Requirements

```
pip install -r code/requirements.txt
```

A MILP solver capable of handling the models is also required. The
reported results were obtained with **SCIP** (via `pyomo`'s
`SolverFactory("scip")`), either through

```
pip install pyscipopt
```

or a standalone install of the SCIP Optimization Suite
(https://www.scipopt.org/). Any other MILP solver Pyomo supports can
be substituted by changing the `SolverFactory(...)` argument; absolute
solve times will differ from those reported in the paper, but all
feasibility/optimality conclusions are solver-independent.

## Running a model

```bash
cd code
python mpmtmc_model_m.py            # edit excel_path in __main__ to point
python mpmtmc_model_r.py            # at an instance under ../instances/
python mpmtmc_model_simplified.py
```

Each script can also be imported directly — see the docstrings of
`load_instance_data`, `build_mpmttmc_model` (Model M) /
`build_mpmttmc_model_repair` (Model R), and `run_scenario` /
`run_benchmark*`.

## Correspondence with the manuscript

- **Model M** — anticipatory model, Section "MILP Formulation of
  Models M and R".
- **Model R** — repair-based model, same section; obtained from Model
  M by removing the `delta` variable and adding the anti-anticipation
  constraint `H <= comp_used`.
- **Simplified model** — Section "Simplified Formulation for Compatible
  Instances"; used for the three Sfax instances, where all transported
  grades are mutually compatible and no cleaning decision is needed.
- **`template5M1`** and **`template_persist3P`** — the two controlled
  instances behind the "Empirical illustration" paragraphs of the
  structural-regime analysis (pure economic anticipation, and
  cross-period persistence of the CLEAN state, respectively).

### Note on `template_persist3P.xlsx`

This specific instance file was not preserved from the original
experiments; it has been reconstructed here from the exact parameters
given in the manuscript (1 vehicle, 1 compartment, 3 periods,
$S^{\mathrm{init}}=L$; demand of grade L in period 1; no demand in
period 2; demand of grade EV in period 3 with $T^{\max}_3=30$ min).
The reconstruction was verified by solving both Model M and Model R on
it: Model M returns the exact objective value reported in the paper
(80 EUR, with the CLEAN state created during an idle tour of period 2
and consumed by the period-3 load), and Model R is correctly proven
infeasible. All other instances in this repository are the original
files used to produce the published results.

## Format of instance files

See `code/FORMAT.md` for the exact sheet/column schema, and
`instances/MPPC-VRP_files_formats.pdf` for a human-readable version
with examples.

## License

MIT — see `LICENSE`.
