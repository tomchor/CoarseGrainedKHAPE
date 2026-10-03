# KHAPE — Kelvin-Helmholtz Available Potential Energy

Computes Available Potential Energy (APE) from three-dimensional Kelvin-Helmholtz instability simulations using the Winters et al. (1995) sorting method.

## Pipeline overview

1. **Julia simulation** (`simulation.pbs`) — runs the KH instability on a GPU, writes NetCDF output, and then draws the three figures that need nothing else (`plot3_b_br_snapshots.py`, `plot5_budgets.py`, `plot6_panels.py`) into `figures/`. Above `NZ=512` it runs on several GPUs, and the same job merges their files before it draws those figures (see [Multi-GPU runs](#multi-gpu-runs))
2. **Post-processing** — `postprocessing/budgeting.pbs` assembles the SFS KE and APE budgets from the terms the simulation computed online and plots them (the offline computation of those terms lives in `postprocessing/offline/` and runs only as a CI cross-check)
3. **Sweep** — parameter sweep over filter scales, split into two jobs:
   - `postprocessing/sweep_filter.pbs` — filters fields at all scales (shared; runs once regardless of `FIXED_REF`)
   - `postprocessing/sweep_transfer.pbs` — computes and plots energy transfer spectra (per `FIXED_REF` variant)

## Post-processing scripts (`postprocessing/`)

Scripts in `postprocessing/` follow a naming convention by purpose:

| Prefix | Purpose |
|--------|---------|
| `01_online_budgets.py`, `02_plot_budgets.py` | **Budget pipeline.** Every term of the SFS KE and APE budgets is computed by the simulation itself; `01` writes the integrated budgets from those online terms (at the simulation's own filter scales, ℓ = 1 and 7 by default) and `02` plots them. |
| `offline/01_…` – `offline/05_…` | **Offline budget pipeline, kept as a test.** The independent Python implementation of every term (scipy filtering on a z-padded domain, a numpy density sort, an FFT-filtered reference profile). It is no longer a product: `offline/run_offline_budgets.sh` runs it into `output/offline/` and `pytest --offline-check` compares each term against the assembled online budgets (see [Tests](#tests)). |
| `sweep1_…` – `sweep3_…` | **Parameter sweep pipeline** over filter scales: filter fields, compute cross-scale transfer at every scale, and plot transfer spectra. |
| `plot2_…`, `plot3_…`, `plot5_…`, `plot6_…` | **Paper figure scripts.** Produce the figures used in the manuscript (cross-scale transfer spectrum, b and b_r snapshots, SFS KE/APE budget time series, local-field snapshot panels). `plots.pbs` runs every `plot*.py`. Output goes to `figures/`. |
| `anim1_…`, `X1_…` – `X11_…` | **Extra material.** Animations (`anim*`, requires `ffmpeg`) and the extra figures outside the manuscript set (`X*`, X for extra: Π hovmöllers, snapshot panels, a thumbnail, sweep-spectrum diagnostics). The `X*` scripts write to `postprocessing/extra_figures/`, not `figures/`. |
| `aux*` (under `src/`) | Shared utilities reused across the pipeline (data loading, Gaussian filtering, spatial derivatives, PE/KE budget terms, plotting helpers). |
| `00_get_budgets.sh`, `inv00_get_sweep.sh` | Local helpers that run the budget pipeline or sweep pipeline end-to-end without PBS (see [Running locally](#running-locally-without-pbs)). |
| `*.pbs`, `submit_*.sh` | PBS job scripts and their wrappers (see [Submitting jobs](#submitting-jobs)). |

All Python scripts accept `--filename`, and most accept `--filter-scales` and `--n-workers`; the sweep scripts also take `--fixed-reference`. Run any script with `--help` for its full argument list.

### Where the budgets come from

The SFS KE budget (Kˢ, ∂ₜKˢ, Π_K, ε_Kˢ, τ(w,b_r)) and the SFS APE budget (S̃, ∂ₜS̃, Π_A, ε_Aˢ, Rˢ, and the same τ) are computed online by `kelvin_helmholtz_instability.jl` at each of its `--filter_ls` scales, as 3D fields and as volume integrals, and written unconditionally. The resolved APE reservoir is measured against the vertically filtered reference profile ⟨b✶⟩ (Wenegrat, Chor & Barkan, Eq. 2.3): the pipeline filters in x, y **and** z, and against the unfiltered b✶ the resolved reservoir would not vanish for a fluid at rest and the sub-filter remainder would go negative over much of the domain. `01_online_budgets.py` writes the volume integrals into the two integrated budget files (`<stem>_sfs_{ke,ape}_budget_integrated.nc`) that `02_plot_budgets.py` and the tests consume, one record per output time. The 3D fields are not copied anywhere: the assembly itself is `online_budgets` in `src/aux04_online_budgets.py`, which the tests, `plot5_budgets.py`, `plot6_panels.py`, `X2_panels.py` and `X4_thumbnail.py` call on the simulation file directly (so `plot5` and `plot6` need no post-processing), and which `anim1_panels.py` calls on the x–z slices of the `_2d.nc` file for its panels and its time series. See CLAUDE.md for the full account.

The offline pipeline that used to compute these terms (`postprocessing/offline/`) recomputes every one of them independently and runs only as the CI cross-check. There is no fixed-in-time reference variant of the budgets; the sweep keeps its own (`sweep2 --fixed-reference`).

## Setup

Create the conda environment for Python post-processing:

```bash
conda env create -f environment.yml   # creates env "py313"
conda activate py313
```

The Julia simulation uses the project's `Project.toml`/`Manifest.toml` — instantiate with `julia --project -e 'using Pkg; Pkg.instantiate()'` on first use.

## Submitting jobs

### File naming convention

| Extension | Role |
|-----------|------|
| `*.pbs`   | PBS job script — passed directly to `qsub`; do not run with `bash` |
| `submit_*.sh` | Wrapper script — constructs job names/log paths and calls `qsub`; this is what you invoke |

Always use the `submit_*.sh` wrappers rather than submitting `*.pbs` files directly — the wrappers ensure job names and log files reflect the run parameters.

Arguments are passed as `KEY=VALUE` pairs in any order. All arguments are optional and fall back to their defaults if omitted.

### Environment

The submit wrappers take the project to charge and the Python to run from two variables whose defaults live in `khape_defaults.sh` (sourced by every `submit_*.sh`, so that is the one place to change them), and two more variables move the output off the repository, for example to scratch. To override any of them, export it in the shell you submit from, or set it in the login environment (e.g. `~/.bashrc`): every wrapper hands the Python and, when they are set, the two output directories to its jobs (`qsub -v`), which PBS would not otherwise pass on, and the jobs run in a login shell. Give absolute paths. No `.pbs` file names an account, a mail address or a Python environment, since the wrapper hands them over at submission; PBS mails its reports to whoever submitted the job.

| Variable | Default | Read by |
|----------|---------|---------|
| `KHAPE_ACCOUNT` | `UMCP0061` | every submit wrapper, which charges each job to it with `qsub -A` |
| `KHAPE_PYTHON` | `$HOME/miniconda3/envs/py313/bin/python` | the wrappers, which check that it exists at submission and pass it to every Python job and to the simulation job, for the figures it draws after the run (the path to your `py313` environment's `python`) |
| `KHAPE_OUTPUT_DIR` | `output/` | the simulation (where it writes), every post-processing PBS job (where they read the run), and the tests |
| `KHAPE_PP_OUTPUT` | `postprocessing/output/` | every post-processing script (through `src/aux00_utils.PP_OUTPUT`) and the tests |
| `KHAPE_MPI_DEPOT` | `$WORK/.julia-mpi` | `setup_mpi_env.sh` and multi-GPU simulation jobs: the Julia depot for the packages built against the system MPI |
| `KHAPE_MPI_ENV` | `$KHAPE_MPI_DEPOT/environments/khape-mpi` | the same: the Julia environment that sends MPI.jl to Casper's CUDA-aware OpenMPI (through MPItrampoline) |

### Run everything (simulation + post-processing + sweep, with optional validation and plots)

```bash
# Default resolution (Nz=512, the largest 3D run one A100 holds), time-varying reference profile
bash submit_all_pbs.sh

# Custom resolution
bash submit_all_pbs.sh NZ=256

# Above Nz=512: on four GPUs by default (see Multi-GPU runs)
bash submit_all_pbs.sh NZ=1024

# Fixed-in-time reference profile for the sweep transfer (the budgets have no such variant)
bash submit_all_pbs.sh NZ=256 FIXED_REF=1

# Add the online-vs-offline validation and/or the final plots (independently toggleable)
bash submit_all_pbs.sh VALIDATE=1            # + validation (figures + animations); runs the sim with --save_tensors and --save_sorted
bash submit_all_pbs.sh PLOTS=1               # + every plot*.py after sweep_transfer
bash submit_all_pbs.sh VALIDATE=1 PLOTS=1    # the whole pipeline

# The same chain from budgeting on, on the run already in $KHAPE_OUTPUT_DIR (no simulation job)
bash submit_all_pbs.sh SIMULATION=0 PLOTS=1
```

Jobs are chained: `budgeting` starts after the simulation, `sweep_filter` after `budgeting`, and `sweep_transfer` after `sweep_filter`. With `SIMULATION=0` there is no simulation job and `budgeting` starts at once, on the run already in `$KHAPE_OUTPUT_DIR` (the wrapper stops if that run does not exist). `budgeting` assembles and plots the integrated budgets from the simulation's online terms (about a minute); when `FIXED_REF=1`, the sweep transfer job builds its own frozen column (see Run sweep only).

`SAVE_SORTED` defaults to `1`, so the simulation also writes the validation-only view of the sorted reference state (the two model-grid z✶ methods, the sorted column, ∫E_b) that `inv06` and `inv07` compare against the offline sort. Every budget term is written regardless, so `SAVE_SORTED=0` gives smaller output and changes no budget number; `VALIDATE=1` turns it back on.

Two optional stages are gated by flags (both default `0`, so the base behavior is simulation + post-processing + sweep):
- `VALIDATE=1` runs the simulation with `--save_tensors` (and with `--save_sorted`, even if `SAVE_SORTED=0`) and submits a parallel **validation** job (`postprocessing/validation/validation.pbs`) after the simulation, writing online-vs-offline comparison figures (`figures/validation/`) and animations (`animations/`).
- `PLOTS=1` submits a **plots** job (`postprocessing/plots.pbs`) after `sweep_transfer` that runs every `postprocessing/plot*.py`, in name order: `plot2_transfer_spectrum.py` (transfer spectra, from the sweep output), `plot3_b_br_snapshots.py` (b and b_r snapshots, from the `_2d.nc` file), `plot4_sweep_spectrum_hovmoller.py` (Hovmöllers of the transfers, from the sweep output), `plot5_budgets.py` (the integrated SFS budgets) and `plot6_panels.py` (the local SFS budget fields), the last two assembled from the simulation output itself, so they need no post-processing. `plot3`, `plot5` and `plot6` are also drawn by the simulation job itself as soon as the run ends (see *Run simulation only*), so the plots job redraws them after the sweep.

### Run simulation only

```bash
# Default (Nz=512)
bash submit_simulation.sh

# Custom resolution
bash submit_simulation.sh NZ=256

# Also write the per-scale strain/stress tensor components (for online-vs-offline validation)
bash submit_simulation.sh NZ=256 SAVE_TENSORS=1

# Also write the validation-only sorted reference state (for inv06/inv07)
bash submit_simulation.sh NZ=256 SAVE_SORTED=1

# Also write each 3D output as a consecutive-iteration pair (for pytest --offline-check; doubles the 3D output)
bash submit_simulation.sh NZ=256 OFFLINE_CHECK=1

# Several GPUs (four by default above NZ=512; see Multi-GPU runs)
bash submit_simulation.sh NZ=1024
bash submit_simulation.sh NZ=1024 NGPUS=8
```

The grid is isotropic (Δx = Δy = Δz) on a domain of one KH wavelength λ in x, λ/3 in y and 25h in z, so `NZ` sets the whole grid: 288 × 96 × 512 cells at `NZ=512`. The Reynolds number scales as Re = Re₀ Nz^(4/3) (Kolmogorov resolution at fixed domain height), with Re₀ = 0.1 by default, i.e. Re = 410 at Nz=512.

When the run ends, the same job draws the three figures that need nothing but the simulation's own files, into `figures/`: `plot3_b_br_snapshots.py` (from the `_2d.nc` file) and `plot5_budgets.py` and `plot6_panels.py` (which assemble the SFS budgets from the 3D file themselves). A figure that fails is logged as a warning rather than failing the job, so jobs chained on the simulation with `afterok` still start. The wrapper checks `KHAPE_PYTHON` at submission and passes it to the job for this.

`SAVE_TENSORS=1` passes `--save_tensors` to the Julia simulation, which additionally outputs the
resolved strain-rate (S̄ⁱʲ) and sub-filter stress (τⁱʲ) tensor components at each filter scale. These
are full 3D fields (off by default to keep production output lean) and are consumed only by the
validation scripts in `postprocessing/validation/`.

`SAVE_SORTED` (default **1**) passes `--save_sorted`, which additionally outputs the adiabatically sorted reference
state in the forms only the validation reads: the reference height `z✶_3dsort` (`ThreeDimensionalSort`) and
`z✶_heaviside` (`HeavisideIntegral`) as 3D fields on the model grid, the sorted column `z✶_1dsort` / `b✶_1dsort`
(what `VerticalSort` builds; see CLAUDE.md, Online sorted reference state) on its own N = Nx·Ny·Nz vertical axis, and ∫E_b. The column itself is built and used by every
budget term whether or not it is written; the flag decides only whether the two extra model-grid sorts run and
whether the column goes to the file. All of these go into the main output file, since one `NetCDFWriter` holds
both grids; the resulting per-grid dimension suffixing is undone at load time by the post-processing loader.
`inv06_compare_sorted_profiles.py` and `inv07_compare_local_ape.py` compare them against the offline sort.

`OFFLINE_CHECK=1` passes `--offline_check`, which makes the 3D writer also write the record one time step after each output (`ConsecutiveIterations`). Only the offline pipeline reads those pairs, to form its own tendencies for `pytest --offline-check`; the online tendencies come from `TimeDerivative` and need no pair, so the default (`0`) writes one record per output time and halves the 3D output. CI's offline-check run passes the flag. The three flags are written to both output files as 0/1 global attributes (`save_tensors`, `save_sorted`, `offline_check`), which is how the post-processing tells what a run contains.

### Multi-GPU runs

An Nz=512 run peaks at 54 GB on one A100-80GB, and Nz=1024 has 8× the cells, so above `NZ=512` the simulation runs on
several GPUs: `NGPUS` (default 1 up to `NZ=512`, 4 up to `NZ=1024`, required beyond) on whole Casper A100-80GB nodes
(so a multiple of 4), one MPI rank per GPU, the domain split into x-slabs (`--ranks`). Once per machine, first:

```bash
bash setup_mpi_env.sh   # sends MPI.jl to Casper's CUDA-aware OpenMPI, from a Julia environment and depot of its own
```

Oceananigans' distributed model needs an MPI that takes GPU arrays, which the one MPI.jl bundles is not; Casper's
OpenMPI (`intel/2025.2.1 openmpi/5.0.8`, built with CUDA and UCX) is. MPI.jl reaches it through MPItrampoline: pointed
at the system library directly, it would have HDF5_jll (under NCDatasets) load a second OpenMPI of its own, which
cannot share a process with Casper's. The script builds MPIwrapper against Casper's OpenMPI, writes MPIPreferences'
choice of MPItrampoline into `$KHAPE_MPI_ENV`, and checks from the project that MPI.jl reaches Open MPI and that the
NetCDF stack loads beside it. Only multi-GPU jobs load any of it, so one-GPU runs and CI are untouched.

What changes when `NGPUS > 1`:
- `simulation.pbs` asks for `NGPUS/4` nodes, loads OpenMPI (and takes the CUDA toolkit it brings back off
  `LD_LIBRARY_PATH`, where it would shadow CUDA.jl's own libraries), precompiles once and launches the ranks with
  `mpiexec`; `logs/kelvin_helmholtz_<NZ>_gpu_memory.csv` records every GPU's memory every 10 s.
- Every rank writes its own slab, `khi_Nz<NZ>_Ri0.10_rank<r>.nc` and `_2d_rank<r>.nc`. Once the ranks have finished,
  the same job, back in the one-GPU environment (no MPI), stitches them into the two files a one-GPU run writes
  (`merge_rank_output.jl`, ~3 min at Nz=1024) and draws the figures and the animation the one-GPU job draws. A failed
  merge fails the job. Budgeting, validation and the sweep wait on the simulation job, as on one GPU, and read the
  merged files as always.
- The 3D file is merged **virtually**, in seconds: its fields are HDF5 virtual datasets that read each rank's slab
  from the rank file it is in, so the merged file takes almost no space and **the rank files must stay beside it**
  (the directory can move as a whole). A missing rank file would read as fill values, so the Python loader refuses a
  merged file whose rank files are not all there. The 2D file (9 GB at Nz=1024) is copied, so Julia reads it as any
  file; Julia readers of the virtual 3D file need `allow_virtual_storage!()` from `merge_rank_output.jl` first.
- For a standalone 3D file, which no longer needs the rank files, copy the virtual one with netCDF's own
  `nccopy -k nc4 khi_Nz<NZ>_Ri0.10.nc <copy>.nc` (hours at Nz=1024, and better on a login node than in a job,
  whose memory limit counts the page cache of all that I/O).
- `SAVE_SORTED` and `VALIDATE` are refused: the two model-grid sorts of the validation-only view would each sort one
  rank's slab.

Every budget term is computed as on one GPU: the filter's x-pass and the sorted reference column, the two parts that
need the whole domain, gather it across ranks (`distributed_diagnostics.jl`), and give the one-GPU numbers bit for
bit (`tests/test_distributed_diagnostics.jl`). Every rank keeps its own copy of the sorted column, which grows 8×
per doubling of Nz, so this tops out near Nz=1024. Four A100-80GB hold an Nz=1024 run: 67 GiB per GPU at the peak, of
80, measured over the first outputs, where a time step takes 125 ms and an output ~67 s, after about an hour of setup
and compilation, so a run to t=200 takes ~4–4.5 h. `NGPUS=8` (two nodes) is the fallback should a run need more.

### Run a simulation + online-vs-offline validation

```bash
# Submit the simulation (with --save_tensors) then a chained validation job (default Nz=512)
bash submit_validation_run.sh
bash submit_validation_run.sh NZ=256
```

`submit_validation_run.sh` submits the simulation with `SAVE_TENSORS=1` and a `validation` job that
runs after it (`afterok`). The validation job recomputes the filtered fields, cross-scale KE transfer
Π_K, the strain/stress tensors, the SFS KE dissipation ε_Kˢ, the sorted reference state, the local APE,
and the sub-filter APE dissipation ε_Aˢ offline and
compares them against the simulation's online diagnostics (`postprocessing/validation/inv01`–`inv09`),
writing comparison figures to `figures/validation/` and online-vs-offline animations to `animations/`. Note that
`inv06` needs a run with `SAVE_SORTED=1` (which `submit_all_pbs.sh VALIDATE=1` sets automatically).

### Run post-processing only

```bash
cd postprocessing
bash submit_budgeting.sh                          # default Nz=512
bash submit_budgeting.sh NZ=256
```

One small job, about a minute: `01_online_budgets.py` writes the two integrated budget files from the simulation's online terms
and `02_plot_budgets.py` plots them.

### Run sweep only

The sweep is split into two PBS jobs to avoid race conditions when running both `FIXED_REF` variants simultaneously: the field-filtering step (`sweep1`) runs once and is shared, while the energy transfer and plotting steps (`sweep2`+`sweep3`) run separately per variant.

```bash
cd postprocessing
bash submit_sweep.sh                          # default Nz=512, FIXED_REF=0
bash submit_sweep.sh NZ=256
bash submit_sweep.sh NZ=512 FIXED_REF=1      # fixed-in-time reference profile
bash submit_sweep.sh NZ=512 FIXED_REF=both   # submit both variants; filter runs only once
bash submit_sweep.sh NZ=512 EXTENSION=odd    # the whole sweep with b oddly reflected past the walls
```

`submit_sweep.sh` refuses any argument it does not know, so a misspelled key cannot silently rerun the default sweep over the production files.

Under the default edge extension the sweep does not build its z padding. Its fields keep one padded cell past each wall, which is what the centred gradients at the wall cells read; the filter's `nearest` mode repeats the wall value from there on, as the padding did; and only the sorted reference column still holds the padded fluid. The filtered fields come out bit for bit as on the padded grid and the integrals agree with the padded computation to roundoff, while `sweep1`'s file and the arrays `sweep2` holds are Nz + 2 levels tall instead of 3.7 Nz. `EXTENSION=odd` is a different extension of b, so it keeps the whole padding in the arrays.

#### Wall-extension test

The manuscript leaves the extension of b and b✶ past the walls free (§2) and uses the wall values (§4). `EXTENSION=odd` reflects b oddly about the wall value instead; every other field keeps the wall value, so the comparison changes the buoyancy rule alone. Its files carry an `_odd` tag. `submit_extension_test.sh` (which submits `extension_test.pbs`) runs the odd rule at one scale, snapped to the nearest of the production sweep's 30, and `compare_extension.py` sets it against the production (edge) sweep:

```bash
cd postprocessing
bash submit_extension_test.sh NZ=512 SCALE=20                            # odd rule at l=20 only
python compare_extension.py --filename output/khi_Nz512_Ri0.10.nc --filter-scale 20
python compare_extension.py --filename output/khi_Nz512_Ri0.10.nc --all   # every scale, after EXTENSION=odd
```

A run over a subset of scales (`sweep1 --filter-scales`, as `extension_test.pbs` does) tags its files with the scales, e.g. `_sweep_l20_odd.nc`, so it never replaces a full sweep; `sweep2` takes the same `--filter-scales` to find it, and `compare_extension.py` prefers it for a single-scale comparison.

`FIXED_REF=both` submits the filter job once and two transfer jobs (one for each variant) that both depend on the single filter job. The transfer step writes only the volume integrals (`∫Π_K dV`, `∫Π_A dV`, ...), which is all the sweep plots read; `sweep2_energy_transfer.py --keep-fields` also writes the 4D fields, which are enormous at production resolution.

When `FIXED_REF=1`, the transfer job builds its own frozen reference column: it sorts t=0 on the grid it loaded and broadcasts that row over the time axis. **The sweep does not need the budgeting pipeline to have run**, and does not read `_sorted_density_fixed_ref.nc`. It cannot: the column's z✶ are the padded grid's own heights, and the sweep pads to 4σ of its widest scale (ℓ=20) while `02_sort_density.py` pads to the budget scales — at Nz=2048, 2784 cells per side against 1024 — so the budgeting pipeline's column belongs to a different grid. Sorting t=0 costs one sort, and is bit-identical to `02`'s output whenever the two paddings do coincide.

## Running locally (without PBS)

For development on a workstation (no PBS scheduler), run the simulation and post-processing pipeline directly.

```bash
# Julia simulation (CPU, small grid; CI's run)
julia --project -t 8 kelvin_helmholtz_instability.jl --Nz 128 --Ri 0.1 --stop_time 70 --Re0 0.4 --output_interval 4

# The same split across 2 MPI ranks (CPU, the MPI MPI.jl bundles), then the rank files stitched together
julia --project -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project kelvin_helmholtz_instability.jl --Nz 32 --ranks 2`)'
julia --project merge_rank_output.jl output/khi_Nz32_Ri0.10

# Budgets (assembled from the online terms) and their plots, for an existing NetCDF file
cd postprocessing
bash 00_get_budgets.sh output/khi_Nz128_Ri0.10.nc --filter-scales 1 7

# The offline pipeline, as the cross-check runs it (writes to postprocessing/output/offline/)
N_WORKERS=4 bash offline/run_offline_budgets.sh output/khi_Nz128_Ri0.10.nc --filter-scales 1 7

# Sweep pipeline (sweep1–sweep3)
bash inv00_get_sweep.sh output/khi_Nz128_Ri0.10.nc
```

Set `N_WORKERS` to control the offline pipeline's parallelism (default 1).

## Tests

The test suite checks SFS KE and APE budget closure (rms residual / mean rms of terms < 6%) on the integrated budget files `01_online_budgets.py` writes, and the sign of the energies that have one on the 3D fields, which it reads from the simulation output itself. It expects the CI run, `khi_Nz128_Ri0.10`, in `output/` with its integrated budget files in `postprocessing/output/`. That name is set once, as `STEM` in `tests/conftest.py` (`$KHAPE_TEST_STEM` overrides it).

```bash
pytest tests/ -v -s                                  # closure, positivity, the synthetic filter and Jensen tests (minutes)
pytest tests/ -v -s --offline-check                  # + the offline pipeline as a cross-check (minutes at CI's Nz=128)
julia --project tests/test_distributed_diagnostics.jl   # the multi-GPU pieces against one process, on 2 and 4 CPU ranks
```

`tests/test_distributed_diagnostics.jl` needs no simulation output and no GPU: it runs itself under `mpiexec` and
checks that the distributed Gaussian filter and the sorted column (`distributed_diagnostics.jl`) give the
one-process results bit for bit.

`--offline-check` runs `postprocessing/offline/run_offline_budgets.sh` (unless its output already exists) and `tests/test_offline_check.py` compares every field and every integral of the offline budgets against the online ones, term by term, with tolerances set from measurement; it also runs the `inv0*` validation scripts (`tests/test_online_vs_offline.py`). Without the flag those tests are skipped. It needs a simulation run with `--offline_check` (`OFFLINE_CHECK=1`), whose consecutive-iteration output pairs the offline pipeline differences for its tendencies; the pipeline refuses a run without them.

CI (`.github/workflows/test.yml`) runs the Julia simulation (Nz=128, `--Re0 0.4` for Re = 258, outputs every 4 time units) twice in parallel, once as production writes it and once with `--save_sorted --offline_check`, then two jobs in parallel: `test-online` (the production run: assemble → pytest → animation, minutes) and `test-offline-check` (the paired run: the offline pipeline → `pytest --offline-check`), on push to `main` and on PR comments starting with `test`.

## Logs

All job logs are written to the `logs/` subdirectory next to the submit script:
- `logs/<job_name>.log` — PBS stdout/stderr (written by PBS after job ends)
- `logs/<job_name>.out` — Python script output (written live via `tee`)
- for a multi-GPU run, also `logs/kelvin_helmholtz_<NZ>_gpu_memory.csv` (every GPU's memory every 10 s); the merge writes to the simulation's own `.out`

Job names follow the pattern `<stage>_Nz<NZ>_Ri0.10[_fixed_ref]`, e.g. `budgeting_Nz2048_Ri0.10`, `sweep_filter_Nz2048_Ri0.10`, `sweep_transfer_Nz2048_Ri0.10_fixed_ref` (the `_fixed_ref` tag exists only for the sweep).
