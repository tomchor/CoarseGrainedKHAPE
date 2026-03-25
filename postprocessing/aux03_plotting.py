"""
Plotting functions for Available Potential Energy (APE) analysis

This module contains visualization functions for energy timeseries data.
"""

import matplotlib.pyplot as plt

budget_colors = {
    "tendency":    "C0",
    "flux":        "C1",
    "dissipation": "C2",
    "exchange":    "C3",
    "residual":    "k",
}

#+++ SFS budget plotting
def plot_sfs_budget(budget_terms, integrated_vars, filter_length_scales, output_filename, repo_root, title):
    """Plot integrated SFS budget terms vs time for each filter scale.

    Parameters
    ----------
    budget_terms : xr.Dataset
        Dataset with budget variables indexed by filter_length_scale and time.
    integrated_vars : dict
        Mapping of variable name → matplotlib color.
    filter_length_scales : array-like
        Filter length scales to iterate over.
    output_filename : str
        Path to the saved NetCDF file; used to derive the plot filename.
    repo_root : Path
        Repository root directory; figures are written to repo_root/figures/.
    title : str
        Base title string, e.g. "Integrated SFS KE Budget Terms".
    """
    import os
    for ℓ in filter_length_scales:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for var, color in integrated_vars.items():
            budget_terms[var].sel(filter_length_scale=ℓ).dropna("time").plot.line(
                ax=ax, x="time", label=var, color=color)
        ax.legend()
        ax.set_ylabel("Budget Terms [W or J s⁻¹]")
        ax.set_title(f"{title}  (ℓ = {ℓ:.4f})")
        ax.grid(True, alpha=0.3)
        plot_filename = str(repo_root / "figures" / os.path.basename(output_filename).replace(
            ".nc", f"_l{ℓ:.4f}.png"))
        fig.savefig(plot_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved to: {plot_filename}")
#---

#+++ Plot energy timeseries
def plot_energy_timeseries(ds, APE=None, TPE=None, RPE=None, KE=None):
    """
    Plot energy components over time

    Only plots quantities that are provided (not None).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing time coordinate
    APE : array-like, optional
        Available Potential Energy time series
    TPE : array-like, optional
        Total Potential Energy time series
    RPE : array-like, optional
        Reference Potential Energy time series
    KE : array-like, optional
        Kinetic Energy time series

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plots
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: TPE and RPE
    ax1 = axes[0]
    if TPE is not None:
        ax1.plot(ds.time, TPE, label="Total PE", linewidth=2)
    if RPE is not None:
        ax1.plot(ds.time, RPE, label="Reference PE", linewidth=2)
    if TPE is not None or RPE is not None:
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Potential Energy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Potential Energy Components")

    # Plot 2: APE and KE
    ax2 = axes[1]
    if APE is not None:
        ax2.plot(ds.time, APE, label="APE", linewidth=2, color="red")
        ax2.set_ylabel("Available Potential Energy", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.legend(loc="upper left")
    if KE is not None:
        ax2_twin = ax2.twinx()
        ax2_twin.plot(ds.time, KE, label="KE", linewidth=2, color="blue", alpha=0.7)
        ax2_twin.set_ylabel("Kinetic Energy", color="blue")
        ax2_twin.tick_params(axis="y", labelcolor="blue")
        ax2_twin.legend(loc="upper right")
    if APE is not None or KE is not None:
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Available Potential Energy and Kinetic Energy")

    # Plot 3: Normalized energy budget
    ax3 = axes[2]
    if APE is not None and KE is not None:
        total_energy = APE + KE
        total_energy_norm = total_energy / abs(total_energy[0])

        ax3.plot(ds.time, APE / APE[0], label="APE (normalized)", linewidth=2, color="red")
        ax3.plot(ds.time, KE / KE[0], label="KE (normalized)", linewidth=2, color="blue")
        ax3.plot(ds.time, total_energy_norm, label="Total (APE + KE, normalized)",
                 linewidth=2, color="black", linestyle="--")

        ax3.set_xlabel("Time")
        ax3.set_ylabel("Normalized Energy")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Energy Budget: APE to KE Conversion")

    plt.tight_layout()
    return fig
#---

#+++ Plot potential energies
def plot_potential_energies(time, TPE=None, RPE=None, APE=None):
    """
    Plot potential energy components

    Only plots quantities that are provided (not None).

    Creates two panels:
    - Panel 1: TPE, RPE, and APE on the same axes (if provided)
    - Panel 2: TPE and (TPE+RPE) for comparison (if both provided)

    Parameters
    ----------
    time : array-like
        Time coordinate
    TPE : array-like, optional
        Total Potential Energy time series
    RPE : array-like, optional
        Reference Potential Energy time series
    APE : array-like, optional
        Available Potential Energy time series

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plots
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Panel 1: All potential energy components
    ax1 = axes[0]
    if TPE is not None:
        ax1.plot(time, TPE, label="TPE (Total PE)", linewidth=2, color="blue")
    if RPE is not None:
        ax1.plot(time, RPE, label="RPE (Reference PE)", linewidth=2, color="green")
    if APE is not None:
        ax1.plot(time, APE, label="APE (Available PE)", linewidth=2, color="red")
    if TPE is not None or RPE is not None or APE is not None:
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Potential Energy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title("Potential Energy Components")

    # Panel 2: TPE and (TPE+RPE) comparison
    ax2 = axes[1]
    if TPE is not None:
        ax2.plot(time, TPE, label="TPE", linewidth=2, color="blue")
    if TPE is not None and RPE is not None:
        ax2.plot(time, TPE + RPE, label="TPE + RPE", linewidth=2, color="purple", linestyle="--")
    if TPE is not None or RPE is not None:
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Potential Energy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title("TPE vs TPE+RPE")

    plt.tight_layout()
    return fig
#---

#+++ Plot all dataset variables
def plot_dataset_variables(ds, time_stride=None, figsize=None, **kwargs):
    """
    Plot all variables in a dataset as spatial-temporal plots

    This function creates facet plots for each variable in the dataset,
    showing multiple time steps in a grid layout.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing variables to plot
    time_stride : int, optional
        Stride for selecting time steps (e.g., 6 means every 6th time step).
        If None, plots all time steps.
    col_wrap : int, optional
        Number of columns in the facet grid, default is 5
    cmap : str, optional
        Colormap to use, default is "RdBu_r"
    robust : bool, optional
        If True, use robust colorbar limits (2nd and 98th percentiles), default is True
    figsize : tuple, optional
        Figure size (width, height) for each variable plot

    Returns
    -------
    dict
        Dictionary mapping variable names to their figure objects
    """
    import matplotlib.pyplot as plt

    figures = {}

    # Determine time selection
    if time_stride is not None:
        time_sel = slice(None, None, time_stride)
    else:
        time_sel = slice(None, None)

    # Loop through all data variables in the dataset
    for var_name in ds.data_vars:
        var = ds[var_name]

        # Check if variable has time dimension
        if "time" not in var.dims:
            print(f"Skipping {var_name}: no time dimension")
            continue

        # Check if variable has spatial dimensions
        spatial_dims = [d for d in var.dims if d not in ["time"]]
        if len(spatial_dims) == 0:
            print(f"Skipping {var_name}: no spatial dimensions")
            continue

        print(f"Plotting {var_name}...")

        try:
            # Squeeze to remove singleton dimensions and select time steps
            var_plot = var.squeeze().sel(time=time_sel)

            # Create the facet plot
            if figsize is not None:
                fig = plt.figure(figsize=figsize)

            plot = var_plot.plot(**kwargs)

            # Get the figure from the facet grid
            fig = plt.gcf()
            fig.suptitle(f"{var_name}", fontsize=14, y=1.02)

            figures[var_name] = fig

        except Exception as e:
            print(f"Error plotting {var_name}: {e}")
            continue

    return figures
#---

