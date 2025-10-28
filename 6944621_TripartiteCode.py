import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# Tripartite Axis Setup
# ----------------------------
def setup_tripartite_axis(ax):
    """
    Configures the axes so that one figure can display the three 
    tripartite response spectrum coordinates:
        - Pseudo-Velocity (main vertical axis)
        - Pseudo-Acceleration (secondary vertical axis)
        - Deformation / Displacement (secondary horizontal axis)
    """
    # Set log-log scaling for the main plot (Period vs. Pseudo-Velocity)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(visible=True, which='both', color='k', linestyle='-', alpha=0.3, lw=0.3)
    ax.set_xlabel('Period (s)')
    ax.set_ylabel('Spectral Velocity (m/s)')

    # Fix limits for consistency with standard tripartite plots
    ax.set_ylim(1e-3, 1e1)   # velocity range (pseudo-velocity)
    ax.set_xlim(1e-2, 1e1)   # period range

    # ------------------------
    # Pseudo-Acceleration axis
    # ------------------------
    # Conversion functions
    def vel_to_acc(v, T):   # Sv → Sa
        return v * (2 * np.pi / T)
    def acc_to_vel(a, T):   # Sa → Sv
        return a / (2 * np.pi / T)

    # Create secondary vertical axis
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(
        vel_to_acc(ax.get_ylim()[0], ax.get_xlim()[1]),  # bottom Sa
        vel_to_acc(ax.get_ylim()[1], ax.get_xlim()[0])   # top Sa
    )
    ax2.set_ylabel('Pseudo-Acceleration (m/s²)')

    # ------------------------
    # Deformation axis
    # ------------------------
    # Conversion functions
    def vel_to_disp(v, T):  # Sv → Sd
        return v * T / (2 * np.pi)
    def disp_to_vel(d, T): # Sd → Sv
        return d * (2 * np.pi) / T

    # Create secondary horizontal axis
    ax3 = ax.twiny()
    ax3.set_xscale('log')
    ax3.set_xlim(
        vel_to_disp(ax.get_xlim()[0], ax.get_ylim()[0]),  # left Sd
        vel_to_disp(ax.get_xlim()[1], ax.get_ylim()[1])   # right Sd
    )
    ax3.set_xlabel('Deformation (m)')

    return ax, ax2, ax3


# ----------------------------
# Plotting Function
# ----------------------------
def plot_tripartite(periods, vel_spectra, damping_ratios):
    """
    Plots the response spectra (pseudo-velocity) for different damping ratios
    on the tripartite coordinate system.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['k', 'blue', 'green', 'red']  # curve colors
    for i, zeta in enumerate(damping_ratios):
        # Plot pseudo-velocity spectrum for each damping ratio
        ax.plot(periods, vel_spectra[zeta], 
                c=colors[i % len(colors)], linewidth=2, 
                label=f'ζ = {zeta}')

    # Add the tripartite axis overlays (Sv, Sa, Sd)
    ax, ax2, ax3 = setup_tripartite_axis(ax)
    ax.legend()
    plt.title('Tripartite Response Spectrum')
    return fig, ax


# ----------------------------
# Load Earthquake Data
# ----------------------------
# Reads ground motion time history from Excel file.
df = pd.read_excel("RSN42.xlsx", sheet_name="Sheet1")
time = df.iloc[:, 0].values        # first column: time (s)
ag = df.iloc[:, 1].values          # second column: accel. (in g)
dt = time[1] - time[0]             # sampling interval
ag_mps2 = ag * 9.81                # convert to m/s²


# ----------------------------
# System Parameters
# ----------------------------
m = 1.0   # assumed unit mass (results are scalable)
damping_ratios = [0.0, 0.02, 0.10, 0.20]   # 0%, 2%, 10%, 20%
Tn_range = np.logspace(-2, 1, 100)         # periods from 0.01s to 10s
omega_n_range = 2 * np.pi / Tn_range       # natural circular frequencies

# Storage dictionary for spectra (per damping ratio)
vel_spectra = {}


# ----------------------------
# Structural Response (CDM)
# ----------------------------
# For each damping ratio, compute the response spectrum using
# Central Difference Method (step-by-step integration).
for zeta in damping_ratios:
    peak_vel = []  # store peak velocities for each oscillator period

    for omega_n in omega_n_range:
        # Define spring and damper properties for SDOF system
        k = m * omega_n**2
        c = 2 * m * omega_n * zeta

        # Central difference method constants
        a = m / dt**2 + c / (2 * dt)
        b = m / dt**2 - c / (2 * dt)
        k_hat = k - 2 * m / dt**2

        # Initialize displacement response array
        u = np.zeros_like(ag_mps2)
        u[0], u[1] = 0, 0  # initial conditions: rest state

        # Time-stepping recursion for displacement response
        for i in range(1, len(ag_mps2) - 1):
            p_eff = -m * ag_mps2[i] - b * u[i-1] - k_hat * u[i]
            u[i+1] = p_eff / a

        # Approximate velocity response from displacement
        v = (u[2:] - u[:-2]) / (2 * dt)

        # Store peak velocity response (absolute)
        peak_vel.append(np.max(np.abs(v)))

    # Save full spectrum curve for this damping ratio
    vel_spectra[zeta] = np.array(peak_vel)


# ----------------------------
# Plot Spectrum
# ----------------------------
fig, ax = plot_tripartite(Tn_range, vel_spectra, damping_ratios)
plt.tight_layout()
plt.show()
