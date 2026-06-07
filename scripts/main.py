from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import Satellite
from src.analysis.solarpower import SolarPowerMonitor
from src.maths.quaternions import Quaternion
from src.simulation import DynamicSimulation


sat = Satellite()
sat.load_from_file(PROJECT_ROOT / "config.json")

solar_power = sat.add_monitor(SolarPowerMonitor())

# Propagate for 100 minutes with 1 second steps.
total_time = 100 * 60
dt = 1
steps = int(total_time / dt)

positions = []
attitudes = []
simulation = DynamicSimulation(sat)

for step_idx in range(steps):
    simulation.step(dt)
    positions.append(sat.position.copy())

    q = Quaternion(*sat.quaternion)
    z_body = np.array([0.0, 0.0, 1.0])
    attitudes.append(q.inverse().rotate_vector(z_body))

    if step_idx % 600 == 0:
        eclipse_status = "ECLIPSE" if solar_power.in_eclipse[-1] else "SUNLIT"
        print(
            f"Time: {sat.time/60:.1f} min, Position (km): {sat.position}, "
            f"Power: {solar_power.powers[-1]:.2f} W [{eclipse_status}]"
        )

pos_array = np.array(positions)
power_array = np.array(solar_power.powers)
time_array = np.array(solar_power.times) / 60.0

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
    subplot_titles=(
        "Satellite Trajectory with Attitude Vectors",
        "Solar Power vs Time",
    ),
)

fig.add_trace(
    go.Scatter3d(
        x=pos_array[:, 0],
        y=pos_array[:, 1],
        z=pos_array[:, 2],
        mode="lines",
        name="Trajectory",
    ),
    row=1,
    col=1,
)

for idx in range(0, len(positions), 100):
    pos = positions[idx]
    att = attitudes[idx]
    fig.add_trace(
        go.Cone(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            u=[att[0]],
            v=[att[1]],
            w=[att[2]],
            sizemode="absolute",
            sizeref=500,
            anchor="tip",
            name=f"Attitude at {time_array[idx]:.1f} min",
            showscale=False,
        ),
        row=1,
        col=1,
    )

fig.add_trace(
    go.Scatter(
        x=time_array,
        y=power_array,
        mode="lines",
        name="Solar Power",
        line=dict(color="orange"),
    ),
    row=1,
    col=2,
)

fig.update_layout(
    scene=dict(xaxis_title="X (km)", yaxis_title="Y (km)", zaxis_title="Z (km)"),
    title="Satellite Trajectory and Solar Power Analysis",
    xaxis_title="Time (minutes)",
    yaxis_title="Power (W)",
)

fig.show()
