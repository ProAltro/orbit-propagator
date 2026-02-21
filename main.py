from src import Satellite
from src.maths.quaternions import Quaternion
import numpy as np
import plotly.graph_objects as go

# Solar panel constants
SOLAR_CONSTANT = 1361  # W/m^2 at 1 AU
SOLAR_PANEL_EFFICIENCY = 0.2  # 30% efficiency
R_EARTH = 6371e3  # Earth radius in meters
R_SUN = 696340e3  # Sun radius in meters

sat = Satellite()
sat.load_from_file("config.json")

# Propagate for 100 minutes with 0.1 second steps
total_time = 100 * 60  # seconds
dt = 1
steps = int(total_time / dt)

r_sun = np.array([1.496e11, 0.0, 0.0])

positions = []
attitudes = []
solar_powers = []

for _ in range(steps):
    sat.propagate(dt)
    positions.append(sat.position.copy())
    q = Quaternion(*sat.quaternion)
    z_body = np.array([0, 0, 1])
    z_eci = q.inverse().rotate_vector(z_body)
    attitudes.append(z_eci)

    r_sat_sun = r_sun - sat.position

    # eclipse check
    r_norm = np.linalg.norm(sat.position)
    r_sun_norm = np.linalg.norm(r_sat_sun)
    theta_earth = np.arcsin(R_EARTH / r_norm)
    theta_sun = np.arcsin(R_SUN / r_sun_norm)
    phi = np.arccos(np.dot(sat.position, r_sat_sun) / (r_norm * r_sun_norm))

    # calculate solar power
    in_eclipse = phi < (theta_earth + theta_sun) and np.dot(sat.position, r_sat_sun) < 0

    if in_eclipse:
        power = 0.0
    else:
        # Get sun direction in ECI frame
        sun_dir_eci = r_sat_sun / r_sun_norm

        # Transform sun direction to body frame using quaternion
        sun_dir_body = q.rotate_vector(sun_dir_eci)

        # Calculate power from each face
        power = 0.0
        for i in range(len(sat.n)):
            # cos(angle) between sun and face normal
            cos_angle = np.dot(sat.n[i], sun_dir_body)

            # Only add power if sun is shining on this face (cos > 0)
            if cos_angle > 0:
                power += (
                    SOLAR_CONSTANT
                    * SOLAR_PANEL_EFFICIENCY
                    * sat.solarp_area[i]
                    * cos_angle
                )

    solar_powers.append(power)

    if _ % 600 == 0:  # Print every 10 minutes
        eclipse_status = "ECLIPSE" if in_eclipse else "SUNLIT"
        print(
            f"Time: {sat.time/60:.1f} min, Position (km): {sat.position}, Power: {power:.2f} W [{eclipse_status}]"
        )

# Plot the positions
pos_array = np.array(positions)
att_array = np.array(attitudes)
power_array = np.array(solar_powers)
time_array = np.arange(len(solar_powers)) * dt / 60  # time in minutes

# Create subplot figure
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
    subplot_titles=(
        "Satellite Trajectory with Attitude Vectors",
        "Solar Power vs Time",
    ),
)

# Add trajectory
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

# Add attitude vectors every 100 steps
step = 100
for i in range(0, len(positions), step):
    pos = positions[i]
    att = attitudes[i]
    fig.add_trace(
        go.Cone(
            x=[pos[0]],
            y=[pos[1]],
            z=[pos[2]],
            u=[att[0]],
            v=[att[1]],
            w=[att[2]],
            sizemode="absolute",
            sizeref=500,  # Adjust size as needed
            anchor="tip",
            name=f"Attitude at {i*dt/60:.1f} min",
            showscale=False,
        ),
        row=1,
        col=1,
    )

# Add solar power plot
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
