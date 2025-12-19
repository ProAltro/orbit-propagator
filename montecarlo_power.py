import numpy as np
from src import Satellite
from src.maths.quaternions import Quaternion
import json

SOLAR_CONSTANT = 1361
SOLAR_PANEL_EFFICIENCY = 0.3
R_EARTH = 6371.0
R_SUN = 696340.0

ALTITUDE_KM = 400
RADIUS_KM = 6371 + ALTITUDE_KM
INCLINATION_DEG = 60
MU = 398600.4418

NUM_SIMULATIONS = 500
ORBIT_PERIOD_MINUTES = 92.5
TOTAL_TIME = ORBIT_PERIOD_MINUTES * 60
DT = 10


def generate_inclined_orbit(
    inclination_deg: float,
    radius_km: float,
    raan_deg: float = None,
    arg_lat_deg: float = None,
    rng: np.random.Generator = None,
):
    if rng is None:
        rng = np.random.default_rng()

    if raan_deg is None:
        raan_deg = rng.uniform(0, 360)

    if arg_lat_deg is None:
        arg_lat_deg = rng.uniform(0, 360)

    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    arg_lat = np.radians(arg_lat_deg)

    r_orbital = radius_km * np.array([np.cos(arg_lat), np.sin(arg_lat), 0])

    v_mag = np.sqrt(MU / radius_km)

    v_orbital = v_mag * np.array([-np.sin(arg_lat), np.cos(arg_lat), 0])

    R_raan = np.array(
        [[np.cos(raan), -np.sin(raan), 0], [np.sin(raan), np.cos(raan), 0], [0, 0, 1]]
    )

    R_inc = np.array(
        [[1, 0, 0], [0, np.cos(inc), -np.sin(inc)], [0, np.sin(inc), np.cos(inc)]]
    )

    R = R_raan @ R_inc

    r_eci = R @ r_orbital
    v_eci = R @ v_orbital

    return r_eci, v_eci


def random_quaternion(rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()

    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    return q


def simulate_orbit(
    initial_r,
    initial_v,
    initial_q,
    initial_omega,
    sat_template,
    r_sun: np.ndarray,
    total_time: float,
    dt: float,
):
    from src.maths.maths import Omega

    steps = int(total_time / dt)
    powers = []
    eclipse_steps = 0
    face_aligns = []

    r_mag = np.linalg.norm(initial_r)
    v_mag = np.linalg.norm(initial_v)

    n = v_mag / r_mag

    h = np.cross(initial_r, initial_v)
    h_hat = h / np.linalg.norm(h)
    r_hat = initial_r / r_mag
    v_hat = np.cross(h_hat, r_hat)

    q = initial_q.copy()
    omega = initial_omega.copy()

    for step in range(steps):
        t = step * dt

        theta = n * t
        r = r_mag * (np.cos(theta) * r_hat + np.sin(theta) * v_hat)

        q_dot = 0.5 * Omega(omega) @ q
        q = q + q_dot * dt
        q = q / np.linalg.norm(q)

        quat = Quaternion(*q)

        r_sat_sun = r_sun - r

        r_norm = np.linalg.norm(r)
        r_sun_norm = np.linalg.norm(r_sat_sun)

        sin_theta_earth = min(1.0, R_EARTH / r_norm)
        sin_theta_sun = min(1.0, R_SUN / r_sun_norm)

        theta_earth = np.arcsin(sin_theta_earth)
        theta_sun = np.arcsin(sin_theta_sun)

        cos_phi = np.dot(r, r_sat_sun) / (r_norm * r_sun_norm)
        cos_phi = np.clip(cos_phi, -1, 1)
        phi = np.arccos(cos_phi)

        sun_dir = r_sun / np.linalg.norm(r_sun)

        r_proj = np.dot(r, sun_dir)

        r_perp = np.sqrt(r_norm**2 - r_proj**2)

        in_eclipse = (r_proj < 0) and (r_perp < R_EARTH)

        if in_eclipse:
            power = 0.0
            eclipse_steps += 1
        else:
            sun_dir_eci = r_sat_sun / r_sun_norm
            sun_dir_body = quat.rotate_vector(sun_dir_eci)

            power = 0.0
            for i in range(len(sat_template.n)):
                cos_angle = np.dot(sat_template.n[i], sun_dir_body)
                if cos_angle > 0:
                    power += (
                        SOLAR_CONSTANT
                        * SOLAR_PANEL_EFFICIENCY
                        * sat_template.solarp_area[i]
                        * cos_angle
                    )

        powers.append(power)

        eci_x = np.array([1.0, 0.0, 0.0])
        body_x_eci = quat.rotate_vector(np.array([1.0, 0.0, 0.0]))
        face_align = np.dot(eci_x, body_x_eci)
        face_aligns.append(abs(face_align))

    powers = np.array(powers)
    eclipse_fraction = eclipse_steps / steps
    average_face_align = np.mean(face_aligns)

    return {
        "mean_power": np.mean(powers),
        "max_power": np.max(powers),
        "min_power": np.min(powers),
        "eclipse_fraction": eclipse_fraction,
        "powers": powers,
        "initial_quaternion": initial_q.copy(),
        "initial_omega": initial_omega.copy(),
        "average_face_align": average_face_align,
    }


def run_monte_carlo():
    rng = np.random.default_rng(42)

    r_sun = np.array([1.496e8, 0.0, 0.0])

    results = []

    sat_template = Satellite()

    print(f"Running Monte Carlo simulation with {NUM_SIMULATIONS} orbits...")
    print(f"Altitude: {ALTITUDE_KM} km, Inclination: {INCLINATION_DEG}°")
    print(f"Simulating {TOTAL_TIME/60:.1f} minutes per orbit with dt={DT}s")
    print("-" * 70)

    for i in range(NUM_SIMULATIONS):
        r, v = generate_inclined_orbit(
            inclination_deg=INCLINATION_DEG, radius_km=RADIUS_KM, rng=rng
        )

        q = random_quaternion(rng)
        omega = rng.normal(size=3) * 0.01

        result = simulate_orbit(r, v, q, omega, sat_template, r_sun, TOTAL_TIME, DT)
        result["raan"] = np.degrees(np.arctan2(r[1], r[0]))
        result["orbit_index"] = i

        results.append(result)

        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{NUM_SIMULATIONS} simulations...")

    return results


def analyze_results(results):
    mean_powers = [r["mean_power"] for r in results]
    eclipse_fractions = [r["eclipse_fraction"] for r in results]
    face_aligns = [r["average_face_align"] for r in results]

    normalized_powers = [
        mp / (1 - ef) if ef < 1 else mp
        for mp, ef in zip(mean_powers, eclipse_fractions)
    ]

    orbit_time_hours = TOTAL_TIME / 3600
    energies = [r["mean_power"] * orbit_time_hours for r in results]

    overall_mean = np.mean(mean_powers)
    overall_std = np.std(mean_powers)
    overall_energy = np.mean(energies)
    overall_normalized_mean = np.mean(normalized_powers)
    overall_normalized_std = np.std(normalized_powers)

    best_idx = np.argmax(normalized_powers)
    best = results[best_idx]
    best_energy = best["mean_power"] * orbit_time_hours

    worst_idx = np.argmin(normalized_powers)
    worst = results[worst_idx]
    worst_energy = worst["mean_power"] * orbit_time_hours

    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION RESULTS")
    print("=" * 70)

    print(f"\nOVERALL STATISTICS ({NUM_SIMULATIONS} orbits)")
    print("-" * 40)
    print(f"  Mean Power (across all orbits):  {overall_mean:.4f} W")
    print(f"  Std Dev:                         {overall_std:.4f} W")
    print(f"  Normalized Mean Power:           {overall_normalized_mean:.4f} W")
    print(f"  Normalized Std Dev:              {overall_normalized_std:.4f} W")
    print(f"  Mean Energy per Orbit:           {overall_energy:.4f} Wh")
    print(f"  Mean Eclipse Fraction:           {np.mean(eclipse_fractions)*100:.2f}%")
    print(
        f"  Eclipse Range:                   {np.min(eclipse_fractions)*100:.2f}% - {np.max(eclipse_fractions)*100:.2f}%"
    )

    print(f"\nBEST CASE ORBIT (Orbit #{best['orbit_index']})")
    print("-" * 40)
    print(f"  Mean Power:      {best['mean_power']:.4f} W")
    print(f"  Energy/Orbit:    {best_energy:.4f} Wh")
    print(f"  Max Power:       {best['max_power']:.4f} W")
    print(f"  Min Power:       {best['min_power']:.4f} W")
    print(f"  Eclipse:         {best['eclipse_fraction']*100:.2f}%")
    print(
        f"  Sunlit Mean:     {best['mean_power'] / (1 - best['eclipse_fraction']) if best['eclipse_fraction'] < 1 else 0:.4f} W"
    )
    print(
        f"  Initial Quaternion: [{best['initial_quaternion'][0]:.4f}, {best['initial_quaternion'][1]:.4f}, {best['initial_quaternion'][2]:.4f}, {best['initial_quaternion'][3]:.4f}]"
    )
    print(f"  Average Face Align: {best['average_face_align']:.4f}")

    print(f"\nWORST CASE ORBIT (Orbit #{worst['orbit_index']})")
    print("-" * 40)
    print(f"  Mean Power:      {worst['mean_power']:.4f} W")
    print(f"  Energy/Orbit:    {worst_energy:.4f} Wh")
    print(f"  Max Power:       {worst['max_power']:.4f} W")
    print(f"  Min Power:       {worst['min_power']:.4f} W")
    print(f"  Eclipse:         {worst['eclipse_fraction']*100:.2f}%")
    print(
        f"  Sunlit Mean:     {worst['mean_power'] / (1 - worst['eclipse_fraction']) if worst['eclipse_fraction'] < 1 else 0:.4f} W"
    )
    print(
        f"  Initial Quaternion: [{worst['initial_quaternion'][0]:.4f}, {worst['initial_quaternion'][1]:.4f}, {worst['initial_quaternion'][2]:.4f}, {worst['initial_quaternion'][3]:.4f}]"
    )
    print(f"  Average Face Align: {worst['average_face_align']:.4f}")

    correlation = np.corrcoef(face_aligns, normalized_powers)[0, 1]
    print(f"\nCorrelation (Face Align vs Normalized Mean Power): {correlation:.4f}")

    omega_yz_mags = [
        np.sqrt(r["initial_omega"][1] ** 2 + r["initial_omega"][2] ** 2)
        for r in results
    ]
    correlation_yz = np.corrcoef(omega_yz_mags, normalized_powers)[0, 1]
    print(f"Correlation (Normalized Power vs sqrt(ωy² + ωz²)): {correlation_yz:.4f}")

    return {
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "best": best,
        "worst": worst,
        "all_results": results,
    }


def plot_results(results):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    mean_powers = [r["mean_power"] for r in results]
    eclipse_fractions = [r["eclipse_fraction"] for r in results]
    normalized_powers = [
        mp / (1 - ef) if ef < 1 else mp
        for mp, ef in zip(mean_powers, eclipse_fractions)
    ]
    face_aligns = [r["average_face_align"] for r in results]
    eclipse_fractions = [r["eclipse_fraction"] * 100 for r in results]
    face_aligns = [r["average_face_align"] for r in results]

    best_idx = np.argmax(normalized_powers)
    worst_idx = np.argmin(normalized_powers)

    best = results[best_idx]
    worst = results[worst_idx]

    fig1 = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Normalized Mean Power Distribution",
            "Normalized Power vs Face Align",
            "Best Case Power Over Time",
            "Worst Case Power Over Time",
        ),
    )

    fig1.add_trace(
        go.Histogram(
            x=normalized_powers,
            nbinsx=20,
            name="Normalized Mean Power",
            marker_color="steelblue",
        ),
        row=1,
        col=1,
    )

    fig1.add_trace(
        go.Scatter(
            x=face_aligns,
            y=normalized_powers,
            mode="markers",
            name="Normalized Power vs Face Align",
            marker=dict(
                color=eclipse_fractions,
                colorscale="RdYlGn_r",
                size=8,
                colorbar=dict(title="Eclipse %", x=1.02),
            ),
        ),
        row=1,
        col=2,
    )

    fig1.add_trace(
        go.Scatter(
            x=[best["average_face_align"]],
            y=[normalized_powers[best_idx]],
            mode="markers",
            name="Best",
            marker=dict(color="green", size=15, symbol="star"),
        ),
        row=1,
        col=2,
    )
    fig1.add_trace(
        go.Scatter(
            x=[worst["average_face_align"]],
            y=[normalized_powers[worst_idx]],
            mode="markers",
            name="Worst",
            marker=dict(color="red", size=15, symbol="x"),
        ),
        row=1,
        col=2,
    )

    time_array = np.arange(len(best["powers"])) * DT / 60
    fig1.add_trace(
        go.Scatter(
            x=time_array,
            y=best["powers"],
            mode="lines",
            name="Best Case",
            line=dict(color="green"),
        ),
        row=2,
        col=1,
    )

    fig1.add_trace(
        go.Scatter(
            x=time_array,
            y=worst["powers"],
            mode="lines",
            name="Worst Case",
            line=dict(color="red"),
        ),
        row=2,
        col=2,
    )

    fig1.update_xaxes(title_text="Normalized Power (W)", row=1, col=1)
    fig1.update_xaxes(title_text="Face Align", row=1, col=2)
    fig1.update_xaxes(title_text="Time (min)", row=2, col=1)
    fig1.update_xaxes(title_text="Time (min)", row=2, col=2)

    fig1.update_yaxes(title_text="Count", row=1, col=1)
    fig1.update_yaxes(title_text="Normalized Mean Power (W)", row=1, col=2)
    fig1.update_yaxes(title_text="Power (W)", row=2, col=1)
    fig1.update_yaxes(title_text="Power (W)", row=2, col=2)

    fig1.update_layout(
        title=f"Monte Carlo Solar Power Analysis ({NUM_SIMULATIONS} orbits, {ALTITUDE_KM}km, {INCLINATION_DEG}° inc)",
        showlegend=True,
        height=800,
    )

    fig1.show()

    omega_yz_mags = [
        np.sqrt(r["initial_omega"][1] ** 2 + r["initial_omega"][2] ** 2)
        for r in results
    ]

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=omega_yz_mags,
            y=normalized_powers,
            mode="markers",
            name="Normalized Power vs sqrt(ωy² + ωz²)",
            marker=dict(
                color=eclipse_fractions,
                colorscale="RdYlGn_r",
                size=8,
                colorbar=dict(title="Eclipse %"),
            ),
        )
    )

    fig2.add_trace(
        go.Scatter(
            x=[omega_yz_mags[best_idx]],
            y=[normalized_powers[best_idx]],
            mode="markers",
            name="Best",
            marker=dict(color="green", size=15, symbol="star"),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=[omega_yz_mags[worst_idx]],
            y=[normalized_powers[worst_idx]],
            mode="markers",
            name="Worst",
            marker=dict(color="red", size=15, symbol="x"),
        )
    )

    fig2.update_xaxes(title_text="sqrt(ωy² + ωz²) (rad/s)")
    fig2.update_yaxes(title_text="Normalized Mean Power (W)")
    fig2.update_layout(
        title="Normalized Mean Power vs sqrt(ωy² + ωz²)",
        showlegend=True,
    )

    fig2.show()


if __name__ == "__main__":
    results = run_monte_carlo()
    analysis = analyze_results(results)
    plot_results(results)
