import numpy as np


def random_circular_orbit_state(
    radius_km: float = 6771.0,
    speed_km_s: float = 7.67,
    rng: np.random.Generator | None = None,
):
    if rng is None:
        rng = np.random.default_rng()

    # 1. Random position direction (uniform on sphere)
    r_hat = rng.normal(size=3)
    r_hat /= np.linalg.norm(r_hat)
    r = radius_km * r_hat

    # 2. Random vector not parallel to r
    u = rng.normal(size=3)

    # 3. Project out radial component -> perpendicular direction
    v_hat = u - np.dot(u, r_hat) * r_hat
    v_hat /= np.linalg.norm(v_hat)

    # 4. Scale to circular orbit speed
    v = speed_km_s * v_hat

    return r, v


print("Generated random circular orbit state:")
r, v = random_circular_orbit_state()
print(f"Position (km): {r}")
print(f"Velocity (km/s): {v}")
