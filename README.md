# orbit-propagator-3d

Orbit propagation and spacecraft analysis sandbox for simulating satellite state, attachable spacecraft hardware, and monitor-based analyses such as solar power and TTC contact.

## Conventions

- Distances use `km` unless a name/comment explicitly says otherwise, such as `areas_m2`.
- Velocities use `km/s`.
- Accelerations use `km/s^2`.
- Quaternions are ordered `w, x, y, z`.
- Position and velocity are ECI-frame values.
- Satellite quaternion represents ECI to body rotation.
- Satellite angular velocity `omega` is body-frame `rad/s`.
- `sat.time` is numeric seconds since simulation start.
- `sat.epoch_julian_date` / `sat.epoch_datetime` are optional metadata for environment models.
- RF/link-budget internals use SI units where RF formulas expect them.

## Running Scripts

Run commands from the repository root.

```powershell
python scripts/main.py
```

Propagates the orbit in `config.json`, samples attached monitors, prints periodic solar-power status, and opens a Plotly trajectory/power visualisation.

```powershell
python scripts/montecarlo_power.py
```

Runs a Monte Carlo solar-power analysis over random inclined circular orbits and random initial attitudes. It uses `SolarPowerMonitor` and `BodyAxisAlignmentMonitor`.

```powershell
python scripts/montecarlo_gps.py
```

Runs a Monte Carlo TTC contact analysis for a ground station over sampled RAAN values. It uses `TtcContactMonitor`, `GroundStation`, and an `Antenna` component.

The GPS/TTC script defaults to a fine `0.1 s` timestep and `1000` samples, so it can be slow. For quick checks, import the script and override `N_SAMPLES` / `DT`.

```powershell
python scripts/ttc_report.py --days 7 --dt 10
```

Runs a structured TTC report using current architecture: link-budget table, LoRa time-on-air table, five Keplerian initial-condition pass reports, beacon timing recommendations, and a cross-IC comparison. For a quick smoke run:

```powershell
python scripts/ttc_report.py --days 0.1 --dt 60 --limit-ics 1
```

```powershell
python scripts/magnetorquer_detumble.py
```

Runs a rough B-dot magnetorquer detumble estimate using the tilted-dipole magnetic environment and the `Magnetorquer` component. Useful options include:

```powershell
python scripts/magnetorquer_detumble.py --initial-omega-deg-s "5,-3,2" --threshold-deg-s 0.1 --max-dipole-a-m2 0.02 --bdot-gain 1e6
```

## Verification

```powershell
python -m unittest discover -s tests
python -m compileall .\src .\tests .\scripts
python -m py_compile .\scripts\main.py .\scripts\montecarlo_power.py .\scripts\montecarlo_gps.py .\scripts\ttc_report.py .\scripts\magnetorquer_detumble.py
```

## Design Decisions

The project is organized around a few core choices:

- `Satellite` is the source of truth for spacecraft state. Position, velocity, attitude, angular velocity, mass, inertia, time, components, and monitors live on the satellite.
- Satellite hardware is modeled as pluggable components. Solar panels, antennas, surface geometry, and future reaction wheels or batteries should be attached to `Satellite`, not hardcoded into the satellite class.
- Analyses are modeled as monitors. A monitor observes satellite state after each propagation/simulation step and records histories such as power, contact, attitude alignment, or future battery/data metrics.
- Monitors should not mutate spacecraft hardware or propagation state. Scenario overrides should be passed into calculation helpers or expressed as components.
- Dynamics query components. Force and torque models should ask the satellite what hardware it has instead of assuming every satellite has the same surfaces, wheels, thrusters, or panels.
- Numeric seconds are the primary time model. Epoch metadata is optional and should only be used by environment models that need it.
- Environment models are reusable external-context services. Sun direction and magnetic field live in `src/external/environment`, not on `Satellite` or in scripts.
- Simulation models advance state but do not decide what to analyze. Orbit models should update `Satellite`; monitors decide what values to record.
- Simulation runners own loops and monitor sampling. Scripts should use `DynamicSimulation` or `PrescribedOrbitSimulation` instead of hand-written propagation loops.
- Dynamics are provider-based. New force/torque models should be added as providers, then registered with a dynamics model. Stateful providers should update actuator/controller state in a pre-step hook, while ODE evaluations should read the prepared force/torque state.
- Generic math stays generic. Coordinate transforms and vector/quaternion helpers belong in `src/maths`, not in analysis modules.
- Ground-side objects are separate from spacecraft hardware. A ground station is not a satellite component, so it lives under `src/external/ground`.
- Scripts are experiments, not libraries. Scripts should configure and run scenarios, while reusable physics, geometry, components, and analyses live in `src`.
- Units should be visible from names/comments whenever they are not the project default. For example, orbit distances are `km`, but component areas are named `areas_m2`.

## Architecture

The project separates spacecraft state, spacecraft hardware, simulation models, force models, and analysis monitors.

### Satellite State

`src/satellite/satellite.py` owns the canonical spacecraft state:

- `position`
- `velocity`
- `quaternion`
- `omega`
- `mass`
- `J`, `J_inv`
- `time`
- attached `components`
- attached `monitors`

`Satellite` should remain the state owner. Simulation code should read and update state on `sat`; it should not carry separate attitude/velocity copies unless there is a specific numerical reason.

### Satellite Components

Satellite-specific hardware belongs in `src/satellite/components/`.

Current components:

- `BodySurfaceModel`: body-frame face normals, drag areas, SRP areas, drag coefficient, and optical coefficients.
- `SolarPanelArray`: solar-panel areas and efficiency.
- `Antenna`: TTC antenna min elevation, boresight, gain, and pointing loss.
- `LoRaRadio`: RF frequency, transmit power, bandwidth, LoRa settings, losses, receiver sensitivity, and Eb/N0 threshold.
- `BeaconSchedule`: blind TX/RX timing and payload sizes.
- `Magnetorquer`: B-dot actuator state, commanded dipole limit, and last torque.

Use components for things the spacecraft physically has:

- solar panels
- antennas
- batteries
- reaction wheels
- thrusters
- magnetorquers
- payloads
- onboard data storage

Attach components with:

```python
sat.add_component(Antenna(min_elevation_deg=30.0))
antenna = sat.get_component("ttc_antenna")
panels = sat.get_components(SolarPanelArray)
```

`Satellite()` installs default `BodySurfaceModel` and `SolarPanelArray` for compatibility with existing analyses. Use `Satellite(install_default_components=False)` when a scenario should start with no implicit hardware.

### Analysis Monitors

Analysis monitors belong in `src/analysis/`.

Current monitors:

- `SolarPowerMonitor`: samples generated solar power and eclipse state.
- `TtcContactMonitor`: samples ground-station contact/elevation.
- `TtcLinkBudgetMonitor`: samples elevation, azimuth, range, Doppler, Eb/N0, link margin, and link-open state.
- `BodyAxisAlignmentMonitor`: samples body-axis alignment against an ECI reference axis.
- `DynamicsEnvironmentMonitor`: samples magnetic field, magnetorquer torque, and SRP acceleration histories.
- `AngularRateMonitor`: samples body angular velocity and reports detumble time against a threshold.

Monitors observe the satellite. They should not own propagation logic or define generic geometry classes.

Use monitors for things we want to record:

- generated solar power
- eclipse history
- TTC contact time
- link margin
- downlinked data
- battery state history
- reaction wheel momentum history

Attach monitors with:

```python
power = sat.add_monitor(SolarPowerMonitor())
contact = sat.add_monitor(TtcContactMonitor(ground_station=gs))
```

During `sat.propagate(dt)` or `CircularOrbit.simulate(...)`, monitors are sampled after each state update.
New code should prefer simulation runners for this sampling loop.

### Simulation Models

Simulation models belong in `src/simulation/`.

Current model:

- `CircularOrbit`: closed-form circular orbit state generator for fast experiments.
- `KeplerianCircularOrbit`: circular orbit generated from altitude, inclination, RAAN, argument of perigee, and true anomaly.
- `DynamicSimulation`: integrates satellite dynamics with force/torque providers.
- `PrescribedOrbitSimulation`: samples monitors along an orbit model without dynamically perturbing the orbit.
- `SimulationRunner`: shared loop and monitor-sampling base for simulation runners.
- `AngularRateBelowStop` and `TimeLimitStop`: reusable stop conditions for simulation runs.

`CircularOrbit` uses `sat.position` and `sat.velocity` to define a prescribed orbit. It exposes `position_at(t)` and `velocity_at(t)`.

Preferred pattern:

```python
sat.position = r0
sat.velocity = v0
sat.quaternion = q0
sat.omega = omega0

orbit = CircularOrbit.from_satellite(sat)
simulation = PrescribedOrbitSimulation(sat, orbit)
simulation.run(total_time_s, dt_s)
```

For coupled forces/torques:

```python
simulation = DynamicSimulation(sat)
simulation.run(total_time_s, dt_s)
```

Add new orbit/simulation approximations here, not in scripts. Keep `orbit.simulate(...)` only for backward compatibility; new scripts should use simulation runners directly.

### Dynamics

Numerical force/torque models belong in `src/dynamics/`.

Current dynamics include:

- atmospheric drag
- J2 acceleration
- RK4 ODE stepping
- surface-model solar radiation pressure
- tilted-dipole magnetic field driven magnetorquer torque

Dynamics should query satellite components for hardware-dependent effects. For example, drag uses `BodySurfaceModel` instead of hardcoded `sat.A` / `sat.n`.
Reaction wheel torque, thrust, data-storage dynamics, and battery dynamics should follow the same query-based pattern.

Dynamics are organized around providers:

- `ForceProvider.acceleration_eci_km_s2(...)`
- `TorqueProvider.torque_body_n_m(...)`
- Optional `prepare_step(...)` hooks for stateful controllers/actuators.
- `DefaultDynamicsModel`: sums default providers for Kepler gravity, J2, drag, SRP, and magnetorquer torque.

Use a custom dynamics model when a scenario needs to disable or replace a provider. Do not add new imports directly to the ODE/integration layer for each future force or torque.

### External Context

External context models belong in `src/external/`.

Environment models live in `src/external/environment/`.

Current models:

- `sun_direction_unit(...)`: approximate inertial Sun direction from numeric seconds plus optional epoch.
- `earth_magnetic_field_eci(...)`: tilted-dipole magnetic field in tesla.

Keep these independent of `Satellite`. Dynamics and monitors may pass `sat.position`, `sat.time`, and optional epoch metadata into environment functions.

Ground-side, non-satellite objects live in `src/external/ground/`.

Current ground object:

- `GroundStation`: latitude, longitude, ECEF position, local vertical.
- `ground.geometry`: elevation/azimuth/range and Doppler helpers.

Ground stations are not satellite hardware and should not live in `src/satellite/components` or `src/analysis`.

### Math And Geometry

Generic math belongs in `src/maths/`.

Current examples:

- quaternion/DCM helpers
- angular velocity matrix `Omega`
- vector-angle helpers
- rotation matrices
- ECI/ECEF conversion
- simple spherical geodetic conversion

Frame conversion and reusable geometry should go here, unless it represents a real-world object.

### Utility Helpers

Experiment setup helpers belong in `src/utils/`.

Current helpers:

- `generate_inclined_circular_orbit`
- `keplerian_to_state_vectors`
- `random_quaternion`

Use this area for reusable initial-condition generation or other small setup helpers. If a helper becomes a physical model, move it to `simulation`, `dynamics`, `ground`, or `satellite/components`.

### Scripts

Scripts belong in `scripts/`.

Scripts should be thin orchestration layers:

- configure constants
- generate initial conditions
- create a `Satellite`
- attach components
- attach monitors
- run a simulation/propagation model
- print and plot results

Scripts should not define reusable physics, coordinate conversion, hardware models, or analysis logic. Move those into `src`.

## Where To Add New Things

Add a new spacecraft hardware item:

- Put it in `src/satellite/components/`.
- Inherit from `Component`.
- Give it a stable `name`.
- Add validation in `__init__` or `attach`.

Add a new analysis:

- Put it in `src/analysis/`.
- Inherit from `AnalysisMonitor`.
- Implement `sample(self, sat)`.
- Store histories as arrays/lists on the monitor.

Add a new force/torque model:

- Put it in `src/dynamics/`.
- Implement a force or torque provider.
- Register it through a dynamics model.
- Query satellite components for geometry/hardware.
- Keep units explicit in names/comments.

Add a new orbit/simulation shortcut:

- Put it in `src/simulation/`.
- Let `Satellite` remain the state owner.
- Sample `sat.monitors` after state updates.
- Prefer adding a simulation runner or stop condition over adding loops inside scripts.

Add a new initial-condition generator:

- Put it in `src/utils/initial_conditions.py` unless it grows into a full simulation model.

Add ground infrastructure:

- Put ground stations and ground-side objects in `src/external/ground/`.

Add a new experiment:

- Put it in `scripts/`.
- Import reusable logic from `src`.
- Avoid adding reusable physics/math inside the script.
