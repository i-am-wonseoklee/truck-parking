# truck-parking

Motion planning for truck parking using Hybrid A* and reinforcement learning.

## I. Abstract

This repository focuses on motion planning for truck parking, where the vehicle consists of a tractor and a single trailer.

The approach implements a two-stage pipeline:

1. **Global Planning:** Generate a collision-free reference path from the initial configuration to the goal configuration using the Hybrid A* algorithm.
2. **Path Following:** Follow the reference path using a neural network trained with reinforcement learning (RL), which handles actuator dynamics, constraint satisfaction, and precise path tracking.

## II. Acknowledgements

The truck kinematic model used in this project is based on the following reference:

> Ljungqvist, Oskar. Motion planning and feedback control techniques with applications to long tractor-trailer vehicles. Linköping University, 2020.

Using this model, the truck configuration is represented as:

```math
\mathbf{q} = [x \ y \ \psi \ \phi]^T
```

where:

- $(x, y)$ is the position of the tractor rear-axle center (global frame).
- $\psi$ is the tractor heading (global frame).
- $\phi$ is the articulation angle between the tractor and the trailer.

The reference provides a continuous-time kinematic model that yields the time derivatives of the configuration under constant control commands: velocity $v$ and steering angle $\delta$.

```math
\dot{\mathbf{q}} =
\begin{bmatrix}
  \dot{x}     \\
  \dot{y}     \\
  \dot{\psi}  \\
  \dot{\phi}
\end{bmatrix}
=
\begin{bmatrix}
  v \cdot \cos(\psi) \\
  v \cdot \sin(\psi) \\
  v \cdot \dfrac{\tan(\delta)}{L_{tractor}} \\
  v \cdot \dfrac{\tan(\delta)}{L_{tractor}}
  -
  \dfrac{\rho \cdot \cos(\phi) \cdot \left(v \cdot \dfrac{\tan(\delta)}{L_{tractor}}\right) + v \cdot \sin(\phi)}{L_{trailer}}
\end{bmatrix}
```

where:

- $L_{tractor}$ is the tractor wheelbase.
- $L_{trailer}$ is the trailer wheelbase.
- $\rho$ is the distance from the tractor rear-axle center to the articulation point.

**Note:** This project uses a sign convention for $\rho$ that is opposite to the one used in the reference for simplicity.

## III. Discrete-time vehicle model

Building on the continuous-time kinematic model introduced in Section II, this project derives a discrete-time vehicle model.

The state vector is defined as:

```math
\mathbf{x}_{k} =
[x_{k} \ y_{k} \ \psi_{k} \ \phi_{k} \ v_{k} \ \delta_{k} \ a_{k} \ w_{k}]^T
```

where:

- $v_k$ is the current longitudinal velocity.
- $\delta_k$ is the current steering angle.
- $a_k$ is the current longitudinal acceleration.
- $w_k$ is the current steering angle rate.

The control input vector is defined as:

```math
\mathbf{u}_{k} =
[u^{v}_{k} \ u^{\delta}_{k}]^T
```

where:

- $u^{v}_k$ is the velocity command.
- $u^{\delta}_k$ is the steering angle command.

### III.1. First-order command response

The vehicle is assumed to track commanded inputs through first-order response dynamics with time constants $\tau_v$ and $\tau_\delta$:

```math
\begin{aligned}
v_{k+1} &= v_k + \frac{\Delta t}{\tau_{v}} \cdot \left(u^{v}_{k} - v_{k}\right) \\
\delta_{k+1} &= \delta_k + \frac{\Delta t}{\tau_{\delta}} \cdot \left(u^{\delta}_{k} - \delta_{k}\right)
\end{aligned}
```

The acceleration and steering-rate terms are included in the state as derived quantities:

```math
\begin{aligned}
a_{k+1} &= \frac{v_{k+1} - v_k}{\Delta t} \\
w_{k+1} &= \frac{\delta_{k+1} - \delta_k}{\Delta t}
\end{aligned}
```

### III.2. Semi-implicit Euler kinematic update

This project uses a semi-implicit Euler discretization for the kinematic update.

In other words, it utilizes the updated velocity $v_{k+1}$ and steering angle $\delta_{k+1}$ to integrate the kinematic model.

First, update $v$ and $\delta$ using the command response model, and compute the derived quantities $a$ and $w$:

```math
\begin{aligned}
v_{k+1} &= v_k + \frac{\Delta t}{\tau_{v}} \cdot \left(u^{v}_{k} - v_{k}\right) \\
\delta_{k+1} &= \delta_k + \frac{\Delta t}{\tau_{\delta}} \cdot \left(u^{\delta}_{k} - \delta_{k}\right) \\
a_{k+1} &= \frac{v_{k+1} - v_k}{\Delta t} \\
w_{k+1} &= \frac{\delta_{k+1} - \delta_k}{\Delta t}
\end{aligned}
```

Then, update the kinematic state using $v_{k+1}$ and $\delta_{k+1}$:

```math
\begin{aligned}
x_{k+1} &= x_k + \Delta t \cdot v_{k+1} \cdot \cos(\psi_k) \\
y_{k+1} &= y_k + \Delta t \cdot v_{k+1} \cdot \sin(\psi_k) \\
\psi_{k+1} &= \psi_k + \Delta t \cdot v_{k+1} \cdot \frac{\tan(\delta_{k+1})}{L_{tractor}} \\
\phi_{k+1} &= \phi_k + \Delta t \cdot \left(
v_{k+1} \cdot \frac{\tan(\delta_{k+1})}{L_{tractor}}
-
\frac{
\rho \cdot \cos(\phi_k) \cdot \left(v_{k+1} \cdot \frac{\tan(\delta_{k+1})}{L_{tractor}}\right)
+
v_{k+1} \cdot \sin(\phi_k)
}{
L_{trailer}
}
\right)
\end{aligned}
```

Finally, the full state at time step $k+1$ is assembled as:

```math
\mathbf{x}_{k+1} =
[x_{k+1} \ y_{k+1} \ \psi_{k+1} \ \phi_{k+1} \ v_{k+1} \ \delta_{k+1} \ a_{k+1} \ w_{k+1}]^T
```

## IV. Hybrid A*

Hybrid A* generates a global reference path from an initial configuration to a goal configuration while avoiding collisions with static obstacles.

The resulting path is geometrically feasible and may include both forward and reverse motions.

### IV.1. Inputs

- **Truck parameters:** Dimensions, articulation geometry, kinematic limits ($v_{min/max}, \delta_{min/max}$), etc.
- **Map boundary:** Defined as $(x_{min}, x_{max}) \times (y_{min}, y_{max})$.
- **Obstacles:** Provided as polygons, each represented by a list of vertices in counterclockwise order.
- **Initial and goal configurations:** $\mathbf{q}_{init}$, $\mathbf{q}_{goal}$.

### IV.2. Output

- A collision-free path from $\mathbf{q}_{init}$ to $\mathbf{q}_{goal}$.
- The path is represented as a sequence of **trajectory segments**, where each segment has a single driving direction (forward or reverse).
- Each segment consists of a sequence of **nodes** $(x, y, \psi, \phi, v, \delta)$.
  - $(v, \delta)$ denotes the constant control applied to propagate to the next node.
  - For the terminal node of each segment, both $v$ and $\delta$ are set to $0$.

### IV.3. Process

- Starting from $\mathbf{q}_{init}$, Hybrid A* expands nodes until it reaches $\mathbf{q}_{goal}$.

- **Motion primitives (fixed-length):**
  - Each expansion applies a constant control $(v, \delta)$ over a fixed travel distance $\Delta s$.
  - In practice, the planner uses a fixed speed magnitude $|v_0|$ and varies only the sign for direction:
    - forward: $v = +v_0$
    - reverse:  $v = -v_0$
  - The corresponding integration time is $\Delta t = \Delta s / |v_0|$.
  - The kinematic model used here is the continuous-time kinematic model from Section II (ignoring actuator dynamics and using forward Euler integration).

- **Sampling:**
  - Steering primitives are sampled over $\delta \in [\delta_{min}, \delta_{max}]$.
  - Direction changes (gear shifts) are allowed but penalized.

- **Cost-to-come:** Accumulated travel time.

- **Cost-to-go:** The Reeds-Shepp path length (computed for the tractor) divided by $v_{max}$, ensuring an admissible heuristic.

- **Penalties:** Direction changes (gear shifts) incur a fixed time penalty.

## V. Neural network policy

The neural network acts as a controller for path following.

Given a reference trajectory segment extracted from the Hybrid A* path, it generates the control commands $(u^v, u^\delta)$ to track the path while handling obstacles and vehicle dynamics.

### V.1. Ego-centric representation

To ensure the policy is invariant to the global position and heading of the vehicle, all spatial inputs are transformed into an **ego-centric frame** before being fed into the network.

At step $k$, define the ego frame $\mathcal{F}_k$ as the coordinate frame:

- Origin at $(x_k, y_k)$.
- Heading aligned with $\psi_k$.

Any global pose or point is converted into this ego frame and denoted by the superscript local.

This implies:

- $x^{local}_k = 0$, $y^{local}_k = 0$, $\psi^{local}_k = 0$.
- $x^{local}_{k+1}$, $y^{local}_{k+1}$, $\psi^{local}_{k+1}$ are generally non-zero (one-step transition expressed in $\mathcal{F}_k$).

### V.2. Reference segment as polyline arc-length

Each Hybrid A* segment is stored as a **polyline** parameterized by the accumulated arc-length.

Let the segment contain reference configurations:

```math
\mathbf{q}_i = [x_i \ y_i \ \psi_i \ \phi_i]^T, \quad i = 0, 1, ..., N
```

Define the accumulated arc-length:

```math
s_0 = 0, \quad s_{i+1} = s_i + \left\| (x_{i+1}, y_{i+1}) - (x_i, y_i) \right\|_2
```

Given any $s \in [0, s_N]$, the reference configuration $\mathbf{q}_{ref}(s)$ is obtained by **piecewise linear interpolation** on the polyline.
Angle components $(\psi, \phi)$ are interpolated using the wrapped shortest-angle difference.

### V.3. Projection progress (stable indexing)

To avoid index oscillation caused by coarse primitives, the environment maintains a continuous progress variable $s_k$ obtained by projecting the current state onto the reference polyline.

- Let $\mathrm{Project}(\mathbf{x}_k)$ return the arc-length progress $s_k$ and the projected point on the polyline.
- To avoid jumps in ambiguous regions, projection is computed only within a local search window around the previous progress $s_{k-1}$ (or previous segment index).
- Since each episode tracks a single direction-fixed segment, the progress is constrained to be non-decreasing:

```math
s_k \leftarrow \max(s_k, s_{k-1})
```

### V.4. Observations (tokens)

The observation is a dictionary of fixed-size tokens.

- **Truck parameters (token_parameters):** Truck parameters serialized into a fixed-size vector.

- **Current state (token_state):**
  - Since $x^{local}_k$, $y^{local}_k$, and $\psi^{local}_k$ are zeros, therefore omitted.
  - The token retains the articulation angle and the remaining states:
    - $[\phi_{curr} \ v_{curr} \ \delta_{curr} \ a_{curr} \ w_{curr}]^T$

- **Obstacles (token_obstacles, mask_obstacles):**
  - Vertices of obstacle polygons transformed into the ego frame $\mathcal{F}_k$.
  - Serialized into a fixed-size vector with a corresponding validity boolean mask.

- **Reference trajectory (token_trajectory):**
  - A local, fixed-horizon reference trajectory in $\mathcal{F}_k$ built from the arc-length polyline.
  - Procedure:
    1. **Projection (anchor progress):** Compute $s_k = \mathrm{Project}(\mathbf{x}_k)$ in global space (stable window + non-decreasing constraint).
    2. **Horizon length:** Define the horizon distance using current speed and maximum acceleration:
       ```math
       D_{horizon} = |v_{curr}| \cdot T_{horizon} + \frac{1}{2} a_{max} \cdot T_{horizon}^2
       ```
    3. **Sampling:** Sample $M$ points at distances $d_m$ along the reference, e.g.,
       ```math
       d_m = \frac{m}{M} D_{horizon}, \quad m=1,2,\dots,M
       ```
       and query $\mathbf{q}_{ref}(s_k + d_m)$ using piecewise linear interpolation.
    4. **Transformation:** Transform sampled configurations into the ego frame $\mathcal{F}_k$.
  - If the remaining segment length is insufficient, the last configuration is repeated.

- **Driving direction (token_direction):** A length-one vector indicating the required motion direction (forward or reverse).

### V.5. Actions

The policy outputs continuous control commands:

```math
\mathbf{u}_k = [u^{v}_{k} \ u^{\delta}_{k}]^T
```

### V.6. Network architecture

- **Input embedding:**
  - Each token (and the obstacle mask) is treated as a distinct token.
  - Linear layers project these vectors into a shared embedding dimension $d_{model}$.

- **Feature extraction:** A stack of Transformer encoder layers generates a high-level feature representation.

- **Control generation:** A MLP outputs $(u^v_k, u^\delta_k)$.

## VI. Training

The neural network is trained using reinforcement learning.

- **Episode:**
  - The agent tracks a single reference trajectory segment (direction-fixed).
  - The episode terminates upon reaching the last configuration of the segment, colliding with an obstacle, or timing out.

- **Action space:** Continuous actions $(u^{v}_k, u^{\delta}_k)$.

- **Simulation:** State transitions are simulated using the discrete-time vehicle model described in Section III.

- **Observation:** token_{parameters|state|obstacles|trajectory|direction} and mask_obstacles.

### VI.1. Reward function

The reward at time step $k$ is:

```math
r_k = r_{tracking} + r_{smoothness} + r_{constraint} + r_{terminal}
```

**Notations:**
- Let $[x^{local}_{k+1} \ y^{local}_{k+1} \ \psi^{local}_{k+1} \ \phi_{k+1} \ v_{k+1} \ \delta_{k+1} \ a_{k+1} \ w_{k+1}]$ denote the next state obtained after applying the given action for one step (expressed in $\mathcal{F}_k$ for the local pose terms).
- Let $s_k$ denote the arc-length progress of the current state $\mathbf{x}_k$ projected onto the reference polyline.
- Let $s_{k+1}$ denote the arc-length progress of the next state $\mathbf{x}_{k+1}$ projected onto the reference polyline.
- Let the **target progress** be defined from the current progress:
  ```math
  s_k^{tar} = s_k + \Delta s_{step}
  ```
  where $\Delta s_{step} > 0$ is a lookahead distance per step (e.g., based on $|v_{curr}| \Delta t$, or a small constant).
- Let the corresponding target configuration be:
  ```math
  \mathbf{q}_k^{tar} = \mathbf{q}_{ref}(s_k^{tar})
  ```
  and let $[x^{local}_{tar} \ y^{local}_{tar} \ \psi^{local}_{tar} \ \phi_{tar}]$ denote this target configuration expressed in $\mathcal{F}_k$.

#### VI.1.1. Tracking reward

The tracking reward penalizes deviations from the reference while also encouraging correct progress along the segment.

Define a scalar tracking error:

```math
\begin{aligned}
e_{track} &=
\lambda_x \left(\frac{x^{local}_{k+1} - x^{local}_{tar}}{\sigma_x}\right)^2
+
\lambda_y \left(\frac{y^{local}_{k+1} - y^{local}_{tar}}{\sigma_y}\right)^2
+
\lambda_\psi \left(\frac{\psi^{local}_{k+1} \ominus \psi^{local}_{tar}}{\sigma_\psi}\right)^2
+
\lambda_\phi \left(\frac{\phi_{k+1} \ominus \phi_{tar}}{\sigma_\phi}\right)^2
+
\lambda_s \left(\frac{s_{k+1} - s_k^{tar}}{\sigma_s}\right)^2
\end{aligned}
```

The tracking reward is:

```math
r_{tracking} = - w_{tracking} \cdot \tanh(e_{track})
```

where:

- $\ominus$ denotes the normalized angle difference in $[-\pi, \pi]$.
- $\sigma_x, \sigma_y, \sigma_\psi, \sigma_\phi, \sigma_s > 0$ are normalization scales.
- $\lambda_x, \lambda_y, \lambda_\psi, \lambda_\phi, \lambda_s > 0$ are relative weights among components.
- $w_{tracking} > 0$ is a final scaling weight.

#### VI.1.2. Smoothness reward (bounded jerk penalty)

The smoothness term penalizes control aggressiveness.

Define the longitudinal jerk:

```math
j_{k+1} = \frac{a_{k+1} - a_k}{\Delta t}
```

The smoothness reward is:

```math
r_{smoothness} = - w_{smoothness} \cdot \tanh\left(\left(\frac{j_{k+1}}{\sigma_{jerk}}\right)^2\right)
```

where:

- $\sigma_{jerk} > 0$ is the normalization scale for longitudinal jerk.
- $w_{smoothness} > 0$ is a final scaling weight.

#### VI.1.3. Constraint violation penalties

Hard constraints often cause early termination and unstable learning.

Instead, this project applies **soft penalties** that are explicitly **bounded** to prevent reward explosion.

Define an over-limit operator:

```math
\mathrm{over}(z, z_{max}) = \max(0, |z| - z_{max})
```

Define a bounded constraint error:

```math
\begin{aligned}
e_{constraint} &=
\lambda_a \left(\frac{\mathrm{over}(a_{k+1}, a_{max})}{\sigma_{a}}\right)^2
+
\lambda_w \left(\frac{\mathrm{over}(w_{k+1}, w_{max})}{\sigma_{w}}\right)^2
+
\lambda_\phi \left(\frac{\mathrm{over}(\phi_{k+1}, \phi_{max})}{\sigma_{\phi}}\right)^2
\end{aligned}
```

Then the constraint penalty is saturated and scaled:

```math
r_{constraint} = - w_{constraint} \cdot \tanh(e_{constraint})
```

where:

- $\phi_{max}$ is the articulation angle limit.
- $\lambda_a, \lambda_w, \lambda_\phi > 0$ are relative weights among constraint types.
- $\sigma_{a}, \sigma_{w}, \sigma_{\phi} > 0$ normalize the magnitude of each violation before squaring.
- $w_{constraint} > 0$ is a final scaling weight.

#### VI.1.4. Terminal reward

Let the terminal configuration (the last node of the reference segment) be $[x^{local}_{goal} \ y^{local}_{goal} \ \psi^{local}_{goal} \ \phi^{local}_{goal}]^T$.

Arrival thresholds are $\epsilon_{pos}$, $\epsilon_{\psi}$, $\epsilon_{\phi}$, and $\epsilon_{v}$.

The terminal reward is:

```math
r_{terminal} =
\begin{cases}
  +C_{success} &
  \text{if }
  ||(x^{local}_{k+1}, y^{local}_{k+1}) - (x^{local}_{goal}, y^{local}_{goal})|| < \epsilon_{pos}
  \text{ and }
  |\psi^{local}_{k+1} \ominus \psi^{local}_{goal}| < \epsilon_{\psi}
  \text{ and }
  |\phi_{k+1} \ominus \phi_{goal}| < \epsilon_{\phi}
  \text{ and }
  |v_{k+1}| < \epsilon_{v} \\
  -C_{timeout} & \text{if timed out} \\
  -C_{collision} & \text{if collision with obstacles} \\
  0 & \text{otherwise}
\end{cases}
```

where:

- $C_{success}$ is the arrival reward.
- $C_{timeout}$ is the timeout penalty.
- $C_{collision}$ is the collision penalty.