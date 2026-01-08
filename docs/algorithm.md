# truck-parking

Motion planning for truck parking using Hybrid A\* and reinforcement learning.

## I. Abstract

This repository focuses on motion planning for truck parking, where the vehicle consists of a tractor and a single trailer.

The approach implements a two-stage pipeline:

1. **Global Planning:** Generate a collision-free reference path from the initial configuration to the goal configuration using the Hybrid A\* algorithm.
2. **Path Following:** Follow the reference path using a neural network trained with reinforcement learning.

## II. Acknowledgements

The truck kinematic model used in this project is based on the following reference:

> Ljungqvist, Oskar. Motion planning and feedback control techniques with applications to long tractor-trailer vehicles. Linköping University, 2020.

Using this model, the truck configuration is represented as:

```math
\mathbf{q} = [x \, y \, \psi \, \phi]^T
```

where:

- $(x, y)$ is the position of the tractor rear-axle center in global frame.
- $\psi$ is the tractor heading in global frame.
- $\phi$ is the articulation angle between the tractor and the trailer.

The reference provides a continuous-time kinematic model that yields the time derivatives of the configuration under constant control commands, velocity $v$ and steering angle $\delta$.

```math
\dot{\mathbf{q}}
=
\begin{bmatrix}
  \dot{x}       \\
  \dot{y}       \\
  \dot{\psi}    \\
  \dot{\phi}
\end{bmatrix}
=
\begin{bmatrix}
  v \cdot \cos(\psi)                        \\
  v \cdot \sin(\psi)                        \\
  v \cdot \dfrac{\tan(\delta)}{L_{tractor}} \\
  v \cdot \dfrac{\tan(\delta)}{L_{tractor}} - \dfrac{\rho \cdot \cos(\phi) \cdot v \cdot \dfrac{\tan(\delta)}{L_{tractor}} + v \cdot \sin(\phi)}{L_{trailer}}
\end{bmatrix}
```

where:

- $L\_{tractor}$ is the tractor wheelbase.
- $L\_{trailer}$ is the trailer wheelbase.
- $\rho$ is the distance from the tractor rear-axle center to the articulation point.

**Note:** This project uses a sign convention for $\rho$ that is opposite to the one used in the reference for implementation simplicity.

## III. Discrete-time vehicle model

Building on the continuous-time kinematic model introduced in Section II, this project derives a discrete-time vehicle model.


**Note:** This project uses a semi-implicit Euler discretization for the kinematic update that utilizes the updated velocity $v\_{k+1}$ and steering angle $\delta\_{k+1}$ to integrate the kinematic model.

The state vector is defined as:

```math
\mathbf{x}_{k}
=
[x_{k} \, y_{k} \, \psi_{k} \, \phi_{k} \, v_{k} \, \delta_{k} \, a_{k} \, w_{k}]^T
```

where:

- $v\_k$ is the current longitudinal velocity.
- $\delta\_k$ is the current steering angle.
- $a\_k$ is the current longitudinal acceleration.
- $w\_k$ is the current steering angle rate.

The control input is defined as:

```math
\mathbf{u}_{k}
=
[u^{v}_{k} \, u^{\delta}_{k}]^T
```

where:

- $u^{v}\_{k}$ is the velocity command at step $k$.
- $u^{\delta}\_{k}$ is the steering angle command at step $k$.

The vehicle is assumed to track commanded inputs through first-order response dynamics with time constants $\tau\_v$ and $\tau\_\delta$:

```math
\begin{aligned}
v_{k+1}       &= v_{k} + (1 - e^{-\frac{\Delta t}{\tau_{v}}})(u^{v}_{k} - v_{k}) \\
\delta_{k+1}  &= \delta_{k} + (1 - e^{-\frac{\Delta t}{\tau_{\delta}}})(u^{\delta}_{k} - \delta_{k})
\end{aligned}
```

The acceleration and steering-rate terms are included in the state as derived quantities:

```math
\begin{aligned}
a_{k+1} &= \frac{v_{k+1} - v_{k}}{\Delta t} \\
w_{k+1} &= \frac{\delta_{k+1} - \delta_{k}}{\Delta t}
\end{aligned}
```

Then, update the kinematic state using $v\_{k+1}$ and $\delta\_{k+1}$:

```math
\begin{aligned}
x_{k+1}     &= x_{k} + \Delta t \cdot v_{k+1} \cdot \cos(\psi_{k})                            \\
y_{k+1}     &= y_{k} + \Delta t \cdot v_{k+1} \cdot \sin(\psi_{k})                            \\
\psi_{k+1}  &= \psi_{k} + \Delta t \cdot v_{k+1} \cdot \frac{\tan(\delta_{k+1})}{L_{tractor}} \\
\phi_{k+1}  &= \phi_{k} + \Delta t \cdot (v_{k+1} \cdot \frac{\tan(\delta_{k+1})}{L_{tractor}} - \frac{\rho \cdot \cos(\phi_{k}) \cdot v_{k+1} \cdot \frac{\tan(\delta_{k+1})}{L_{tractor}} + v_{k+1} \cdot \sin(\phi_{k})}{L_{trailer}})
\end{aligned}
```

Finally, the full state at time step $k+1$ is assembled as:

```math
\mathbf{x}_{k+1}
=
[x_{k+1} \, y_{k+1} \, \psi_{k+1} \, \phi_{k+1} \, v_{k+1} \, \delta_{k+1} \, a_{k+1} \, w_{k+1}]^T
```

## IV. Hybrid A\*

Hybrid A\* generates a global reference path from an initial configuration to a goal configuration while avoiding collisions with static obstacles.

The resulting path is **geometrically** feasible and may include both forward and reverse motions.

The procedure takes the followings as inputs:

- **Truck parameters:** Dimensions, kinematic limits ($v\_{max}, \delta\_{max}$), etc.
- **Map boundary:** Defined as $(x\_{min}, x\_{max}) \times (y\_{min}, y\_{max})$.
- **Obstacles:** Provided as polygons, each represented by a list of vertices in counterclockwise order.
- **Initial and goal configurations:** $\mathbf{q}\_{init}$, $\mathbf{q}\_{goal}$.

The procedure outputs a collision-free path from $\mathbf{q}\_{init}$ to $\mathbf{q}\_{goal}$.

The path is represented as a sequence of **trajectories**, where each trajectory has a single driving direction (forward or reverse).

Each trajectory consists of a sequence of **nodes** where each node represents the representative configuration $(x, y, \psi, \phi)$.

Starting from $\mathbf{q}\_{init}$, Hybrid A\* expands nodes until it reaches $\mathbf{q}\_{goal}$.

The **motion primitives** used for the expansion are defined as follows:
- Each expansion applies a constant control $(v, \delta)$ over a **fixed travel distance** $\Delta s$.
- The control $v$ and $\delta$ are sampled within $[-v\_{max}, v\_{max}]$ and $[-\delta\_{max}, \delta\_{max}]$, respectively.
- The kinematic model used here is the continuous-time kinematic model from Section II (ignoring actuator dynamics and using forward Euler integration).

The "cost-to-come" is defined as the accumulated travel time, and the "cost-to-go" is defined as the Reeds-Shepp path length of the tractor divided by $v\_{max}$.

For driving direction change, the fixed time penalty will be added.

At the end of the procedure, each discrete trajectory is converted into a spline representation, denoted by $\gamma$, parameterized by the accumulated arc length.

Let the trajectory consist of configurations

```math
\mathbf{q}_{i} = [x_{i} \, y_{i} \, \psi_{i} \, \phi_{i}]^T, \quad i = 0, 1, \ldots, N
```

Define the accumulated arc length as

```math
s_{0} = 0, \quad s_{i+1} = s_{i} + \| (x_{i+1}, y_{i+1}) - (x_{i}, y_{i}) \|_{2}, \quad i = 0, 1, \ldots, N-1
```

The spline $\gamma$ supports the following queries:

- $\gamma.\mathrm{length}$: Returns the length of the whole spline.
- $\gamma.\mathrm{closest}(x, y, k)$: Returns the index $i (\ge k)$ of the closest configuration from the given $(x, y)$.
- $\gamma.\mathrm{projection}(x, y, k)$: Returns the $s\_{i (\ge k)}$ of the closest configuration from the given $(x, y)$.
- $\gamma.\mathbf{q}(s)$ for $s \in [0, s\_{N}]$: Returns the configuration at arc length $s$ obtained by piecewise linear interpolation along the polyline.

**Note:** The angle components $(\psi, \phi)$ are interpolated using the wrapped shortest-angle difference.

## V. Neural network policy

The neural network serves as a feedback controller for path following.

Given a reference spline $\gamma$ produced by Hybrid A\*, the policy outputs control commands $(u^v, u^\delta)$ to follow the spline while accounting for vehicle dynamics and obstacle interactions.

Each episode corresponds to tracking a single direction-fixed spline.

The episode terminates when the vehicle reaches the final configuration of the spline, a collision occurs, or a timeout is triggered.

**Note:** To make the policy invariant to the vehicle’s global position and heading, all spatial observations are expressed in an ego-centric frame $\mathcal{F}\_{k}$ before being fed into the network (origin at $(x\_{k}, y\_{k})$, $x$-axis aligned with the vehicle heading $\psi\_{k}$).

During training, the environment evolves as follows:

1. At step $k$, the environment provides the current state $\mathbf{x}\_{k}$.
2. A step-specific local reference is extracted from the trajectory spline $\gamma$ (see V.1 for details).
3. The observation at step $k$ is constructed as follows:
    - `token_parameters`: Truck parameters serialized into a fixed-size vector.
    - `token_state`: $[\phi\_{k}, v\_{k}, \delta\_{k}, a\_{k}, w\_{k}]$, since the pose components are zero in $\mathcal{F}\_{k}$ by construction.
    - `token_obstacles`, `mask_obstacles`: Obstacle polygon vertices expressed in $\mathcal{F}\_{k}$, along with the corresponding boolean mask for padding.
    - `token_reference`: The local reference expressed in $\mathcal{F}\_{k}$.
    - `token_direction`: A length-one vector encoding the driving direction (forward or reverse).
- The action $(u^{v}\_k, u^{\delta}\_k)$ is applied to the vehicle model to obtain the next state $\mathbf{x}\_{k+1}$.
- The reward $r\_{k}$ is computed from the transition (typically using $\mathbf{x}\_{k}$, $\mathbf{x}\_{k+1}$, and the local reference), and the termination conditions are evaluated (see V.2 for details).

### V.1. Local reference

Local reference $\overline{\mathbf{q}}\_{k}$ is a step-specific subset of the global trajectory spline $\gamma$ that provides the policy with a finite look-ahead reference at time step $k$.

First, the closest spline index is updated with a non-decreasing constraint to prevent backward jumps along the spline:

```math
i_{k} = \gamma.\mathrm{closest}(x_{k}, y_{k}, i_{k-1})
```

Second, the current position is projected onto the spline to obtain the corresponding arc-length coordinate:

```math
s_{k} = \gamma.\mathrm{projection}(x_{k}, y_{k}, i_{k-1})
```

Finally, the local reference is constructed by sampling $M$ configurations on the interval $[s\_{k}, s\_{k}^{\mathrm{end}}]$, where the end arc-length is defined using a distance horizon:

```math
\begin{aligned}
D_{\mathrm{horizon}}
& =
|v_{k}|\,T_{\mathrm{horizon}}
+
\frac{1}{2}\,a_{max}\,T_{\mathrm{horizon}}^{2}
\\
s_{k}^{\mathrm{end}}
& =
\min\!\left(\gamma.\mathrm{length},\, s_{k} + D_{\mathrm{horizon}}\right)
\end{aligned}
```

The arc-length samples and the resulting local reference configurations are:

```math
\begin{aligned}
\overline{S}_{k}
& =
\mathrm{linspace}\!\left(s_{k},\, s_{k}^{\mathrm{end}},\, M\right)
=
\left\{
s_{k} + \frac{j}{M-1}\left(s_{k}^{\mathrm{end}} - s_{k}\right)
\;\middle|\;
j \in \{0, 1, \ldots, M-1\}
\right\}

\\

\overline{\mathbf{q}}_{k}
& =
\left\{
\gamma.\mathbf{q}(s)
=
(\overline{x}_{kj},\, \overline{y}_{kj},\, \overline{\psi}_{kj},\, \overline{\phi}_{kj})
\;\middle|\;
s \in \overline{S}_{k}
\right\}
\end{aligned}
```

### V.2. Reward

The reward at time step $k$ is:

```math
r_k = r_{tracking} + r_{smoothness} + r_{constraints} + r_{terminal}
```

The **tracking reward** penalizes deviations from the reference while also encouraging correct progress along the segment.

Define a scalar tracking error:

```math
e_{track} =
\lambda_x (\frac{x_{k+1} - \overline{x}_{k+1}}{\sigma_x})^2
+
\lambda_y (\frac{y_{k+1} - \overline{y}_{k+1}}{\sigma_y})^2
+
\lambda_\psi (\frac{\psi_{k+1} \ominus \overline{\psi}_{k+1}}{\sigma_\psi})^2
+
\lambda_\phi (\frac{\phi_{k+1} \ominus \overline{\phi}_{k+1}}{\sigma_\phi})^2
```

The tracking reward is:

```math
r_{tracking} = - w_{tracking} \cdot \tanh(e_{track})
```

where:

- $(\overline{x}\_{k+1}, \overline{y}\_{k+1}, \overline{\psi}\_{k+1}, \overline{\phi}\_{k+1})$ denotes the closest configuration in $\overline{\mathbf{q}}\_{k}$ from $(x\_{k+1}, y\_{k+1})$.
- $\ominus$ denotes the normalized angle difference in $[-\pi, \pi]$.
- $\sigma_x, \sigma_y, \sigma_\psi, \sigma_\phi > 0$ are normalization scales.
- $\lambda_x, \lambda_y, \lambda_\psi, \lambda_\phi > 0$ are relative weights among components.
- $w_{tracking} > 0$ is a final scaling weight.

The **smoothness reward** penalizes control aggressiveness.

Define the longitudinal jerk:

```math
j_{k+1} = \frac{a_{k+1} - a_k}{\Delta t}
```

The smoothness reward is:

```math
r_{smoothness} = - w_{smoothness} \cdot \tanh((\frac{j_{k+1}}{\sigma_{j}})^2)
```

where:

- $\sigma\_{j} > 0$ is the normalization scale.
- $w\_{smoothness} > 0$ is a final scaling weight.

Hard constraints often cause early termination and unstable learning.

Instead, this project applies **soft penalties** as a **constraints** reward.

Define an over-limit operator:

```math
\mathrm{over}(z, z_{max}) = \max(0, |z| - z_{max})
```

Define a bounded constraint error:

```math
e_{constraints} =
\lambda_a (\frac{\mathrm{over}(a_{k+1}, a_{max})}{\sigma_{a}})^2
+
\lambda_w (\frac{\mathrm{over}(w_{k+1}, w_{max})}{\sigma_{w}})^2
+
\lambda_\phi (\frac{\mathrm{over}(\phi_{k+1}, \phi_{max})}{\sigma_{\phi}})^2
```

Then the constraints reward is:

```math
r_{constraints} = - w_{constraints} \cdot \tanh(e_{constraints})
```

where:

- $\phi\_{max}$ is the articulation angle limit.
- $\lambda\_{a}, \lambda\_{w}, \lambda\_{\phi} > 0$ are relative weights among constraint types.
- $\sigma\_{a}, \sigma\_{w}, \sigma\_{\phi} > 0$ are normalization scales.
- $w_{constraints} > 0$ is a final scaling weight.

The sparse **terminal reward** is defined.

The agent is assumed to arrive at the goal if:

```math
||(x_{k}, y_{k}) - (\overline{x}_{-1}, \overline{y}_{-1})|| < \epsilon_{pos}
\text{ and }
|\psi_{k} \ominus \overline{\psi}_{-1}| < \epsilon_{\psi}
\text{ and }
|\phi_{k} \ominus \overline{\phi}_{-1}| < \epsilon_{\phi}
\text{ and }
|v_{k}| < \epsilon_{v}
```

where:

- $(\overline{x}\_{-1}, \overline{y}\_{-1}, \overline{\psi}\_{-1}, \overline{\phi}\_{-1})$ denotes the last configuration in $\gamma$.
- $\epsilon\_{pos}$, $\epsilon\_{\psi}$, $\epsilon\_{\phi}$, and $\epsilon\_{v}$ denote the pre-defined arrival thresholds.

The terminal reward is:

```math
r_{terminal} =
\begin{cases}
  +C_{arrival} & \text{if arrives} \\
  -C_{timeout} & \text{if timed out} \\
  -C_{collision} & \text{if collision with obstacles} \\
  0 & \text{otherwise}
\end{cases}
```

where:

- $C\_{arrival}$ is the arrival reward.
- $C\_{timeout}$ is the timeout penalty.
- $C\_{collision}$ is the collision penalty.

### V.3. Network architecture

- **Input embedding:**
  - Each token (and the obstacle mask) is treated as a distinct token.
  - Linear layers project these vectors into a shared embedding dimension $d\_{model}$.

- **Feature extraction:** A stack of Transformer encoder layers generates a high-level feature representation.

- **Control generation:** A MLP outputs $(u^v\_k, u^\delta\_k)$.
