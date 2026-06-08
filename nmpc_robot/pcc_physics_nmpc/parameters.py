import numpy as np

# material
E = 23e6
rho = 1220

# arm
r_o = 0.04 / 2
t_wall = 0.002
r_i = r_o - t_wall
r_d = 0.04 / 2
L = 0.315

A = np.pi * (r_o**2 - r_i**2)
I = np.pi * (r_o**4 - r_i**4) / 4
m = rho * np.pi * (r_o**2) * L * 0.5

print("Mass of each segment:", m)

k_phi = 0.0
k_theta = 0.2
beta = 0.05

rho_water = 1000
rho_air = 1.225
rho_liquid = rho_water

horizon_time = 1.0
dt = 0.1
num_segments = 1

tension_min_n = 0.3
tension_max_n = 4.0

q_bend = 300.0
q_phi_dot = 0.2
q_theta_dot = 3.0

qf_bend = 600.0
qf_phi_dot = 0.2
qf_theta_dot = 3.0

MPC_PARAMETERS = {
    "N": int(np.ceil(horizon_time / dt)),

    "Q": np.diag([
        250.0,
        250.0,
        1.0,
        6.0,
    ]),

    "Qf": np.diag([
        500.0,
        500.0,
        1.0,
        6.0,
    ]),

    "R": np.diag([
        8.0,
        8.0,
        8.0,
    ]),

    "R_rate": np.diag([
        8.0,
        8.0,
        8.0,
    ]),

    "u_bound": [0.45, 4.0],
    "v_bound": [-0.8, 0.8],
    "N_p_adaptative": 20,
}



circle_theta_deg = 45.0
circle_loop_time = 45.0
circle_duration = 45.0
circle_phi_start_deg = -45.0
circle_direction = "cw"

theta_circle = np.deg2rad(circle_theta_deg)

RUN_PARAMETERS = {
    "dt": dt,
    "T": circle_duration,

    "x0": np.array([
        0.0,
        0.0,
        0.0,
        0.0,
    ]),

    "T_loop": circle_loop_time,
    "shape": "circle",

    "radius_trajectory": L * (1.0 - np.cos(theta_circle)) / theta_circle,
    "center_trajectory": np.array([
        0.0,
        0.0,
        L * np.sin(theta_circle) / theta_circle,
    ]),
    "rotation_angles_trajectory": np.array([
        np.deg2rad(0.0),
        np.deg2rad(0.0),
        np.deg2rad(circle_phi_start_deg),
    ]),

    "theta_trajectory_deg": circle_theta_deg,
    "phi_start_deg": circle_phi_start_deg,
    "direction": circle_direction,
}


sigma_k = np.deg2rad([
    -170.0,
    -58.0,
    62.0,
]).tolist()

ARM_PARAMETERS = {
    "L_segs": [L] * num_segments,
    "r_o": r_o,
    "r_i": r_i,
    "sigma_k": sigma_k,
    "rho_arm": rho,
    "beta": [beta] * num_segments,
    "K": np.diag([k_phi, k_theta] * num_segments),
    "num_segments": num_segments,
    "rho_liquid": rho_liquid,
    "r_d": r_d,
    "m": m,
    "tension_pretension": tension_min_n,
}
