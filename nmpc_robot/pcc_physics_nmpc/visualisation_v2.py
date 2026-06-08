import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import shutil

out_dir = "csv_and_plots_adapt/"
colored_line = True


def xyz_to_phi_theta_reference(xyz):
    """
    Convert a task-space tip trajectory to an approximate one-segment PCC
    phi/theta reference for plotting only.

    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        return None, None

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r_xy = np.sqrt(x**2 + y**2)

    phi_ref = np.unwrap(np.arctan2(y, x))
    theta_ref = 2.0 * np.arctan2(r_xy, np.maximum(z, 1e-9))

    return phi_ref, theta_ref


def build_reference_xyz_for_plot(pcc_arm, history_xyz_meas, target_xyz=None, xyz_traj=None):
    """
    Build the exact reference used for time plots.
    """
    n_meas = history_xyz_meas.shape[0]

    if target_xyz is not None:
        target_xyz_single = np.asarray(target_xyz, dtype=float).reshape(1, 3)
        xyz_target_arr = np.repeat(target_xyz_single, n_meas, axis=0)
        return xyz_target_arr, "Tip error to fixed target"

    if hasattr(pcc_arm, "history_xyz_ref"):
        logged_ref = np.asarray(
            pcc_arm.history_xyz_ref[:, :pcc_arm.history_index],
            dtype=float,
        ).T

        if logged_ref.ndim == 2 and logged_ref.shape[1] == 3:
            valid = np.isfinite(logged_ref).all(axis=1)
            if np.any(valid):
                logged_ref = logged_ref[:n_meas]
                valid = valid[:n_meas]

                if not np.all(valid):
                    idx_valid = np.where(valid)[0]
                    for i in range(n_meas):
                        if not valid[i]:
                            nearest = idx_valid[np.argmin(np.abs(idx_valid - i))]
                            logged_ref[i] = logged_ref[nearest]

                return logged_ref, "Tip error to trajectory"

    if xyz_traj is not None:
        xyz_target_arr = np.asarray(xyz_traj, dtype=float)
        n = min(n_meas, xyz_target_arr.shape[0])
        return xyz_target_arr[:n], "Tip error to trajectory"

    return None, None


def history_plot(
    pcc_arm,
    u_bound,
    xyz_traj=None,
    save=False,
    opti_index=None,
    sim_parameters=None,
    target_phi_rad=None,
    target_theta_rad=None,
    target_xyz=None,
    current_gain_mA_per_N=64.7,
    current_offset_mA=0.0,
    current_limit_mA=120.0,
    current_signs=None,
    plot_adaptive=False,
    output_dir="csv_and_plots_adapt/hardware_tests/latest",
):
    if opti_index is None:
        opti_index = [0]
    
    output_dir = Path(output_dir)

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    history = pcc_arm.history[:, :pcc_arm.history_index].T
    history_u_tendon = pcc_arm.history_u_tendon[:, :pcc_arm.history_index].T
    history_param = pcc_arm.history_adaptive_param[:, :pcc_arm.history_index].T
    history_pred = pcc_arm.history_pred[:, :pcc_arm.history_index].T
    history_meas = pcc_arm.history_meas[:, :pcc_arm.history_index].T

    if save:
        np.savetxt(output_dir / "history_angles.csv", history, delimiter=",") 

    if save:
        np.savetxt(output_dir / "history_u_tendon.csv", history_u_tendon, delimiter=",")
    if current_signs is None:
        current_signs = np.ones(history_u_tendon.shape[1])

    current_signs = np.asarray(current_signs, dtype=float).reshape(1, -1)

    history_u_tendon_current = (
        current_signs *
        np.clip(
            current_gain_mA_per_N * history_u_tendon + current_offset_mA,
            0.0,
            current_limit_mA
        )
    )
    time = np.arange(history.shape[0]) * pcc_arm.dt

    # ============================================================
    # Measured tip position and reference used for trajectory plots
    # ============================================================
    history_xyz_meas = np.array([
        pcc_arm.end_effector(x[:2 * pcc_arm.num_segments]).full().flatten()
        for x in history_meas
    ])

    xyz_target_arr, error_label = build_reference_xyz_for_plot(
        pcc_arm,
        history_xyz_meas,
        target_xyz=target_xyz,
        xyz_traj=xyz_traj,
    )

    if xyz_target_arr is not None:
        n_ref = min(history_xyz_meas.shape[0], xyz_target_arr.shape[0])
        history_xyz_meas = history_xyz_meas[:n_ref]
        xyz_target_arr = xyz_target_arr[:n_ref]
        time_ref = time[:n_ref]
        phi_ref_plot, theta_ref_plot = xyz_to_phi_theta_reference(xyz_target_arr)
    else:
        time_ref = time
        phi_ref_plot, theta_ref_plot = None, None

    # ============================================================
    # States + tensions + currents
    # One segment only
    # ============================================================

    fig, axs = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    fig.suptitle("States and Tendon Inputs for One Segment")

        # phi
    axs[0].set_title("Segment 1 Phi")

    phi_deg = np.rad2deg(history[:, 0])
    phi_dot_deg_s = np.rad2deg(history[:, 2])

    axs[0].plot(time, phi_deg, label=r"$\phi$", linestyle="-", color="b")

    if target_phi_rad is not None:
        axs[0].axhline(
            np.rad2deg(target_phi_rad),
            linestyle="--",
            color="b",
            label=r"$\phi_{ref}$"
        )
        axs[0].legend(loc="upper right")
    elif phi_ref_plot is not None:
        axs[0].plot(
            time_ref,
            np.rad2deg(phi_ref_plot),
            linestyle="--",
            color="b",
            label=r"$\phi_{ref}$ from xyz"
        )
        axs[0].legend(loc="upper right")

    axs[0].set_ylabel("deg")
    axs[0].tick_params(axis="y", labelcolor="b")

    phi_ylim_values = [phi_deg]
    if phi_ref_plot is not None:
        phi_ylim_values.append(np.rad2deg(phi_ref_plot))
    phi_ylim_values = np.concatenate([np.asarray(v).flatten() for v in phi_ylim_values])
    phi_ylim_values = phi_ylim_values[np.isfinite(phi_ylim_values)]
    if phi_ylim_values.size > 0:
        phi_min = float(np.min(phi_ylim_values))
        phi_max = float(np.max(phi_ylim_values))
        phi_margin = max(20.0, 0.1 * (phi_max - phi_min + 1e-9))
        axs[0].set_ylim(phi_min - phi_margin, phi_max + phi_margin)
    else:
        axs[0].set_ylim(-220, 220)

    ax0b = axs[0].twinx()
    ax0b.plot(
        time,
        phi_dot_deg_s,
        label=r"$\dot{\phi}$",
        linestyle="-",
        color="c"
    )
    ax0b.set_ylabel("deg/s")
    ax0b.tick_params(axis="y", labelcolor="c")
    
        # theta
    axs[1].set_title("Segment 1 Theta")

    theta_deg = np.rad2deg(history[:, 1])
    theta_dot_deg_s = np.rad2deg(history[:, 3])

    axs[1].plot(time, theta_deg, label=r"$\theta$", linestyle="-", color="b")

    if target_theta_rad is not None:
        axs[1].axhline(
            np.rad2deg(target_theta_rad),
            linestyle="--",
            color="b",
            label=r"$\theta_{ref}$"
        )
        axs[1].legend(loc="upper right")
    elif theta_ref_plot is not None:
        axs[1].plot(
            time_ref,
            np.rad2deg(theta_ref_plot),
            linestyle="--",
            color="b",
            label=r"$\theta_{ref}$ from xyz"
        )
        axs[1].legend(loc="upper right")

    axs[1].set_ylabel("deg")
    axs[1].tick_params(axis="y", labelcolor="b")

    theta_ylim_values = [theta_deg]
    if theta_ref_plot is not None:
        theta_ylim_values.append(np.rad2deg(theta_ref_plot))
    theta_ylim_values = np.concatenate([np.asarray(v).flatten() for v in theta_ylim_values])
    theta_ylim_values = theta_ylim_values[np.isfinite(theta_ylim_values)]
    if theta_ylim_values.size > 0:
        theta_min = min(-20.0, float(np.min(theta_ylim_values)) - 10.0)
        theta_max = max(80.0, float(np.max(theta_ylim_values)) + 10.0)
        axs[1].set_ylim(theta_min, theta_max)
    else:
        axs[1].set_ylim(-20, 80)

    ax1b = axs[1].twinx()
    ax1b.plot(
        time,
        theta_dot_deg_s,
        label=r"$\dot{\theta}$",
        linestyle="-",
        color="c"
    )
    ax1b.set_ylabel("deg/s")
    ax1b.tick_params(axis="y", labelcolor="c")

    # tendon tensions
    labels = [r"$T_1$", r"$T_2$", r"$T_3$"]
    colors = ["b", "r", "m"]

    for k in range(3):
        axs[2].plot(
            time,
            history_u_tendon[:, k],
            label=labels[k],
            linestyle="-",
            color=colors[k],
        )

    axs[2].set_title("Tendon Tensions")
    axs[2].set_ylabel("Force [N]")
    axs[2].set_ylim(0, u_bound[1] * 1.1)
    axs[2].legend()

    # motor currents
    labels = [r"$I_1$", r"$I_2$", r"$I_3$"]

    for k in range(3):
        axs[3].plot(
            time,
            history_u_tendon_current[:, k],
            label=labels[k],
            linestyle="-",
            color=colors[k],
        )

    axs[3].set_title("Motor Currents")
    axs[3].set_xlabel("Time [s]")
    axs[3].set_ylabel("Current [mA]")
    axs[3].axhline(0.0, linestyle="--", linewidth=0.8)
    axs[3].set_ylim(-1.1 * current_limit_mA, 1.1 * current_limit_mA)
    axs[3].legend()

    if save:
        plt.savefig(output_dir / "states_tensions_currents_1seg.png", dpi=200)

    # ============================================================
    # Tendon-rate commands from reformulated NMPC
    # ============================================================
    if hasattr(pcc_arm, "history_u_rate"):
        history_u_rate = pcc_arm.history_u_rate[:, :pcc_arm.history_index].T

        if history_u_rate.shape[0] == history_u_tendon.shape[0]:
            fig_rate, ax_rate = plt.subplots(1, 1, figsize=(10, 4))

            for k in range(3):
                ax_rate.plot(
                    time,
                    history_u_rate[:, k],
                    label=rf"$\dot{{T}}_{k + 1}$",
                    linestyle="-",
                    color=colors[k],
                )

            ax_rate.axhline(0.0, linestyle="--", linewidth=0.8)
            ax_rate.set_title("Tendon Rate Commands")
            ax_rate.set_xlabel("Time [s]")
            ax_rate.set_ylabel("Rate [N/s]")
            ax_rate.grid(True)
            ax_rate.legend()

            if save:
                np.savetxt(
                    output_dir / "history_u_rate.csv",
                    history_u_rate,
                    delimiter=",",
                    header="dT1_dt_N_per_s,dT2_dt_N_per_s,dT3_dt_N_per_s",
                    comments=""
                )
                fig_rate.savefig(output_dir / "tendon_rate_commands.png", dpi=200)

    # ============================================================
    # Adaptive parameters plot
    # One segment: mass, damping segment 1, stiffness segment 1
    # ============================================================
    if plot_adaptive:
        titles = [
            "Mass",
            "Damping Segment 1",
            "Stiffness Segment 1",
        ]

        initial_param = np.concatenate((
            [pcc_arm.m],
            pcc_arm.beta,
            np.diag(pcc_arm.K)[1::2],
        ))

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        for i in range(pcc_arm.num_adaptive_params):
            axes[i].plot(time, history_param[:, i], label=f"Param {i + 1}")
            axes[i].axhline(initial_param[i], linestyle="--", label="Initial Value")
            axes[i].set_title(f"Adaptive {titles[i]} over Time")
            axes[i].set_ylabel("Value")
            axes[i].legend()

        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()

        if save:
            fig.savefig(output_dir / "adaptive_parameters_1seg.png", dpi=200)

    # ============================================================
    # Error plot over time
    # ============================================================

    # ============================================================
# Tip target error plot over time
# ============================================================

    try:
        if xyz_target_arr is None:
            raise ValueError("No target_xyz, logged reference, or xyz_traj provided for tip error plot.")

        time_axis = time_ref
        tip_error = np.linalg.norm(history_xyz_meas - xyz_target_arr, axis=1)

        # ============================================================
        # Plot tip XYZ reference vs measured
        # ============================================================
        fig_xyz, axs_xyz = plt.subplots(
            3, 1, figsize=(10, 8), sharex=True, constrained_layout=True
        )

        coord_labels = ["X", "Y", "Z"]

        for i in range(3):
            axs_xyz[i].plot(
                time_axis,
                history_xyz_meas[:, i],
                label=f"{coord_labels[i]} measured",
                linestyle="-",
            )

            axs_xyz[i].plot(
                time_axis,
                xyz_target_arr[:, i],
                label=f"{coord_labels[i]} reference",
                linestyle="--",
            )

            axs_xyz[i].set_ylabel(f"{coord_labels[i]} [m]")
            axs_xyz[i].grid(True)
            axs_xyz[i].legend()

        axs_xyz[-1].set_xlabel("Time [s]")
        fig_xyz.suptitle("Tip Position: Reference vs Measured")

        if save:
            np.savetxt(
                output_dir / "tip_xyz_reference_vs_measured.csv",
                np.column_stack((
                    time_axis,
                    history_xyz_meas[:, 0],
                    history_xyz_meas[:, 1],
                    history_xyz_meas[:, 2],
                    xyz_target_arr[:, 0],
                    xyz_target_arr[:, 1],
                    xyz_target_arr[:, 2],
                )),
                delimiter=",",
                header="time_s,x_meas,y_meas,z_meas,x_ref,y_ref,z_ref",
                comments=""
            )

            fig_xyz.savefig(
                output_dir / "tip_xyz_reference_vs_measured.png",
                dpi=200
            )

        # ============================================================
        # Top-down XY tip path: reference vs measured
        # ============================================================
        fig_xy, ax_xy = plt.subplots(1, 1, figsize=(7, 7))

        ax_xy.plot(
            xyz_target_arr[:, 0],
            xyz_target_arr[:, 1],
            linestyle="--",
            label="Reference XY",
        )

        ax_xy.plot(
            history_xyz_meas[:, 0],
            history_xyz_meas[:, 1],
            linestyle="-",
            label="Measured XY",
        )

        # Mark start and end points so the direction of motion is easy to see.
        ax_xy.scatter(
            [xyz_target_arr[0, 0]],
            [xyz_target_arr[0, 1]],
            marker="o",
            s=60,
            label="Reference start",
        )
        ax_xy.scatter(
            [xyz_target_arr[-1, 0]],
            [xyz_target_arr[-1, 1]],
            marker="x",
            s=80,
            label="Reference end",
        )
        ax_xy.scatter(
            [history_xyz_meas[0, 0]],
            [history_xyz_meas[0, 1]],
            marker="o",
            s=60,
            label="Measured start",
        )
        ax_xy.scatter(
            [history_xyz_meas[-1, 0]],
            [history_xyz_meas[-1, 1]],
            marker="x",
            s=80,
            label="Measured end",
        )

        ax_xy.set_title("Top-Down Tip Path: Reference vs Measured")
        ax_xy.set_xlabel("X [m]")
        ax_xy.set_ylabel("Y [m]")
        ax_xy.grid(True)
        ax_xy.axis("equal")
        ax_xy.legend()

        if save:
            np.savetxt(
                output_dir / "tip_top_down_xy_reference_vs_measured.csv",
                np.column_stack((
                    time_axis,
                    history_xyz_meas[:, 0],
                    history_xyz_meas[:, 1],
                    xyz_target_arr[:, 0],
                    xyz_target_arr[:, 1],
                )),
                delimiter=",",
                header="time_s,x_meas,y_meas,x_ref,y_ref",
                comments=""
            )

            fig_xy.savefig(
                output_dir / "tip_top_down_xy_reference_vs_measured.png",
                dpi=200
            )

        # ============================================================
        # Tip error plot over time
        # ============================================================
        if sim_parameters is not None:
            n_mean = int(sim_parameters["T_loop"] // pcc_arm.dt)
        else:
            n_mean = max(1, int(1.0 // pcc_arm.dt))

        n_mean = max(1, n_mean)

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        ax.plot(time_axis, tip_error, label=error_label)

        if len(tip_error) >= n_mean:
            mean_error = np.convolve(
                tip_error,
                np.ones(n_mean) / n_mean,
                mode="valid"
            )
            ax.plot(
                time_axis[-len(mean_error):],
                mean_error,
                label="Moving average"
            )

        ax.set_title("Tip Tracking Error Over Time")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Tip error [m]")
        ax.legend()
        ax.grid(True)

        if save:
            np.savetxt(
                output_dir / "tip_error_to_target.csv",
                np.column_stack((time_axis, tip_error)),
                delimiter=",",
                header="time_s,tip_error_m",
                comments=""
            )

            fig.savefig(output_dir / "tip_error_to_target.png", dpi=200)

    except Exception as e:
        print(f"Could not plot tip target error: {e}")
    
    

    # ============================================================
    # 3D animation
    # ============================================================

    ani = animate_3d_1seg(
        pcc_arm,
        history,
        xyz_traj=xyz_traj,
        target_xyz=target_xyz,
        save=save,
        opti_index=opti_index,
        output_dir=output_dir,
    )

    plt.show()

    return ani


def animate_3d_1seg(
    pcc_arm,
    history,
    xyz_traj=None,
    target_xyz=None,
    save=False,
    opti_index=None,
    output_dir="csv_and_plots_adapt/hardware_tests/latest",
):
    if opti_index is None:
        opti_index = [0]

    output_dir = Path(output_dir)

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    points1 = []

    for x in history:
        q = x[:2 * pcc_arm.num_segments]
        segment1 = pcc_arm.shape_func(q)
        points1.append(segment1.full())

    tip_trajectory = np.array(points1)[:, :, -1]

    if colored_line and len(tip_trajectory) > 1:
        p_start = tip_trajectory[:-1]
        p_end = tip_trajectory[1:]
        tip_segments = np.stack((p_start, p_end), axis=1)

        segment_colors = np.zeros((len(tip_segments), 4))

        opti_index = list(opti_index)
        opti_index.append(len(tip_trajectory))

        cmap = plt.cm.jet(np.linspace(0, 1, len(opti_index) - 1))

        for i in range(len(opti_index) - 1):
            start_idx = opti_index[i]
            end_idx = opti_index[i + 1]

            color = cmap[i]
            safe_end = min(end_idx, len(tip_segments))

            if start_idx < len(tip_segments):
                segment_colors[start_idx:safe_end] = color
    else:
        tip_segments = None
        segment_colors = None

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("PCC Arm Hardware / Simulation - One Segment")

    line1 = ax.plot(
        points1[0][0, :],
        points1[0][1, :],
        points1[0][2, :],
        "b-",
        label="Segment 1",
    )

    if colored_line and tip_segments is not None:
        tip_line_collection = Line3DCollection(
            tip_segments[:1],
            colors=segment_colors[:1],
            linewidth=2,
        )
        ax.add_collection3d(tip_line_collection)
        tip_line = None
    else:
        tip_line = ax.plot(
            tip_trajectory[0, 0],
            tip_trajectory[0, 1],
            tip_trajectory[0, 2],
            "g-",
            label="Tip trajectory",
        )
        tip_line_collection = None

    lines = [line1[0]]

    max_length = np.sum(pcc_arm.L_segs)

    ax.set(
        xlim3d=(-1.1 * max_length, 1.1 * max_length),
        xlabel="X (m)",
    )
    ax.set(
        ylim3d=(-1.1 * max_length, 1.1 * max_length),
        ylabel="Y (m)",
    )
    ax.set(
        zlim3d=(-0.1 * max_length, 1.1 * max_length),
        zlabel="Z (m)",
    )

    if xyz_traj is not None:
        xyz_traj = np.asarray(xyz_traj)

        if xyz_traj.ndim == 2 and xyz_traj.shape[0] > 1:
            xyz_traj_closed = np.vstack((xyz_traj, xyz_traj[0]))

            ax.plot(
                xyz_traj_closed[:, 0],
                xyz_traj_closed[:, 1],
                xyz_traj_closed[:, 2],
                "k--",
                label="Target trajectory",
            )

    if target_xyz is not None:
        target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)

        ax.scatter(
            [target_xyz[0]],
            [target_xyz[1]],
            [target_xyz[2]],
            marker="x",
            s=80,
            label="Fixed target",
        )

    ax.legend()

    fargs = (
        points1,
        lines,
        tip_line[0] if tip_line is not None else None,
        tip_trajectory,
        tip_line_collection,
        tip_segments,
        segment_colors,
        ax,
        pcc_arm.dt,
    )

    ani = animation.FuncAnimation(
        fig,
        func=update_line_1seg,
        frames=len(history),
        interval=pcc_arm.dt * 1000,
        fargs=fargs,
    )

    if save:
        ffmpeg_path = shutil.which("ffmpeg")

        if ffmpeg_path is None:
            print("ffmpeg not found. Skipping mp4 animation save.")
        else:
            print("Saving animation")
            mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path

            ani.save(
                str(output_dir / "pcc_arm_1seg.mp4"),
                writer="ffmpeg",
                fps=int(round(1.0 / pcc_arm.dt)),
                dpi=200,
            )

            print("Animation saved")

    return ani


def update_line_1seg(
    num,
    points1,
    lines,
    tip_line=None,
    tip_trajectory=None,
    tip_line_collection=None,
    tip_segments=None,
    tip_colors=None,
    ax=None,
    dt=0.1,
):
    pts1 = points1[num]

    lines[0].set_data(pts1[0, :], pts1[1, :])
    lines[0].set_3d_properties(pts1[2, :])

    ax.set_title(f"PCC Arm One Segment - Time: {num * dt:.2f} s")

    if colored_line and tip_line_collection is not None:
        current_idx = max(1, min(num, len(tip_segments)))
        tip_line_collection.set_segments(tip_segments[:current_idx])
        tip_line_collection.set_color(tip_colors[:current_idx])
        tip_line_collection.set_alpha(0.7)

        return lines + [tip_line_collection]

    if tip_line is not None:
        tip_line.set_data(tip_trajectory[:num + 1, 0], tip_trajectory[:num + 1, 1])
        tip_line.set_3d_properties(tip_trajectory[:num + 1, 2])

    return lines