import torch
from pathlib import Path
import plotly.graph_objects as go
from bayes_phase.acquisition.peak_finding import find_slice_peak_idxs


@torch.no_grad()
def plot3d(learner: "ActiveLearner"):
    n_temp = int(
        learner.config.diagram_generator_kwargs.get("n_temp_grid", 300)
    )
    device = learner.train_x.device
    dtype = learner.train_x.dtype
    temp_grid = torch.linspace(0.0, 1.0, n_temp, device=device, dtype=dtype)

    # Denormalizing using known minima/maxima in this case
    theta_vals = torch.arange(
        learner.data_generator.min_theta,
        learner.data_generator.max_theta + 1,
        device=device,
        dtype=dtype,
    )
    theta_grid = (
        theta_vals - learner.data_generator.min_theta
    ) / learner.data_generator.theta_range

    x, y = torch.meshgrid(temp_grid, theta_grid, indexing="ij")
    grid = torch.stack([x.ravel(), y.ravel()]).T

    y_pred = learner.model.likelihood(learner.model(grid))
    y_pred_mean = (y_pred.mean * learner._y_std + learner._y_mean).reshape(-1)
    y_pred_std = (y_pred.stddev * learner._y_std).reshape(-1)

    extent = (
        learner.data_generator.min_temp,
        learner.data_generator.max_temp,
        learner.data_generator.min_theta,
        learner.data_generator.max_theta,
    )

    # Output folder
    out_dir = Path(learner.config.base_savepath) / "3D plots"
    out_dir.mkdir(exist_ok=True, parents=True)
    savepath_pdf = out_dir / f"iteration_{learner._i:03d}.pdf"
    savepath_html = out_dir / f"iteration_{learner._i:03d}.html"

    # Original-scale coordinates
    grid_raw = learner.data_generator.untransform_inputs(grid.detach())
    T_raw = grid_raw[:, 0]
    cov_raw = grid_raw[:, 1]

    # Vertical offset for plotting peak points
    z_offset = 0.05

    # Find peaks (or valleys) independently on each theta slice of the
    # posterior mean surface, using the same settings as the configured
    # diagram generator.
    peak_u = []
    peak_z = []

    y_pred_mean_grid = y_pred_mean.reshape(n_temp, len(theta_grid))
    peak_kwargs = learner.config.diagram_generator_kwargs

    for theta_idx in range(len(theta_grid)):
        y_slice = y_pred_mean_grid[:, theta_idx]

        peak_idxs = find_slice_peak_idxs(
            y_slice,
            peak_threshold=peak_kwargs.get("peak_threshold"),
            max_n_peaks=peak_kwargs.get("max_n_peaks"),
            peak_height_rule=peak_kwargs.get("peak_height_rule"),
            extrema=peak_kwargs.get("extrema"),
        )

        if len(peak_idxs) == 0:
            continue

        peak_idxs_t = torch.as_tensor(
            peak_idxs,
            device=device,
            dtype=torch.long,
        )

        # Normalized coordinates of the detected peaks on this theta slice.
        slice_peak_u = torch.stack(
            [
                temp_grid[peak_idxs_t],
                theta_grid[theta_idx].expand(len(peak_idxs_t)),
            ],
            dim=1,
        )

        peak_u.append(slice_peak_u)
        peak_z.append(y_slice[peak_idxs_t] + z_offset)

    if peak_u:
        peak_u = torch.cat(peak_u, dim=0)
        peak_z = torch.cat(peak_z, dim=0)
        peak_raw = learner.data_generator.untransform_inputs(peak_u.detach())
    else:
        peak_u = None
        peak_z = None
        peak_raw = None

    # Method-dependent peak color
    method = learner.config.acquisition_function
    peak_color = {
        "mutual_information": "green",
        "eig": "blue",
        "random": "red",
    }.get(method, "white")

    # Title text
    METHOD_LABEL = {
        "mutual_information": "Mutual Information",
        "eig": "EIG",
        "random": "Random Sampling",
    }

    n_init = int(learner.config.init_n_points)
    xs_all = learner.data_generator.untransform_inputs(learner.train_x.detach())
    z_all = learner.train_y_orig.detach().reshape(-1)

    # Points sampled during AL
    # Only points sampled during AL, excluding the most recent point
    # (plotted separately in orange)
    n_new = int(learner.next_x.shape[0])
    xs = xs_all[n_init:-n_new]
    z_samp = z_all[n_init:-n_new]

    new_x_u = learner.next_x.detach()  # (1,2) in unit space
    new_xs = learner.data_generator.untransform_inputs(
        new_x_u
    )  # raw coords for plotting

    # Posterior mean at next_x (denormalized to original y-scale)
    z_next = learner.next_y_orig.detach().reshape(-1)

    # Plotly figure
    z_label = "Log(scaled Cv)" if learner.config.use_log_transform else "Scaled Cv"

    fig = go.Figure()

    # Point cloud
    z_cloud = y_pred_mean.detach().cpu().numpy()
    fig.add_trace(
        go.Scatter3d(
            x=T_raw.detach().cpu().numpy(),
            y=cov_raw.detach().cpu().numpy(),
            z=y_pred_mean.detach().cpu().numpy(),
            mode="markers",
            marker=dict(
                size=6,
                opacity=0.4,
                color=z_cloud,
                colorscale="Jet",
                showscale=True,
                colorbar=dict(title=z_label),
            ),
            showlegend=False,
            hovertemplate="Temp: %{x}<br>Theta: %{y}<br>"
            + z_label
            + ": %{z}<extra></extra>",
        )
    )

    # Peaks: halo + core so they remain visible
    if peak_raw is not None and peak_z is not None and peak_raw.numel() > 0:
        px = peak_raw[:, 0].detach().cpu().numpy()
        py = peak_raw[:, 1].detach().cpu().numpy()
        pz = peak_z.detach().cpu().numpy()

        # Halo
        fig.add_trace(
            go.Scatter3d(
                x=px,
                y=py,
                z=pz,
                mode="markers",
                marker=dict(size=9, color="white", opacity=1.0),
                hovertemplate=None,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Core
        fig.add_trace(
            go.Scatter3d(
                x=px,
                y=py,
                z=pz,
                mode="markers",
                marker=dict(size=6, color=peak_color, opacity=1.0),
                hovertemplate=(
                    "Peak Point - Temp: %{x}<br>"
                    "Theta: %{y}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Sampled points: black X
    if xs.numel() > 0:
        fig.add_trace(
            go.Scatter3d(
                x=xs[:, 0].detach().cpu().numpy(),
                y=xs[:, 1].detach().cpu().numpy(),
                z=z_samp.detach().cpu().numpy(),
                mode="markers",
                marker=dict(symbol="x", size=4, color="black", opacity=1.0),
                showlegend=False,
                hovertemplate="Sampled Point - Temp: %{x}<br>Theta: %{y}<br>"
                + z_label
                + ": %{z}<extra></extra>",
            )
        )

    # Next point: orange X
    fig.add_trace(
        go.Scatter3d(
            x=new_xs[:, 0].detach().cpu().numpy(),
            y=new_xs[:, 1].detach().cpu().numpy(),
            z=z_next.detach().cpu().numpy(),
            mode="markers",
            marker=dict(symbol="x", size=5, color="orange", opacity=1.0),
            showlegend=False,
            hovertemplate="Temp: %{x}<br>Theta: %{y}<br>"
            + z_label
            + ": %{z}<extra></extra>",
        )
    )

    # Camera
    camera = dict(eye=dict(x=-1.6, y=-1.8, z=0.8))

    # Axis ranges
    x_range = [float(extent[0]), float(extent[1])]
    y_range = [float(extent[2]), float(extent[3])]

    zmin = float(y_pred_mean.min().detach().cpu().item())
    zmax = float(y_pred_mean.max().detach().cpu().item())
    pad = 0.05 * (zmax - zmin + 1e-12)
    z_range = [zmin - pad, zmax + pad]

    fig.update_layout(
        title_text=f"{METHOD_LABEL.get(method, method)} Posterior | Iteration {learner._i}",
        scene=dict(
            xaxis=dict(title="Temperature", range=x_range),
            yaxis=dict(title="Coverage", range=y_range),
            zaxis=dict(title=z_label, range=z_range),
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Save
    fig.write_html(str(savepath_html), include_plotlyjs=True, full_html=True)