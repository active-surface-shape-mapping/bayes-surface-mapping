import torch
from pathlib import Path
import plotly.graph_objects as go
from bayes_phase.acquisition import find_ridges

@torch.no_grad()
def plot3d(learner: "ActiveLearner"):
    n_temp = 300
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
    theta_grid = (theta_vals - learner.data_generator.min_theta) / learner.data_generator.theta_range
    x, y = torch.meshgrid(temp_grid, theta_grid, indexing="ij")
    grid = torch.stack([x.ravel(), y.ravel()]).T

    y_pred = learner.model.likelihood(learner.model(grid))
    y_pred_mean = (y_pred.mean * learner._y_std + learner._y_mean).reshape(-1)
    y_pred_std  = (y_pred.stddev * learner._y_std).reshape(-1)
    
    extent = (
        learner.data_generator.min_temp,
        learner.data_generator.max_temp,
        learner.data_generator.min_theta,
        learner.data_generator.max_theta,
    )

    # Output folder
    out_dir = Path(learner.config.base_savepath) / "3D plots"
    out_dir.mkdir(exist_ok=True, parents=True)
    savepath_png = out_dir / f"iteration_{learner._i:03d}.png"
    savepath_html = out_dir / f"iteration_{learner._i:03d}.html"

    # original-scale coordinates
    grid_raw = learner.data_generator.untransform_inputs(grid.detach())
    T_raw = grid_raw[:, 0]
    cov_raw = grid_raw[:, 1]


    # vertical offset for plotting ridge points
    z_offset = 0.05

    # ridge points on predicted mean surface
    ridge_raw = None
    ridge_z = None
    try:
        ridge_idx, ridge_u = find_ridges(grid, y_pred_mean)
        ridge_raw = learner.data_generator.untransform_inputs(ridge_u.detach())
        ridge_z = y_pred_mean[ridge_idx] + z_offset # Visual offset
    except Exception:
        ridge_raw = None
        ridge_z = None

    # method-dependent ridge color
    method = learner.config.acquisition_function
    ridge_color = {
        "mutual_information": "green",
        "eig": "blue",
        "random": "red",
    }.get(method, "white")
    # title text
    METHOD_LABEL = {
    "mutual_information": "Mutual Information",
    "eig": "EIG",
    "random": "Random Sampling",
    }


    n_init = int(learner.config.init_n_points)
    xs_all = learner.data_generator.untransform_inputs(learner.train_x.detach())
    z_all = learner.train_y_orig.detach().reshape(-1)
    # points sampled during AL
    # Only points sampled during AL, excluding the most recent point (plotted separately in orange)
    xs = xs_all[n_init:-1]
    z_samp = z_all[n_init:-1]
    new_x_u = learner.next_x.detach()  # (1,2) in unit space
    new_xs = learner.data_generator.untransform_inputs(new_x_u)  # raw coords for plotting

    # posterior mean at next_x (denormalized to original y-scale)
    z_next = learner.next_y_orig.detach().reshape(-1)

    # Plotly figure
    z_label = "log(Cv)" if learner.config.use_log_transform else "Cv"

    fig = go.Figure()

    # point cloud
    z_cloud = y_pred_mean.detach().cpu().numpy()
    fig.add_trace(
        go.Scatter3d(
            x=T_raw.detach().cpu().numpy(),
            y=cov_raw.detach().cpu().numpy(),
            z=y_pred_mean.detach().cpu().numpy(),
            mode="markers",
            marker=dict(
                size=6,
                opacity=0.7,
                color=z_cloud,
                colorscale="Jet",
                showscale=True,
                colorbar=dict(title=z_label),
            ),
            showlegend=False,
            hovertemplate="Temp: %{x}<br>Theta: %{y}<br>" + z_label + ": %{z}<extra></extra>",
        )
    )

    # ridges: halo + core so they remain visible
    if ridge_raw is not None and ridge_z is not None and ridge_raw.numel() > 0:
        rx = ridge_raw[:, 0].detach().cpu().numpy()
        ry = ridge_raw[:, 1].detach().cpu().numpy()
        rz = ridge_z.detach().cpu().numpy()

        # halo (white)
        fig.add_trace(
            go.Scatter3d(
                x=rx, y=ry, z=rz,
                mode="markers",
                marker=dict(size=9, color="white", opacity=1.0),
                hovertemplate=None,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # core (method color)
        fig.add_trace(
            go.Scatter3d(
                x=rx, y=ry, z=rz,
                mode="markers",
                marker=dict(size=6, color=ridge_color, opacity=1.0),
                hovertemplate="Ridge Point - Temp: %{x}<br>Theta: %{y}<br>",
                showlegend=False,
            )
        )

    # sampled points: black X
    if xs.numel() > 0:
        fig.add_trace(
            go.Scatter3d(
                x=xs[:, 0].detach().cpu().numpy(),
                y=xs[:, 1].detach().cpu().numpy(),
                z=z_samp.detach().cpu().numpy(),
                mode="markers",
                marker=dict(symbol="x", size=4, color="black", opacity=1.0),
                showlegend=False,
                hovertemplate="Sampled Point - Temp: %{x}<br>Theta: %{y}<br>" + z_label + ": %{z}<extra></extra>",
            )
        )

    # next point: orange X
    fig.add_trace(
        go.Scatter3d(
            x=new_xs[:, 0].detach().cpu().numpy(),
            y=new_xs[:, 1].detach().cpu().numpy(),
            z=z_next.detach().cpu().numpy(),
            mode="markers",
            marker=dict(symbol="x", size=4, color="orange", opacity=1.0),
            showlegend=False,
            hovertemplate="Temp: %{x}<br>Theta: %{y}<br>" + z_label + ": %{z}<extra></extra>",
        )
    )

    # camera
    camera = dict(eye=dict(x=-1.6, y=-1.8, z=0.8))

    # axis ranges
    x_range = [float(extent[0]), float(extent[1])]
    y_range = [float(extent[2]), float(extent[3])]

    zmin = float(y_pred_mean.min().detach().cpu().item())
    zmax = float(y_pred_mean.max().detach().cpu().item())
    pad = 0.05 * (zmax - zmin + 1e-12)
    z_range = [zmin - pad, zmax + pad]

    fig.update_layout(
        title_text=f"{METHOD_LABEL.get(method, method)} Posterior | Iteration {learner._i}",
        scene=dict(
            xaxis=dict(title="Temp", range=x_range),
            yaxis=dict(title="Coverage", range=y_range),
            zaxis=dict(title=z_label, range=z_range),
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # save
    fig.write_html(str(savepath_html), include_plotlyjs=True, full_html=True)
    fig.write_image(str(savepath_png), width=1600, height=900, scale=2)

