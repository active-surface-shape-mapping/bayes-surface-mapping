import torch
from pathlib import Path
import plotly.graph_objects as go


@torch.no_grad()
def plot_acq(learner: "ActiveLearner"):
    # candidate points (unit space)
    cand_u = learner.cand_x.detach()  # (Nc,2)

    # denormalize candidates via data_generator
    cand_raw = learner.data_generator.untransform_inputs(cand_u)
    T_raw = cand_raw[:, 0]
    theta_raw = cand_raw[:, 1]

    # posterior mean z-value at candidates
    y_pred = learner.model.likelihood(learner.model(cand_u))
    z_mean = (y_pred.mean * learner._y_std + learner._y_mean).reshape(-1)

    # acquisition values for color/tooltips
    acq = learner.last_scores
    if acq is None:
        acq = torch.full((cand_u.shape[0],), float("nan"), device=cand_u.device)
    acq = acq.reshape(-1)

    # output folder
    out_dir = Path(learner.config.base_savepath) / "Acquisition plots"
    out_dir.mkdir(exist_ok=True, parents=True)
    savepath_pdf = out_dir / f"iteration_{learner._i:03d}.pdf"
    savepath_html = out_dir / f"iteration_{learner._i:03d}.html"

    # sampled points
    z_label = "log(Cv)" if learner.config.use_log_transform else "Cv"

    n_init = int(learner.config.init_n_points)
    xs_all = learner.data_generator.untransform_inputs(learner.train_x.detach())
    z_all = learner.train_y_orig.detach().reshape(-1)

    xs = xs_all[:-1]
    z_samp = z_all[:-1]

    new_xs = learner.data_generator.untransform_inputs(learner.next_x.detach())
    z_next = learner.next_y_orig.detach().reshape(-1)

    # figure
    fig = go.Figure()

    # candidate cloud colored by acquisition, z = posterior mean
    fig.add_trace(
        go.Scatter3d(
            x=T_raw.detach().cpu().numpy(),
            y=theta_raw.detach().cpu().numpy(),
            z=z_mean.detach().cpu().numpy(),
            mode="markers",
            marker=dict(
                size=6,
                opacity=0.75,
                color=acq.detach().cpu().numpy(),
                colorscale="Magma",
                showscale=True,
                colorbar=dict(title="Acquisition"),
            ),
            showlegend=False,
            hovertemplate=(
                "Temp: %{x}<br>"
                "Theta: %{y}<br>"
                f"{z_label} (posterior mean): %{{z}}<br>"
                "Acquisition: %{marker.color}<extra></extra>"
            ),
        )
    )

    # sampled points: blue X (observed z)
    if xs.numel() > 0:
        fig.add_trace(
            go.Scatter3d(
                x=xs[:, 0].detach().cpu().numpy(),
                y=xs[:, 1].detach().cpu().numpy(),
                z=z_samp.detach().cpu().numpy(),
                mode="markers",
                marker=dict(symbol="x", size=4, color="blue", opacity=1.0),
                showlegend=False,
                hovertemplate=(
                    "Sampled Point<br>"
                    "Temp: %{x}<br>"
                    "Theta: %{y}<br>"
                    f"{z_label} (observed): %{{z}}<extra></extra>"
                ),
            )
        )

    # most recent point: green X (observed z)
    fig.add_trace(
        go.Scatter3d(
            x=new_xs[:, 0].detach().cpu().numpy(),
            y=new_xs[:, 1].detach().cpu().numpy(),
            z=z_next.detach().cpu().numpy(),
            mode="markers",
            marker=dict(symbol="x", size=4, color="green", opacity=1.0),
            showlegend=False,
            hovertemplate=(
                "Most Recent Sample<br>"
                "Temp: %{x}<br>"
                "Theta: %{y}<br>"
                f"{z_label} (observed): %{{z}}<extra></extra>"
            ),
        )
    )

    # camera
    camera = dict(eye=dict(x=-1.6, y=-1.8, z=0.8))

    # axis ranges (temp/theta from generator, z from posterior mean)
    x_range = [
        float(learner.data_generator.min_temp),
        float(learner.data_generator.max_temp),
    ]
    y_range = [
        float(learner.data_generator.min_theta),
        float(learner.data_generator.max_theta),
    ]

    zmin = float(z_mean.min().detach().cpu().item())
    zmax = float(z_mean.max().detach().cpu().item())
    pad = 0.05 * (zmax - zmin + 1e-12)
    z_range = [zmin - pad, zmax + pad]

    # title with literate method name
    method = learner.config.acquisition_function
    METHOD_LABEL = {
        "mutual_information": "Mutual Information",
        "eig": "EIG",
        "random": "Random Sampling",
    }

    fig.update_layout(
        title_text=f"{METHOD_LABEL.get(method, method)} Acquisition | Iteration {learner._i}",
        scene=dict(
            xaxis=dict(title="Temperature", range=x_range),
            yaxis=dict(title="Coverage", range=y_range),
            zaxis=dict(title=z_label, range=z_range),
        ),
        scene_camera=camera,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # save
    fig.write_html(str(savepath_html), include_plotlyjs=True, full_html=True)
