import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from pydantic import BaseModel

from dpyverification.datamodel import OutputDataset


class Theme(BaseModel):
    """Theme settings for app-wide styling tokens."""

    font_family: str = "Segoe UI, Arial, sans-serif"
    font_color: str = "#022B6D"
    page_bg: str = "#F8FAFC"
    card_bg: str = "#FFFFFF"
    card_border: str = "#E2E8F0"
    accent: str = "#7A491C"
    accent_muted: str = "#61BD4D"
    reference: str = "#818181"


THEME = Theme()
MISSING_VALUE_MARKER = -999

STYLES = {
    "page": {
        "maxWidth": "1200px",
        "margin": "24px auto",
        "padding": "16px",
        "fontFamily": THEME.font_family,
        "color": THEME.font_color,
        "backgroundColor": THEME.page_bg,
    },
    "filters": {
        "display": "flex",
        "gap": "12px",
        "flexWrap": "wrap",
        "marginBottom": "12px",
        "backgroundColor": THEME.card_bg,
        "border": f"1px solid {THEME.card_border}",
        "borderRadius": "10px",
        "padding": "12px",
    },
    "card": {
        "backgroundColor": THEME.card_bg,
        "border": f"1px solid {THEME.card_border}",
        "borderRadius": "10px",
        "padding": "12px",
    },
}


PLOT_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font={"family": THEME.font_family, "color": THEME.font_color},
        paper_bgcolor=THEME.card_bg,
        plot_bgcolor=THEME.card_bg,
    ),
)


def create_app(output_dataset: OutputDataset) -> Dash:
    """Create a Dash app for visualizing observations vs simulations from an output dataset."""
    verification_pairs = output_dataset.verification_pairs
    pair_lookup = {pair.id: pair for pair in verification_pairs}

    if not verification_pairs:
        msg = "No verification pairs found in the output dataset."
        raise ValueError(msg)

    default_left_pair_id = verification_pairs[0].id
    default_right_pair_id = (
        verification_pairs[1].id if len(verification_pairs) > 1 else verification_pairs[0].id
    )

    def get_pair_controls(pair_id):
        pair = pair_lookup[pair_id]
        pair_ds = output_dataset.get(pair)
        pair_stations = [str(v) for v in pair_ds.coords["station"].values]
        pair_variables = [str(v) for v in pair_ds.coords["variable"].values]
        pair_forecast_period_values = list(pair_ds.coords["forecast_period"].values)
        pair_forecast_period_labels = [
            f"{int(v / np.timedelta64(1, 'h'))} h" for v in pair_forecast_period_values
        ]
        return pair_stations, pair_variables, pair_forecast_period_labels

    left_stations, left_variables, left_forecast_period_labels = get_pair_controls(
        default_left_pair_id,
    )
    right_stations, right_variables, right_forecast_period_labels = get_pair_controls(
        default_right_pair_id,
    )

    panel_defaults = {
        "left": {
            "default_pair_id": default_left_pair_id,
            "stations": left_stations,
            "variables": left_variables,
            "forecast_period_labels": left_forecast_period_labels,
        },
        "right": {
            "default_pair_id": default_right_pair_id,
            "stations": right_stations,
            "variables": right_variables,
            "forecast_period_labels": right_forecast_period_labels,
        },
    }

    def make_panel(panel_key, title):
        panel_config = panel_defaults[panel_key]
        return html.Div(
            [
                html.H4(title, style={"marginTop": "0", "marginBottom": "10px"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Verification Pair"),
                                dcc.Dropdown(
                                    id=f"verification-pair-dropdown-{panel_key}",
                                    options=[
                                        {
                                            "label": f"{pair.id} ({pair.obs} vs {pair.sim})",
                                            "value": pair.id,
                                        }
                                        for pair in verification_pairs
                                    ],
                                    value=panel_config["default_pair_id"],
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "280px"},
                        ),
                        html.Div(
                            [
                                html.Label("Station"),
                                dcc.Dropdown(
                                    id=f"station-dropdown-{panel_key}",
                                    options=[
                                        {"label": s, "value": s} for s in panel_config["stations"]
                                    ],
                                    value=panel_config["stations"][0],
                                    clearable=False,
                                    searchable=True,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "260px"},
                        ),
                        html.Div(
                            [
                                html.Label("Forecast Period"),
                                dcc.Dropdown(
                                    id=f"forecast-period-dropdown-{panel_key}",
                                    options=[
                                        {"label": p, "value": p}
                                        for p in panel_config["forecast_period_labels"]
                                    ],
                                    value=panel_config["forecast_period_labels"][0],
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "220px"},
                        ),
                        html.Div(
                            [
                                html.Label("Variable"),
                                dcc.Dropdown(
                                    id=f"variable-dropdown-{panel_key}",
                                    options=[
                                        {"label": v, "value": v} for v in panel_config["variables"]
                                    ],
                                    value=panel_config["variables"][0],
                                    clearable=False,
                                ),
                            ],
                            style={"flex": "1", "minWidth": "180px"},
                        ),
                    ],
                    style=STYLES["filters"],
                ),
                dcc.Graph(
                    id=f"obs-sim-scatter-{panel_key}",
                    style={"height": "620px", **STYLES["card"]},
                ),
            ],
            style={
                "flex": "1 1 560px",
                "minWidth": "460px",
                "display": "flex",
                "flexDirection": "column",
                "gap": "8px",
            },
        )

    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.H3("Observations vs Simulations Comparison"),
            html.Div(
                [
                    make_panel(
                        panel_key="left",
                        title="Left Plot",
                    ),
                    make_panel(
                        panel_key="right",
                        title="Right Plot",
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
        ],
        style=STYLES["page"],
    )

    def make_figure(
        selected_pair_id,
        selected_station,
        selected_forecast_period,
        selected_variable,
    ):
        selected_pair = pair_lookup[selected_pair_id]
        ds = output_dataset.get(selected_pair)
        pair_forecast_period_values = list(ds.coords["forecast_period"].values)
        pair_forecast_period_labels = [
            f"{int(v / np.timedelta64(1, 'h'))} h" for v in pair_forecast_period_values
        ]
        pair_forecast_period_lookup = dict(
            zip(pair_forecast_period_labels, pair_forecast_period_values, strict=True),
        )
        forecast_period = pair_forecast_period_lookup[selected_forecast_period]

        obs_selected = ds[str(selected_pair.obs)].sel(
            station=selected_station,
            variable=selected_variable,
            forecast_period=forecast_period,
            drop=True,
        )
        sim_selected = ds[str(selected_pair.sim)].sel(
            station=selected_station,
            variable=selected_variable,
            forecast_period=forecast_period,
            drop=True,
        )

        obs_values = obs_selected.values
        sim_mean = sim_selected.mean(dim="realization").values

        # Stack obs and ensemble mean into (obs, fcst) pairs
        obs_flat = np.repeat(obs_values, sim_selected.sizes["realization"])
        sim_flat = sim_selected.values.flatten()

        # Remove missing markers and non-finite values before computing axis limits.
        valid_mask = (
            (sim_flat != MISSING_VALUE_MARKER) & np.isfinite(obs_flat) & np.isfinite(sim_flat)
        )
        obs_flat = obs_flat[valid_mask]
        sim_flat = sim_flat[valid_mask]

        if obs_flat.size and sim_flat.size:
            axis_values = np.concatenate([obs_flat, sim_flat])
        else:
            fallback_values = np.concatenate([np.ravel(obs_values), np.ravel(sim_mean)])
            axis_values = fallback_values[np.isfinite(fallback_values)]

        if axis_values.size:
            axis_min = float(np.min(axis_values))
            axis_max = float(np.max(axis_values))
        else:
            axis_min, axis_max = 0.0, 1.0

        span = axis_max - axis_min
        if span == 0:
            base = max(abs(axis_min), 1.0)
            padding = base * 0.05
        else:
            padding = span * 0.05

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=obs_flat,
                y=sim_flat,
                mode="markers",
                name=f"{selected_pair.sim} ensemble members",
                marker={
                    "size": 5,
                    "color": THEME.accent_muted,
                    "symbol": "circle",
                },
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=obs_values,
                y=sim_mean,
                mode="markers",
                name=f"{selected_pair.sim} ensemble mean",
                marker={
                    "size": 7,
                    "color": THEME.accent,
                    "symbol": "circle",
                },
            ),
        )
        # 1:1 reference line
        fig.add_trace(
            go.Scatter(
                x=[axis_min - padding, axis_max + padding],
                y=[axis_min - padding, axis_max + padding],
                mode="lines",
                name="1:1 line",
                line={"color": THEME.reference, "dash": "dash", "width": 1},
                showlegend=True,
            ),
        )
        fig.update_layout(
            title=(
                f"Pair: {selected_pair.id} ({selected_pair.obs} vs {selected_pair.sim}) "
                f"| Station: {selected_station} | Variable: {selected_variable} "
                f"| Forecast period: {selected_forecast_period}"
            ),
            xaxis_title=str(selected_pair.obs),
            yaxis_title=str(selected_pair.sim),
            xaxis={"range": [axis_min - padding, axis_max + padding]},
            yaxis={
                "range": [axis_min - padding, axis_max + padding],
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            template=PLOT_TEMPLATE,
            hovermode="closest",
        )
        return fig

    def register_panel_callbacks(panel_key):
        @app.callback(
            Output(f"station-dropdown-{panel_key}", "options"),
            Output(f"station-dropdown-{panel_key}", "value"),
            Output(f"forecast-period-dropdown-{panel_key}", "options"),
            Output(f"forecast-period-dropdown-{panel_key}", "value"),
            Output(f"variable-dropdown-{panel_key}", "options"),
            Output(f"variable-dropdown-{panel_key}", "value"),
            Input(f"verification-pair-dropdown-{panel_key}", "value"),
        )
        def update_controls(selected_pair_id):
            pair_stations, pair_variables, pair_forecast_period_labels = get_pair_controls(
                selected_pair_id,
            )
            return (
                [{"label": s, "value": s} for s in pair_stations],
                pair_stations[0],
                [{"label": p, "value": p} for p in pair_forecast_period_labels],
                pair_forecast_period_labels[0],
                [{"label": v, "value": v} for v in pair_variables],
                pair_variables[0],
            )

        @app.callback(
            Output(f"obs-sim-scatter-{panel_key}", "figure"),
            Input(f"verification-pair-dropdown-{panel_key}", "value"),
            Input(f"station-dropdown-{panel_key}", "value"),
            Input(f"forecast-period-dropdown-{panel_key}", "value"),
            Input(f"variable-dropdown-{panel_key}", "value"),
        )
        def update_figure(
            selected_pair_id,
            selected_station,
            selected_forecast_period,
            selected_variable,
        ):
            return make_figure(
                selected_pair_id,
                selected_station,
                selected_forecast_period,
                selected_variable,
            )

    register_panel_callbacks("left")
    register_panel_callbacks("right")

    return app
