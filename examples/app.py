import numpy as np
import plotly.graph_objects as go
import xarray as xr
from dash import Dash, Input, Output, dcc, html
from pydantic import BaseModel


class Theme(BaseModel):
    """Theme settings for app-wide styling tokens."""

    font_family: str = "Segoe UI, Arial, sans-serif"
    font_color: str = "#022B6D"
    page_bg: str = "#F8FAFC"
    card_bg: str = "#FFFFFF"
    card_border: str = "#E2E8F0"
    accent: str = "#7A491CA9"
    accent_muted: str = "#6DBD5BAC"
    reference: str = "#818181"


THEME = Theme()

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


def create_app(ds: xr.Dataset) -> Dash:
    """Create a Dash app for visualizing the observations vs simulations for a given dataset."""
    stations = [str(v) for v in ds.coords["station"].values]
    variables = [str(v) for v in ds.coords["variable"].values]
    forecast_period_values = list(ds.coords["forecast_period"].values)
    forecast_period_labels = [
        f"{int(v / np.timedelta64(1, 'h'))} h" for v in forecast_period_values
    ]
    forecast_period_lookup = dict(zip(forecast_period_labels, forecast_period_values))

    app = Dash(__name__)

    app.layout = html.Div(
        [
            html.H3("Observations vs Simulations"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Station"),
                            dcc.Dropdown(
                                id="station-dropdown",
                                options=[{"label": s, "value": s} for s in stations],
                                value=stations[0],
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
                                id="forecast-period-dropdown",
                                options=[{"label": p, "value": p} for p in forecast_period_labels],
                                value=forecast_period_labels[0],
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "220px"},
                    ),
                    html.Div(
                        [
                            html.Label("Variable"),
                            dcc.Dropdown(
                                id="variable-dropdown",
                                options=[{"label": v, "value": v} for v in variables],
                                value=variables[0],
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "180px"},
                    ),
                ],
                style=STYLES["filters"],
            ),
            dcc.Graph(id="obs-sim-scatter", style={"height": "620px", **STYLES["card"]}),
        ],
        style=STYLES["page"],
    )

    @app.callback(
        Output("obs-sim-scatter", "figure"),
        Input("station-dropdown", "value"),
        Input("forecast-period-dropdown", "value"),
        Input("variable-dropdown", "value"),
    )
    def update_figure(selected_station, selected_forecast_period, selected_variable):
        forecast_period = forecast_period_lookup[selected_forecast_period]

        obs_selected = ds["obs"].sel(
            station=selected_station,
            variable=selected_variable,
            forecast_period=forecast_period,
            drop=True,
        )
        raw_selected = ds["raw"].sel(
            station=selected_station,
            variable=selected_variable,
            forecast_period=forecast_period,
            drop=True,
        )

        obs_values = obs_selected.values
        raw_mean = raw_selected.mean(dim="realization").values

        # Stack obs and ensemble mean into (obs, fcst) pairs
        obs_flat = np.repeat(obs_values, raw_selected.sizes["realization"])
        raw_flat = raw_selected.values.flatten()

        # Ensemble mean pairs sometimes are -999, which is a common missing value indicator. Remove these pairs from the plot.
        valid_mask = raw_flat != -999
        obs_flat = obs_flat[valid_mask]
        raw_flat = raw_flat[valid_mask]

        axis_min = min(obs_flat.min(), raw_flat.min())
        axis_max = max(obs_flat.max(), raw_flat.max())
        padding = (axis_max - axis_min) * 0.05

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=obs_flat,
                y=raw_flat,
                mode="markers",
                name="Ensemble members",
                marker={"size": 5, "color": THEME.accent_muted, "symbol": "circle"},
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=obs_values,
                y=raw_mean,
                mode="markers",
                name="Ensemble mean",
                marker={"size": 7, "color": THEME.accent, "symbol": "diamond"},
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
                f"Station: {selected_station} | Variable: {selected_variable} "
                f"| Forecast period: {selected_forecast_period}"
            ),
            xaxis_title="Observed",
            yaxis_title="Simulated",
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

    return app
