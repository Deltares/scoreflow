"""Reusable Plotly figures for verification results.

This module provides plain functions that build Plotly figures from pre-sliced
verification datasets. There is no interactive layer (no Dash): every selection
that used to be an interactive control - verification pair, station, lead time -
is expected to already be applied to the ``xarray.Dataset`` passed in.

Typical usage::

    ds = output_dataset.get(pair)

    # Scatter / rank histogram: slice down to a single station and lead time.
    sliced = ds.sel(station="some-station", lead_time=ds.coords["lead_time"][0])
    fig = scatter_plot(sliced, obs_var=str(pair.obs), sim_var=str(pair.sim))

    # CRPS: keep the lead_time axis, pre-select the station(s) of interest.
    crps_sliced = ds.sel(station=["station-a", "station-b"])
    fig = crps_plot(crps_sliced, score_var=find_crps_variable(crps_sliced))

    # Reanalysis / historical simulations: plot each verification pair over time.
    fig = reanalysis_timeseries_plot(output_dataset, station="some-station", error_var="mean_error")
"""

from typing import Protocol

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import xarray as xr
from plotly.subplots import make_subplots
from pydantic import BaseModel

# Serialize Plotly figures as self-contained HTML (with plotly.js loaded from the
# CDN) instead of only the ``application/vnd.plotly.v1+json`` MIME bundle. The MIME
# bundle renders in JupyterLab/VS Code but shows up as
# "Data type cannot be displayed: application/vnd.plotly.v1+json" in the static HTML
# produced by nbsphinx/nbconvert for the documentation site.
pio.renderers.default = "notebook_connected"


class VerificationPairLike(Protocol):
    """Minimal protocol for a verification pair used by the plotting functions."""

    id: str
    obs: str
    sim: str


class OutputDatasetLike(Protocol):
    """Minimal protocol for an ``OutputDataset`` used by the plotting functions."""

    @property
    def verification_pairs(self) -> list[VerificationPairLike]:
        """Return the verification pairs stored in the output dataset."""
        ...

    def get(self, verification_pair: VerificationPairLike) -> xr.Dataset:
        """Return the dataset for a given verification pair."""
        ...


class Theme(BaseModel):
    """Theme settings for figure-wide styling tokens."""

    font_family: str = "Segoe UI, Arial, sans-serif"
    font_color: str = "#022B6D"
    card_bg: str = "#FFFFFF"
    accent: str = "#7A491C"
    accent_muted: str = "#61BD4D"
    reference: str = "#818181"


THEME = Theme()
MISSING_VALUE_MARKER = -999

PLOT_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font={"family": THEME.font_family, "color": THEME.font_color},
        paper_bgcolor=THEME.card_bg,
        plot_bgcolor=THEME.card_bg,
    ),
)


def lead_time_hours(ds: xr.Dataset) -> np.ndarray:
    """Return the ``lead_time`` coordinate of ``ds`` expressed in hours."""
    values = ds.coords["lead_time"].values
    return np.array([float(v / np.timedelta64(1, "h")) for v in np.atleast_1d(values)])


def _format_lead_time_label(value: object) -> str:
    """Return a human-readable lead-time label in hours for a scalar ``lead_time`` value.

    Returns an empty string if ``value`` is not a scalar timedelta (e.g. an array left after
    slicing). Whole-hour values are rendered without a decimal part.
    """
    array = np.atleast_1d(np.asarray(value))
    if array.size != 1:
        return ""
    hours = float(array[0] / np.timedelta64(1, "h"))
    hours_text = f"{int(hours)}" if hours.is_integer() else f"{hours:g}"
    return f"lead time {hours_text} h"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a ``#RRGGBB`` string to a Plotly ``rgba(...)`` string."""
    cleaned = hex_color.lstrip("#")
    red, green, blue = (int(cleaned[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _squeeze_to_single_station(ds: xr.Dataset) -> xr.Dataset:
    """Return ``ds`` with the ``station`` dimension dropped.

    If ``station`` is not present the dataset is returned unchanged. If it is present
    with more than one entry a ``ValueError`` is raised, since the time-series plot can
    only render one station at a time.
    """
    if "station" not in ds.dims:
        return ds
    if ds.sizes["station"] != 1:
        msg = (
            "forecast_timeseries_plot expects a single station; pass station=... or "
            "pre-select one with ds.sel(station=...) before calling."
        )
        raise ValueError(msg)
    return ds.isel(station=0)


def _observed_series(obs: xr.DataArray, time_coord: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the observed array against the ``time`` coordinate into a unique series.

    The observed variable is stored on the ``(forecast_reference_time, lead_time)`` grid,
    so the same valid time appears in several forecasts. The values are flattened, sorted
    by valid time and de-duplicated to build one continuous historical series.
    """
    times = np.ravel(np.asarray(time_coord.values))
    values = np.ravel(np.asarray(obs.values, dtype=float))
    values[values == MISSING_VALUE_MARKER] = np.nan

    order = np.argsort(times)
    times = times[order]
    values = values[order]

    unique_times, first_idx = np.unique(times, return_index=True)
    return unique_times, values[first_idx]


def _as_clean_float_array(data: xr.DataArray) -> np.ndarray:
    """Return ``data`` values as floats with the configured missing marker replaced by NaN."""
    values = np.asarray(data.values, dtype=float)
    values[values == MISSING_VALUE_MARKER] = np.nan
    return values


def _iter_station_slices(
    ds: xr.Dataset,
) -> list[tuple[object | None, xr.Dataset]]:
    """Return datasets split by station, or the original dataset when stationless."""
    if "station" not in ds.dims:
        return [(None, ds)]

    return [
        (station_value, ds.sel(station=station_value))
        for station_value in np.atleast_1d(ds.coords["station"].values)
    ]


def find_crps_variable(ds: xr.Dataset, *, exclude_vars: tuple[str, ...] = ()) -> str:
    """Return the name of a CRPS-like score variable in ``ds``.

    A score variable is any data variable that has a ``lead_time`` dimension and is not
    one of ``exclude_vars`` (typically the obs/sim input variables). CRPS-like names are
    preferred; if none match, the first score candidate is returned.
    """
    excluded = set(exclude_vars)
    score_candidates = [
        str(name)
        for name, data_array in ds.data_vars.items()
        if str(name) not in excluded and "lead_time" in data_array.dims
    ]
    if not score_candidates:
        msg = "No score variable with a 'lead_time' dimension found in the dataset."
        raise ValueError(msg)
    crps_candidates = [name for name in score_candidates if "crps" in name.lower()]
    return (crps_candidates or score_candidates)[0]


def _score_variables_to_plot(
    ds: xr.Dataset,
    *,
    score_var: str | None,
    score_vars: str | list[str] | tuple[str, ...] | None,
    exclude_vars: tuple[str, ...],
) -> list[str]:
    """Return score variables requested for a lead-time score plot."""
    if score_var is not None and score_vars is not None:
        msg = "Pass either score_var or score_vars, not both."
        raise ValueError(msg)

    if score_vars is None:
        variables = [score_var or find_crps_variable(ds, exclude_vars=exclude_vars)]
    elif isinstance(score_vars, str):
        variables = [score_vars]
    else:
        variables = list(score_vars)

    if not variables:
        msg = "At least one score variable must be provided."
        raise ValueError(msg)

    missing = [variable for variable in variables if variable not in ds.data_vars]
    if missing:
        msg = (
            f"Score variable(s) {missing} not found in the dataset. "
            f"Available variables: {sorted(ds.data_vars)}."
        )
        raise ValueError(msg)

    return variables


def _subplot_axis_suffix(index: int) -> str:
    """Return the Plotly axis-id suffix for the ``index``-th subplot (1-based)."""
    return "" if index == 1 else str(index)


def _add_scatter_traces(
    fig: go.Figure,
    ds: xr.Dataset,
    obs_var: str,
    sim_var: str,
    *,
    row: int,
    col: int,
    axis_index: int,
    legend_state: dict[str, bool],
) -> None:
    """Add observation-vs-simulation scatter traces for one pair to ``fig``.

    Traces are placed on the subplot at (``row``, ``col``). ``legend_state`` shares a single
    legend entry per trace type across all subplots. The simulation variable may carry a
    ``realization`` dimension (an ensemble) or not (a deterministic/historical simulation).
    """
    ds = _squeeze_to_single_station(ds)
    obs = ds[obs_var]
    sim = ds[sim_var]
    is_ensemble = "realization" in sim.dims

    obs_values = obs.values

    if is_ensemble:
        sim_mean = sim.mean(dim="realization").values
        # Ensure realization is the last axis so the flattened simulation values line up
        # with the observations repeated per realization.
        sim_ordered = sim.transpose(*[d for d in sim.dims if d != "realization"], "realization")
        n_realization = sim_ordered.sizes["realization"]
        obs_flat = np.repeat(np.ravel(obs_values), n_realization)
        sim_flat = sim_ordered.values.flatten()
    else:
        sim_mean = sim.values
        obs_flat = np.ravel(obs_values)
        sim_flat = np.ravel(sim.values)

    valid_mask = (sim_flat != MISSING_VALUE_MARKER) & np.isfinite(obs_flat) & np.isfinite(sim_flat)
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
    padding = max(abs(axis_min), 1.0) * 0.05 if span == 0 else span * 0.05

    if is_ensemble:
        fig.add_trace(
            go.Scatter(
                x=obs_flat,
                y=sim_flat,
                mode="markers",
                name="ensemble members",
                legendgroup="members",
                marker={"size": 5, "color": THEME.accent_muted, "symbol": "circle"},
                showlegend=legend_state["members"],
            ),
            row=row,
            col=col,
        )
        legend_state["members"] = False
        fig.add_trace(
            go.Scatter(
                x=np.ravel(obs_values),
                y=np.ravel(sim_mean),
                mode="markers",
                name="ensemble mean",
                legendgroup="mean",
                marker={"size": 7, "color": THEME.accent, "symbol": "circle"},
                showlegend=legend_state["mean"],
            ),
            row=row,
            col=col,
        )
        legend_state["mean"] = False
    else:
        fig.add_trace(
            go.Scatter(
                x=np.ravel(obs_values),
                y=np.ravel(sim_mean),
                mode="markers",
                name="simulation",
                legendgroup="simulation",
                marker={"size": 6, "color": THEME.accent, "symbol": "circle"},
                showlegend=legend_state["simulation"],
            ),
            row=row,
            col=col,
        )
        legend_state["simulation"] = False

    fig.add_trace(
        go.Scatter(
            x=[axis_min - padding, axis_max + padding],
            y=[axis_min - padding, axis_max + padding],
            mode="lines",
            name="1:1 line",
            legendgroup="reference",
            line={"color": THEME.reference, "dash": "dash", "width": 1},
            showlegend=legend_state["reference"],
        ),
        row=row,
        col=col,
    )
    legend_state["reference"] = False

    axis_range = [axis_min - padding, axis_max + padding]
    fig.update_xaxes(title_text=obs_var, range=axis_range, row=row, col=col)
    fig.update_yaxes(
        title_text=sim_var,
        range=axis_range,
        scaleanchor=f"x{_subplot_axis_suffix(axis_index)}",
        scaleratio=1,
        row=row,
        col=col,
    )


def scatter_plot(
    output_dataset: OutputDatasetLike,
    *,
    lead_time: object | None = None,
    station: object | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build observation-vs-simulation scatter subplots for every verification pair.

    One subplot is drawn per verification pair in ``output_dataset``, laid out in a grid of
    at most two columns (filling left-to-right, then top-to-bottom). For each pair the
    dataset is obtained with ``output_dataset.get(pair)`` and the observed/simulated variable
    names are taken from ``pair.obs`` and ``pair.sim``.

    If ``lead_time`` is given and a dataset has a ``lead_time`` dimension, that dataset is
    sliced with ``ds.sel(lead_time=lead_time)`` before plotting. Datasets without a
    ``lead_time`` dimension (e.g. historical simulations) are left untouched, so the same
    call works for both forecast and historical setups.

    The simulated variable may carry a ``realization`` dimension (an ensemble) or not (a
    deterministic/historical simulation); the ensemble case additionally shows the ensemble
    mean. Each pair's dataset must be sliced to a single station (or have no ``station``
    dimension).
    """
    pairs = list(output_dataset.verification_pairs)
    if not pairs:
        msg = "The output dataset contains no verification pairs to plot."
        raise ValueError(msg)

    n_cols = min(2, len(pairs))
    n_rows = -(-len(pairs) // n_cols)  # ceil division

    # Pre-fetch (and lead-time slice) each dataset so subplot titles can carry the lead time.
    sliced: list[tuple[VerificationPairLike, xr.Dataset]] = []
    titles: list[str] = []
    for pair in pairs:
        ds = output_dataset.get(pair)
        title = str(pair.id)
        if lead_time is not None and "lead_time" in ds.dims:
            ds = ds.sel(lead_time=lead_time)
            label = _format_lead_time_label(ds.coords["lead_time"].values)
            if label:
                title = f"{title} ({label})"
        if station is not None:
            ds = ds.sel(station=station)
            title = f"{title} - station {station}"
        sliced.append((pair, ds))
        titles.append(title)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=min(0.12, 0.3 / n_cols),
        vertical_spacing=min(0.12, 0.6 / n_rows),
    )

    # Shared across all subplots so each trace type yields a single legend entry.
    legend_state = {
        "members": True,
        "mean": True,
        "simulation": True,
        "reference": True,
    }

    for index, (pair, ds) in enumerate(sliced):
        row = index // n_cols + 1
        col = index % n_cols + 1
        _add_scatter_traces(
            fig,
            ds,
            str(pair.obs),
            str(pair.sim),
            row=row,
            col=col,
            axis_index=index + 1,
            legend_state=legend_state,
        )

    fig.update_layout(
        template=template,
        hovermode="closest",
        height=420 * n_rows,
    )
    return fig


def crps_plot(
    output_dataset: OutputDatasetLike,
    *,
    stations: list[object] | None = None,
    score_var: str | None = None,
    score_vars: str | list[str] | tuple[str, ...] | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build a score-vs-lead-time figure with one line per verification pair.

    One line is drawn per verification pair in ``output_dataset`` in a single shared figure.
    For each pair the dataset is obtained with ``output_dataset.get(pair)`` and the score
    variable is located automatically (excluding the ``pair.obs``/``pair.sim`` input
    variables); pass ``score_var`` to use a specific variable name for every pair instead.
    Pass ``score_vars`` to plot multiple score variables, for example
    ``score_vars=["crps", "mae"]``.

    Each pair's dataset must retain its ``lead_time`` dimension. If a ``station`` dimension is
    present, one line is drawn per station (labelled ``"<pair id> - <station>"``); otherwise a
    single line labelled by the pair id is drawn. Any remaining dimensions other than
    ``lead_time`` and ``station`` are averaged out.
    """
    pairs = list(output_dataset.verification_pairs)
    if not pairs:
        msg = "The output dataset contains no verification pairs to plot."
        raise ValueError(msg)

    fig = go.Figure()
    all_hours: list[np.ndarray] = []
    plotted_variables: list[str] = []

    for pair in pairs:
        ds = output_dataset.get(pair)
        if stations is not None and "station" in ds.dims:
            ds = ds.sel(station=stations)

        variables = _score_variables_to_plot(
            ds,
            score_var=score_var,
            score_vars=score_vars,
            exclude_vars=(str(pair.obs), str(pair.sim)),
        )
        plotted_variables.extend(variables)

        for variable in variables:
            score_data = ds[variable]
            if "lead_time" not in score_data.dims:
                msg = f"Score variable '{variable}' must have a 'lead_time' dimension."
                raise ValueError(msg)

            dims_to_reduce = [
                dim_name for dim_name in score_data.dims if dim_name not in {"lead_time", "station"}
            ]
            if dims_to_reduce:
                score_data = score_data.mean(dim=dims_to_reduce, skipna=True)

            hours = lead_time_hours(ds)
            all_hours.append(hours)
            single_pair = len(pairs) == 1
            single_variable = len(variables) == 1 and (
                score_vars is None or isinstance(score_vars, str)
            )

            if "station" in score_data.dims:
                for station in score_data.coords["station"].values:
                    station_values = np.ravel(score_data.sel(station=station).values)
                    valid = np.isfinite(station_values)
                    if single_pair and single_variable:
                        name = str(station)
                    elif single_pair:
                        name = f"{variable} - {station}"
                    elif single_variable:
                        name = f"{pair.id} - {station}"
                    else:
                        name = f"{variable} - {pair.id} - {station}"
                    fig.add_trace(
                        go.Scatter(
                            x=hours[valid],
                            y=station_values[valid],
                            mode="lines+markers",
                            name=name,
                            legendgroup=variable,
                        ),
                    )
            else:
                score_values = np.ravel(score_data.values)
                valid = np.isfinite(score_values)
                name = str(pair.id) if single_variable else f"{variable} - {pair.id}"
                fig.add_trace(
                    go.Scatter(
                        x=hours[valid],
                        y=score_values[valid],
                        mode="lines+markers",
                        name=name,
                        legendgroup=variable,
                    ),
                )

    # Build a tick set covering the union of lead times across all pairs.
    tick_hours = np.unique(np.concatenate(all_hours)) if all_hours else np.array([])
    tick_labels = [f"{int(h)} h" for h in tick_hours]
    unique_variables = list(dict.fromkeys(plotted_variables))
    yaxis_title = unique_variables[0].upper() if len(unique_variables) == 1 else "Score"

    fig.update_layout(
        xaxis={
            "title": "Lead Time (h)",
            "tickmode": "array",
            "tickvals": tick_hours,
            "ticktext": tick_labels,
        },
        yaxis_title=yaxis_title,
        template=template,
        hovermode="x unified",
        title=f"CRPS vs Lead Time for {len(pairs)} Verification Pair(s)",
    )
    return fig


def rank_histogram_plot(
    output_dataset: OutputDatasetLike,
    *,
    rank_var: str = "histogram_rank",
    station: object | None = None,
    lead_time: object | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build rank-histogram subplots for every verification pair.

    One subplot is drawn per verification pair in ``output_dataset``, laid out in a grid of
    at most two columns (filling left-to-right, then top-to-bottom). For each pair the dataset
    is obtained with ``output_dataset.get(pair)`` and the ``rank_var`` variable is plotted.

    ``station`` and ``lead_time`` select the slice to plot: when given (and present on the
    dataset) the dataset is sliced with ``ds.sel(station=...)`` and/or ``ds.sel(lead_time=...)``
    so that ``rank_var`` is one-dimensional over the ``rank`` coordinate. The chosen station and
    lead time are shown in each subplot title.
    """
    pairs = list(output_dataset.verification_pairs)
    if not pairs:
        msg = "The output dataset contains no verification pairs to plot."
        raise ValueError(msg)

    n_cols = min(2, len(pairs))
    n_rows = -(-len(pairs) // n_cols)  # ceil division

    # Pre-fetch (and slice) each dataset so subplot titles can carry station/lead time.
    sliced: list[tuple[VerificationPairLike, xr.Dataset]] = []
    titles: list[str] = []
    for pair in pairs:
        ds = output_dataset.get(pair)
        if station is not None and "station" in ds.dims:
            ds = ds.sel(station=station)
        if lead_time is not None and "lead_time" in ds.dims:
            ds = ds.sel(lead_time=lead_time)

        title_parts = [str(pair.id)]
        if "station" in ds.coords:
            title_parts.append(f"station {np.atleast_1d(ds.coords['station'].values)[0]}")
        if "lead_time" in ds.coords:
            label = _format_lead_time_label(ds.coords["lead_time"].values)
            if label:
                title_parts.append(label)
        sliced.append((pair, ds))
        titles.append(", ".join(title_parts))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        horizontal_spacing=min(0.12, 0.3 / n_cols),
        vertical_spacing=min(0.2, 1.0 / n_rows),
    )

    for index, (_pair, ds) in enumerate(sliced):
        row = index // n_cols + 1
        col = index % n_cols + 1
        hist = ds[rank_var]
        fig.add_trace(
            go.Bar(
                x=hist.coords["rank"].values,
                y=np.ravel(hist.values),
                marker_color=THEME.accent,
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="Rank", row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)

    fig.update_layout(
        template=template,
        height=380 * n_rows,
        title=f"Rank Histograms for {len(pairs)} Verification Pair(s)",
    )
    return fig


def rank_histogram_3d_plot(
    output_dataset: OutputDatasetLike,
    *,
    rank_var: str = "histogram_rank",
    station: object | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Build 3D rank-histogram surfaces for every verification pair.

    This is a three-dimensional version of :func:`rank_histogram_plot`. Instead of collapsing
    the dataset to a single lead time, the ``lead_time`` axis is kept and shown as a third
    dimension: for each verification pair a 3D surface is drawn with ``rank`` on the x-axis,
    ``lead_time`` (in hours) on the y-axis, and the rank count on the z-axis. This shows all
    rank histograms across lead times in one figure, so the evolution of the histogram shape
    with increasing lead time becomes visible.

    One 3D subplot is drawn per verification pair, laid out in a grid of at most two columns
    (filling left-to-right, then top-to-bottom). ``station`` selects the station to plot when
    the dataset has a ``station`` dimension; it must reduce ``rank_var`` to two dimensions
    (``lead_time`` by ``rank``). The chosen station is shown in each subplot title.
    """
    pairs = list(output_dataset.verification_pairs)
    if not pairs:
        msg = "The output dataset contains no verification pairs to plot."
        raise ValueError(msg)

    n_cols = min(2, len(pairs))
    n_rows = -(-len(pairs) // n_cols)  # ceil division

    # Pre-fetch (and slice) each dataset so subplot titles can carry the station.
    sliced: list[tuple[VerificationPairLike, xr.Dataset]] = []
    titles: list[str] = []
    for pair in pairs:
        ds = output_dataset.get(pair)
        if station is not None and "station" in ds.dims:
            ds = ds.sel(station=station)

        title_parts = [str(pair.id)]
        if "station" in ds.coords:
            title_parts.append(f"station {np.atleast_1d(ds.coords['station'].values)[0]}")
        sliced.append((pair, ds))
        titles.append(", ".join(title_parts))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        specs=[[{"type": "surface"} for _ in range(n_cols)] for _ in range(n_rows)],
        horizontal_spacing=min(0.12, 0.3 / n_cols),
        vertical_spacing=min(0.2, 1.0 / n_rows),
    )

    for index, (_pair, ds) in enumerate(sliced):
        row = index // n_cols + 1
        col = index % n_cols + 1
        hist = ds[rank_var]
        if "lead_time" not in hist.dims or "rank" not in hist.dims:
            msg = (
                "rank_histogram_3d_plot expects the rank variable to have both 'lead_time' "
                f"and 'rank' dimensions; got dimensions {tuple(hist.dims)}."
            )
            raise ValueError(msg)
        extra_dims = set(hist.dims) - {"lead_time", "rank"}
        if extra_dims:
            msg = (
                "rank_histogram_3d_plot expects the rank variable to reduce to two dimensions "
                f"(lead_time x rank); got extra dimensions {sorted(extra_dims)}. Pass 'station' "
                "to select a single station."
            )
            raise ValueError(msg)
        # Orient as (lead_time, rank) so the surface z-grid matches y=lead_time, x=rank.
        hist = hist.transpose("lead_time", "rank")

        ranks = hist.coords["rank"].values
        leads = lead_time_hours(ds)
        fig.add_trace(
            go.Surface(
                x=ranks,
                y=leads,
                z=np.asarray(hist.values, dtype=float),
                colorscale="Earth",
                showscale=index == 0,
                colorbar={"title": "Count"} if index == 0 else None,
                hovertemplate=("Rank: %{x}<br>Lead time: %{y} h<br>Count: %{z}<extra></extra>"),
            ),
            row=row,
            col=col,
        )

    fig.update_scenes(
        xaxis_title_text="Rank",
        yaxis_title_text="Lead time (h)",
        zaxis_title_text="Count",
    )
    fig.update_layout(
        template=template,
        height=480 * n_rows,
        title=f"3D Rank Histograms for {len(pairs)} Verification Pair(s)",
    )
    return fig


def _add_reanalysis_traces(
    fig: go.Figure,
    ds: xr.Dataset,
    obs_var: str,
    sim_var: str,
    *,
    row: int,
    legend_state: dict[str, bool],
    error_var: str | None,
) -> str:
    """Add observed and simulated historical time-series traces for one pair to ``fig``."""
    if "time" not in ds.coords:
        msg = "reanalysis_timeseries_plot expects datasets with a 'time' coordinate."
        raise ValueError(msg)
    if error_var is not None and error_var not in ds.data_vars:
        msg = (
            f"Error variable '{error_var}' not found in the dataset. "
            f"Available variables: {sorted(ds.data_vars)}."
        )
        raise ValueError(msg)

    obs = ds[obs_var]
    time_values = np.ravel(ds.coords["time"].values)
    station_slices = _iter_station_slices(ds)
    multi_station = len(station_slices) > 1

    for station_value, station_ds in station_slices:
        station_label = "" if station_value is None else str(station_value)
        hover_station = "station: %{customdata}<br>" if station_value is not None else ""
        customdata = np.full(time_values.shape, station_label, dtype=object)
        obs_values = np.ravel(_as_clean_float_array(station_ds[obs_var]))
        sim_values = np.ravel(_as_clean_float_array(station_ds[sim_var]))

        obs_valid = np.isfinite(obs_values)
        fig.add_trace(
            go.Scatter(
                x=time_values[obs_valid],
                y=obs_values[obs_valid],
                customdata=customdata[obs_valid],
                mode="lines+markers",
                name="observed",
                legendgroup=f"{row}-observed",
                line={"color": THEME.font_color, "width": 1.5},
                marker={"size": 5},
                opacity=0.45 if multi_station else 1.0,
                showlegend=legend_state["observed"],
                hovertemplate=(
                    f"{hover_station}time: %{{x}}<br>{obs_var}: %{{y}}<extra>observed</extra>"
                ),
            ),
            row=row,
            col=1,
        )
        legend_state["observed"] = False

        sim_valid = np.isfinite(sim_values)
        fig.add_trace(
            go.Scatter(
                x=time_values[sim_valid],
                y=sim_values[sim_valid],
                customdata=customdata[sim_valid],
                mode="lines+markers",
                name=sim_var,
                legendgroup=f"{row}-{sim_var}",
                line={"color": THEME.accent, "width": 1.5},
                marker={"size": 5},
                opacity=0.65 if multi_station else 1.0,
                showlegend=legend_state[sim_var],
                hovertemplate=(
                    f"{hover_station}time: %{{x}}<br>{sim_var}: %{{y}}<extra>{sim_var}</extra>"
                ),
            ),
            row=row,
            col=1,
        )
        legend_state[sim_var] = False

        if error_var is None:
            continue

        error_values = np.ravel(_as_clean_float_array(station_ds[error_var]))
        error_valid = np.isfinite(error_values)
        fig.add_trace(
            go.Scatter(
                x=time_values[error_valid],
                y=error_values[error_valid],
                customdata=customdata[error_valid],
                mode="lines+markers",
                name=error_var,
                legendgroup=f"{row}-{error_var}",
                line={"color": THEME.reference, "width": 1.5, "dash": "dot"},
                marker={"size": 5, "symbol": "diamond"},
                opacity=0.7 if multi_station else 1.0,
                showlegend=legend_state[error_var],
                hovertemplate=(
                    f"{hover_station}time: %{{x}}<br>{error_var}: %{{y}}<extra>{error_var}</extra>"
                ),
            ),
            row=row,
            col=1,
            secondary_y=True,
        )
        legend_state[error_var] = False

    return str(obs.attrs.get("units", ""))


def reanalysis_timeseries_plot(
    output_dataset: OutputDatasetLike,
    *,
    station: object | None = None,
    error_var: str | None = None,
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Plot observed and simulated historical time series for every verification pair.

    This is intended for reanalysis or other historical verification runs where the pair
    datasets use a direct ``time`` dimension instead of the forecast
    ``(forecast_reference_time, lead_time)`` grid.

    One stacked subplot is drawn per verification pair in ``output_dataset``. By default,
    all stations are plotted with translucent traces. Pass a scalar station id or a list of
    station ids to pre-select the station dimension before plotting. Pass ``error_var`` to
    draw one score/error variable, such as ``"mean_error"`` or ``"mae"``, on a secondary
    y-axis.
    """
    pairs = list(output_dataset.verification_pairs)
    if not pairs:
        msg = "The output dataset contains no verification pairs to plot."
        raise ValueError(msg)

    sliced: list[tuple[VerificationPairLike, xr.Dataset]] = []
    titles: list[str] = []
    for pair in pairs:
        ds = output_dataset.get(pair)
        if station is not None:
            if "station" not in ds.dims:
                msg = "Cannot select station because this dataset has no 'station' dimension."
                raise ValueError(msg)
            ds = ds.sel(station=station)

        title_parts = [str(pair.id)]
        if "station" in ds.coords and "station" not in ds.dims:
            title_parts.append(f"station {np.asarray(ds.coords['station'].values).item()}")
        elif "station" in ds.dims and station is not None:
            title_parts.append(f"{ds.sizes['station']} stations")

        sliced.append((pair, ds))
        titles.append(", ".join(title_parts))

    fig = make_subplots(
        rows=len(pairs),
        cols=1,
        shared_xaxes=False,
        subplot_titles=titles,
        vertical_spacing=min(0.12, 1.0 / len(pairs)),
        specs=[[{"secondary_y": error_var is not None}] for _ in pairs],
    )

    legend_state = {"observed": True}
    for pair, _ds in sliced:
        legend_state[str(pair.sim)] = True
    if error_var is not None:
        legend_state[error_var] = True

    for index, (pair, ds) in enumerate(sliced, start=1):
        obs_var = str(pair.obs)
        sim_var = str(pair.sim)
        units = _add_reanalysis_traces(
            fig,
            ds,
            obs_var,
            sim_var,
            row=index,
            legend_state=legend_state,
            error_var=error_var,
        )
        y_title = f"{obs_var} / {sim_var}" + (f" ({units})" if units else "")
        fig.update_yaxes(title_text=y_title, row=index, col=1, secondary_y=False)
        if error_var is not None:
            error_units = str(ds[error_var].attrs.get("units", ""))
            error_title = error_var + (f" ({error_units})" if error_units else "")
            fig.update_yaxes(title_text=error_title, row=index, col=1, secondary_y=True)

    fig.update_xaxes(title_text="Time", row=len(pairs), col=1)
    for index in range(1, len(pairs) + 1):
        fig.update_yaxes(matches="y", row=index, col=1, secondary_y=False)
    fig.update_layout(
        template=template,
        hovermode="closest",
        height=340 * len(pairs),
    )
    return fig


def _add_forecast_traces(
    fig: go.Figure,
    ds: xr.Dataset,
    obs_var: str,
    sim_var: str,
    *,
    row: int,
    legend_state: dict[str, bool],
    show_members: bool,
    show_spread: bool,
    spread_quantiles: tuple[float, float],
) -> str:
    """Add the observed series and forecast trajectories of one pair to ``fig``.

    Traces are placed on subplot ``row`` (column 1). ``legend_state`` tracks which legend
    entries have already been shown across the whole figure so that each trace type
    contributes only a single, shared legend item. Returns the unit string of ``obs_var``
    for use in the axis title.
    """
    ds = _squeeze_to_single_station(ds)
    obs = ds[obs_var]
    sim = ds[sim_var]
    time_coord = ds.coords["time"]

    # Continuous historical observed series.
    obs_times, obs_values = _observed_series(obs, time_coord)
    obs_valid = np.isfinite(obs_values)
    fig.add_trace(
        go.Scatter(
            x=obs_times[obs_valid],
            y=obs_values[obs_valid],
            mode="lines",
            name="observed",
            legendgroup="observed",
            line={"color": THEME.font_color, "width": 2},
            showlegend=legend_state["observed"],
        ),
        row=row,
        col=1,
    )
    legend_state["observed"] = False

    is_ensemble = "realization" in sim.dims
    reference_times = np.atleast_1d(ds.coords["forecast_reference_time"].values)

    spread_label = (
        f"ensemble spread ({int(spread_quantiles[0] * 100)}-{int(spread_quantiles[1] * 100)}%)"
    )

    for reference_time in reference_times:
        sim_f = sim.sel(forecast_reference_time=reference_time)
        valid_time = np.ravel(time_coord.sel(forecast_reference_time=reference_time).values)

        if is_ensemble:
            sim_ordered = sim_f.transpose(
                *[dim for dim in sim_f.dims if dim != "realization"],
                "realization",
            )
            members = np.asarray(sim_ordered.values, dtype=float).reshape(len(valid_time), -1)
            members[members == MISSING_VALUE_MARKER] = np.nan

            if show_members:
                for column in range(members.shape[1]):
                    member = members[:, column]
                    finite = np.isfinite(member)
                    fig.add_trace(
                        go.Scatter(
                            x=valid_time[finite],
                            y=member[finite],
                            mode="lines",
                            line={"color": THEME.accent_muted, "width": 1},
                            opacity=0.25,
                            legendgroup="members",
                            name="ensemble members",
                            showlegend=legend_state["member"],
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=1,
                    )
                    legend_state["member"] = False

            if show_spread:
                lower = np.nanquantile(members, spread_quantiles[0], axis=1)
                upper = np.nanquantile(members, spread_quantiles[1], axis=1)
                finite = np.isfinite(lower) & np.isfinite(upper)
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([valid_time[finite], valid_time[finite][::-1]]),
                        y=np.concatenate([upper[finite], lower[finite][::-1]]),
                        fill="toself",
                        fillcolor=_hex_to_rgba(THEME.accent, 0.15),
                        line={"color": "rgba(0, 0, 0, 0)"},
                        legendgroup="spread",
                        name=spread_label,
                        showlegend=legend_state["spread"],
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=1,
                )
                legend_state["spread"] = False

            mean = np.nanmean(members, axis=1)
            finite = np.isfinite(mean)
            fig.add_trace(
                go.Scatter(
                    x=valid_time[finite],
                    y=mean[finite],
                    mode="lines",
                    line={"color": THEME.accent, "width": 1.5},
                    legendgroup="mean",
                    name="ensemble mean forecast",
                    showlegend=legend_state["mean"],
                ),
                row=row,
                col=1,
            )
            legend_state["mean"] = False
        else:
            values = np.asarray(np.ravel(sim_f.values), dtype=float)
            values[values == MISSING_VALUE_MARKER] = np.nan
            finite = np.isfinite(values)
            fig.add_trace(
                go.Scatter(
                    x=valid_time[finite],
                    y=values[finite],
                    mode="lines",
                    line={"color": THEME.accent, "width": 1},
                    legendgroup="forecast",
                    name="forecast",
                    showlegend=legend_state["forecast"],
                ),
                row=row,
                col=1,
            )
            legend_state["forecast"] = False

    return str(obs.attrs.get("units", ""))


def forecast_timeseries_plot(
    output_dataset: OutputDatasetLike,
    *,
    station: object | None = None,
    show_members: bool = False,
    show_spread: bool = True,
    spread_quantiles: tuple[float, float] = (0.1, 0.9),
    template: go.layout.Template = PLOT_TEMPLATE,
) -> go.Figure:
    """Plot historical observed series and forecast trajectories for every pair.

    One stacked subplot is drawn per verification pair in ``output_dataset``. For each pair
    the dataset is obtained with ``output_dataset.get(pair)`` and the observed/simulated
    variable names are taken from ``pair.obs`` and ``pair.sim``.

    Both variables are stored on the ``(forecast_reference_time, lead_time)`` grid and share
    an auxiliary ``time(forecast_reference_time, lead_time)`` coordinate giving each value's
    valid time. The observed values are flattened against ``time`` into one continuous line,
    while each ``forecast_reference_time`` produces one forecast trajectory drawn along its
    valid times.

    The simulated variable may carry a ``realization`` dimension (an ensemble) or not
    (deterministic):

    - Ensemble: the ensemble mean is drawn per forecast; with ``show_spread`` a shaded band
      between ``spread_quantiles`` is added, and with ``show_members`` every member line is
      drawn faintly.
    - Deterministic: a single line is drawn per forecast.

    Pass ``station`` to select one station from multi-station datasets. If ``station`` is not
    given, each pair's dataset must already be sliced to a single station (or have no
    ``station`` dimension).
    """
    pairs = list(output_dataset.verification_pairs)
    if not pairs:
        msg = "The output dataset contains no verification pairs to plot."
        raise ValueError(msg)

    sliced: list[tuple[VerificationPairLike, xr.Dataset]] = []
    titles: list[str] = []
    for pair in pairs:
        ds = output_dataset.get(pair)
        if station is not None:
            if "station" not in ds.dims:
                msg = "Cannot select station because this dataset has no 'station' dimension."
                raise ValueError(msg)
            ds = ds.sel(station=station)

        title_parts = [str(pair.id)]
        if "station" in ds.coords and "station" not in ds.dims:
            title_parts.append(f"station {np.asarray(ds.coords['station'].values).item()}")

        sliced.append((pair, ds))
        titles.append(", ".join(title_parts))

    fig = make_subplots(
        rows=len(pairs),
        cols=1,
        shared_xaxes=False,
        subplot_titles=titles,
        vertical_spacing=min(0.12, 1.0 / len(pairs)),
    )

    # Shared across all subplots so each trace type yields a single legend entry.
    legend_state = {
        "observed": True,
        "member": True,
        "spread": True,
        "mean": True,
        "forecast": True,
    }

    for index, (pair, ds) in enumerate(sliced, start=1):
        obs_var = str(pair.obs)
        sim_var = str(pair.sim)
        units = _add_forecast_traces(
            fig,
            ds,
            obs_var,
            sim_var,
            row=index,
            legend_state=legend_state,
            show_members=show_members,
            show_spread=show_spread,
            spread_quantiles=spread_quantiles,
        )
        y_title = f"{obs_var} / {sim_var}" + (f" ({units})" if units else "")
        fig.update_yaxes(title_text=y_title, row=index, col=1)

    # Only label the x-axis of the bottom subplot so the "Time" label does not collide
    # with the title of the subplot below it.
    fig.update_xaxes(title_text="Time", row=len(pairs), col=1)

    # Share a single y-axis scale across all subplots so values are directly comparable.
    fig.update_yaxes(matches="y")

    fig.update_layout(
        template=template,
        hovermode="x unified",
        height=340 * len(pairs),
    )
    return fig
