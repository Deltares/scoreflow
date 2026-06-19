(gallery-delft-fews)=
# The Delft-FEWS (OpenFEWS) Case

This tutorial follows a recent (June 2026) high-flow event in the **Elbow
Watershed** in Alberta, Canada — a catchment whose floods caused catastrophic
damage to the City of Calgary in 2013. Across the notebooks below you will verify
precipitation reanalyses, deterministic and ensemble forecasts, and Raven
discharge forecasts, all accessed through the **Delft-FEWS PI Webservice**.

Work through the notebooks in order, or jump straight to a topic. Each notebook
page includes a banner to **download** it or **open it in Binder**.

::::{grid} 1 2 2 3
:gutter: 3
:class-container: veriflow-gallery

:::{grid-item-card} 00 · Introduction
:img-top: ../_static/gallery/openfews.png
:link: ../examples/The%20Delft-FEWS%20(OpenFEWS)%20Case/00_introduction.html
:link-type: url

Meet the Elbow Watershed event, OpenFEWS hindcasts, and how _veriflow_ integrates
with Delft-FEWS.
:::

:::{grid-item-card} 01 · Precipitation Reanalysis (RDPA, HRDPA)
:img-top: ../_static/gallery/elbowrainfall.png
:link: ../examples/The%20Delft-FEWS%20(OpenFEWS)%20Case/01_elbow_precipitation_analysis.html
:link-type: url

Compare observed precipitation with the RDPA and HRDPA reanalyses and visualize
simple error metrics.
:::

:::{grid-item-card} 02 · Precipitation Deterministic Forecasts (GDPS)
:img-top: ../_static/gallery/elbowwatershed.png
:link: ../examples/The%20Delft-FEWS%20(OpenFEWS)%20Case/02_elbow_deterministic_forecast.html
:link-type: url

Verify deterministic precipitation forecasts (GDPS) for the Elbow Watershed.
:::

:::{grid-item-card} 03 · Precipitation Ensemble Forecasts (GEPS, GEFS, IFS)
:img-top: ../_static/gallery/elbowforecastens.png
:link: ../examples/The%20Delft-FEWS%20(OpenFEWS)%20Case/03_elbow_precipitation_ensemble.html
:link-type: url

Move from deterministic to ensemble precipitation forecasts and verify forecast
uncertainty.
:::

:::{grid-item-card} 04 · Raven Discharge Ensemble Forecasts (GEPS, GEFS, IFS)
:img-top: ../_static/gallery/ravenforecast.png
:link: ../examples/The%20Delft-FEWS%20(OpenFEWS)%20Case/04_raven_elbow_discharge_ensemble.html
:link-type: url

Compare ensemble discharge forecasts and evaluate probabilistic skill.
:::

::::

```{toctree}
:hidden:

../examples/The Delft-FEWS (OpenFEWS) Case/00_introduction
../examples/The Delft-FEWS (OpenFEWS) Case/01_elbow_precipitation_analysis
../examples/The Delft-FEWS (OpenFEWS) Case/02_elbow_deterministic_forecast
../examples/The Delft-FEWS (OpenFEWS) Case/03_elbow_precipitation_ensemble
../examples/The Delft-FEWS (OpenFEWS) Case/04_raven_elbow_discharge_ensemble
```
