# Gallery

Explore _veriflow_ through hands-on, case-based tutorials. Each case bundles a
set of Jupyter notebooks that walk through a complete verification workflow, from
accessing data to computing scores and interpreting the results.

Every notebook can be **downloaded** or **run live in Binder** — look for the
banner at the top of each notebook page.

::::{grid} 1 2 2 2
:gutter: 4
:class-container: veriflow-gallery

:::{grid-item-card} The Delft-FEWS (OpenFEWS) Case
:img-top: _static/gallery/openfews.png
:link: gallery/delft_fews
:link-type: doc
:class-card: sd-shadow-md

A step-by-step tutorial built around a real flood event in the Elbow Watershed
(Alberta, Canada). Verify precipitation reanalyses, deterministic and ensemble
forecasts, and Raven discharge forecasts — all accessed through the Delft-FEWS PI
Webservice.

+++
7 notebooks · beginner friendly
:::

:::{grid-item-card} The Rhine Case
:img-top: _static/gallery/rhine.jpg
:link: gallery/rhine
:link-type: doc
:class-card: sd-shadow-md

A compact, end-to-end example comparing multiple models for the Rhine by running
one integrated verification pipeline. Learn the core _veriflow_ workflow and how
to interpret the resulting verification metrics.

+++
2 notebooks · core workflow
:::

::::

```{toctree}
:hidden:

gallery/delft_fews
gallery/rhine
```
