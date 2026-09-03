=========================
Datamodel
=========================

.. autosummary::
   :toctree: _generated

   veriflow.datamodel.input
   veriflow.datamodel.output

VeriflowDataTree
================

``veriflow.datamodel.output.VeriflowDataTree`` is a typing-only helper, not a real runtime
class. It exists so that static type checkers (mypy, pyright) know about the ``veriflow``
accessor added to ``xr.DataTree`` by :class:`~veriflow.datamodel.output.VeriflowAccessor`,
without veriflow having to subclass or monkeypatch xarray's actual ``DataTree`` type.

At runtime, ``VeriflowDataTree`` is simply an alias for ``xr.DataTree`` (there is no separate
class, and no ``isinstance`` distinction). When you receive or construct a plain ``xr.DataTree``
and want typed access to ``.veriflow``, cast it once:

.. code-block:: python

    from typing import cast
    from veriflow.datamodel.output import VeriflowDataTree

    dt = cast("VeriflowDataTree", xr.DataTree(name="veriflow_output"))
    dt.veriflow.add_input_data(...)  # now statically typed

See :class:`~veriflow.datamodel.output.VeriflowAccessor` for the actual runtime API surface
exposed through ``.veriflow``.
