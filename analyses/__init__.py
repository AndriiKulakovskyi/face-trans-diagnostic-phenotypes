"""Sensitivity / exploration arms that consume the core ``face`` package but are not part of it.

Each subpackage backs a published supplementary claim and is physically separated from the core
M1->M5 pipeline (own drivers, own results under ``results/analyses/<arm>/``, own findings doc under
``docs/sensitivity/``). Kept off the ``face`` CLI so the core surface is exactly the paper's main pipeline.

  * ``variational_gllvm`` — PyTorch SVI re-estimation of M1 (the torch arm; docs/sensitivity/variational_gllvm.md).
"""
