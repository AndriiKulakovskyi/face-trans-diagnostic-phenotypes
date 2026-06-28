"""Variational mixed-likelihood GLLVM atlas engine.

A parallel, SVI-trained re-estimation of the FACE M1 measurement model that keeps the same
scientific contract (ontology-constrained linear decoder, positive home loadings, hard-zero
forbidden cells, mixed per-item likelihoods, observed-cell likelihood, patient coordinates
with uncertainty) but replaces NUTS with stochastic variational optimization.  It is an
exploration / acceleration arm calibrated against — never replacing — the NUTS authority.

See ``docs/VGLLVM_MODEL.md``.
"""
from face.models.variational.generative import (
    correlation_block,
    generate_synthetic,
    marginal_summary,
)
from face.models.variational.gllvm import (
    CorrelationPrior,
    GLLVMTrainer,
    LoadingOntology,
    TrainingConfig,
    VariationalGLLVM,
    synthetic_gllvm_dataset,
)
from face.models.variational.gllvm_model_oop import (
    F8_FIT,
    MODEL_VERSION,
    GLLVMConfig,
    GLLVMData,
    GLLVMDataset,
    GLLVMProjector,
    GLLVMRunner,
    GLLVMStage,
    GLLVMVisualizer,
)

__all__ = [
    "CorrelationPrior",
    "GLLVMTrainer",
    "LoadingOntology",
    "TrainingConfig",
    "VariationalGLLVM",
    "synthetic_gllvm_dataset",
    "generate_synthetic",
    "marginal_summary",
    "correlation_block",
    "F8_FIT",
    "MODEL_VERSION",
    "GLLVMConfig",
    "GLLVMData",
    "GLLVMDataset",
    "GLLVMProjector",
    "GLLVMRunner",
    "GLLVMStage",
    "GLLVMVisualizer",
]
