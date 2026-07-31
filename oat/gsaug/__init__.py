"""Compositional 3D Gaussian Splatting render source for the SE(2) augmentation.

Phase 0 (see IMPLEMENTATION_PLAN_gs_render_phase0.md): replace the MuJoCo oracle
re-render pixel source with a GS composite renderer built from sim-captured
assets. Everything downstream of pixels (labels, normalization, validity,
matched budget) is reused unchanged.

Import discipline: this package is imported by the pre-render script and by
probes; keep robosuite/mujoco imports out of module scope wherever possible
(`cameras`, `sh_rotation`, `gaussian_asset`, `components` are sim-free;
`capture`/`compose` touch the sim only through duck-typed env handles).
"""
