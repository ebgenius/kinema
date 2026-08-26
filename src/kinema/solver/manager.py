"""Per-rig solver state: pick a backend, keep it warm, run a solve.

One :class:`RigSolver` is cached per armature. Building one is not free -- it
extracts the kinematic chain, and for PyRoki it reloads the URDF and pays a JIT
compile -- so the cache is what makes live viewport IK viable at all.

Backend selection is deliberately forgiving. PyRoki is preferred, but a rig
whose description cannot be found, or a session where JAX failed to import,
silently falls back to the NumPy backend rather than refusing to solve. The
panel says which backend actually answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..rig import builder
from . import chain as chain_mod
from . import numpy_backend, pyroki_backend
from .base import SolverError, SolveResult

MODE_PYROKI = "PYROKI"
MODE_NUMPY = "NUMPY"
MODE_OFF = "OFF"


def _np4(matrix) -> np.ndarray:
    return np.array([[matrix[r][c] for c in range(4)] for r in range(4)])


@dataclass
class RigSolver:
    """Everything needed to run IK on one rig."""

    rig_name: str
    chain: chain_mod.Chain
    ik_bone: str
    tcp_bone: str
    #: bone-space goal -> URDF link-space goal, and the link PyRoki should hit.
    link_target: tuple[str, np.ndarray] | None = None
    _pyroki: pyroki_backend.PyrokiSolver | None = None
    _pyroki_failed: str | None = None
    #: chain index -> index into PyRoki's full actuated vector.
    _chain_to_full: np.ndarray | None = None
    last_result: SolveResult | None = None
    #: How many solves this rig has run. The first PyRoki solve includes
    #: JIT compilation, so callers time it differently.
    solve_count: int = 0
    _warned: set = field(default_factory=set)

    # ---------------------------------------------------------------- PyRoki
    def pyroki(self, rig) -> pyroki_backend.PyrokiSolver | None:
        """Build the PyRoki solver on first use, or explain why it is absent."""
        if self._pyroki is not None or self._pyroki_failed:
            return self._pyroki
        try:
            self._pyroki = self._build_pyroki(rig)
        except Exception as exc:  # noqa: BLE001 - always falls back to NumPy
            self._pyroki_failed = str(exc)
            return None
        return self._pyroki

    def _build_pyroki(self, rig) -> pyroki_backend.PyrokiSolver:
        if self.link_target is None:
            raise SolverError("this rig has no URDF link recorded for its TCP")
        urdf = _load_source_urdf(rig)
        if urdf is None:
            raise SolverError("the robot description for this rig is not available")

        link_name, _ = self.link_target
        solver = pyroki_backend.build(urdf, link_name)

        # Map our chain's joints onto PyRoki's full actuated vector by name.
        order = {name: index for index, name in enumerate(solver.actuated_names)}
        missing = [n for n in self.chain.bone_names if n not in order]
        if missing:
            raise SolverError(f"joints not found in the description: {missing[:3]}")
        self._chain_to_full = np.array([order[n] for n in self.chain.bone_names])
        return solver

    @property
    def pyroki_error(self) -> str | None:
        return self._pyroki_failed

    # ----------------------------------------------------------------- solve
    def solve(self, rig, mode: str = MODE_PYROKI) -> SolveResult | None:
        """Solve for the current IK target pose and write the result to the rig.

        Returns None when there is nothing to do (no IK bone, or mode OFF).
        """
        if mode == MODE_OFF:
            return None
        pose = rig.pose
        if self.ik_bone not in pose.bones:
            return None

        goal = _np4(pose.bones[self.ik_bone].matrix)
        seed = chain_mod.read_configuration(rig, self.chain)

        result = None
        if mode == MODE_PYROKI:
            solver = self.pyroki(rig)
            if solver is not None:
                result = self._solve_pyroki(solver, seed, goal)

        if result is None:
            result = numpy_backend.solve(self.chain, seed, goal)

        chain_mod.write_configuration(rig, self.chain, result.q)
        self.last_result = result
        self.solve_count += 1
        return result

    def _solve_pyroki(self, solver, seed: np.ndarray, goal: np.ndarray) -> SolveResult | None:
        try:
            # PyRoki targets a URDF link; the IK bone expresses a *bone* goal,
            # so apply the fixed bone->link correction recorded at build time.
            _, correction = self.link_target
            link_goal = goal @ correction

            full_seed = np.zeros(solver.dof)
            full_seed[self._chain_to_full] = seed
            full = solver.solve(full_seed, link_goal)

            result = pyroki_backend.measure(solver, full, link_goal)
            # Write back only the joints on our chain; anything else PyRoki
            # moved (a gripper finger, say) does not affect the tool.
            result.q = full[self._chain_to_full]
            return result
        except Exception as exc:  # noqa: BLE001
            self._pyroki_failed = str(exc)
            return None


# --------------------------------------------------------------------------
# construction and cache
# --------------------------------------------------------------------------
_cache: dict[str, RigSolver] = {}


def _load_source_urdf(rig):
    """Reload the description this rig was built from, if we still can."""
    kind = rig.get(builder.PROP_SOURCE_KIND)
    source = rig.get(builder.PROP_SOURCE)
    if not kind or not source:
        return None
    try:
        if kind == "catalog":
            from ..catalog.index import load_urdf

            return load_urdf(source)
        import os

        if not os.path.isfile(source):
            return None
        import yourdfpy

        from ..io.resolve import make_mesh_resolver

        resolver = make_mesh_resolver(source)
        return yourdfpy.URDF.load(
            source, build_scene_graph=False, load_meshes=False,
            filename_handler=lambda name: resolver(name),
        )
    except Exception:  # noqa: BLE001
        return None


def _link_target_for(rig, tcp_bone_name: str) -> tuple[str, np.ndarray] | None:
    """Work out which URDF link the TCP rides, and the bone->link correction.

    The TCP bone hangs off a joint bone. That joint's child link is what PyRoki
    should aim at, and the correction chains the TCP bone's offset from the
    joint bone onto the joint bone's own bone->link correction::

        link_goal = tcp_goal · M_tcp⁻¹ · M_joint · C_joint
    """
    bones = rig.data.bones
    tcp = bones.get(tcp_bone_name)
    if tcp is None:
        return None

    node = tcp.parent
    while node is not None and builder.PROP_CHILD_LINK not in node:
        node = node.parent
    if node is None:
        return None

    stored = node.get(builder.PROP_LINK_CORRECTION)
    if stored is None or len(stored) != 16:
        return None
    joint_correction = np.array([float(v) for v in stored]).reshape(4, 4)

    correction = (
        np.linalg.inv(_np4(tcp.matrix_local)) @ _np4(node.matrix_local) @ joint_correction
    )
    return str(node[builder.PROP_CHILD_LINK]), correction


def build_solver(rig, ik_bone: str, tcp_bone: str | None = None) -> RigSolver | None:
    """Create the solver state for one rig, or None if it cannot be rigged."""
    tcp_bone = tcp_bone or rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE
    chain = chain_mod.chain_from_rig(rig, tcp_bone)
    if chain is None:
        return None
    return RigSolver(
        rig_name=rig.name,
        chain=chain,
        ik_bone=ik_bone,
        tcp_bone=tcp_bone,
        link_target=_link_target_for(rig, tcp_bone),
    )


def get_solver(rig, ik_bone: str | None = None) -> RigSolver | None:
    """Fetch (or build) the cached solver for ``rig``."""
    ik_bone = ik_bone or rig.get(builder.PROP_IK_BONE)
    if not ik_bone:
        return None

    cached = _cache.get(rig.name)
    if cached is not None and cached.ik_bone == ik_bone and cached.tcp_bone in rig.pose.bones:
        return cached
    # Anything that changes the rig's bones -- rebuilding it, or moving the
    # TCP -- calls invalidate(), so a stale entry here is not silently reused.

    solver = build_solver(rig, ik_bone)
    if solver is not None:
        _cache[rig.name] = solver
    return solver


def invalidate(rig_name: str | None = None) -> None:
    """Drop cached solvers -- after a rig rebuild, or on unregister."""
    if rig_name is None:
        _cache.clear()
    else:
        _cache.pop(rig_name, None)
