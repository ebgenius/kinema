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

from collections import OrderedDict
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
    #: The bone the solver aims at. Usually the TCP marker, but any joint bone
    #: can be the tip -- see :func:`tip_bone`.
    tip_bone: str
    #: Which rig this was built for, checked on every cache hit -- a name alone
    #: is handed back to the next object to take it. See :func:`rig_identity`.
    identity: int | str = ""
    #: bone-space goal -> URDF link-space goal, and the link PyRoki should hit.
    link_target: tuple[str, np.ndarray] | None = None
    #: Deliberately *not* a field. The compiled solver is owned by
    #: ``_pyroki_cache`` and nothing else, so its size limit is the real bound
    #: on how many JAX kernels can be alive at once. A reference held here too
    #: would keep a kernel reachable for as long as this RigSolver sat in
    #: ``_cache`` -- which is per rig name, indefinitely -- and the limit would
    #: bound nothing.
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
        """Fetch the PyRoki solver from the cache, building it on a miss.

        Looked up every time rather than held on the instance, so the cache is
        the only strong owner of a compiled kernel and its size limit actually
        bounds memory. The lookup is a dict hit; the rebuild it risks is the
        price of that bound, and with a limit of four it does not come up for
        the case this cache exists for.
        """
        if self._pyroki_failed:
            return None
        cached = _pyroki_cache_get(self.identity, self.link_target)
        # Length check, not trust: the cached mapping was derived from whatever
        # chain reached this link first. Today that is always this same chain,
        # but a mapping of the wrong width would silently scatter joint values
        # into the wrong slots, and that is not a failure worth risking to save
        # a rebuild.
        if cached is not None and len(cached[1]) == self.chain.dof:
            self._chain_to_full = cached[1]
            return cached[0]
        try:
            solver = self._build_pyroki(rig)
        except Exception as exc:  # noqa: BLE001 - always falls back to NumPy
            self._pyroki_failed = str(exc)
            return None
        _pyroki_cache_put(
            self.identity, self.rig_name, self.link_target,
            solver, self._chain_to_full,
        )
        return solver

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

def rig_identity(rig) -> int | str:
    """Something that identifies this rig and is never reused for another.

    Not the name. Blender hands a deleted object's name straight back to the
    next import, so a new robot can arrive as "arm6" holding the previous
    "arm6"'s cache entries -- and if it happens to share a link name and joint
    count, a compiled solver built for a model that no longer exists would be
    accepted and would silently solve the wrong robot. ``session_uid`` is unique
    for the life of the session; the name is only a fallback for a build that
    does not expose it.
    """
    return getattr(rig, "session_uid", None) or rig.name


#: (rig identity, URDF link) -> a built PyRoki solver, its chain->full mapping,
#: and the rig name, which is kept only so invalidate() can purge by name.
#: Separate from ``_cache`` because moving the IK tip throws the RigSolver away
#: but not the thing that was expensive to make: building a PyRoki solver
#: reloads the description and pays a JAX compile, tens of seconds the first
#: time. Keyed by link so switching the tip back and forth -- which a keyframed
#: tip does on every scrub -- costs nothing after the first visit to each.
_pyroki_cache: OrderedDict[
    tuple[int | str, str], tuple[pyroki_backend.PyrokiSolver, np.ndarray, str]
] = OrderedDict()

#: How many compiled solvers to keep, least-recently-used evicted first.
#:
#: Small on purpose. Each entry pins a JAX-compiled kernel, which is tens of
#: megabytes, and nothing else would ever drop them: rigs are keyed by name, so
#: deleting a rig or opening a new file leaves entries behind that no longer
#: describe anything. An unbounded cache here grew until the machine ran out of
#: memory. Four covers the case this exists for -- an animator flipping between
#: a couple of tips -- and anything beyond that pays one rebuild.
_PYROKI_CACHE_LIMIT = 4


def _pyroki_cache_get(identity, link_target) -> tuple | None:
    if link_target is None:
        return None
    key = (identity, link_target[0])
    entry = _pyroki_cache.get(key)
    if entry is not None:
        _pyroki_cache.move_to_end(key)
    return entry


def _pyroki_cache_put(identity, rig_name: str, link_target, solver, chain_to_full) -> None:
    if link_target is None or solver is None or chain_to_full is None:
        return
    key = (identity, link_target[0])
    _pyroki_cache[key] = (solver, chain_to_full, rig_name)
    _pyroki_cache.move_to_end(key)
    while len(_pyroki_cache) > _PYROKI_CACHE_LIMIT:
        _pyroki_cache.popitem(last=False)


def tip_bone_for(rig, index: int) -> str:
    """Which bone ``index`` names, without reading or writing the rig's own tip.

    Split out from :func:`tip_bone` so a caller can find out where a tip *would*
    land before committing to it. Writing ``kinema_ik_tip`` is not inert: it
    invalidates the cached goal, and the next depsgraph update then solves the
    new chain. Anything that needs the new tip's *current* pose has to ask
    before that happens.
    """
    if index >= 0:
        joints = builder.joint_bones(rig)
        if index < len(joints):
            return joints[index].name
    return rig.get(builder.PROP_TCP_BONE) or builder.TCP_BONE


def tip_bone(rig) -> str:
    """The bone the solver aims at.

    ``kinema_ik_tip`` is an index into the rig's joint bones, and it is a real
    RNA property rather than a bone reference precisely so that it can be
    keyframed: an animator can hand the goal from the wrist to the elbow
    mid-shot. Out of range -- including the -1 default -- means "use the TCP
    marker", which is the behaviour every rig had before the property existed.
    """
    return tip_bone_for(rig, getattr(rig, "kinema_ik_tip", -1))


def _load_source_urdf(rig):
    """Reload the description this rig was built from, if we still can.

    Raises SolverError with a readable reason; the panel shows it. Returns None
    only when the rig records no source at all.
    """
    kind = rig.get(builder.PROP_SOURCE_KIND)
    source = rig.get(builder.PROP_SOURCE)
    if not kind or not source:
        return None

    if kind in ("catalog", "catalog-mjcf"):
        # Rigs built by Kinema 0.2.0 and earlier, when the catalog downloaded
        # descriptions itself. The file may well still be in the old cache, but
        # nothing records where, so there is nothing honest to do but say so.
        raise SolverError(
            f"'{source}' came from the old downloading catalog. Re-import the "
            "description from disk to restore the PyRoki solver"
        )

    try:
        if kind == "mjcf":
            # PyRoki only reads URDF, so re-parse the MJCF and render the
            # kinematic tree back out as one. Without this every MJCF rig would
            # be stuck on the NumPy fallback.
            from ..io.mjcf import model_from_mjcf
            from .urdf_bridge import urdf_from_model

            return urdf_from_model(model_from_mjcf(source))

        import os
        from pathlib import Path

        if not os.path.isfile(source):
            raise SolverError(f"the description file is missing: {source}")

        import yourdfpy

        from ..io.loader import load_xacro_urdf, looks_like_xacro
        from ..io.resolve import make_mesh_resolver
        from ..prefs import package_search_paths

        path = Path(source)
        resolver = make_mesh_resolver(path, extra_search_paths=package_search_paths())

        if looks_like_xacro(path):
            # A xacro has to be rendered before yourdfpy sees it, and this path
            # used to hand it over raw -- so `$(arg …)` reached a float parser
            # and every xacro rig fell back to the NumPy solver with "could not
            # convert string to float: '$(arg'". Silently, because falling back
            # is what the manager does with any reload failure, and 38 of the
            # catalogue's robots ship only a xacro.
            #
            # The arguments matter here for the same reason: a description that
            # needs `name:=ur5e` needs it now as much as it did at import, and
            # the same silence would swallow the failure.
            from ..io.xacro_args import parse_args

            return load_xacro_urdf(
                path,
                resolver,
                xacro_args=parse_args(rig.get(builder.PROP_XACRO_ARGS, "")),
                extra_search_paths=package_search_paths(),
            )

        return yourdfpy.URDF.load(
            source, build_scene_graph=True, load_meshes=False,
            filename_handler=lambda name: resolver(name),
        )
    except SolverError:
        raise
    except Exception as exc:  # noqa: BLE001
        # Was a bare `return None`, which reported every failure as the same
        # opaque "description not available". _build_pyroki's caller funnels
        # this into _pyroki_failed, which the sidebar already displays.
        raise SolverError(f"could not reload the robot description: {exc}") from exc


def _link_target_for(rig, tip_bone_name: str) -> tuple[str, np.ndarray] | None:
    """Work out which URDF link the tip rides, and the bone->link correction.

    The TCP bone hangs off a joint bone. That joint's child link is what PyRoki
    should aim at, and the correction chains the TCP bone's offset from the
    joint bone onto the joint bone's own bone->link correction::

        link_goal = tcp_goal · M_tcp⁻¹ · M_joint · C_joint

    A joint bone can also be the tip in its own right, in which case the walk
    starts on the bone itself and the correction reduces to that bone's own.
    """
    bones = rig.data.bones
    tcp = bones.get(tip_bone_name)
    if tcp is None:
        return None

    node = tcp
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


def build_solver(rig, ik_bone: str, tip: str | None = None) -> RigSolver | None:
    """Create the solver state for one rig, or None if it cannot be rigged."""
    tip = tip or tip_bone(rig)
    chain = chain_mod.chain_from_rig(rig, tip)
    if chain is None:
        return None
    return RigSolver(
        rig_name=rig.name,
        chain=chain,
        ik_bone=ik_bone,
        tip_bone=tip,
        identity=rig_identity(rig),
        link_target=_link_target_for(rig, tip),
    )


def get_solver(rig, ik_bone: str | None = None) -> RigSolver | None:
    """Fetch (or build) the cached solver for ``rig``."""
    ik_bone = ik_bone or rig.get(builder.PROP_IK_BONE)
    if not ik_bone:
        return None

    # The tip is compared, not just checked for existence: it is keyframable,
    # so it can change between two solves with nothing else about the rig
    # having moved, and a cached solver would then drive the wrong chain.
    tip = tip_bone(rig)
    cached = _cache.get(rig.name)
    if (
        cached is not None
        # Same object, not merely the same name: deleting a rig frees its name
        # for the next import, which would otherwise inherit its solver.
        and cached.identity == rig_identity(rig)
        and cached.ik_bone == ik_bone
        and cached.tip_bone == tip
        and tip in rig.pose.bones
    ):
        return cached
    # Anything that changes the rig's bones -- rebuilding it, or moving the
    # TCP -- calls invalidate(), so a stale entry here is not silently reused.

    solver = build_solver(rig, ik_bone, tip)
    if solver is not None:
        _cache[rig.name] = solver
    return solver


def invalidate(rig_name: str | None = None) -> None:
    """Drop cached solvers -- after a rig rebuild, or on unregister.

    The compiled PyRoki solvers go too. They are expensive to rebuild, but the
    reasons to invalidate -- the bones changed underneath us -- are exactly the
    reasons a compiled kernel for the old bones must not be reused.
    """
    if rig_name is None:
        _cache.clear()
        _pyroki_cache.clear()
    else:
        _cache.pop(rig_name, None)
        # Matched on the stored name rather than the key, which is now an
        # identity: callers only ever have a name to give.
        for key, entry in list(_pyroki_cache.items()):
            if entry[2] == rig_name:
                del _pyroki_cache[key]
