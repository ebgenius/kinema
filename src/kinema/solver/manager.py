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


def root_pose(rig) -> np.ndarray:
    """How far the Root bone is posed from its rest, as a transform in armature space.

    Both backends solve with Root at rest: the chain is read off the bones' rest
    matrices, and PyRoki's model is rooted at the armature origin. Everything under
    Root is then displayed moved by this, the IK target and swivel included, since
    they hang off Root too. So a goal read off a posed bone has to have it taken out
    before it is solved for, or the arm lands off by exactly Root's own pose.
    """
    root = rig.pose.bones.get(builder.ROOT_BONE)
    if root is None:
        return np.eye(4)
    return _np4(root.matrix) @ np.linalg.inv(_np4(root.bone.matrix_local))


def solver_goal(rig, bone_name: str) -> np.ndarray:
    """``bone_name``'s posed matrix, in the frame the solvers work in: Root at rest."""
    return np.linalg.inv(root_pose(rig)) @ _np4(rig.pose.bones[bone_name].matrix)


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
        if self._pyroki_failed or self.rig_name in _deferred:
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

        goal = solver_goal(rig, self.ik_bone)
        seed = chain_mod.read_configuration(rig, self.chain)
        held = held_mask(rig, self.chain)
        if held.any():
            # A held joint stands where the rig shows it, limit constraint and
            # all, not at its channel value -- see chain.displayed_configuration.
            seed = chain_mod.displayed_configuration(rig, self.chain, seed, only=held)

        result = None
        if mode == MODE_PYROKI:
            solver = self.pyroki(rig)
            if solver is not None:
                result = self._solve_pyroki(
                    solver, seed, goal, elbow=self.elbow_goal(rig, solver), held=held
                )

        if result is None:
            result = numpy_backend.solve(self.chain, seed, goal, held=held)

        # Held joints are not written back. They are the animator's input, and
        # writing the displayed value over a channel dragged past its limit would
        # quietly move it.
        chain_mod.write_configuration(rig, self.chain, result.q, skip=held)
        self.last_result = result
        self.solve_count += 1
        return result

    def compile(self, rig) -> bool:
        """Build PyRoki and pay its compile, without touching the pose.

        One solve for where the tool already stands, with the elbow goal a live
        solve would use, so the kernel compiled is the one the next drag needs. The
        result is thrown away: a warm-up nobody asked for must not move a robot --
        its IK goal may be stale, left behind while the arm was posed by hand.
        """
        solver = self.pyroki(rig)
        if solver is None:
            return False
        tool = solver_goal(rig, self.tip_bone)
        seed = chain_mod.read_configuration(rig, self.chain)
        held = held_mask(rig, self.chain)
        if held.any():
            seed = chain_mod.displayed_configuration(rig, self.chain, seed, only=held)
        result = self._solve_pyroki(
            solver, seed, tool, elbow=self.elbow_goal(rig, solver), held=held
        )
        return result is not None

    def elbow_goal(self, rig, solver) -> tuple[int, np.ndarray, float] | None:
        """The swivel's elbow goal as PyRoki wants it, or None if there is none.

        The point lies on the circle the elbow can actually sweep -- see
        ``solver/swivel.py`` -- at the angle the swivel is turned to. That
        circle is built about the wrist centre *at the goal*, not where the
        wrist is now: on the update that moves the target the arm has not got
        there yet, and a circle about the old wrist would pull the elbow toward
        a point it cannot reach from the new one.

        Only on a chain with more than six unheld joints. With six there is no
        freedom left over, and the goal could only be met by moving the tool.
        """
        from . import swivel

        handle_name = rig.get(builder.PROP_SWIVEL_BONE)
        if not handle_name:
            return None
        if self.chain.dof - int(np.sum(held_mask(rig, self.chain))) <= 6:
            return None
        weight = float(getattr(rig, "kinema_elbow_strength", 0.0))
        if weight <= 0.0:
            return None

        pose = rig.pose.bones
        names = [
            rig.get(key, "")
            for key in (
                builder.PROP_SWIVEL_SHOULDER,
                builder.PROP_ELBOW_JOINT,
                builder.PROP_SWIVEL_WRIST,
            )
        ]
        handle, tip, target = (
            pose.get(handle_name), pose.get(self.tip_bone), pose.get(self.ik_bone)
        )
        if handle is None or tip is None or target is None:
            return None
        if not all(name and name in pose for name in names):
            return None
        shoulder, elbow, wrist = (pose[name] for name in names)

        link = rig.data.bones[elbow.name].get(builder.PROP_CHILD_LINK)
        index = solver.link_index(str(link)) if link else None
        if index is None:
            return None

        # Rigid links, so the rest pose gives both lengths once and for all.
        rest = [np.array(rig.data.bones[name].head_local, dtype=float) for name in names]
        upper = float(np.linalg.norm(rest[1] - rest[0]))
        fore = float(np.linalg.norm(rest[2] - rest[1]))

        # The wrist centre rides the tool: every wrist axis passes through it,
        # so it is fixed in the tool's frame however the wrist is turned.
        wrist_in_tip = np.linalg.inv(_np4(tip.matrix)) @ np.append(
            np.array(wrist.matrix.translation, dtype=float), 1.0
        )
        wrist_at_goal = (_np4(target.matrix) @ wrist_in_tip)[:3]

        point = swivel.elbow_goal(
            np.array(shoulder.matrix.translation, dtype=float),
            wrist_at_goal,
            upper,
            fore,
            np.array(handle.matrix.col[2][:3], dtype=float),
            np.array(elbow.matrix.translation, dtype=float),
        )
        # Built from posed bones, so in the posed frame; the solver's has Root at rest.
        point = (np.linalg.inv(root_pose(rig)) @ np.append(point, 1.0))[:3]
        return (index, point, weight)

    def _solve_pyroki(
        self, solver, seed: np.ndarray, goal: np.ndarray, elbow=None, held=None
    ) -> SolveResult | None:
        try:
            # PyRoki targets a URDF link; the IK bone expresses a *bone* goal,
            # so apply the fixed bone->link correction recorded at build time.
            _, correction = self.link_target
            link_goal = goal @ correction

            full_seed = np.zeros(solver.dof)
            full_seed[self._chain_to_full] = seed
            full_held = _full_mask(held, self._chain_to_full, solver.dof)
            full = solver.solve(
                full_seed, link_goal, elbow=elbow,
                held=(full_held, full_seed) if full_held is not None else None,
            )
            if full_held is not None:
                # The hold is a heavy cost, not a lock, so put held joints back
                # exactly. Seeding the next update from a value that crept
                # would let a rail drift a hair per solve until it visibly had.
                full[full_held] = full_seed[full_held]

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


def held_mask(rig, chain: chain_mod.Chain) -> np.ndarray:
    """Which of ``chain``'s joints the animator is positioning by hand.

    Read off each joint bone's ``kinema_ik_hold``. A bone without the property
    -- a rig saved before it existed -- is simply not held.
    """
    bones = rig.pose.bones
    return np.array(
        [bool(getattr(bones[name], "kinema_ik_hold", False)) for name in chain.bone_names],
        dtype=bool,
    )


def _full_mask(held, chain_to_full, dof: int) -> np.ndarray | None:
    """A chain-joint mask spread onto PyRoki's full actuated vector, or None."""
    if held is None or not np.any(held):
        return None
    mask = np.zeros(dof, dtype=bool)
    mask[chain_to_full] = held
    return mask


def _load_source_urdf(rig):
    """Reload the description this rig was built from, if we still can.

    Raises SolverError with a readable reason; the panel shows it. Returns None
    only when the rig records no source at all.

    A rig with an external axis is no longer the robot in its description, so it
    is described from its own bones instead -- see ``solver/rig_model.py``.
    """
    from . import rig_model

    if rig_model.has_external_axes(rig):
        from .urdf_bridge import urdf_from_model

        try:
            return urdf_from_model(rig_model.model_from_rig(rig))
        except Exception as exc:  # noqa: BLE001
            raise SolverError(f"could not describe this rig for the solver: {exc}") from exc

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


def find_solutions(rig, solver: RigSolver, seeds: int = 0, seed_value: int = 0):
    """Every distinct configuration these seeds could reach the current goal in.

    A lower bound, not an enumeration -- see ``solver/branches.py``. Runs the
    seeds through whichever backend the rig is on, then hands the results to
    :func:`branches.collect`, which decides what counts and what is a duplicate.

    The rig is left exactly as it was found. Searching should not be a way to
    lose the pose you were looking at, and the caller applies a solution by
    choosing one.
    """
    from . import branches

    pose = rig.pose
    if solver.ik_bone not in pose.bones:
        return []
    goal = solver_goal(rig, solver.ik_bone)
    chain = solver.chain

    mode = getattr(rig, "kinema_solver_mode", MODE_PYROKI)
    if mode == MODE_OFF:
        mode = MODE_PYROKI
    pyroki = solver.pyroki(rig) if mode == MODE_PYROKI else None
    elbow = solver.elbow_goal(rig, pyroki) if pyroki is not None else None

    # The goal in the frame each backend wants it in. PyRoki aims at a URDF
    # link, the NumPy fallback at the tool frame the chain already describes.
    link_goal = goal @ solver.link_target[1] if solver.link_target else goal

    started = chain_mod.read_configuration(rig, chain)
    held = held_mask(rig, chain)
    if held.any():
        started = chain_mod.displayed_configuration(rig, chain, started, only=held)
    rng = np.random.default_rng(seed_value)
    candidates = []
    for q_seed in branches.seed_configurations(
        chain, seeds or branches.DEFAULT_SEEDS, rng
    ):
        # A held joint is not the search's to vary: every alternative it finds
        # is an arm reaching from where the rail already stands.
        q_seed = np.where(held, started, q_seed)
        if pyroki is not None:
            full = np.zeros(pyroki.dof)
            full[solver._chain_to_full] = q_seed
            mask = _full_mask(held, solver._chain_to_full, pyroki.dof)
            try:
                solved = pyroki.solve(
                    full, link_goal, elbow=elbow,
                    held=(mask, full) if mask is not None else None,
                )
            except Exception:  # noqa: BLE001 - one bad seed must not end the search
                continue
            if mask is not None:
                solved[mask] = full[mask]
            candidates.append(solved[solver._chain_to_full])
        else:
            candidates.append(numpy_backend.solve(chain, q_seed, goal, held=held).q)

    # Include where the arm already is, so the configuration the user is
    # looking at is in the list rather than conspicuously missing from it.
    candidates.insert(0, started)
    return branches.collect(chain, goal, candidates)


#: Rigs whose PyRoki build is put off: their model is still being edited, and each
#: edit would pay a fresh JAX compile. They solve on NumPy meanwhile -- no compile,
#: and good enough to keep the arm on its target while the edit goes on.
_deferred: set[str] = set()


def defer(rig_name: str) -> None:
    """Put off building PyRoki for ``rig_name`` until :func:`release`."""
    _deferred.add(rig_name)


def release(rig_name: str) -> None:
    """Let ``rig_name`` build PyRoki again, on its next solve."""
    _deferred.discard(rig_name)


def deferred(rig_name: str) -> bool:
    return rig_name in _deferred


def invalidate(rig_name: str | None = None) -> None:
    """Drop cached solvers -- after a rig rebuild, or on unregister.

    The compiled PyRoki solvers go too. They are expensive to rebuild, but the
    reasons to invalidate -- the bones changed underneath us -- are exactly the
    reasons a compiled kernel for the old bones must not be reused.
    """
    if rig_name is None:
        _cache.clear()
        _pyroki_cache.clear()
        _deferred.clear()
    else:
        _cache.pop(rig_name, None)
        # Matched on the stored name rather than the key, which is now an
        # identity: callers only ever have a name to give.
        for key, entry in list(_pyroki_cache.items()):
            if entry[2] == rig_name:
                del _pyroki_cache[key]
