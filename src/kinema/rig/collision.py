"""Collision: the robot as capsules, the cell as boxes, spheres and floors, and how far apart.

**The robot.** Each joint bone carries the meshes of the link it moves, so
the meshes on one bone move together, and a few capsules around them stand in
for the link. One capsule a link is what PyRoki fits from a description, and
it is far too coarse for an industrial arm: a KR210's forearm, with its motor
housing, came out 824 mm in radius, 8.7 times the volume of its convex hull.
So :func:`fit` splits a link's points into clusters and bounds each with the
tightest capsule it finds (:func:`_capsule`). More capsules fit tighter and
cost more in every solve; it takes the fewest that come within :data:`SLACK`
of the tightest of up to :data:`MAX_PARTS`. On that KR210: 4 on the forearm
(401 mm at most), 14 on the whole arm.

A capsule is fitted as a capsule, not as PyRoki's ``Capsule.from_trimesh``
fits one: the narrowest cylinder round the points (trimesh's
``minimum_cylinder``) with its ends rounded. Each rounded end then stands a
whole radius past a face the cylinder fitted exactly, and on a squat link the
narrowest cylinder stands on its shortest side.

Each capsule is kept in its bone's frame, as the segment between two points
and a radius. Blender places the bone on every frame, so the capsules move
with whatever plays, IK or keys, without a solver.

**The cell.** What stands in the robot's way is a box, a sphere, or a floor
(everything below a plane). That is what the solver can measure fast, and how
most cells are described anyway; a complicated fixture is a few of them.

**Distances** are exact and signed: negative is how far into an obstacle a
capsule reaches. A box is measured by its signed distance field, which is
convex, so its minimum along a capsule's axis is found by golden-section search
to well under a micron. PyRoki's own capsule-box distance refines its closest
point twice and can read a corner as further away than it is; this one is the
check the solver's results are held to.

Deliberately free of ``bpy``: reading meshes and obstacles off the scene lives
in ``ops/collision.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The most capsules one link is split into.
MAX_PARTS = 4
#: How much more volume than the tightest fit tried is accepted, for fewer capsules.
SLACK = 1.25
#: A link's points are thinned to one per cell of this size, in metres, before
#: fitting: what clusters them is their spread, and a mesh is dense where it
#: is detailed, not where it is large.
CELL = 0.02
#: How far apart the points sampled over a mesh's surface are, about, in
#: metres. A mesh's vertices alone leave its flat faces bare: a box is eight
#: corners, and capsules round its two ends hold every one of them while the
#: middle of the box sticks out between.
SPACING = 0.01
#: The most points sampled over one link, so a huge mesh can't take minutes.
MAX_SAMPLES = 100_000
#: Golden-section steps for a capsule's closest approach to a box. Each keeps
#: 0.618 of the interval: 40 take a 2 m capsule's search to a few microns.
STEPS = 40

KIND_BOX = "BOX"
KIND_SPHERE = "SPHERE"
KIND_FLOOR = "FLOOR"


@dataclass(frozen=True)
class Capsule:
    """A segment and a radius, in the frame of the bone it rides on."""

    bone: str
    a: np.ndarray
    b: np.ndarray
    radius: float


@dataclass(frozen=True)
class Obstacle:
    """Something in the robot's way, in world space.

    A box is ``matrix`` (rotation and centre, unscaled) and ``size``, its
    half-extents. A sphere is the centre of ``matrix`` and ``size[0]``, its
    radius. A floor is the plane through the centre of ``matrix`` with its Z
    axis as the normal; everything below it is solid.
    """

    name: str
    kind: str
    matrix: np.ndarray
    size: np.ndarray


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------
def fit(
    points: np.ndarray,
    max_parts: int = MAX_PARTS,
    slack: float = SLACK,
    parts: int | None = None,
) -> list:
    """Capsules around ``points``, as ``(a, b, radius)``, in the points' own frame.

    Every point is inside one of them. They are fitted to the points thinned,
    which bounds only what thinning kept, so each is then widened to take in
    the points nearest it that it doesn't.

    ``parts`` asks for that many, as set on a link by hand, and gets them where
    the points can be split that far: the same split the count would be tried
    with here. Otherwise the count is chosen by volume, which can't see where
    the bulk sits: a link can still come out with fewer capsules than suit it.
    """
    points = np.asarray(points, dtype=float)
    thinned = thin(points)
    if len(thinned) == 0:
        return []
    if parts is not None:
        # One cluster always forms, so this always returns.
        for count in range(max(int(parts), 1), 0, -1):
            clusters = _clusters(thinned, count)
            if clusters is not None:
                return _covering(points, [_capsule(cluster) for cluster in clusters])
    tries = []
    for parts in range(1, max_parts + 1):
        clusters = _clusters(thinned, parts)
        if clusters is None:
            break
        capsules = [_capsule(cluster) for cluster in clusters]
        tries.append((sum(volume(radius, a, b) for a, b, radius in capsules), capsules))
    best = min(total for total, _ in tries)
    chosen = next(capsules for total, capsules in tries if total <= best * slack)
    return _covering(points, chosen)


def _covering(points: np.ndarray, capsules: list) -> list:
    """``capsules`` widened until every one of ``points`` is inside one of them.

    A point outside them all goes to the capsule it is least outside of.
    """
    gaps = np.stack(
        [_point_segment_distance(points, a, b) - radius for a, b, radius in capsules], axis=1
    )
    nearest = np.argmin(gaps, axis=1)
    outside = gaps[np.arange(len(points)), nearest] > 0.0
    widened = []
    for index, (a, b, radius) in enumerate(capsules):
        mine = outside & (nearest == index)
        if mine.any():
            radius = radius + float(gaps[mine, index].max())
        widened.append((a, b, radius))
    return widened


def surface_points(
    vertices: np.ndarray, triangles: np.ndarray, spacing: float = SPACING
) -> np.ndarray:
    """``vertices`` and points spread evenly over the triangles between them.

    Seeded, so the same mesh always gives the same points, and the same capsules.
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if not len(triangles):
        return vertices
    corners = vertices[triangles]
    edge_1, edge_2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge_1, edge_2), axis=1)
    total = float(areas.sum())
    if total <= 0.0:
        return vertices
    count = int(min(MAX_SAMPLES, math.ceil(total / spacing**2)))
    random = np.random.default_rng(0)
    which = random.choice(len(triangles), count, p=areas / total)
    # Uniform over each triangle: the square root keeps them off its first corner.
    root = np.sqrt(random.random(count))[:, None]
    other = random.random(count)[:, None]
    samples = (
        corners[which, 0] * (1.0 - root)
        + corners[which, 1] * (root * (1.0 - other))
        + corners[which, 2] * (root * other)
    )
    return np.concatenate([vertices, samples])


def thin(points: np.ndarray, cell: float = CELL) -> np.ndarray:
    """One point per ``cell``-sized cube, the first in each, in their original order."""
    if len(points) == 0:
        return points
    _, first = np.unique(np.floor(points / cell).astype(np.int64), axis=0, return_index=True)
    return points[np.sort(first)]


def volume(radius: float, a, b) -> float:
    length = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
    return math.pi * radius * radius * length + 4.0 / 3.0 * math.pi * radius**3


def _clusters(points: np.ndarray, parts: int) -> list | None:
    """``points`` split into ``parts`` clusters, or None if they can't be."""
    if parts == 1:
        return [points]
    if len(points) < parts * 4:
        return None
    from scipy.cluster.vq import kmeans2

    # Seeded, so the same meshes always give the same capsules.
    _, labels = kmeans2(points, parts, minit="++", seed=0)
    clusters = [points[labels == index] for index in range(parts)]
    clusters = [cluster for cluster in clusters if len(cluster)]
    return clusters if len(clusters) == parts else None


def _capsule(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """The tightest capsule round ``points`` along either of two axes.

    The axes: the points' principal one, and the narrowest cylinder's
    (trimesh's ``minimum_cylinder``, the fit PyRoki's ``Capsule.from_trimesh``
    uses). Along each, :func:`_along` fits a capsule as a capsule; the smaller
    is kept. The principal axis is the better one for most links. The
    cylinder's covers a squat link, which has no clear long direction.

    A capsule is not that cylinder with its ends rounded. Round them and each
    end stands a whole radius past a face the cylinder fitted exactly. On a
    squat link the narrowest cylinder stands on the shortest side: a KR120's
    wrist, 210 mm thick, got a capsule 579 mm tall, which this fits at under
    half the volume.

    trimesh is imported outside the fallback: without it every capsule would
    quietly take the crude fit below, so a missing trimesh has to fail loudly.
    Only geometry with no volume falls back: a hull of flat or collinear points
    fails to diagonalise, which numpy reports as a LinAlgError.
    """
    import trimesh
    from scipy.spatial import QhullError

    if len(points) >= 4:
        try:
            hull = trimesh.convex.convex_hull(points)
            if hull.volume > 1e-12:
                corners = np.asarray(hull.vertices, dtype=float)
                cylinder = trimesh.bounds.minimum_cylinder(hull)
                axes = [
                    _principal_axis(points),
                    np.asarray(cylinder["transform"], dtype=float)[:3, 2],
                ]
                fits = []
                for axis in axes:
                    try:
                        fits.append(_along(corners, axis))
                    except (np.linalg.LinAlgError, QhullError, ValueError):
                        continue  # flat seen along it: the other axis may do
                fits = [fit for fit in fits if fit is not None]
                if fits:
                    _, a, b, radius = min(fits, key=lambda fit: fit[0])
                    return a, b, radius
        except (np.linalg.LinAlgError, QhullError, ValueError):
            pass
    return _segment_around(points)


#: How far past the least radius round an axis :func:`_along` looks: a wider
#: capsule can be shorter, and for a squat set of points that pays.
WIDEST = 1.8
#: Radii tried between the least and :data:`WIDEST` times it, then again
#: between the best one's neighbours.
RADII = 41


def _principal_axis(points: np.ndarray) -> np.ndarray:
    """The direction ``points`` spread furthest along."""
    return np.linalg.svd(points - points.mean(axis=0), full_matrices=False)[2][0]


def _along(points: np.ndarray, axis: np.ndarray):
    """The least-volume capsule round ``points`` whose segment runs along ``axis``.

    Returns ``(volume, a, b, radius)``, or None if it can't be fitted. The
    segment lies on the line through the centre of the smallest circle round
    the points seen along the axis, which gives the least radius. For a given
    radius, the segment is trimmed to what keeps every point inside: a point
    ``d`` from the line is inside a rounded end reaching ``sqrt(r^2 - d^2)``
    past the segment's end. A wider radius trims more, so radii past the least
    are tried, for the least volume.
    """
    from trimesh import nsphere

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    flat = np.c_[points @ u, points @ v]
    centre, least = nsphere.minimum_nsphere(flat)
    least = float(least)
    distance = np.linalg.norm(flat - centre, axis=1)
    height = points @ axis
    line = centre[0] * u + centre[1] * v

    def capsule(radius: float):
        reach = np.sqrt(np.maximum(radius * radius - distance * distance, 0.0))
        low, high = float(np.min(height + reach)), float(np.max(height - reach))
        if high < low:
            # Wider than needs any segment. The least radius never is: the
            # points furthest from the line reach no way past their own height.
            return None
        held = math.pi * radius * radius * (high - low) + 4.0 / 3.0 * math.pi * radius**3
        return held, line + low * axis, line + high * axis, radius

    radii = least * np.linspace(1.0, WIDEST, RADII)
    tried = [(capsule(float(radius)), index) for index, radius in enumerate(radii)]
    tried = [(fit, index) for fit, index in tried if fit is not None]
    if not tried:
        return None
    best, index = min(tried, key=lambda item: item[0][0])
    # Finer, between the best radius's neighbours.
    finer = np.linspace(radii[max(index - 1, 0)], radii[min(index + 1, RADII - 1)], RADII)
    refined = [fit for fit in (capsule(float(radius)) for radius in finer) if fit is not None]
    return min([best, *refined], key=lambda fit: fit[0])


def _segment_around(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """A capsule along the points' longest direction: for points with no volume.

    A flat plate has no convex hull to take a cylinder of, but still has to be
    stood clear of.
    """
    centre = points.mean(axis=0)
    spread = points - centre
    if len(points) > 1:
        axis = np.linalg.svd(spread, full_matrices=False)[2][0]
    else:
        axis = np.array([0.0, 0.0, 1.0])
    along = spread @ axis
    a, b = centre + along.min() * axis, centre + along.max() * axis
    radius = float(np.max(np.linalg.norm(spread - np.outer(along, axis), axis=1), initial=0.0))
    return a, b, radius


# --------------------------------------------------------------------------
# distances
# --------------------------------------------------------------------------
def clearances(a: np.ndarray, b: np.ndarray, radius: np.ndarray, obstacles: list) -> np.ndarray:
    """Signed clearance between every capsule and every obstacle.

    ``a`` and ``b`` are ``(..., C, 3)`` world-space ends and ``radius`` is
    ``(C,)``; the result is ``(..., C, O)``, in metres, negative where a capsule
    reaches into an obstacle.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    radius = np.asarray(radius, dtype=float)
    out = np.empty(a.shape[:-1] + (len(obstacles),))
    for index, obstacle in enumerate(obstacles):
        out[..., index] = segment_distance(a, b, obstacle) - radius
    return out


def segment_distance(a: np.ndarray, b: np.ndarray, obstacle: Obstacle) -> np.ndarray:
    """Signed distance from each segment ``a``-``b`` to ``obstacle``, over leading axes."""
    matrix = np.asarray(obstacle.matrix, dtype=float)
    centre, rotation = matrix[:3, 3], matrix[:3, :3]
    if obstacle.kind == KIND_FLOOR:
        normal = rotation[:, 2] / np.linalg.norm(rotation[:, 2])
        return np.minimum((a - centre) @ normal, (b - centre) @ normal)
    if obstacle.kind == KIND_SPHERE:
        return _point_segment_distance(centre, a, b) - float(obstacle.size[0])
    # A box: into its own frame, where it is centred and axis-aligned.
    local_a = (a - centre) @ rotation
    local_b = (b - centre) @ rotation
    half = np.asarray(obstacle.size, dtype=float)
    return _segment_box(local_a, local_b, half)


def box_sdf(points: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Signed distance from points to the axis-aligned box of half-extents ``half``."""
    q = np.abs(points) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def _segment_box(a: np.ndarray, b: np.ndarray, half: np.ndarray) -> np.ndarray:
    """The least signed distance along each segment: the box's field is convex along it."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0

    def at(t):
        return box_sdf(a + t[..., None] * (b - a), half)

    low = np.zeros(a.shape[:-1])
    high = np.ones(a.shape[:-1])
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    f_left, f_right = at(left), at(right)
    for _ in range(STEPS):
        # The minimum is left of `right` where f_left is the lower, else right of `left`.
        lower = f_left < f_right
        low = np.where(lower, low, left)
        high = np.where(lower, right, high)
        kept, f_kept = np.where(lower, left, right), np.where(lower, f_left, f_right)
        new = np.where(lower, high - ratio * (high - low), low + ratio * (high - low))
        f_new = at(new)
        left, f_left = np.where(lower, new, kept), np.where(lower, f_new, f_kept)
        right, f_right = np.where(lower, kept, new), np.where(lower, f_kept, f_new)
    # The ends too: the minimum of a convex function on [0, 1] may sit on one.
    ends = np.minimum(at(np.zeros_like(low)), at(np.ones_like(low)))
    return np.minimum(np.minimum(f_left, f_right), ends)


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    span = b - a
    length = np.einsum("...i,...i->...", span, span)
    along = np.einsum("...i,...i->...", point - a, span)
    t = np.where(length > 0.0, along / np.where(length > 0.0, length, 1.0), 0.0)
    closest = a + np.clip(t, 0.0, 1.0)[..., None] * span
    return np.linalg.norm(point - closest, axis=-1)
