"""How fast a traced tool moved and turned: the part of Trace Tool that needs no Blender.

A trace is the tool's pose on every frame. From those poses come four rates per
frame, measured as the Motion Check measures a joint's (``rig/velocity.py``):

* **speed**: how far the tool moved since the frame before, times the frame rate;
* **acceleration**: how much its velocity changed from the frame before to the
  frame after, the second difference of its position, times the frame rate
  squared. A change of direction at a steady speed counts, as it does for a
  controller;
* **angular speed**: the angle it turned through since the frame before, times
  the frame rate;
* **angular acceleration**: how much its rate of turn changed, as a vector,
  between the frame before and the frame after, times the frame rate squared.

Each needs the frame before, and the accelerations the frame after too, so a
trace over frames ``first..last`` is measured from poses over
``first - 1..last + 1``.
"""

from __future__ import annotations

import math

import numpy as np

from .motion_check import quaternion

SPEED = "Speed"
ACCELERATION = "Acceleration"
ANGULAR_SPEED = "Angular Speed"
ANGULAR_ACCELERATION = "Angular Acceleration"
#: Each rate and its unit, in the order a trace lists them.
UNITS = {
    SPEED: "m/s",
    ACCELERATION: "m/s²",
    ANGULAR_SPEED: "rad/s",
    ANGULAR_ACCELERATION: "rad/s²",
}


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """The axis times the angle, in radians, of a rotation matrix: at most a half-turn."""
    w, *v = quaternion(rotation)
    v = np.array(v)
    length = float(np.linalg.norm(v))
    if length < 1e-12:
        return np.zeros(3)
    return v / length * 2.0 * math.atan2(length, w)


def rates(poses, fps: float) -> dict[str, np.ndarray]:
    """The four rates on each pose but the first and last, from 4x4 ``poses`` a frame apart.

    Positions in metres give metres per second; rotations are read from the
    upper 3x3, which must be a rotation, without scale.
    """
    poses = np.asarray(poses, dtype=float)
    if len(poses) < 3:
        raise ValueError("A rate needs a pose either side of each frame it is measured on")
    positions = poses[:, :3, 3]
    steps = np.diff(positions, axis=0) * fps
    # Each frame's turn since the one before, in world axes: R_now R_before^T.
    turns = np.array([
        rotation_vector(now[:3, :3] @ before[:3, :3].T) * fps
        for before, now in zip(poses, poses[1:], strict=False)
    ])
    return {
        SPEED: np.linalg.norm(steps[:-1], axis=1),
        ACCELERATION: np.linalg.norm(np.diff(steps, axis=0), axis=1) * fps,
        ANGULAR_SPEED: np.linalg.norm(turns[:-1], axis=1),
        ANGULAR_ACCELERATION: np.linalg.norm(np.diff(turns, axis=0), axis=1) * fps,
    }
