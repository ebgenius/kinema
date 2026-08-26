"""Synthesise a minimal URDF from a :class:`RobotModel`.

PyRoki builds its robot model from a ``yourdfpy.URDF``, so an MJCF-sourced rig
would otherwise be stuck with the NumPy fallback -- and MJCF is 60 of the
catalog's 186 robots, a third of everything Kinema can import. Rather than
teach PyRoki a second input format, the kinematic tree Kinema already parsed is
written back out as URDF.

Only kinematics is emitted: links, joints, axes and limits. Visual geometry is
deliberately omitted, because PyRoki never looks at it and mesh paths would
just be a way to fail. The result is parsed from a temporary file and thrown
away; nothing is written into the user's project.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np

from ..rig.kinematics import RobotModel


def _rpy_from_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    """Inverse of URDF's fixed-axis XYZ convention, R = Rz(y)·Ry(p)·Rx(r)."""
    sy = -float(np.clip(rotation[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sy))
    # cos(pitch) ~ 0 is gimbal lock: yaw and roll become degenerate, so pin
    # roll to zero and fold the whole rotation into yaw.
    if abs(np.cos(pitch)) < 1e-9:
        roll = 0.0
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
    else:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return roll, pitch, yaw


def _origin_element(parent: ElementTree.Element, transform: np.ndarray) -> None:
    roll, pitch, yaw = _rpy_from_matrix(transform[:3, :3])
    position = transform[:3, 3]
    ElementTree.SubElement(
        parent,
        "origin",
        {
            "xyz": f"{position[0]:.12g} {position[1]:.12g} {position[2]:.12g}",
            "rpy": f"{roll:.12g} {pitch:.12g} {yaw:.12g}",
        },
    )


def urdf_xml(model: RobotModel) -> str:
    """Render ``model`` as a kinematics-only URDF document."""
    robot = ElementTree.Element("robot", {"name": model.name or "robot"})

    for name in model.links:
        ElementTree.SubElement(robot, "link", {"name": name})

    for joint in model.joints:
        element = ElementTree.SubElement(
            robot, "joint", {"name": joint.name, "type": joint.joint_type}
        )
        ElementTree.SubElement(element, "parent", {"link": joint.parent_link})
        ElementTree.SubElement(element, "child", {"link": joint.child_link})
        _origin_element(element, joint.origin)

        if joint.joint_type != "fixed":
            axis = joint.axis
            ElementTree.SubElement(
                element,
                "axis",
                {"xyz": f"{axis[0]:.12g} {axis[1]:.12g} {axis[2]:.12g}"},
            )
            # effort and velocity are always emitted. URDF makes <limit>
            # optional on a continuous joint, but PyRoki rejects one without
            # velocity limits outright -- which is how every unlimited MJCF
            # hinge failed to load before.
            limit = {"effort": "100", "velocity": "3.14"}
            if joint.has_limits:
                limit["lower"] = f"{joint.lower:.12g}"
                limit["upper"] = f"{joint.upper:.12g}"
            elif joint.joint_type == "revolute":
                # An unlimited revolute joint is spelled "continuous" in URDF.
                element.set("type", "continuous")
            ElementTree.SubElement(element, "limit", limit)

        if joint.mimic_joint:
            ElementTree.SubElement(
                element,
                "mimic",
                {
                    "joint": joint.mimic_joint,
                    "multiplier": f"{joint.mimic_multiplier:.12g}",
                    "offset": f"{joint.mimic_offset:.12g}",
                },
            )

    return ElementTree.tostring(robot, encoding="unicode")


def urdf_from_model(model: RobotModel):
    """Build a ``yourdfpy.URDF`` from ``model``, for PyRoki to consume."""
    import yourdfpy

    xml = urdf_xml(model)
    # yourdfpy wants a path it can resolve relative filenames against. There
    # are none here, but a real file keeps it on its happy path.
    with tempfile.TemporaryDirectory(prefix="kinema-urdf-") as directory:
        path = Path(directory) / "model.urdf"
        path.write_text(xml, encoding="utf-8")
        return yourdfpy.URDF.load(
            str(path), build_scene_graph=True, load_meshes=False
        )
