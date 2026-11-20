"""kinema.api (#102): waypoints, motions and jobs. Needs a real ``bpy``.

The claims:

* the README's script, the issue's "done when", teaches, generates and checks a
  job on arm6 through kinema.api alone;
* waypoints are taught at joint values without moving the rig, relative to an
  object or the base, and read back where they are;
* motions are added between waypoints by name, and what isn't one is refused;
* removing a waypoint takes its motions with it;
* generating or checking a job that can't be is refused, writing nothing.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy

HOME = [0.0, -0.6, 1.2, 0.0, 0.6, 0.0]
AWAY = [0.5, -0.4, 1.0, 0.2, 0.5, -0.3]


@pytest.fixture
def api(addon):
    return importlib.import_module(f"{addon.__name__}.api")


@pytest.fixture
def arm6(api, addon, fixture_dir, clean_scene):
    import bpy

    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(fixture_dir / "arm6.urdf"))
    rig = bpy.context.view_layer.objects.active
    manager = importlib.import_module(f"{addon.__name__}.solver.manager")
    rig.kinema_solver_mode = manager.MODE_NUMPY  # The same rule; no compile.
    return rig


def _readme_job() -> str:
    readme = (Path(__file__).parents[2] / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Scripting Kinema", 1)[1].split("\n## ", 1)[0]
    return section.split("```python\n")[2].split("```", 1)[0]


class TestTheScript:
    def test_the_readme_job_runs_through_the_api_alone(self, api, arm6):
        """The issue's "done when": a script of about 30 lines, outside the add-on."""
        import bpy

        script = _readme_job()
        assert 25 <= len([line for line in script.splitlines() if line.strip()]) <= 40
        assert "bpy.ops" not in script, "kinema.api alone"
        exec(compile(script, "README.md", "exec"), {"__name__": "readme"})

        assert [m.name for m in api.motions(arm6)] == ["above", "pick", "home"]
        by_name = {w.name: w for w in api.waypoints(arm6)}
        bpy.context.scene.frame_set(49)
        np.testing.assert_allclose(api.tool_frame(arm6), by_name["pick"].pose, atol=1e-3)
        bpy.context.scene.frame_set(37)
        midway = (by_name["above"].pose[:3, 3] + by_name["pick"].pose[:3, 3]) / 2
        np.testing.assert_allclose(api.tool_frame(arm6)[:3, 3], midway, atol=1e-3)


class TestWaypoints:
    def test_taught_at_values_without_moving_the_rig(self, api, arm6):
        before = api.joint_values(arm6)
        taught = api.add_waypoint(arm6, "away", AWAY)
        np.testing.assert_allclose(api.joint_values(arm6), before)
        np.testing.assert_allclose(taught.values, AWAY)
        api.set_joint_values(arm6, AWAY)
        np.testing.assert_allclose(taught.pose, api.tool_frame(arm6), atol=1e-5)
        assert [w.name for w in api.waypoints(arm6)] == ["away"]

    def test_where_the_rig_stands_and_a_name_taken_numbered(self, api, arm6):
        api.set_joint_values(arm6, HOME)
        first = api.add_waypoint(arm6, "home")
        again = api.add_waypoint(arm6)
        twin = api.add_waypoint(arm6, "home")
        np.testing.assert_allclose(first.values, HOME, atol=1e-6)
        assert (again.name, twin.name) == ("point2", "home.001")

    def test_relative_to_an_object_it_moves_with(self, api, arm6):
        import bpy
        from mathutils import Vector

        table = bpy.data.objects.new("Table", None)
        bpy.context.scene.collection.objects.link(table)
        bpy.context.view_layer.update()
        taught = api.add_waypoint(arm6, "on table", AWAY, reference=table)
        assert taught.reference == "Table"
        table.location += Vector((0.1, 0.0, 0.0))
        bpy.context.view_layer.update()
        (moved,) = api.waypoints(arm6)
        np.testing.assert_allclose(moved.pose[:3, 3] - taught.pose[:3, 3], (0.1, 0, 0), atol=1e-6)

    def test_the_robot_is_refused_as_a_reference(self, api, arm6):
        with pytest.raises(api.Error, match="can't hold a waypoint"):
            api.add_waypoint(arm6, "on itself", reference=arm6)
        assert api.waypoints(arm6) == []

    def test_removing_one_takes_its_motions_and_marker(self, api, arm6):
        import bpy

        for name, values in (("home", HOME), ("away", AWAY), ("back", HOME)):
            api.add_waypoint(arm6, name, values)
        api.add_motion(arm6, "home", "away", 1, 25)
        api.add_motion(arm6, "away", "back", 25, 49)
        markers = len(bpy.data.objects)
        api.remove_waypoint(arm6, "back")
        assert [m.name for m in api.motions(arm6)] == ["away"]
        assert len(bpy.data.objects) == markers - 1
        with pytest.raises(api.Error, match="no waypoint 'back'"):
            api.remove_waypoint(arm6, "back")


class TestMotions:
    def test_added_and_read_by_name(self, api, arm6):
        api.add_waypoint(arm6, "home", HOME)
        api.add_waypoint(arm6, "away", AWAY)
        added = api.add_motion(arm6, "home", "away", 5, 29, move="LINEAR", name="reach")
        assert added == api.Motion("reach", "home", "away", "LINEAR", 5, 29)
        assert api.motions(arm6) == [added]
        assert api.add_motion(arm6, "away", "home", 29, 53).name == "home"

    def test_what_isnt_a_motion_is_refused(self, api, arm6):
        api.add_waypoint(arm6, "home", HOME)
        api.add_waypoint(arm6, "away", AWAY)
        with pytest.raises(api.Error, match="no waypoint 'nowhere'"):
            api.add_motion(arm6, "home", "nowhere", 1, 25)
        with pytest.raises(api.Error, match="one of JOINT, LINEAR"):
            api.add_motion(arm6, "home", "away", 1, 25, move="ARC")
        with pytest.raises(api.Error, match="ends after it starts"):
            api.add_motion(arm6, "home", "away", 25, 25)
        assert api.motions(arm6) == []

    def test_removed_by_a_name_that_says_which(self, api, arm6):
        api.add_waypoint(arm6, "home", HOME)
        api.add_waypoint(arm6, "away", AWAY)
        api.add_motion(arm6, "home", "away", 1, 25, name="go")
        api.add_motion(arm6, "away", "home", 25, 49, name="go")
        with pytest.raises(api.Error, match="2 motions called 'go'"):
            api.remove_motion(arm6, "go")
        api.add_motion(arm6, "home", "away", 49, 73, name="again")
        api.remove_motion(arm6, "again")
        assert [m.name for m in api.motions(arm6)] == ["go", "go"]


class TestJobs:
    def _job(self, api, rig, move="JOINT"):
        api.add_waypoint(rig, "home", HOME)
        api.add_waypoint(rig, "away", AWAY)
        api.add_motion(rig, "home", "away", 1, 25, move=move)

    def test_generated_and_checked(self, api, arm6):
        import bpy

        self._job(api, arm6)
        report = api.generate(arm6)
        assert (report.done, report.level, report.flagged) == (True, "INFO", 0), report
        bpy.context.scene.frame_set(25)
        np.testing.assert_allclose(api.joint_values(arm6), AWAY, atol=1e-5)
        checked = api.check(arm6)
        assert checked.done and checked.message.startswith("Checked 1 move"), checked

    def test_a_linear_move_without_an_ik_target_is_refused(self, api, arm6):
        self._job(api, arm6, move="LINEAR")
        report = api.generate(arm6)
        assert (report.done, report.level) == (False, "ERROR")
        assert "need an IK target" in report.message
        assert arm6.animation_data is None or arm6.animation_data.action is None, "nothing keyed"
        api.add_ik_target(arm6)
        from mathutils import Vector

        goal = arm6.pose.bones[arm6["kinema_ik_bone"]]
        goal.location += Vector((0.0, 0.0, 0.05))
        moved = goal.location.copy()
        api.add_ik_target(arm6)  # a second time changes nothing: the goal stays moved
        assert (goal.location - moved).length < 1e-9
        assert api.generate(arm6).done

    def test_motions_that_overlap_are_refused(self, api, arm6):
        self._job(api, arm6)
        api.add_motion(arm6, "away", "home", 20, 40)
        for action in (api.generate, api.check):
            report = action(arm6)
            assert (report.done, report.level) == (False, "WARNING"), report
            assert "Nothing was generated or checked" in report.message
