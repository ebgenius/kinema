"""Locate Description (#92): ops/locate.py. Needs a real ``bpy``.

The claims:

* a rig whose description file is missing says so, and the solver's reason
  says how to find it again;
* Locate Description points the rig at a file describing the same robot, and
  drops the cached solver, so the next solve builds from it;
* another robot, or this one changed, is refused with what differs, and the
  rig is left as it was;
* a description whose limits differ is accepted, and the rig keeps its own;
* an MJCF is located as a URDF is, and a rig with an external axis, which
  describes itself, has nothing to locate.
"""

from __future__ import annotations

import importlib
import shutil

import pytest

from ..conftest import requires_bpy

pytestmark = requires_bpy


@pytest.fixture
def modules(addon):
    return lambda name: importlib.import_module(f"{addon.__name__}.{name}")


def _build(modules, source):
    """A rig built from ``source``, active."""
    import bpy

    builder = modules("rig.builder")
    assert "FINISHED" in bpy.ops.kinema.build_robot(filepath=str(source))
    rig = next(o for o in bpy.data.objects if builder.is_kinema_rig(o))
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    return rig


@pytest.fixture
def moved(tmp_path):
    """A description copied somewhere, to build from and then take away."""

    def copy(fixture):
        path = tmp_path / "gone" / fixture.name
        path.parent.mkdir(exist_ok=True)
        shutil.copy(fixture, path)
        return path

    return copy


def _limits(modules, rig) -> dict:
    model = modules("solver.rig_model").model_from_rig(rig)
    return {j.name: (j.lower, j.upper, j.velocity) for j in model.actuated_joints}


def _edited(tmp_path, fixture, old, new):
    text = fixture.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path = tmp_path / "edited" / fixture.name
    path.parent.mkdir(exist_ok=True)
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def _drawn(modules) -> list:
    """What the Kinema panel draws, as ("label", text) and ("operator", idname)."""
    import types

    import bpy

    log = []

    class Layout:
        use_property_split = enabled = alert = active = False
        scale_y = 1.0
        alignment = "EXPAND"

        def row(self, **_):
            return self

        column = box = split = row

        def separator(self, **_):
            pass

        def panel(self, *_args, **_kwargs):
            return self, None

        def label(self, *, text="", icon="NONE"):
            log.append(("label", text))

        def prop(self, *_args, **_kwargs):
            pass

        def operator(self, idname, **_):
            log.append(("operator", idname))
            return types.SimpleNamespace()

    modules("ui.panel").KINEMA_PT_main.draw(types.SimpleNamespace(layout=Layout()), bpy.context)
    return log


class TestLocating:
    def test_the_panel_offers_it_where_the_file_is_missing(
        self, modules, fixture_dir, clean_scene, moved
    ):
        copy = moved(fixture_dir / "arm6.urdf")
        _build(modules, copy)
        offer = ("operator", "kinema.locate_description")
        assert offer not in _drawn(modules), "precondition: not while the file is there"
        copy.unlink()
        drawn = _drawn(modules)
        assert offer in drawn
        assert ("label", "Description file missing") in drawn


    def test_a_missing_description_is_found_again(
        self, modules, fixture_dir, clean_scene, moved
    ):
        import bpy

        builder, locate = modules("rig.builder"), modules("ops.locate")
        manager = modules("solver.manager")
        copy = moved(fixture_dir / "arm6.urdf")
        rig = _build(modules, copy)
        assert not locate.description_missing(rig), "precondition: where it was"
        copy.unlink()
        assert locate.description_missing(rig)
        with pytest.raises(manager.SolverError, match="missing.*Locate Description"):
            manager._load_source_urdf(rig)

        manager._cache[rig.name] = object()
        target = fixture_dir / "arm6.urdf"
        assert bpy.ops.kinema.locate_description(filepath=str(target)) == {"FINISHED"}
        assert rig[builder.PROP_SOURCE] == str(target)
        assert not locate.description_missing(rig)
        assert rig.name not in manager._cache, "the solver built from the missing file stays"
        assert manager._load_source_urdf(rig) is not None

    def test_another_robot_is_refused(self, modules, fixture_dir, clean_scene, moved):
        import bpy

        builder = modules("rig.builder")
        copy = moved(fixture_dir / "arm6.urdf")
        rig = _build(modules, copy)
        copy.unlink()
        with pytest.raises(RuntimeError, match="Not the robot this rig was built from"):
            bpy.ops.kinema.locate_description(filepath=str(fixture_dir / "arm3.urdf"))
        assert rig[builder.PROP_SOURCE] == str(copy)

    def test_a_changed_robot_is_refused(
        self, modules, fixture_dir, clean_scene, moved, tmp_path
    ):
        import bpy

        builder = modules("rig.builder")
        copy = moved(fixture_dir / "arm6.urdf")
        rig = _build(modules, copy)
        copy.unlink()
        changed = _edited(
            tmp_path, fixture_dir / "arm6.urdf",
            '<origin xyz="0.4 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>',
            '<origin xyz="0.4 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>',
        )
        with pytest.raises(RuntimeError, match="'joint3' turns about an axis"):
            bpy.ops.kinema.locate_description(filepath=str(changed))
        assert rig[builder.PROP_SOURCE] == str(copy)

    def test_limits_that_differ_are_accepted_and_kept(
        self, modules, fixture_dir, clean_scene, moved, tmp_path
    ):
        import bpy

        builder = modules("rig.builder")
        copy = moved(fixture_dir / "arm6.urdf")
        rig = _build(modules, copy)
        copy.unlink()
        before = _limits(modules, rig)
        loosened = _edited(
            tmp_path, fixture_dir / "arm6.urdf",
            '<limit lower="-2.5" upper="2.5" effort="150" velocity="3.15"/>',
            '<limit lower="-2.9" upper="2.9" effort="150" velocity="2.0"/>',
        )
        assert bpy.ops.kinema.locate_description(filepath=str(loosened)) == {"FINISHED"}
        assert rig[builder.PROP_SOURCE] == str(loosened)
        assert _limits(modules, rig) == before

    def test_an_mjcf_is_found_again(self, modules, fixture_dir, clean_scene, moved):
        import bpy

        builder, manager = modules("rig.builder"), modules("solver.manager")
        copy = moved(fixture_dir / "mjcf_arm.xml")
        rig = _build(modules, copy)
        assert rig[builder.PROP_SOURCE_KIND] == "mjcf", "precondition: an MJCF rig"
        copy.unlink()
        with pytest.raises(manager.SolverError, match="missing"):
            manager._load_source_urdf(rig)
        target = fixture_dir / "mjcf_arm.xml"
        assert bpy.ops.kinema.locate_description(filepath=str(target)) == {"FINISHED"}
        assert (rig[builder.PROP_SOURCE_KIND], rig[builder.PROP_SOURCE]) == ("mjcf", str(target))

    def test_a_rig_with_an_external_axis_has_nothing_to_locate(
        self, modules, fixture_dir, clean_scene, moved, monkeypatch
    ):
        import bpy

        locate, rig_model = modules("ops.locate"), modules("solver.rig_model")
        copy = moved(fixture_dir / "arm6.urdf")
        rig = _build(modules, copy)
        copy.unlink()
        monkeypatch.setattr(rig_model, "has_external_axes", lambda _rig: True)
        assert not locate.description_missing(rig)
        with pytest.raises(RuntimeError, match="external axis"):
            bpy.ops.kinema.locate_description(filepath=str(fixture_dir / "arm6.urdf"))
