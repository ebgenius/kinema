"""Xacro substitution arguments, and the package identity they depend on.

Both come from one report: `ur.urdf.xacro` would not import. Three defects were
stacked behind that single error, and the fixtures here reproduce each shape
without needing the 700 MB description checked out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ..conftest import load_addon_module

xacro_args = load_addon_module("io.xacro_args")
resolve = load_addon_module("io.resolve")
loader = load_addon_module("io.loader")

FIXTURE_WS = Path(__file__).resolve().parents[1] / "fixtures" / "ros_ws"
RENAMED = FIXTURE_WS / "Renamed_Package_Dir"
NEEDS_ARGS = RENAMED / "urdf" / "needs_args.urdf.xacro"


class TestParsing:
    def test_ros_assignment_syntax(self):
        assert xacro_args.parse_args("name:=ur5e ur_type:=ur5e") == {
            "name": "ur5e",
            "ur_type": "ur5e",
        }

    def test_empty_input(self):
        assert xacro_args.parse_args("") == {}
        assert xacro_args.parse_args(None) == {}

    def test_an_empty_value_is_still_an_assignment(self):
        """`tf_prefix:=` is how you deliberately pass an empty prefix."""
        assert xacro_args.parse_args("tf_prefix:=") == {"tf_prefix": ""}

    def test_tokens_without_an_assignment_are_ignored(self):
        """Rather than guessed at -- a bare word has no defensible meaning."""
        assert xacro_args.parse_args("name:=ur5e rubbish") == {"name": "ur5e"}

    def test_a_quoted_value_may_contain_spaces(self):
        """`str.split` truncated this to `"left`, silently."""
        assert xacro_args.parse_args('prefix:="left arm "') == {
            "prefix": "left arm "
        }

    def test_quotes_are_stripped_not_passed_through(self):
        assert xacro_args.parse_args('name:="ur5e"') == {"name": "ur5e"}

    def test_a_windows_path_survives(self):
        """The reason the lexer's escape character is cleared. Four of the
        Universal Robots arguments take file paths, and shlex's default escape
        turns C:\\ws\\config.yaml into C:wsconfig.yaml."""
        assert xacro_args.parse_args(
            r"joint_limit_params:=C:\ws\ur\config\joint_limits.yaml"
        ) == {"joint_limit_params": r"C:\ws\ur\config\joint_limits.yaml"}

    def test_a_quoted_path_with_spaces(self):
        assert xacro_args.parse_args(
            r'kinematics_params:="C:\Program Files\ur\kin.yaml"'
        ) == {"kinematics_params": r"C:\Program Files\ur\kin.yaml"}

    def test_an_unbalanced_quote_does_not_raise(self):
        """The import's own error about the missing argument is more useful
        than a complaint about the field's syntax."""
        assert xacro_args.parse_args('name:="ur5e') == {"name": '"ur5e'}

    def test_round_trip(self):
        text = "name:=ur5e ur_type:=ur5e"
        assert xacro_args.format_args(xacro_args.parse_args(text)) == text


class TestDeclaredArgs:
    def test_it_reads_names_and_defaults(self):
        found = xacro_args.declared_args(NEEDS_ARGS)
        assert found == {"name": "unused_default", "segment_length": "0.4"}

    def test_the_namespace_uri_is_not_assumed(self):
        """`http://wiki.ros.org/xacro` in the UR descriptions,
        `http://www.ros.org/wiki/xacro` in KUKA's. Keying on a literal namespace
        finds every argument in one vendor's files and none in the other's."""
        other = FIXTURE_WS / "fixture_robot" / "urdf" / "arm.urdf.xacro"
        assert "prefix" in xacro_args.declared_args(other)

    def test_a_missing_or_broken_file_yields_nothing(self, tmp_path):
        assert xacro_args.declared_args(tmp_path / "nope.xacro") == {}
        broken = tmp_path / "broken.xacro"
        broken.write_text("<robot", encoding="utf-8")
        assert xacro_args.declared_args(broken) == {}


class TestErrorMessage:
    def test_it_names_the_argument_xacro_asked_for(self):
        error = "Undefined substitution argument name"
        assert xacro_args.missing_argument(error) == "name"

    def test_an_unrelated_error_is_left_alone(self):
        assert xacro_args.missing_argument("something else entirely") is None
        assert xacro_args.describe(NEEDS_ARGS, "boom") .startswith("boom")

    def test_the_description_lists_what_the_file_declares(self):
        message = xacro_args.describe(
            NEEDS_ARGS, "Undefined substitution argument name"
        )
        assert "'name'" in message
        assert "segment_length" in message
        assert "Xacro Arguments" in message


class TestPackageIdentity:
    """The directory name is not the package name, and assuming so is a bug."""

    def test_the_declared_name_is_read(self):
        assert resolve.declared_package_name(RENAMED) == "fixture_renamed"

    def test_a_non_ascii_package_xml_does_not_break_it(self):
        """The fixture's maintainer name is non-ASCII on purpose. Reading it with
        the system codec rather than parsing it as XML is what killed the UR
        import on Windows with 'charmap' codec can't decode byte 0x9d."""
        assert resolve.declared_package_name(RENAMED) is not None

    def test_both_names_are_indexed(self):
        """Descriptions in the wild reference either, so both must resolve."""
        index = resolve._index_packages(FIXTURE_WS)
        assert index.get("fixture_renamed") == RENAMED
        assert index.get("Renamed_Package_Dir") == RENAMED

    def test_a_package_without_a_declaration_keeps_its_directory_name(self):
        index = resolve._index_packages(FIXTURE_WS)
        assert "fixture_common" in index

    @pytest.mark.parametrize("content", ["", "<not-xml", "<package/>"])
    def test_an_unusable_package_xml_falls_back(self, tmp_path, content):
        package = tmp_path / "some_dir"
        package.mkdir()
        (package / "package.xml").write_text(content, encoding="utf-8")
        assert resolve.declared_package_name(package) is None
        assert resolve.package_names(package) == ["some_dir"]


class TestImportingWithArgs:
    def test_it_fails_without_the_argument(self):
        result = loader.load_file(NEEDS_ARGS)
        assert result.error is not None
        assert "'name'" in result.error
        assert "segment_length" in result.error

    def test_it_imports_with_the_argument(self):
        result = loader.load_file(NEEDS_ARGS, xacro_args={"name": "fixture_arm"})
        assert result.error is None, result.error
        assert result.model.name == "fixture_arm"

    def test_the_argument_reaches_the_render(self):
        """Not just accepted -- actually substituted. The robot's name comes
        from $(arg name), so this is the value arriving where it was needed."""
        result = loader.load_file(NEEDS_ARGS, xacro_args={"name": "chosen_name"})
        assert result.model.name == "chosen_name"

    def test_the_cross_package_include_used_the_declared_name(self):
        """The include is $(find fixture_renamed), the declared name. Resolving
        it means the index read package.xml rather than trusting the folder."""
        result = loader.load_file(NEEDS_ARGS, xacro_args={"name": "x"})
        assert result.error is None
        assert "link_1" in set(result.model.links)
