"""Xacro substitution arguments: reading them, and asking the user for them.

A xacro can require arguments, and one of the most widely used descriptions
does exactly that. ``ur.urdf.xacro`` opens::

    <robot xmlns:xacro="http://wiki.ros.org/xacro" name="$(arg name)">
       <xacro:arg name="name" default="ur"/>

Line 2 uses the argument that line 4 declares, and xacro evaluates the element's
attributes before it reaches the declaration -- so the default never applies and
the file cannot be rendered without ``name:=something``. In ROS it is always
invoked as ``xacro ur.urdf.xacro ur_type:=ur5e name:=ur5e``, which is why
upstream never trips over it.

That has a consequence worth stating, because it is not obvious: **"required"
cannot be derived from "has no default".** Every argument in that file has one.
So this module does not try to guess which arguments are needed; it reports what
a file *declares*, and the import surfaces whichever one xacro actually asked
for.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from xml.etree import ElementTree

#: How ROS spells an argument on the command line, and what the import field
#: takes: ``name:=ur5e ur_type:=ur5e``.
_ASSIGNMENT = ":="


def _tokenise(text: str) -> list[str]:
    """Split on whitespace, honouring quotes, without eating backslashes.

    ``str.split`` cannot do the first: ``prefix:="left arm "`` becomes three
    tokens and the value silently truncates to ``"left``.

    ``shlex`` can, but two of its defaults are wrong for this. Its escape
    character is the backslash, which turns
    ``joint_limit_params:=C:\\ws\\config.yaml`` into ``C:wsconfig.yaml`` -- not
    hypothetical, since four of the Universal Robots arguments take file paths.
    And it treats ``#`` as starting a comment, which xacro does not: left on,
    ``color:=#ff0000`` arrives empty and ``label:=foo#bar`` truncates to
    ``foo``.

    Both cleared. The cost is no way to escape a quote inside a value, which no
    xacro argument has ever needed.
    """
    lexer = shlex.shlex(text or "", posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    # `#` starts a comment for shlex, and does not for xacro. Left on,
    # `color:=#ff0000` arrives as an empty value and `label:=foo#bar` truncates
    # to `foo` -- both silently, and both valid on the real command line.
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        # An unbalanced quote. The import will report the missing argument,
        # which is more useful than complaining about the field's syntax.
        return (text or "").split()


def parse_args(text: str) -> dict[str, str]:
    """Parse ``name:=value`` pairs, in the syntax the xacro CLI uses.

    Quote a value that contains spaces, as on the command line. An entry
    without ``:=`` is ignored rather than guessed at.
    """
    parsed: dict[str, str] = {}
    for token in _tokenise(text):
        name, sep, value = token.partition(_ASSIGNMENT)
        if sep and name:
            parsed[name] = value
    return parsed


def format_args(args: dict[str, str]) -> str:
    """The inverse, for storing on a rig and showing it back."""
    return " ".join(f"{name}{_ASSIGNMENT}{value}" for name, value in args.items())


def declared_args(path: str | Path) -> dict[str, str]:
    """Every ``<xacro:arg>`` a file declares, name -> default (may be empty).

    Matched on the tag's **local name**. The namespace URI is not fixed in
    practice -- ``http://wiki.ros.org/xacro`` in the Universal Robots
    descriptions, ``http://www.ros.org/wiki/xacro`` in KUKA's -- so keying on a
    literal namespace finds every argument in one vendor's files and none in the
    other's.

    Only this file, not the ones it includes: reaching those means rendering,
    which is the thing that has already failed by the time anyone wants this
    list. Insertion-ordered, so the display matches the file.

    Never raises. A malformed or unreadable file yields nothing and lets the
    import report its own error, which will be more specific than anything here.
    """
    try:
        root = ElementTree.parse(Path(path)).getroot()
    except (ElementTree.ParseError, OSError, ValueError):
        return {}

    found: dict[str, str] = {}
    for element in root.iter():
        tag = element.tag
        if not isinstance(tag, str) or tag.rsplit("}", 1)[-1] != "arg":
            continue
        name = element.get("name")
        if name:
            found.setdefault(name, element.get("default", ""))
    return found


def missing_argument(error: str) -> str | None:
    """The argument name out of xacro's complaint, if that is what it is.

    xacro says ``Undefined substitution argument name``, which tells the user
    almost nothing on its own -- "name" reads as a noun. Pulling it out lets the
    import say which argument, and offer the rest of the file's declarations
    beside it.
    """
    marker = "Undefined substitution argument"
    index = error.find(marker)
    if index == -1:
        return None
    remainder = error[index + len(marker):].strip()
    return remainder.split()[0] if remainder else None


def describe(path: str | Path, error: str) -> str:
    """Turn a failed render into something a user can act on."""
    declared = declared_args(path)
    missing = missing_argument(error)

    if missing is None and not declared:
        return error

    lines = []
    if missing:
        lines.append(
            f"'{Path(path).name}' needs the xacro argument '{missing}', which "
            "has no usable value."
        )
    else:
        lines.append(error)

    if declared:
        lines.append("")
        lines.append("Arguments this file declares:")
        for name, default in declared.items():
            shown = default if default else '""'
            lines.append(f"    {name} = {shown}")
        example = " ".join(f"{name}:=…" for name in list(declared)[:2])
        lines.append("")
        lines.append(f"Set them in Import Options -> Xacro Arguments, e.g. {example}")
        lines.append(
            "A default shown here may still be too late to apply: a file that "
            "uses an argument above the line that declares it, as the Universal "
            "Robots descriptions do, must be given one."
        )
    return "\n".join(lines)
