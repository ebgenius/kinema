"""What an action did, for an operator to report or a script to read (#102).

The work of the operators that kinema.api shares -- generating a job, checking
it, adding an IK target -- is done by plain functions returning an
:class:`Outcome`. An operator reports it; the API hands it on. Neither needs the
other: a function needs no UI context to run, and nothing is said twice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Outcome:
    """Whether it was done, how to report it, and what to say. ``flagged`` is how
    many moves the Motion Check flagged, where it ran."""

    done: bool
    level: str
    message: str
    flagged: int = 0


def reported(operator, outcome: Outcome) -> set[str]:
    """``outcome`` reported by ``operator``, and its return value."""
    operator.report({outcome.level}, outcome.message)
    return {"FINISHED"} if outcome.done else {"CANCELLED"}
