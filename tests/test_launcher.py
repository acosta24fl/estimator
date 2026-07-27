"""Static checks on MNQ.bat.

The launcher is the one file that never runs in CI - it only ever executes on
the user's Windows box, where a mistake costs a round trip. These tests catch
the failure modes that are invisible when reading the file on Linux.
"""

import re
from pathlib import Path

import pytest

BAT = Path(__file__).resolve().parent.parent / "MNQ.bat"


@pytest.fixture(scope="module")
def lines():
    return BAT.read_text(encoding="utf-8", errors="replace").splitlines()


def _strip_comment(line):
    stripped = line.strip()
    if stripped.lower().startswith("rem ") or stripped.lower() == "rem":
        return ""
    return line


def test_no_variable_read_in_the_block_that_sets_it(lines):
    """Guard the parse-time expansion trap.

    cmd.exe expands every %VAR% in a parenthesised block once, when it parses
    the block - so a variable assigned inside the block still reads as its
    pre-block value on later lines of that same block. This silently turned
    "%PYCMD% -m venv" into "-m venv", which reported the baffling error
    "'-m' is not recognized as an internal or external command".

    Setting a variable inside a block is fine. Reading it back there is not.
    """
    depth = 0
    assigned = []  # one set of names per open block
    offenders = []

    for number, raw in enumerate(lines, start=1):
        line = _strip_comment(raw)

        for name in re.findall(r"%(\w+)%", line):
            for scope in assigned:
                if name.upper() in scope:
                    offenders.append(f"  line {number}: %{name}% - {raw.strip()}")
                    break

        opens = line.count("(") - line.count(")")

        match = re.search(r"\bset\s+\"?(\w+)=", line, re.IGNORECASE)
        if match and depth + max(opens, 0) > 0:
            target = assigned[-1] if assigned else None
            if opens > 0 and target is None:
                pass  # assignment on the line that opens the block
            elif target is not None:
                target.add(match.group(1).upper())

        for _ in range(max(opens, 0)):
            assigned.append(set())
        for _ in range(max(-opens, 0)):
            if assigned:
                assigned.pop()

        depth = max(depth + opens, 0)

    assert not offenders, (
        "Variable read inside the same parenthesised block that assigns it; "
        "cmd.exe will expand it to its old value. Use labels and goto "
        "instead:\n" + "\n".join(offenders)
    )


def test_every_goto_has_a_label(lines):
    labels = set()
    for raw in lines:
        match = re.match(r"\s*:(\w+)", raw)
        if match:
            labels.add(match.group(1).lower())

    missing = []
    for number, raw in enumerate(lines, start=1):
        for target in re.findall(r"\bgoto\s+:?(\w+)", _strip_comment(raw), re.IGNORECASE):
            if target.lower() not in labels and target.lower() != "eof":
                missing.append(f"  line {number}: goto :{target}")

    assert not missing, "goto jumps to a label that does not exist:\n" + "\n".join(missing)


def test_every_menu_choice_is_wired_up(lines):
    """A number offered in the menu must be dispatched, and vice versa."""
    text = "\n".join(lines)

    offered = set(re.findall(r"^\s*echo\s+(\d)\s{2,}\S", text, re.MULTILINE))
    offered |= set(re.findall(r"^\s*echo\s+(\d)\s+\w", text, re.MULTILINE))
    dispatched = set(re.findall(r'if\s+"%CHOICE%"=="(\d)"', text))

    assert offered, "no menu entries found - has the menu format changed?"
    assert offered == dispatched, (
        f"menu offers {sorted(offered)} but dispatches {sorted(dispatched)}"
    )


def test_python_is_invoked_through_the_venv_after_setup(lines):
    """Every mnq command must use the venv interpreter, not bare `python`."""
    offenders = []
    for number, raw in enumerate(lines, start=1):
        line = _strip_comment(raw).strip()
        if re.search(r"(^|\s)(python|py)\s+-m\s+(mnq|pytest)", line, re.IGNORECASE):
            offenders.append(f"  line {number}: {line}")

    assert not offenders, (
        "use \"%VPY%\" so the command runs inside the environment:\n"
        + "\n".join(offenders)
    )


def test_paths_with_spaces_and_parentheses_are_quoted(lines):
    """The user's folder is literally 'mnqtradingsystem (8)'."""
    unquoted = []
    for number, raw in enumerate(lines, start=1):
        line = _strip_comment(raw)
        if re.match(r"^\s*echo\b", line, re.IGNORECASE):
            continue  # display only, never executed as a path
        for var in ("VPY", "VENV", "MNQ_HOME", "REQ_MARKER"):
            for match in re.finditer(rf"%{var}%", line):
                before = line[: match.start()]
                after = line[match.end() :]
                if before.count('"') % 2 == 1:
                    continue  # already inside a quoted string
                if after.startswith('"') or before.endswith('"'):
                    continue
                if re.match(r"^\s*(set|if)\b", line.strip(), re.IGNORECASE):
                    continue
                unquoted.append(f"  line {number}: {line.strip()}")

    assert not unquoted, (
        "path variable used unquoted - breaks on spaces and parentheses:\n"
        + "\n".join(unquoted)
    )
