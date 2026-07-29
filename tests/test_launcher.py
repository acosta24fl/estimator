"""Static checks on the .bat files.

These are the only files that never run in CI - they execute exclusively on
the user's Windows box, where a mistake costs a round trip measured in hours.
These tests catch the failure modes that are invisible when reading the files
on Linux.

Two files are covered. MNQ.bat is the launcher. UPDATE.bat applies a patch,
and it carries a particular obligation: it is the script that runs *when
something is already wrong*, so it has to fail legibly rather than half-apply
an update and leave the checkout in a state the user cannot describe.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BAT = ROOT / "MNQ.bat"
UPDATE = ROOT / "UPDATE.bat"


def _lines(path):
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


@pytest.fixture(scope="module")
def lines():
    return _lines(BAT)


@pytest.fixture(scope="module")
def update_lines():
    return _lines(UPDATE)


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


class TestUpdater:
    """UPDATE.bat runs when something is already wrong. It must fail legibly."""

    def test_it_does_not_repeat_the_parse_time_expansion_bug(self, update_lines):
        """The same trap that broke MNQ.bat, checked on the new file too."""
        test_no_variable_read_in_the_block_that_sets_it(update_lines)

    def test_every_goto_has_a_label(self, update_lines):
        test_every_goto_has_a_label(update_lines)

    def test_it_refuses_to_run_outside_a_clone(self, update_lines):
        """The actual failure: running from an extracted zip, not a checkout.

        Without this check `git am` prints "not a git repository" and the user
        is left guessing. The script has to name the cause and give the clone
        command.
        """
        text = "\n".join(update_lines)
        assert "git rev-parse --git-dir" in text
        assert "not a git checkout" in text
        assert "git clone" in text

    def test_it_says_the_data_survives_switching_folders(self, update_lines):
        """The reason someone would hesitate to move folders is losing work.

        The models and price history live under MNQ_HOME, not in the checkout,
        so nothing is lost - but that has to be said at the moment the user is
        being asked to switch.
        """
        text = "\n".join(update_lines)
        assert "mnq-data" in text and "Nothing is lost" in text

    def test_a_dirty_tree_is_refused_before_anything_is_touched(self, update_lines):
        """A half-applied patch over local edits is the worst outcome."""
        text = "\n".join(update_lines)
        assert "git diff --quiet" in text
        assert ":dirty" in text
        assert "git stash" in text

    def test_a_failed_apply_is_rolled_back(self, update_lines):
        text = "\n".join(update_lines)
        assert text.count("git am --abort") >= 2, (
            "abort before applying (clearing an earlier half-run) and after a "
            "failure (leaving the checkout clean)"
        )

    def test_quotes_from_a_dragged_path_are_stripped(self, update_lines):
        """Dragging a path with spaces into a console wraps it in quotes."""
        text = "\n".join(update_lines)
        assert 'set "PATCHFILE=%PATCHFILE:"=%"' in text

    def test_it_pulls_before_applying(self, update_lines):
        """Applying onto a stale checkout is the most likely way this fails."""
        text = "\n".join(update_lines)
        assert "git pull --ff-only" in text
        assert text.index("git pull --ff-only") < text.index('git am "%PATCHFILE%"')

    def test_every_exit_path_pauses(self, update_lines):
        """Double-clicked scripts close instantly; an unpaused error is unread."""
        exits = [n for n, l in enumerate(update_lines) if l.strip().startswith("exit /b")]
        for n in exits:
            window = "\n".join(update_lines[max(0, n - 4):n])
            assert "pause" in window, (
                f"line {n + 1}: `exit /b` with no `pause` above it - the window "
                f"would vanish before the message could be read"
            )
