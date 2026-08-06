"""The launcher scripts are the only files whose *bytes* matter.

cmd.exe does not read a batch file line by line into memory. It seeks to a byte
offset, reads one command, runs it, then seeks to the next stored offset. With
Unix line endings every line is one byte shorter than the parser accounts for,
the offsets drift, and eventually one lands in the middle of a word — so cmd
tries to execute a fragment of a comment. The symptom is baffling
("'erything' is not recognized as an internal or external command") and it
appears only once a file grows past some length, which means an innocent edit
can trigger it long after the real mistake.

Nothing else in this project cares about line endings, so nothing else would
catch it. These tests do.
"""

from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
WINDOWS_SCRIPTS = sorted(PROJECT.glob("*.bat")) + sorted(PROJECT.glob("*.cmd"))
UNIX_SCRIPTS = sorted(PROJECT.glob("*.sh"))


@pytest.mark.parametrize("path", WINDOWS_SCRIPTS, ids=lambda p: p.name)
class TestWindowsLaunchers:
    def test_every_line_ends_crlf(self, path):
        data = path.read_bytes()
        assert data.count(b"\r\n") > 0, f"{path.name} has no CRLF at all"
        assert data.count(b"\n") == data.count(b"\r\n"), (
            f"{path.name} mixes bare LF into CRLF; cmd.exe will mis-seek and "
            "execute a fragment of a comment line"
        )

    def test_it_is_pure_ascii(self, path):
        """The console runs in an OEM codepage, not UTF-8. A smart quote or an
        ellipsis renders as mojibake and shifts every byte offset after it."""
        data = path.read_bytes()
        offenders = [(i, hex(b)) for i, b in enumerate(data) if b > 127]
        assert not offenders, f"{path.name} has non-ASCII bytes at {offenders[:5]}"

    def test_it_does_not_end_mid_line(self, path):
        assert path.read_bytes().endswith(b"\r\n")


@pytest.mark.parametrize("path", UNIX_SCRIPTS, ids=lambda p: p.name)
class TestUnixLaunchers:
    def test_no_carriage_returns(self, path):
        """A CR before the shebang's newline makes the kernel look for an
        interpreter whose name ends in \\r, which does not exist."""
        assert b"\r" not in path.read_bytes()


def test_line_endings_are_pinned_in_git():
    """Without .gitattributes, a clone on another machine can renormalise the
    launchers back to LF and reintroduce the bug silently."""
    attributes = (PROJECT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat text eol=crlf" in attributes
    assert "*.sh  text eol=lf" in attributes or "*.sh text eol=lf" in attributes


def test_both_launchers_exist():
    assert WINDOWS_SCRIPTS and UNIX_SCRIPTS
