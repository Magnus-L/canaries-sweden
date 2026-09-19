#!/usr/bin/env python3
"""test_tee.py -- the log must survive a blocked terminal."""
import os, sys, tempfile
from pathlib import Path
os.environ["CANARIES_DRYRUN"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mona"))
import mona_common as mc
FAILS = []


def check(n, c, d=""):
    print(("PASS " if c else "FAIL ") + n + (f"  [{d}]" if d else ""))
    if not c:
        FAILS.append(n)


class Blocked:
    """A stdout that raises once a pipe's worth has been written, standing in
    for BatchClient's pipe, which blocks instead."""
    def __init__(self, limit=4096):
        self.n, self.limit = 0, limit
    def write(self, s):
        self.n += len(s)
        if self.n > self.limit:
            raise BlockingIOError("pipe full")
        return len(s)
    def flush(self):
        pass


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    real, blocked = sys.stdout, Blocked()
    sys.stdout = blocked
    log = tmp / "t_log.txt"
    # pin the cap in the test: inheriting CANARIES_ECHO_LIMIT from the shell
    # (which the other local tests raise so their output stays readable) would
    # silently disable the very thing under test
    mc.Tee.TERMINAL_ECHO_LIMIT = 2048
    t = mc.Tee(log)
    for i in range(2000):
        print(f"line {i} " + "y" * 60)
    sys.stdout = real
    text = log.read_text()
    check("every line reached the log despite the blocked terminal",
          "line 1999" in text, f"log {len(text)} bytes")
    check("the echo stopped before the pipe would have blocked",
          blocked.n <= 4096, f"{blocked.n} bytes written to the terminal")
    check("the log says the echo was capped", "echo capped" in text)
    check("the log is much larger than the echo", len(text) > 10 * blocked.n,
          f"log {len(text)} vs echo {blocked.n}")
    print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nALL PASS")
    sys.exit(1 if FAILS else 0)
