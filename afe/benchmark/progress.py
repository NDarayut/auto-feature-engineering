"""Terminal progress for a benchmark run.

A full sweep is 22 datasets x N methods and takes hours; without output it
is indistinguishable from a hang. This prints one line per (dataset, method)
as it completes -- status, generation time, and the score from each model
family -- so a run can be watched, and a method that is failing everywhere
is obvious immediately rather than at the end.

Everything goes to stderr, leaving stdout clean for anything a caller pipes
(a report written to stdout, for instance).
"""

from __future__ import annotations

import sys
import time

# Status markers kept ASCII: these lines land in nohup logs and CI output
# where a non-UTF-8 locale would turn fancier glyphs into mojibake.
_MARK = {"ok": "ok", "timeout": "TIMEOUT", "oom": "OOM",
         "crashed": "CRASHED", "error": "ERROR", "model_error": "MODEL-ERR"}


def _fmt_secs(s: float | None) -> str:
    if s is None:
        return "-"
    return f"{s / 60:.1f}m" if s >= 120 else f"{s:.1f}s"


class ProgressReporter:
    """Print one line per completed (dataset, method) pair.

    ``total`` is the number of pairs the run will attempt, used only for the
    ``[i/N]`` counter. Pass ``enabled=False`` for a silent run.
    """

    def __init__(self, total: int | None = None, enabled: bool = True,
                 stream=None):
        self.total = total
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.done = 0
        self._t0 = time.time()

    def _write(self, line: str) -> None:
        if not self.enabled:
            return
        print(line, file=self.stream, flush=True)

    def start(self, methods: list[str], datasets: list[str]) -> None:
        def _plural(n: int, word: str) -> str:
            return f"{n} {word}" if n == 1 else f"{n} {word}s"

        self._write(f"benchmarking {_plural(len(methods), 'method')} on "
                    f"{_plural(len(datasets), 'dataset')}"
                    + (f" -- {_plural(self.total, 'pair')}" if self.total else ""))

    def pair_done(self, key: str, method: str, status: str,
                  gen_elapsed_s: float | None = None,
                  scores: dict[str, float] | None = None,
                  error: str | None = None) -> None:
        """Report one finished (dataset, method) pair."""
        self.done += 1
        counter = f"[{self.done}/{self.total}]" if self.total else f"[{self.done}]"
        head = f"{counter} {key} / {method}"
        mark = _MARK.get(status, status)
        if status == "ok":
            detail = "  ".join(f"{fam} {val:.3f}"
                               for fam, val in sorted((scores or {}).items()))
            self._write(f"{head}: {mark} in {_fmt_secs(gen_elapsed_s)}"
                        + (f" -- {detail}" if detail else ""))
        else:
            # Failures are the reason to be watching; keep the message, but
            # bounded so one huge traceback can't flood the log.
            msg = (error or "").splitlines()[0][:120] if error else ""
            self._write(f"{head}: {mark}" + (f" -- {msg}" if msg else ""))

    def finish(self, n_rows: int, out_path=None, report_path=None) -> None:
        elapsed = _fmt_secs(time.time() - self._t0)
        self._write(f"done: {n_rows} result rows in {elapsed}"
                    + (f" -> {out_path}" if out_path else ""))
        if report_path:
            self._write(f"report: {report_path}")
