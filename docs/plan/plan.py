"""Remaining work: items, owners, dependencies and the critical path to the poster.

    uv run python docs/plan/plan.py              # redraw docs/plan/schedule.svg
    uv run python docs/plan/plan.py --checklist  # print the README checklist

The README checklist is the record of what is done. Tick an item there
("- [x] **I1** ...") and rerun this script: the chart greys it out and
recomputes the critical path. Items, owners, dependencies and hours live in
TASKS below. No dependencies beyond the standard library.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
README = HERE.parents[1] / "README.md"
POSTER = "P4"  # the item whose finish is the poster deadline
OWNERS = ("Andrew", "Luke", "Bao", "Justin", "Everyone")


@dataclass
class Task:
    id: str
    owner: str
    name: str
    deps: tuple[str, ...]
    hands: float  # a person's hours: prompting, reviewing, launching, checking
    machine: float = 0.0  # unattended compute hours on one M4 Pro laptop
    done: bool = False
    es: float = 0.0
    ef: float = 0.0
    lf: float | None = None  # None: only feeds the report, not the poster
    extra: dict = field(default_factory=dict)

    @property
    def dur(self) -> float:
        return 0.0 if self.done else self.hands + self.machine

    @property
    def slack(self) -> float | None:
        return None if self.lf is None else self.lf - self.ef

    @property
    def critical(self) -> bool:
        return not self.done and self.lf is not None and self.lf - self.ef < 1e-6


# Machine hours are extrapolated from a 1,350-question MuSiQue run (28 Nov 2026).
# G3 is a guess: no language model has been timed.
TASKS = [
    Task("I1", "Andrew", "Merge Bao's branch into master", (), 0.5),
    Task("I2", "Andrew", "Merge the cache and hop-count fixes", ("I1",), 0.5),
    Task("I3", "Andrew", "Full-run script with resume", ("I2",), 1, 0.2),
    Task("M1", "Andrew", "MuSiQue: embeddings and baselines", ("I3",), 0.2, 0.7),
    Task("M2", "Andrew", "MuSiQue: cross-encoder, train and score", ("M1",), 0.3, 1.1),
    Task("M3", "Andrew", "MuSiQue: GNN, no-edge GNN, hybrid", ("M2",), 0.2, 0.1),
    Task("M4", "Andrew", "MuSiQue results table, scores to Drive", ("M3",), 0.5),
    Task("O1", "Andrew", "HotpotQA and 2Wiki: embeddings", ("M3",), 0.2, 3.7),
    Task("O2", "Andrew", "HotpotQA and 2Wiki: cross-encoder", ("O1",), 0.3, 7.1),
    Task("O3", "Andrew", "HotpotQA and 2Wiki: GNN, hybrid, tables", ("O2",), 0.5, 0.3),
    Task("G3", "Andrew", "Generate answers, 500 MuSiQue questions", ("G2", "M4"), 0.3, 1.2),
    Task("Q1", "Luke", "RQ2: connectivity definition and code", ("I1",), 1),
    Task("Q2", "Luke", "RQ3: edge-noise sweep code", ("Q1",), 1),
    Task("Q3", "Luke", "RQ2 on MuSiQue, figure", ("Q2", "M4"), 1, 0.1),
    Task("Q4", "Luke", "RQ3 on MuSiQue, figure", ("Q3",), 1, 0.2),
    Task("Q5", "Luke", "RQ2 and RQ3 on the other two datasets", ("Q4", "O3"), 1, 0.2),
    Task("P1", "Bao", "Poster layout, intro, method, data panels", (), 2),
    Task("P2", "Bao", "Poster results panels, MuSiQue", ("P1", "M4", "Q3", "Q4", "C1", "G4"), 2),
    Task("P3", "Bao", "Add HotpotQA and 2Wiki to the poster", ("P2", "O3", "Q5"), 1),
    Task("G1", "Justin", "Choose the answer model and sample size", (), 0.5),
    Task("G2", "Justin", "Generation and EM/F1 code", ("G1", "I1"), 1.5, 0.3),
    Task("C1", "Justin", "Qualitative cases and graph pictures", ("M4",), 1.5),
    Task("G4", "Justin", "Answer metrics table", ("G3",), 0.5),
    Task("W1", "Justin", "Report: method, data, related work", (), 3),
    Task("P4", "Everyone", "Review, export, submit the poster", ("P3",), 1),
    Task("W2", "Everyone", "Report: results and discussion", ("P4", "W1", "O3", "Q5"), 3),
    Task("W3", "Everyone", "Report: edit and submit", ("W2",), 2),
]


def read_done(readme: Path = README) -> set[str]:
    """Item ids ticked in the README checklist."""
    if not readme.exists():
        return set()
    return set(re.findall(r"^- \[[xX]\] \*\*([A-Z]\d+)\*\*", readme.read_text(), flags=re.M))


def schedule(tasks: list[Task], done: set[str]) -> tuple[list[Task], float]:
    """Critical-path method: earliest start and finish, then slack against the poster."""
    by = {t.id: t for t in tasks}
    for t in tasks:
        t.done = t.id in done
    order: list[Task] = []
    seen: set[str] = set()

    def visit(t: Task) -> None:
        if t.id in seen:
            return
        for d in t.deps:
            visit(by[d])
        seen.add(t.id)
        order.append(t)

    for t in tasks:
        visit(t)
    for t in order:
        t.es = max([0.0] + [by[d].ef for d in t.deps])
        t.ef = t.es + t.dur
        t.lf = None
    end = by[POSTER].ef
    by[POSTER].lf = end
    for t in reversed(order):
        if t.lf is None:
            continue
        for d in t.deps:
            latest = t.lf - t.dur
            by[d].lf = latest if by[d].lf is None else min(by[d].lf, latest)
    return order, end


def hours(v: float) -> str:
    return f"{round(v, 1):g} h"


def draw(tasks: list[Task], order: list[Task], end: float) -> str:
    width, x0, row = 1000, 340, 22
    total = max(t.ef for t in tasks)
    step = 4 if total > 30 else 2
    span = max(step, -(-total // step) * step)
    plot = width - x0 - 20

    def x(v: float) -> float:
        return round(x0 + plot * v / span, 1)

    rows: list[Task | str] = []
    for owner in OWNERS:
        mine = sorted((t for t in tasks if t.owner == owner), key=lambda t: (t.es, t.id))
        if mine:
            left = sum(t.hands for t in mine if not t.done)
            rows.append(f"{owner.upper()} · {hours(left)} HANDS-ON LEFT")
            rows.extend(mine)
    top = 34
    height = top + len(rows) * row + 66
    critical = [t.id for t in order if t.critical]
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 12px; fill: #1f2328; }',
        ".mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11px; }",
        ".note { font-size: 11px; fill: #59636e; }",
        ".lane { font-size: 10.5px; font-weight: 700; fill: #59636e; letter-spacing: 0.06em; }",
        ".strong { font-weight: 700; }",
        ".faded { fill: #8c959f; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    v = 0
    while v <= span:
        out.append(f'<line x1="{x(v)}" x2="{x(v)}" y1="{top - 6}" y2="{top + len(rows) * row}" stroke="#eaeef2"/>')
        out.append(f'<text x="{x(v)}" y="{top - 12}" text-anchor="middle" class="mono note">{v:g} h</text>')
        v += step
    for i, r in enumerate(rows):
        y = top + i * row
        if isinstance(r, str):
            out.append(f'<rect x="0" y="{y + 2}" width="{width}" height="{row - 4}" fill="#f6f8fa"/>')
            out.append(f'<text x="8" y="{y + 15}" class="lane">{escape(r)}</text>')
            continue
        cls = "faded" if r.done else "strong" if r.critical else ""
        out.append(f'<text x="8" y="{y + 15}" class="mono {cls}">{r.id}</text>')
        out.append(f'<text x="38" y="{y + 15}" class="{cls}">{escape(r.name)}</text>')
        if r.done:
            out.append(f'<text x="{x(r.es) + 4}" y="{y + 15}" class="note">✓ done</text>')
            continue
        if r.lf is not None and not r.critical:
            out.append(f'<line x1="{x(r.ef)}" x2="{x(r.lf)}" y1="{y + 11}" y2="{y + 11}" stroke="#8c959f" stroke-width="1.5" stroke-dasharray="2 3"/>')
        fill = "#1f2328" if r.critical else "#afb8c1"
        out.append(f'<rect x="{x(r.es)}" y="{y + 5}" width="{max(3, round(x(r.ef) - x(r.es), 1))}" height="{row - 10}" rx="2" fill="{fill}"/>')
    bottom = top + len(rows) * row
    out.append(f'<line x1="{x(end)}" x2="{x(end)}" y1="{top - 6}" y2="{bottom}" stroke="#1f2328" stroke-width="1.5" stroke-dasharray="5 4"/>')
    ly = bottom + 22
    out += [
        f'<rect x="8" y="{ly - 9}" width="26" height="10" rx="2" fill="#1f2328"/><text x="40" y="{ly}" class="note">critical</text>',
        f'<rect x="100" y="{ly - 9}" width="26" height="10" rx="2" fill="#afb8c1"/><text x="132" y="{ly}" class="note">has slack</text>',
        f'<line x1="206" x2="232" y1="{ly - 4}" y2="{ly - 4}" stroke="#8c959f" stroke-width="1.5" stroke-dasharray="2 3"/><text x="238" y="{ly}" class="note">how far it can slip</text>',
        f'<line x1="356" x2="382" y1="{ly - 4}" y2="{ly - 4}" stroke="#1f2328" stroke-width="1.5" stroke-dasharray="5 4"/><text x="388" y="{ly}" class="note">poster ready: {hours(end)} of back-to-back work and compute from now</text>',
        f'<text x="8" y="{ly + 22}" class="note">Critical path: <tspan class="mono">{" → ".join(critical)}</tspan></text>',
        "</svg>",
    ]
    return "\n".join(out) + "\n"


def checklist(tasks: list[Task]) -> str:
    lines = []
    for owner in OWNERS:
        mine = [t for t in tasks if t.owner == owner]
        lines.append(f"**{owner}**\n")
        for t in mine:
            waits = ", ".join(t.deps) or "nothing"
            time = hours(t.hands) + (f" + {hours(t.machine)} machine" if t.machine else "")
            lines.append(f"- [{'x' if t.done else ' '}] **{t.id}** {t.name} · waits on {waits} · {time}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    order, end = schedule(TASKS, read_done())
    if "--checklist" in sys.argv:
        print(checklist(TASKS))
        return
    (HERE / "schedule.svg").write_text(draw(TASKS, order, end))
    left = [t for t in TASKS if not t.done and t.lf is not None]
    print(f"wrote {HERE / 'schedule.svg'}")
    print(f"poster ready after {hours(end)} | hands-on left {hours(sum(t.hands for t in left))} | machine left {hours(sum(t.machine for t in left))}")
    print("critical path:", " -> ".join(t.id for t in order if t.critical))


if __name__ == "__main__":
    main()
