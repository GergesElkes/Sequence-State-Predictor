from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "out.csv"
EDGE_CSV = ROOT / "out_state_machine_edges_clean.csv"
DIAGRAM_MD = ROOT / "out_state_machine_diagram.md"
SUMMARY_MD = ROOT / "out_state_machine_summary.md"

# Keep the exported edge list exhaustive, but hide extremely rare edges in the
# diagram so the main machine remains legible.
DIAGRAM_MIN_COUNT = 50
DIAGRAM_MIN_PROB = 0.005


@dataclass(frozen=True)
class Row:
    csv_line: int
    state_id: int
    state_name: str
    regime: tuple[str, str, str]


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for csv_line, row in enumerate(reader, start=2):
            rows.append(
                Row(
                    csv_line=csv_line,
                    state_id=int(row["current_state_id"]),
                    state_name=row["current_state_name"],
                    regime=tuple(row["regime"].split()),
                )
            )
    return rows


def find_segment_breaks(rows: list[Row]) -> list[int]:
    breaks: list[int] = []
    for idx, (left, right) in enumerate(zip(rows, rows[1:])):
        if left.regime[1:] != right.regime[:2]:
            breaks.append(idx)
    return breaks


def segment_bounds(rows: list[Row], breaks: list[int]) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for idx in breaks:
        bounds.append((start, idx))
        start = idx + 1
    if rows:
        bounds.append((start, len(rows) - 1))
    return bounds


def build_transition_counts(
    rows: list[Row], breaks: set[int]
) -> tuple[Counter[tuple[int, int]], Counter[int]]:
    transitions: Counter[tuple[int, int]] = Counter()
    state_out: Counter[int] = Counter()
    for idx in range(len(rows) - 1):
        if idx in breaks:
            continue
        source = rows[idx].state_id
        target = rows[idx + 1].state_id
        transitions[(source, target)] += 1
        state_out[source] += 1
    return transitions, state_out


def build_dwell_stats(rows: list[Row]) -> dict[int, dict[str, float | int]]:
    runs: defaultdict[int, list[int]] = defaultdict(list)
    if not rows:
        return {}
    current = rows[0].state_id
    run_len = 1
    for row in rows[1:]:
        if row.state_id == current:
            run_len += 1
        else:
            runs[current].append(run_len)
            current = row.state_id
            run_len = 1
    runs[current].append(run_len)

    stats: dict[int, dict[str, float | int]] = {}
    for state_id, values in runs.items():
        sorted_values = sorted(values)
        stats[state_id] = {
            "runs": len(values),
            "mean": mean(values),
            "median": sorted_values[len(sorted_values) // 2],
            "p90": sorted_values[int((len(sorted_values) - 1) * 0.9)],
            "max": max(values),
        }
    return stats


def entropy_from_counts(counts: Counter[int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        value -= p * math.log2(p)
    return value


def write_edge_csv(
    path: Path,
    rows: list[Row],
    transitions: Counter[tuple[int, int]],
    state_out: Counter[int],
) -> None:
    state_name = {row.state_id: row.state_name for row in rows}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_id",
                "source_name",
                "target_id",
                "target_name",
                "transition_count",
                "source_out_count",
                "transition_probability",
                "edge_class",
            ]
        )
        for (source, target), count in sorted(
            transitions.items(),
            key=lambda item: (
                item[0][0],
                -(item[1] / state_out[item[0][0]]),
                -item[1],
                item[0][1],
            ),
        ):
            probability = count / state_out[source]
            edge_class = "rare" if count < DIAGRAM_MIN_COUNT or probability < DIAGRAM_MIN_PROB else "core"
            writer.writerow(
                [
                    source,
                    state_name[source],
                    target,
                    state_name[target],
                    count,
                    state_out[source],
                    f"{probability:.6f}",
                    edge_class,
                ]
            )


def state_node_id(state_id: int, state_name: str) -> str:
    short = state_name.lower().replace(" ", "_")
    return f"s{state_id}_{short}"


def state_style(state_name: str) -> str:
    if state_name.startswith("BULL"):
        return "fill:#d7f2c2,stroke:#2f6b1d,color:#16300d"
    if state_name.startswith("BEAR"):
        return "fill:#f8d3cb,stroke:#8f2f1f,color:#4a160f"
    return "fill:#d9e6f5,stroke:#24507a,color:#12283d"


def write_diagram_md(
    path: Path,
    rows: list[Row],
    transitions: Counter[tuple[int, int]],
    state_out: Counter[int],
    breaks: list[int],
) -> None:
    state_name = {row.state_id: row.state_name for row in rows}
    lines: list[str] = []
    lines.append("# Out.csv State Machine Diagram")
    lines.append("")
    lines.append("This diagram uses cleaned in-segment transitions only.")
    lines.append(
        f"Edges shown below require count >= {DIAGRAM_MIN_COUNT} and probability >= {DIAGRAM_MIN_PROB:.3f}."
    )
    lines.append(f"Detected stitched sequence breaks: {len(breaks)}")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart LR")
    for state_id in sorted(state_name):
        node = state_node_id(state_id, state_name[state_id])
        label = f"{state_id}: {state_name[state_id]}"
        lines.append(f'    {node}["{label}"]')
    lines.append("")
    for state_id in sorted(state_name):
        node = state_node_id(state_id, state_name[state_id])
        lines.append(f"    style {node} {state_style(state_name[state_id])}")
    lines.append("")
    for (source, target), count in sorted(
        transitions.items(),
        key=lambda item: (item[0][0], -(item[1] / state_out[item[0][0]]), -item[1], item[0][1]),
    ):
        probability = count / state_out[source]
        if count < DIAGRAM_MIN_COUNT or probability < DIAGRAM_MIN_PROB:
            continue
        source_node = state_node_id(source, state_name[source])
        target_node = state_node_id(target, state_name[target])
        lines.append(f"    {source_node} -->|{probability:.3f}| {target_node}")
    lines.append("```")
    lines.append("")
    lines.append("## Core Cycles")
    lines.append("")
    lines.append("- Bull loop: `BULL BREAK -> BULL TREND -> BULL PAUSE -> BULL BREAK`")
    lines.append("- Bear loop: `BEAR BREAK -> BEAR TREND -> BEAR PAUSE -> BEAR BREAK`")
    lines.append("- Cross-cycle handoff: `BEAR PAUSE -> BULL BREAK` and `BULL PAUSE -> BEAR BREAK`")
    lines.append("- Neutral gate: `FLATLINE -> BULL BREAK` or `FLATLINE -> BEAR BREAK`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(
    path: Path,
    rows: list[Row],
    transitions: Counter[tuple[int, int]],
    state_out: Counter[int],
    breaks: list[int],
    bounds: list[tuple[int, int]],
) -> None:
    state_name = {row.state_id: row.state_name for row in rows}
    state_counts = Counter(row.state_id for row in rows)
    regime_counts = Counter(" ".join(row.regime) for row in rows)
    dwell = build_dwell_stats(rows)

    conditional_state_entropy = 0.0
    total_edges = sum(state_out.values())
    state_successors: defaultdict[int, Counter[int]] = defaultdict(Counter)
    for (source, target), count in transitions.items():
        state_successors[source][target] = count
    for state_id, counts in state_successors.items():
        conditional_state_entropy += (state_out[state_id] / total_edges) * entropy_from_counts(counts)

    ctx_counts: Counter[tuple[int, str]] = Counter()
    ctx_successors: defaultdict[tuple[int, str], Counter[int]] = defaultdict(Counter)
    for idx in range(len(rows) - 1):
        if idx in set(breaks):
            continue
        key = (rows[idx].state_id, " ".join(rows[idx].regime))
        ctx_counts[key] += 1
        ctx_successors[key][rows[idx + 1].state_id] += 1

    conditional_ctx_entropy = 0.0
    for key, total in ctx_counts.items():
        conditional_ctx_entropy += (total / total_edges) * entropy_from_counts(ctx_successors[key])

    top_state_only_acc = 0
    for idx in range(len(rows) - 1):
        if idx in set(breaks):
            continue
        source = rows[idx].state_id
        target = rows[idx + 1].state_id
        if target == state_successors[source].most_common(1)[0][0]:
            top_state_only_acc += 1

    top_ctx_acc = 0
    for idx in range(len(rows) - 1):
        if idx in set(breaks):
            continue
        key = (rows[idx].state_id, " ".join(rows[idx].regime))
        target = rows[idx + 1].state_id
        if target == ctx_successors[key].most_common(1)[0][0]:
            top_ctx_acc += 1

    bull = sum(count for sid, count in state_counts.items() if state_name[sid].startswith("BULL"))
    bear = sum(count for sid, count in state_counts.items() if state_name[sid].startswith("BEAR"))
    neutral = len(rows) - bull - bear

    lines: list[str] = []
    lines.append("# Out.csv Sequence Reverse-Engineering Summary")
    lines.append("")
    lines.append(f"- Rows: {len(rows):,}")
    lines.append(f"- Distinct states: {len(state_counts)}")
    lines.append(f"- Distinct regime triplets: {len(regime_counts)}")
    lines.append(f"- Stitched sequence breaks: {len(breaks)}")
    lines.append(f"- Segments: {len(bounds)}")
    lines.append(
        f"- Clean transition edges: {len(transitions)} (break-crossing transitions removed)"
    )
    lines.append(f"- Bull / Bear / Neutral share: {bull / len(rows):.3f} / {bear / len(rows):.3f} / {neutral / len(rows):.3f}")
    lines.append("")
    lines.append("## Segment Lengths")
    lines.append("")
    lengths = [end - start + 1 for start, end in bounds]
    lines.append(
        f"- Min / Median / Mean / Max: {min(lengths)} / {sorted(lengths)[len(lengths) // 2]} / {mean(lengths):.1f} / {max(lengths)}"
    )
    lines.append("")
    lines.append("## State Summary")
    lines.append("")
    lines.append("| State | Count | Share | Mean Dwell | Median | P90 | Max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for state_id in sorted(state_name):
        stats = dwell[state_id]
        lines.append(
            f"| {state_id}: {state_name[state_id]} | {state_counts[state_id]:,} | {state_counts[state_id] / len(rows):.3f} | "
            f"{stats['mean']:.3f} | {stats['median']} | {stats['p90']} | {stats['max']} |"
        )
    lines.append("")
    lines.append("## Core Transition Rules")
    lines.append("")
    for state_id in sorted(state_name):
        top_edges = state_successors[state_id].most_common(5)
        formatted = ", ".join(
            f"{state_name[target]} {count / state_out[state_id]:.3f}" for target, count in top_edges
        )
        lines.append(f"- {state_name[state_id]} -> {formatted}")
    lines.append("")
    lines.append("## Predictability")
    lines.append("")
    lines.append(f"- Conditional entropy `H(next_state | current_state)`: {conditional_state_entropy:.4f} bits")
    lines.append(
        f"- Conditional entropy `H(next_state | current_state, regime)`: {conditional_ctx_entropy:.4f} bits"
    )
    lines.append(
        f"- Regime memory gain: {conditional_state_entropy - conditional_ctx_entropy:.4f} bits"
    )
    lines.append(
        f"- Top-1 next-state accuracy from current state only: {top_state_only_acc / total_edges:.4f}"
    )
    lines.append(
        f"- Top-1 next-state accuracy from current state + regime: {top_ctx_acc / total_edges:.4f}"
    )
    lines.append("")
    lines.append("## Dominant Regimes")
    lines.append("")
    for regime, count in regime_counts.most_common(10):
        lines.append(f"- {regime}: {count:,} ({count / len(rows):.3f})")
    lines.append("")
    lines.append("## Break Locations")
    lines.append("")
    for idx in breaks:
        left = rows[idx]
        right = rows[idx + 1]
        lines.append(
            "- "
            f"line {left.csv_line}: {left.state_name} [{ ' '.join(left.regime) }] -> "
            f"line {right.csv_line}: {right.state_name} [{ ' '.join(right.regime) }]"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows(INPUT_CSV)
    breaks = find_segment_breaks(rows)
    bounds = segment_bounds(rows, breaks)
    transitions, state_out = build_transition_counts(rows, set(breaks))
    write_edge_csv(EDGE_CSV, rows, transitions, state_out)
    write_diagram_md(DIAGRAM_MD, rows, transitions, state_out, breaks)
    write_summary_md(SUMMARY_MD, rows, transitions, state_out, breaks, bounds)
    print(f"Wrote {EDGE_CSV.name}")
    print(f"Wrote {DIAGRAM_MD.name}")
    print(f"Wrote {SUMMARY_MD.name}")


if __name__ == "__main__":
    main()
