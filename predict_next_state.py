from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from reverse_engineer_out_sequences import INPUT_CSV, find_segment_breaks, load_rows


VALID_REGIME_TOKENS = {"EARLY", "LATE", "CLOCK"}
FUTURE_HORIZONS = (3, 4)


def normalize_state_name(value: str) -> str:
    return " ".join(value.replace("_", " ").upper().split())


def normalize_regime(value: str) -> str:
    tokens = value.replace(",", " ").upper().split()
    if len(tokens) != 3:
        raise ValueError("regime must contain exactly 3 tokens, e.g. 'CLOCK EARLY LATE'")
    invalid = [token for token in tokens if token not in VALID_REGIME_TOKENS]
    if invalid:
        raise ValueError(f"invalid regime tokens: {', '.join(invalid)}")
    return " ".join(tokens)


def build_model(csv_path: Path) -> dict[str, object]:
    rows = load_rows(csv_path)
    breaks = set(find_segment_breaks(rows))

    state_names: dict[int, str] = {}
    state_ids_by_name: dict[str, int] = {}
    state_counts: Counter[int] = Counter()
    state_only: defaultdict[int, Counter[int]] = defaultdict(Counter)
    state_regime: defaultdict[tuple[int, str], Counter[int]] = defaultdict(Counter)
    state_regime_counts: Counter[tuple[int, str]] = Counter()
    state_regimes_seen: defaultdict[int, Counter[str]] = defaultdict(Counter)
    future_state_only: dict[int, defaultdict[int, Counter[tuple[int, ...]]]] = {
        horizon: defaultdict(Counter) for horizon in FUTURE_HORIZONS
    }
    future_state_regime: dict[int, defaultdict[tuple[int, str], Counter[tuple[int, ...]]]] = {
        horizon: defaultdict(Counter) for horizon in FUTURE_HORIZONS
    }

    for row in rows:
        state_names[row.state_id] = row.state_name
        state_ids_by_name[normalize_state_name(row.state_name)] = row.state_id
        state_counts[row.state_id] += 1

    for idx in range(len(rows) - 1):
        if idx in breaks:
            continue
        source = rows[idx].state_id
        target = rows[idx + 1].state_id
        regime = " ".join(rows[idx].regime)
        state_only[source][target] += 1
        state_regime[(source, regime)][target] += 1
        state_regime_counts[(source, regime)] += 1
        state_regimes_seen[source][regime] += 1

        for horizon in FUTURE_HORIZONS:
            if idx + horizon >= len(rows):
                continue
            crosses_break = any(step in breaks for step in range(idx, idx + horizon))
            if crosses_break:
                continue
            future_path = tuple(rows[idx + step].state_id for step in range(1, horizon + 1))
            future_state_only[horizon][source][future_path] += 1
            future_state_regime[horizon][(source, regime)][future_path] += 1

    return {
        "csv_path": csv_path,
        "rows": rows,
        "state_names": state_names,
        "state_ids_by_name": state_ids_by_name,
        "state_counts": state_counts,
        "state_only": state_only,
        "state_regime": state_regime,
        "state_regime_counts": state_regime_counts,
        "state_regimes_seen": state_regimes_seen,
        "future_state_only": future_state_only,
        "future_state_regime": future_state_regime,
        "break_count": len(breaks),
    }


def resolve_state(model: dict[str, object], state_arg: str) -> int:
    state_names: dict[int, str] = model["state_names"]  # type: ignore[assignment]
    state_ids_by_name: dict[str, int] = model["state_ids_by_name"]  # type: ignore[assignment]

    if state_arg.isdigit():
        state_id = int(state_arg)
        if state_id not in state_names:
            raise ValueError(f"unknown state id: {state_id}")
        return state_id

    normalized = normalize_state_name(state_arg)
    if normalized not in state_ids_by_name:
        known = ", ".join(f"{sid}:{name}" for sid, name in sorted(state_names.items()))
        raise ValueError(f"unknown state name: {state_arg}. Known states: {known}")
    return state_ids_by_name[normalized]


def distribution_rows(
    counts: Counter[int], state_names: dict[int, str], top: int, min_prob: float
) -> list[dict[str, object]]:
    total = sum(counts.values())
    if total == 0:
        return []
    rows: list[dict[str, object]] = []
    for target_id, count in counts.most_common():
        probability = count / total
        if probability < min_prob:
            continue
        rows.append(
            {
                "state_id": target_id,
                "state_name": state_names[target_id],
                "count": count,
                "probability": probability,
            }
        )
        if len(rows) >= top:
            break
    return rows


def future_path_rows(
    counts: Counter[tuple[int, ...]], state_names: dict[int, str], top: int, min_prob: float
) -> list[dict[str, object]]:
    total = sum(counts.values())
    if total == 0:
        return []
    rows: list[dict[str, object]] = []
    for sequence, count in counts.most_common():
        probability = count / total
        if probability < min_prob:
            continue
        rows.append(
            {
                "sequence_ids": list(sequence),
                "sequence_names": [state_names[state_id] for state_id in sequence],
                "path_label": " -> ".join(f"{state_id}:{state_names[state_id]}" for state_id in sequence),
                "count": count,
                "probability": probability,
            }
        )
        if len(rows) >= top:
            break
    return rows


def predict(
    model: dict[str, object], state_id: int, regime: str | None, top: int, min_prob: float
) -> dict[str, object]:
    state_names: dict[int, str] = model["state_names"]  # type: ignore[assignment]
    state_only: defaultdict[int, Counter[int]] = model["state_only"]  # type: ignore[assignment]
    state_regime: defaultdict[tuple[int, str], Counter[int]] = model["state_regime"]  # type: ignore[assignment]
    state_regime_counts: Counter[tuple[int, str]] = model["state_regime_counts"]  # type: ignore[assignment]
    state_regimes_seen: defaultdict[int, Counter[str]] = model["state_regimes_seen"]  # type: ignore[assignment]

    mode = "state_only"
    training_count = sum(state_only[state_id].values())
    selected_regime = None
    counts = state_only[state_id]
    fallback_reason = None

    if regime is not None:
        selected_regime = normalize_regime(regime)
        key = (state_id, selected_regime)
        if key in state_regime:
            counts = state_regime[key]
            training_count = state_regime_counts[key]
            mode = "state_plus_regime"
        else:
            fallback_reason = "exact state+regime context not seen; using state-only distribution"

    results = distribution_rows(counts, state_names, top=top, min_prob=min_prob)
    seen_regimes = [
        {"regime": regime_name, "count": count}
        for regime_name, count in state_regimes_seen[state_id].most_common(5)
    ]

    return {
        "input_state_id": state_id,
        "input_state_name": state_names[state_id],
        "input_regime": selected_regime,
        "mode": mode,
        "training_count": training_count,
        "break_count": model["break_count"],
        "fallback_reason": fallback_reason,
        "top_predictions": results,
        "common_regimes_for_state": seen_regimes,
    }


def predict_future_paths(
    model: dict[str, object],
    state_id: int,
    regime: str | None,
    horizon: int,
    top: int,
    min_prob: float,
) -> dict[str, object]:
    if horizon not in FUTURE_HORIZONS:
        raise ValueError(f"unsupported horizon: {horizon}. Supported values: {', '.join(map(str, FUTURE_HORIZONS))}")

    state_names: dict[int, str] = model["state_names"]  # type: ignore[assignment]
    future_state_only: dict[int, defaultdict[int, Counter[tuple[int, ...]]]] = model["future_state_only"]  # type: ignore[assignment]
    future_state_regime: dict[int, defaultdict[tuple[int, str], Counter[tuple[int, ...]]]] = model["future_state_regime"]  # type: ignore[assignment]

    mode = "state_only"
    selected_regime = None
    counts = future_state_only[horizon][state_id]
    training_count = sum(counts.values())
    fallback_reason = None

    if regime is not None:
        selected_regime = normalize_regime(regime)
        key = (state_id, selected_regime)
        if key in future_state_regime[horizon]:
            counts = future_state_regime[horizon][key]
            training_count = sum(counts.values())
            mode = "state_plus_regime"
        else:
            fallback_reason = "exact state+regime future-path context not seen; using state-only paths"

    paths = future_path_rows(counts, state_names, top=top, min_prob=min_prob)

    return {
        "input_state_id": state_id,
        "input_state_name": state_names[state_id],
        "input_regime": selected_regime,
        "mode": mode,
        "horizon": horizon,
        "training_count": training_count,
        "fallback_reason": fallback_reason,
        "paths": paths,
    }


def print_prediction(result: dict[str, object]) -> None:
    print(f"State: {result['input_state_name']} ({result['input_state_id']})")
    if result["input_regime"] is not None:
        print(f"Regime: {result['input_regime']}")
    print(f"Mode: {result['mode']}")
    print(f"Training transitions used: {result['training_count']}")
    print(f"Detected stitched breaks removed: {result['break_count']}")
    if result["fallback_reason"]:
        print(f"Fallback: {result['fallback_reason']}")
        common = result["common_regimes_for_state"]
        if common:
            print("Most common regimes for this state:")
            for item in common:
                print(f"  - {item['regime']}: {item['count']}")

    predictions = result["top_predictions"]
    if not predictions:
        print("No predictions available for that context.")
        return

    print("Top next-state probabilities:")
    for idx, item in enumerate(predictions, start=1):
        print(
            f"  {idx}. {item['state_name']} ({item['state_id']})"
            f" -> p={item['probability']:.4f}  count={item['count']}"
        )


def list_states(model: dict[str, object]) -> None:
    state_names: dict[int, str] = model["state_names"]  # type: ignore[assignment]
    print("Known states:")
    for state_id, state_name in sorted(state_names.items()):
        print(f"  {state_id}: {state_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict the next state from out.csv using the cleaned sequence model."
    )
    parser.add_argument("--csv", type=Path, default=INPUT_CSV, help="Path to the source CSV.")
    parser.add_argument(
        "--state",
        help="Current state id or name, e.g. 8 or 'BULL PAUSE'. Required unless --list-states is used.",
    )
    parser.add_argument(
        "--regime",
        help="Optional regime triplet, e.g. 'CLOCK EARLY LATE'. If unseen, the script falls back to state-only.",
    )
    parser.add_argument("--top", type=int, default=5, help="Number of predictions to print.")
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.0,
        help="Hide predictions below this probability threshold.",
    )
    parser.add_argument(
        "--list-states",
        action="store_true",
        help="Print the known state ids/names and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the prediction result as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = build_model(args.csv)

    if args.list_states:
        list_states(model)
        return 0

    if not args.state:
        print("error: --state is required unless --list-states is used", file=sys.stderr)
        return 2

    try:
        state_id = resolve_state(model, args.state)
        result = predict(
            model=model,
            state_id=state_id,
            regime=args.regime,
            top=max(1, args.top),
            min_prob=max(0.0, args.min_prob),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_prediction(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
