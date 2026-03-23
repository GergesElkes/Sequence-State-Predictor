from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from predict_next_state import INPUT_CSV, build_model, predict, predict_future_paths


REGIME_TOKENS = ("EARLY", "LATE", "CLOCK")


def state_label(state_id: int, state_name: str) -> str:
    return f"{state_id}: {state_name}"


def state_palette(state_name: str) -> tuple[str, str]:
    if state_name.startswith("BULL"):
        return "#1c3620", "#d6f5b0"
    if state_name.startswith("BEAR"):
        return "#40211d", "#ffd4c9"
    return "#173347", "#d4ebff"


class PredictorGUI:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.model: dict[str, object] | None = None
        self.last_row = None
        self.regimes: list[str] = []
        self.state_labels: list[str] = []
        self._load_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._input_widgets: list[tk.Widget] = []
        self.results_canvas: tk.Canvas | None = None
        self.results_window_id: int | None = None

        self.root = tk.Tk()
        self.root.title("Sequence State Predictor")
        self.root.geometry("1180x860")
        self.root.minsize(1020, 760)
        self.root.configure(bg="#0f1217")

        self.state_var = tk.StringVar()
        self.use_regime_var = tk.BooleanVar(value=True)
        self.regime_vars = [tk.StringVar(value="CLOCK") for _ in range(3)]
        self.top_var = tk.IntVar(value=5)
        self.min_prob_var = tk.DoubleVar(value=0.02)

        self.status_var = tk.StringVar(value=f"Loading model from {self.csv_path.name}...")
        self.meta_var = tk.StringVar(value="Building cleaned transition model...")
        self.headline_var = tk.StringVar(value="Model loading...")
        self.detail_var = tk.StringVar(value="The predictor will be ready in a few seconds.")
        self.context_var = tk.StringVar(value="Context: --")
        self.mode_var = tk.StringVar(value="Mode: --")
        self.training_var = tk.StringVar(value="Training transitions: --")
        self.breaks_var = tk.StringVar(value="Detected stitched breaks removed: --")
        self.common_regime_title_var = tk.StringVar(value="Common regimes for state")
        self.future_note_var = tk.StringVar(value="Future paths will appear here once the model loads.")

        self._configure_styles()
        self._build_ui()
        self._bind_auto_predict()
        self._set_controls_enabled(False)
        self._start_load()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("App.TFrame", background="#0f1217")
        style.configure("Card.TFrame", background="#181c23", borderwidth=0)
        style.configure("Title.TLabel", background="#0f1217", foreground="#eef4fb", font=("Segoe UI", 18, "bold"))
        style.configure("Body.TLabel", background="#181c23", foreground="#d8e0eb", font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background="#181c23", foreground="#eef4fb", font=("Segoe UI", 11, "bold"))
        style.configure("Meta.TLabel", background="#0f1217", foreground="#98a8ba", font=("Segoe UI", 10))
        style.configure("Accent.TButton", background="#2f9d77", foreground="#0f1217", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#43b68d")])
        style.configure("Subtle.TButton", background="#232a35", foreground="#e7eef7", font=("Segoe UI", 10))
        style.map("Subtle.TButton", background=[("active", "#313948")])
        style.configure("Dark.TCheckbutton", background="#181c23", foreground="#d8e0eb")
        style.configure(
            "Predictor.Treeview",
            background="#12161c",
            fieldbackground="#12161c",
            foreground="#e7eef7",
            rowheight=30,
            borderwidth=0,
        )
        style.configure(
            "Predictor.Treeview.Heading",
            background="#232a35",
            foreground="#e7eef7",
            font=("Segoe UI", 10, "bold"),
        )
        style.map("Predictor.Treeview", background=[("selected", "#263345")])

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, style="App.TFrame", padding=18)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text="Sequence State Predictor", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="GUI wrapper for the cleaned out.csv next-state model",
            style="Meta.TLabel",
        ).pack(anchor="w", pady=(2, 0))
        ttk.Label(header, textvariable=self.status_var, style="Meta.TLabel").pack(anchor="w", pady=(8, 0))

        body = ttk.Frame(outer, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True, pady=(18, 0))

        controls = ttk.Frame(body, style="Card.TFrame", padding=16)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        results_shell = ttk.Frame(body, style="App.TFrame")
        results_shell.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(18, 0))

        results_scrollbar = ttk.Scrollbar(results_shell, orient=tk.VERTICAL)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.results_canvas = tk.Canvas(
            results_shell,
            bg="#0f1217",
            highlightthickness=0,
            borderwidth=0,
            yscrollcommand=results_scrollbar.set,
        )
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.configure(command=self.results_canvas.yview)

        results = ttk.Frame(self.results_canvas, style="App.TFrame")
        self.results_window_id = self.results_canvas.create_window((0, 0), window=results, anchor="nw")
        results.bind("<Configure>", self._on_results_frame_configure)
        self.results_canvas.bind("<Configure>", self._on_results_canvas_configure)
        self.results_canvas.bind("<MouseWheel>", self._on_results_mousewheel)
        results.bind("<MouseWheel>", self._on_results_mousewheel)

        self._build_controls(controls)
        self._build_results(results)

    def _build_controls(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Controls", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(parent, text=f"CSV: {self.csv_path.name}", style="Body.TLabel").pack(anchor="w", pady=(6, 14))

        ttk.Label(parent, text="Current state", style="Body.TLabel").pack(anchor="w")
        self.state_combo = ttk.Combobox(parent, textvariable=self.state_var, state="readonly", width=28)
        self.state_combo.pack(anchor="w", fill=tk.X, pady=(6, 12))
        self._input_widgets.append(self.state_combo)

        ttk.Checkbutton(
            parent,
            text="Use regime memory",
            variable=self.use_regime_var,
            command=self._toggle_regime_mode,
            style="Dark.TCheckbutton",
        ).pack(anchor="w")
        ttk.Label(parent, text="Regime tokens in CSV order", style="Body.TLabel").pack(anchor="w", pady=(6, 8))

        regime_row = ttk.Frame(parent, style="Card.TFrame")
        regime_row.pack(fill=tk.X)
        self.regime_boxes: list[ttk.Combobox] = []
        for idx in range(3):
            box = ttk.Combobox(
                regime_row,
                textvariable=self.regime_vars[idx],
                values=REGIME_TOKENS,
                state="readonly",
                width=8,
            )
            box.pack(side=tk.LEFT, padx=(0, 8))
            self.regime_boxes.append(box)
            self._input_widgets.append(box)

        ttk.Label(parent, text="Top predictions", style="Body.TLabel").pack(anchor="w", pady=(16, 4))
        self.top_spin = ttk.Spinbox(parent, from_=1, to=13, textvariable=self.top_var, width=6, state="readonly")
        self.top_spin.pack(anchor="w")
        self._input_widgets.append(self.top_spin)

        self.min_prob_text = tk.StringVar(value=f"Minimum probability: {self.min_prob_var.get():.2f}")
        ttk.Label(parent, textvariable=self.min_prob_text, style="Body.TLabel").pack(anchor="w", pady=(16, 4))
        self.min_prob_scale = ttk.Scale(
            parent,
            from_=0.0,
            to=0.50,
            variable=self.min_prob_var,
            command=self._on_prob_scale,
        )
        self.min_prob_scale.pack(fill=tk.X)
        self._input_widgets.append(self.min_prob_scale)

        btn_row = ttk.Frame(parent, style="Card.TFrame")
        btn_row.pack(fill=tk.X, pady=(18, 0))
        self.predict_btn = ttk.Button(btn_row, text="Predict", command=self._predict_now, style="Accent.TButton")
        self.predict_btn.pack(side=tk.LEFT)
        self.reload_btn = ttk.Button(btn_row, text="Reload CSV", command=self._start_load, style="Subtle.TButton")
        self.reload_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.last_row_btn = ttk.Button(btn_row, text="Use Last Row", command=self._use_last_row, style="Subtle.TButton")
        self.last_row_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._input_widgets.extend([self.predict_btn, self.reload_btn, self.last_row_btn])

        ttk.Separator(parent).pack(fill=tk.X, pady=18)
        ttk.Label(parent, textvariable=self.common_regime_title_var, style="CardTitle.TLabel").pack(anchor="w")
        self.regime_list = tk.Listbox(
            parent,
            height=8,
            bg="#12161c",
            fg="#dbe7f5",
            highlightthickness=0,
            borderwidth=0,
            selectbackground="#263345",
            activestyle="none",
        )
        self.regime_list.pack(fill=tk.X, pady=(8, 0))

    def _build_results(self, parent: ttk.Frame) -> None:
        hero = tk.Frame(parent, bg="#173347", padx=18, pady=18)
        hero.pack(fill=tk.X)
        self.hero_frame = hero

        self.hero_title = tk.Label(
            hero,
            textvariable=self.headline_var,
            bg="#173347",
            fg="#d4ebff",
            font=("Segoe UI", 18, "bold"),
            anchor="w",
        )
        self.hero_title.pack(fill=tk.X)
        self.hero_detail = tk.Label(
            hero,
            textvariable=self.detail_var,
            bg="#173347",
            fg="#d4ebff",
            font=("Segoe UI", 10),
            anchor="w",
            justify=tk.LEFT,
        )
        self.hero_detail.pack(fill=tk.X, pady=(6, 0))

        meta = ttk.Frame(parent, style="Card.TFrame", padding=16)
        meta.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(meta, textvariable=self.context_var, style="Body.TLabel").pack(anchor="w")
        ttk.Label(meta, textvariable=self.mode_var, style="Body.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(meta, textvariable=self.training_var, style="Body.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(meta, textvariable=self.breaks_var, style="Body.TLabel").pack(anchor="w", pady=(6, 0))
        ttk.Label(meta, textvariable=self.meta_var, style="Body.TLabel", wraplength=760).pack(anchor="w", pady=(10, 0))

        table_card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        table_card.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        ttk.Label(table_card, text="Next-State Probabilities", style="CardTitle.TLabel").pack(anchor="w")

        self.tree = ttk.Treeview(
            table_card,
            columns=("rank", "state", "probability", "count"),
            show="headings",
            style="Predictor.Treeview",
            height=10,
        )
        self.tree.heading("rank", text="#")
        self.tree.heading("state", text="Next state")
        self.tree.heading("probability", text="Probability")
        self.tree.heading("count", text="Count")
        self.tree.column("rank", width=60, anchor=tk.CENTER)
        self.tree.column("state", width=320, anchor=tk.W)
        self.tree.column("probability", width=160, anchor=tk.CENTER)
        self.tree.column("count", width=120, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.tree.tag_configure("bull", foreground="#c7f0a3")
        self.tree.tag_configure("bear", foreground="#ffd0c6")
        self.tree.tag_configure("neutral", foreground="#d4ebff")

        bars_card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        bars_card.pack(fill=tk.X, pady=(14, 0))
        ttk.Label(bars_card, text="Visual Mix", style="CardTitle.TLabel").pack(anchor="w")
        self.bar_rows = ttk.Frame(bars_card, style="Card.TFrame")
        self.bar_rows.pack(fill=tk.X, pady=(10, 0))

        future_card = ttk.Frame(parent, style="Card.TFrame", padding=16)
        future_card.pack(fill=tk.BOTH, expand=True, pady=(14, 0))
        ttk.Label(future_card, text="Likely Future Paths", style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(future_card, textvariable=self.future_note_var, style="Body.TLabel", wraplength=760).pack(
            anchor="w", pady=(6, 12)
        )

        future_wrap = ttk.Frame(future_card, style="Card.TFrame")
        future_wrap.pack(fill=tk.BOTH, expand=True)

        self.future_text: dict[int, tk.Text] = {}
        for horizon in (3, 4):
            pane = ttk.Frame(future_wrap, style="Card.TFrame")
            pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10) if horizon == 3 else (0, 0))
            ttk.Label(pane, text=f"Next {horizon} states", style="Body.TLabel").pack(anchor="w", pady=(0, 6))
            text = tk.Text(
                pane,
                height=8,
                wrap="word",
                bg="#12161c",
                fg="#dbe7f5",
                insertbackground="#dbe7f5",
                highlightthickness=0,
                borderwidth=0,
                relief="flat",
                padx=8,
                pady=8,
            )
            text.pack(fill=tk.BOTH, expand=True)
            text.configure(state="disabled")
            self.future_text[horizon] = text

    def _bind_auto_predict(self) -> None:
        self.state_var.trace_add("write", lambda *_: self._predict_now())
        self.use_regime_var.trace_add("write", lambda *_: self._predict_now())
        self.top_var.trace_add("write", lambda *_: self._predict_now())
        for var in self.regime_vars:
            var.trace_add("write", lambda *_: self._predict_now())

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        button_state = "!disabled" if enabled else "disabled"

        self.state_combo.configure(state=state)
        for box in self.regime_boxes:
            box.configure(state=state if self.use_regime_var.get() and enabled else "disabled")
        self.top_spin.configure(state="readonly" if enabled else "disabled")
        self.min_prob_scale.configure(state="normal" if enabled else "disabled")
        for btn in (self.predict_btn, self.reload_btn, self.last_row_btn):
            btn.state([button_state] if enabled else ["disabled"])

    def _start_load(self) -> None:
        self._set_controls_enabled(False)
        self.status_var.set(f"Loading model from {self.csv_path.name}...")
        self.meta_var.set("Rebuilding cleaned state transitions from out.csv.")
        self.headline_var.set("Model loading...")
        self.detail_var.set("Reading rows, removing stitched breaks, and rebuilding probabilities.")
        worker = threading.Thread(target=self._load_worker, daemon=True)
        worker.start()
        self.root.after(100, self._poll_load_queue)

    def _load_worker(self) -> None:
        try:
            model = build_model(self.csv_path)
            self._load_queue.put(("ok", model))
        except Exception as exc:
            self._load_queue.put(("error", exc))

    def _poll_load_queue(self) -> None:
        try:
            status, payload = self._load_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_load_queue)
            return

        if status == "error":
            self.status_var.set("Model load failed.")
            self.meta_var.set(str(payload))
            messagebox.showerror("Load error", str(payload))
            return

        self.model = payload  # type: ignore[assignment]
        rows = self.model["rows"]  # type: ignore[index]
        self.last_row = rows[-1] if rows else None
        state_names = self.model["state_names"]  # type: ignore[index]
        self.state_labels = [state_label(state_id, state_names[state_id]) for state_id in sorted(state_names)]
        self.regimes = sorted({" ".join(row.regime) for row in rows})

        self.state_combo.configure(values=self.state_labels)
        if self.state_labels:
            self.state_var.set(self.state_labels[0])

        if self.last_row is not None:
            self._use_last_row()
        else:
            self._set_controls_enabled(True)

        self.status_var.set(f"Loaded {len(rows):,} rows from {self.csv_path.name}")
        self._set_controls_enabled(True)
        self._toggle_regime_mode()
        self._predict_now()

    def _selected_state_id(self) -> int | None:
        value = self.state_var.get().strip()
        if not value:
            return None
        return int(value.split(":", 1)[0])

    def _current_regime(self) -> str:
        return " ".join(var.get().strip().upper() for var in self.regime_vars)

    def _toggle_regime_mode(self) -> None:
        enabled = self.use_regime_var.get() and self.model is not None
        for box in self.regime_boxes:
            box.configure(state="readonly" if enabled else "disabled")
        self._predict_now()

    def _on_prob_scale(self, _value: str) -> None:
        self.min_prob_text.set(f"Minimum probability: {self.min_prob_var.get():.2f}")
        self._predict_now()

    def _use_last_row(self) -> None:
        if self.last_row is None or self.model is None:
            return
        label = state_label(self.last_row.state_id, self.last_row.state_name)
        if label in self.state_labels:
            self.state_var.set(label)
        for idx, token in enumerate(self.last_row.regime):
            self.regime_vars[idx].set(token)
        self._predict_now()

    def _predict_now(self) -> None:
        if self.model is None:
            return
        state_id = self._selected_state_id()
        if state_id is None:
            return

        regime = self._current_regime() if self.use_regime_var.get() else None
        try:
            top_n = max(1, int(self.top_var.get() or 5))
        except (TypeError, ValueError):
            top_n = 5
        try:
            result = predict(
                model=self.model,
                state_id=state_id,
                regime=regime,
                top=top_n,
                min_prob=max(0.0, float(self.min_prob_var.get())),
            )
            future_results = {
                horizon: predict_future_paths(
                    model=self.model,
                    state_id=state_id,
                    regime=regime,
                    horizon=horizon,
                    top=5,
                    min_prob=max(0.0, float(self.min_prob_var.get())),
                )
                for horizon in (3, 4)
            }
        except ValueError as exc:
            self.status_var.set(f"Prediction error: {exc}")
            return

        self._render_result(result, future_results)

    def _render_result(
        self, result: dict[str, object], future_results: dict[int, dict[str, object]]
    ) -> None:
        state_name = result["input_state_name"]
        state_id = result["input_state_id"]
        regime = result["input_regime"] or "state-only"
        mode = result["mode"]
        training_count = result["training_count"]
        break_count = result["break_count"]
        fallback_reason = result["fallback_reason"]
        predictions = result["top_predictions"]
        common_regimes = result["common_regimes_for_state"]

        self.context_var.set(f"Context: {state_name} ({state_id}) | Regime: {regime}")
        self.mode_var.set(f"Mode: {mode}")
        self.training_var.set(f"Training transitions: {training_count}")
        self.breaks_var.set(f"Detected stitched breaks removed: {break_count}")
        self.common_regime_title_var.set(f"Common regimes for {state_name}")

        self.regime_list.delete(0, tk.END)
        for item in common_regimes:
            self.regime_list.insert(tk.END, f"{item['regime']}  ({item['count']})")

        for item in self.tree.get_children():
            self.tree.delete(item)

        if not predictions:
            self.headline_var.set("No predictions available")
            self.detail_var.set("That context does not have any outgoing transitions after cleaning.")
            self.meta_var.set("Try reducing the minimum probability or selecting a different state.")
            self._paint_hero("#173347", "#d4ebff")
            self._render_bars([])
            self.future_note_var.set("No future paths are available because this context has no outgoing predictions.")
            self._render_future_paths({})
            return

        top_pick = predictions[0]
        prob = float(top_pick["probability"])
        next_name = str(top_pick["state_name"])
        bg, fg = state_palette(next_name)
        self.headline_var.set(f"Most likely next state: {next_name}")
        self.detail_var.set(
            f"Top prediction is {prob:.2%} from {training_count} matching transitions."
        )
        if fallback_reason:
            self.meta_var.set(fallback_reason)
        else:
            self.meta_var.set("Exact state + regime context found in the cleaned sequence model.")
        self._paint_hero(bg, fg)
        self.future_note_var.set(self._future_note_text(result, future_results))

        for rank, item in enumerate(predictions, start=1):
            next_state_name = str(item["state_name"])
            tag = "neutral"
            if next_state_name.startswith("BULL"):
                tag = "bull"
            elif next_state_name.startswith("BEAR"):
                tag = "bear"
            self.tree.insert(
                "",
                tk.END,
                values=(
                    rank,
                    f"{item['state_name']} ({item['state_id']})",
                    f"{float(item['probability']):.2%}",
                    item["count"],
                ),
                tags=(tag,),
            )

        self._render_bars(predictions)
        self._render_future_paths(future_results)

    def _paint_hero(self, bg: str, fg: str) -> None:
        self.hero_frame.configure(bg=bg)
        self.hero_title.configure(bg=bg, fg=fg)
        self.hero_detail.configure(bg=bg, fg=fg)

    def _render_bars(self, predictions: list[dict[str, object]]) -> None:
        for child in self.bar_rows.winfo_children():
            child.destroy()

        for item in predictions[:5]:
            row = ttk.Frame(self.bar_rows, style="Card.TFrame")
            row.pack(fill=tk.X, pady=4)

            label = ttk.Label(
                row,
                text=f"{item['state_name']} ({item['state_id']})",
                style="Body.TLabel",
                width=24,
            )
            label.pack(side=tk.LEFT)

            bar_bg = "#2a3039"
            fill_bg, _ = state_palette(str(item["state_name"]))
            track = tk.Frame(row, bg=bar_bg, height=18)
            track.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
            track.pack_propagate(False)

            fill = tk.Frame(track, bg=fill_bg)
            fill.place(relheight=1.0, relwidth=float(item["probability"]))

            pct = ttk.Label(row, text=f"{float(item['probability']):.2%}", style="Body.TLabel", width=10)
            pct.pack(side=tk.LEFT)

    def _future_note_text(
        self, result: dict[str, object], future_results: dict[int, dict[str, object]]
    ) -> str:
        input_regime = result["input_regime"]
        top_predictions = result["top_predictions"]
        future3 = future_results.get(3, {})
        if not top_predictions:
            return "No future-path summary is available for this context."

        top_state = top_predictions[0]
        next_state_name = str(top_state["state_name"])
        next_state_id = int(top_state["state_id"])
        prob = float(top_state["probability"])
        strong_collapse_regs = {"EARLY CLOCK EARLY", "CLOCK CLOCK EARLY"}

        if input_regime in strong_collapse_regs and next_state_id in {0, 4, 8}:
            return (
                f"Compression signal: {input_regime} is one of the strong EARLY regimes. "
                f"This context first funnels to {next_state_id}:{next_state_name} at {prob:.2%}, "
                "then the future paths below show how the machine re-expands into break/trend loops."
            )

        if next_state_id in {0, 4, 8}:
            return (
                f"Gate-state warning: the most likely next state is {next_state_id}:{next_state_name} at {prob:.2%}. "
                "That usually means short-term compression before the machine branches again."
            )

        if future3.get("fallback_reason"):
            return str(future3["fallback_reason"])

        return "Future paths below are empirical next-state sequences from the same cleaned context."

    def _render_future_paths(self, future_results: dict[int, dict[str, object]]) -> None:
        for horizon, widget in self.future_text.items():
            widget.configure(state="normal")
            widget.delete("1.0", tk.END)

            result = future_results.get(horizon)
            if not result:
                widget.insert("1.0", "No future-path data available.")
                widget.configure(state="disabled")
                continue

            paths = result.get("paths", [])
            if not paths:
                widget.insert("1.0", "No paths match the current filters.")
                widget.configure(state="disabled")
                continue

            lines = []
            for idx, item in enumerate(paths, start=1):
                lines.append(
                    f"{idx}. {float(item['probability']):.2%} | {item['path_label']} "
                    f"(count={item['count']})"
                )
            widget.insert("1.0", "\n\n".join(lines))
            widget.configure(state="disabled")

    def _on_results_frame_configure(self, _event: tk.Event) -> None:
        if self.results_canvas is None:
            return
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _on_results_canvas_configure(self, event: tk.Event) -> None:
        if self.results_canvas is None or self.results_window_id is None:
            return
        self.results_canvas.itemconfigure(self.results_window_id, width=event.width)
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _on_results_mousewheel(self, event: tk.Event) -> str:
        if self.results_canvas is None:
            return "break"
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return "break"
        step = -1 * int(delta / 120) if delta else 0
        if step:
            self.results_canvas.yview_scroll(step, "units")
        return "break"

    def run(self) -> None:
        self.root.mainloop()


def run_check(csv_path: Path) -> int:
    model = build_model(csv_path)
    rows = model["rows"]  # type: ignore[index]
    if not rows:
        print("Model loaded, but no rows were found.")
        return 1
    tail = rows[-1]
    result = predict(model, tail.state_id, " ".join(tail.regime), top=3, min_prob=0.0)
    future3 = predict_future_paths(model, tail.state_id, " ".join(tail.regime), horizon=3, top=3, min_prob=0.0)
    print(f"Loaded {len(rows):,} rows from {csv_path.name}")
    print(
        f"Tail context: {tail.state_name} ({tail.state_id}) | {' '.join(tail.regime)}"
    )
    top = result["top_predictions"][0]
    print(
        f"Top prediction: {top['state_name']} ({top['state_id']}) at {float(top['probability']):.4f}"
    )
    if future3["paths"]:
        path = future3["paths"][0]
        print(f"Top 3-step future: {path['path_label']} at {float(path['probability']):.4f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tkinter GUI for the out.csv next-state predictor.")
    parser.add_argument("--csv", type=Path, default=INPUT_CSV, help="Path to the source CSV.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load the model and print one sample prediction without opening the GUI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        return run_check(args.csv)
    app = PredictorGUI(args.csv)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
