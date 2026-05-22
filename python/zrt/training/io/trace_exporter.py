"""Chrome Trace exporter for spec-based training estimation.

Generate a trace.json that can be loaded by chrome://tracing or Perfetto UI.

The trace is intentionally derived from:
- IR Graph ops / collectives
- per-op analytical OpCost
- model / system / strategy

It does not change the estimator path. It only provides a profiling-like
timeline view for easier validation of one training step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zrt.training.compose.stage import _cost_phase_time, _resolve_compute_dtype
from zrt.training.compose.schedules import _assign_stages
from zrt.training.ir.training_graph import Collective, Graph, Op
from zrt.training.models.comm import total_comm_time
from zrt.training.models.flops import OpCost, op_cost
from zrt.training.spec.model import ModelSpec
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import SystemSpec


_US = 1_000_000.0


def _sec_to_us(x: float) -> int:
    return max(0, int(round(float(x) * _US)))


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def _op_stage_map(graph: Graph, model: ModelSpec, strategy: Strategy) -> dict[str, int]:
    """Map op.name -> PP stage id.

    Uses the same private stage assignment helper as pipeline_step_time().
    Non-layer ops are placed at the beginning or end:
    - embedding / hc_expand -> stage 0
    - final_ln / lm_head / mhc_head -> last stage
    """
    pp = max(1, strategy.pp)
    stage_ids = _assign_stages(model, strategy)

    layer_to_stage: dict[int, int] = {}
    for stage_id, layer_ids in enumerate(stage_ids):
        for layer_id in layer_ids:
            layer_to_stage[layer_id] = stage_id

    out: dict[str, int] = {}
    for op in graph.ops:
        layer_id = getattr(op, "layer_id", None)
        if layer_id is not None and layer_id in layer_to_stage:
            out[op.name] = layer_to_stage[layer_id]
            continue

        # Heuristic for non-layer ops.
        name = op.name.lower()
        if "embed" in name or "hc_expand" in name:
            out[op.name] = 0
        else:
            out[op.name] = pp - 1

    return out


def _ops_by_stage(graph: Graph, op_to_stage: dict[str, int], pp: int) -> list[list[Op]]:
    result: list[list[Op]] = [[] for _ in range(pp)]
    for op in graph.ops:
        s = op_to_stage.get(op.name, 0)
        s = max(0, min(pp - 1, s))
        result[s].append(op)
    return result


def _collectives_by_anchor(
    graph: Graph,
    op_to_stage: dict[str, int],
    comm_times: dict[str, float],
) -> dict[str, list[tuple[Collective, float, int]]]:
    """Index collectives by inserted_after / inserted_before op name."""
    by_anchor: dict[str, list[tuple[Collective, float, int]]] = {}

    for c in graph.collectives:
        anchor = getattr(c, "inserted_after", None) or getattr(c, "inserted_before", None)
        if not anchor:
            continue

        stage_id = op_to_stage.get(anchor, 0)
        t = comm_times.get(c.name, 0.0)
        if t <= 0:
            continue

        by_anchor.setdefault(anchor, []).append((c, t, stage_id))

    return by_anchor


def _op_phase_time(
    op: Op,
    cost: OpCost,
    phase: str,
    model: ModelSpec,
    system: SystemSpec,
) -> float:
    """Return op phase latency in seconds using the same roofline helper."""
    gpu = system.gpu
    overlap = gpu.overlap_ratio.get(op.kind, 0.0)
    dtype = _resolve_compute_dtype(op, model)
    return _cost_phase_time(cost, phase, system, gpu.name, overlap, dtype)


def _event(
    *,
    name: str,
    cat: str,
    pid: int,
    tid: int,
    ts_s: float,
    dur_s: float,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "cat": cat,
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": _sec_to_us(ts_s),
        "dur": max(1, _sec_to_us(dur_s)),
        "args": args or {},
    }


def _metadata_events(pp: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    for s in range(pp):
        events.append({
            "name": "process_name",
            "ph": "M",
            "pid": s,
            "tid": 0,
            "args": {"name": f"pp_stage_{s}"},
        })
        events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": s,
            "tid": 0,
            "args": {"name": "compute_fwd"},
        })
        events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": s,
            "tid": 1,
            "args": {"name": "compute_bwd"},
        })
        events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": s,
            "tid": 2,
            "args": {"name": "comm"},
        })

    events.append({
        "name": "process_name",
        "ph": "M",
        "pid": 9999,
        "tid": 0,
        "args": {"name": "step_tail"},
    })
    events.append({
        "name": "thread_name",
        "ph": "M",
        "pid": 9999,
        "tid": 0,
        "args": {"name": "optimizer_and_dp"},
    })

    return events


def _op_args(op: Op, cost: OpCost, phase: str, stage_id: int) -> dict[str, Any]:
    cube = getattr(cost, f"{phase}_cube_flops", 0.0)
    vector = getattr(cost, f"{phase}_vector_flops", 0.0)
    bytes_ = getattr(cost, f"{phase}_bytes", 0.0)

    return {
        "stage": stage_id,
        "layer_id": getattr(op, "layer_id", None),
        "layer_kind": _safe_str(getattr(op, "layer_kind", "")),
        "component": _safe_str(getattr(op, "component", "")),
        "kind": op.kind,
        "phase": phase,
        "cube_flops": cube,
        "vector_flops": vector,
        "bytes": bytes_,
        "bound": getattr(cost, "bound", ""),
        "meta": getattr(op, "meta", {}),
    }


def export_estimate_trace(
    *,
    report: TrainingReport,
    graph: Graph,
    model: ModelSpec,
    system: SystemSpec,
    strategy: Strategy,
    op_costs: dict[str, OpCost] | None,
    output_path: str | Path,
) -> None:
    """Export one-step Chrome trace JSON.

    Timeline model:
    - Forward is scheduled stage-by-stage with simple pipeline dependencies.
    - Backward is scheduled in reverse stage order after last-stage forward.
    - Per-op duration comes from analytical op_cost + heterogeneous roofline.
    - Graph collectives are shown on a separate comm thread.
    - DP / Muon tail comm is appended as step-tail events when available.

    This gives users a profiling-like view for validation. It is intentionally
    visualization-oriented and does not replace pipeline_step_time().
    """
    pp = max(1, strategy.pp)
    microbatches = max(1, strategy.num_microbatches())

    op_costs = op_costs or {
        op.name: op_cost(op, model, system)
        for op in graph.ops
    }

    op_to_stage = _op_stage_map(graph, model, strategy)
    stage_ops = _ops_by_stage(graph, op_to_stage, pp)

    comm_times = total_comm_time(graph, model, system, strategy)
    collectives_by_anchor = _collectives_by_anchor(graph, op_to_stage, comm_times)

    events: list[dict[str, Any]] = []
    events.extend(_metadata_events(pp))

    # Thread availability.
    fwd_ready = [0.0 for _ in range(pp)]
    bwd_ready = [0.0 for _ in range(pp)]
    comm_ready = [0.0 for _ in range(pp)]

    # Forward completion time per microbatch/stage.
    fwd_done = [[0.0 for _ in range(pp)] for _ in range(microbatches)]
    bwd_done = [[0.0 for _ in range(pp)] for _ in range(microbatches)]

    # Forward pipeline.
    for mb in range(microbatches):
        for s in range(pp):
            t = fwd_ready[s]
            if s > 0:
                t = max(t, fwd_done[mb][s - 1])

            for op in stage_ops[s]:
                cost = op_costs.get(op.name) or op_cost(op, model, system)
                dur = _op_phase_time(op, cost, "fwd", model, system)
                if dur > 0:
                    events.append(_event(
                        name=f"mb{mb}.fwd.{op.name}",
                        cat=f"fwd/{op.kind}",
                        pid=s,
                        tid=0,
                        ts_s=t,
                        dur_s=dur,
                        args=_op_args(op, cost, "fwd", s),
                    ))
                    t += dur

                # Communication anchored after this op.
                for c, c_dur, c_stage in collectives_by_anchor.get(op.name, []):
                    c_start = max(comm_ready[c_stage], t)
                    events.append(_event(
                        name=f"mb{mb}.comm.{c.name}",
                        cat=f"comm/{c.group}/{c.kind}",
                        pid=c_stage,
                        tid=2,
                        ts_s=c_start,
                        dur_s=c_dur,
                        args={
                            "stage": c_stage,
                            "group": c.group,
                            "kind": c.kind,
                            "bytes": c.bytes_,
                            "anchor": op.name,
                            "phase": getattr(c, "phase", ""),
                            "overlap": getattr(c, "overlap", None),
                        },
                    ))
                    comm_ready[c_stage] = c_start + c_dur

            fwd_ready[s] = t
            fwd_done[mb][s] = t

    # Backward pipeline, reverse stage order.
    for mb in reversed(range(microbatches)):
        for s in reversed(range(pp)):
            t = max(bwd_ready[s], fwd_done[mb][s])
            if s < pp - 1:
                t = max(t, bwd_done[mb][s + 1])

            for op in reversed(stage_ops[s]):
                cost = op_costs.get(op.name) or op_cost(op, model, system)

                dx_dur = _op_phase_time(op, cost, "dx", model, system)
                if dx_dur > 0:
                    events.append(_event(
                        name=f"mb{mb}.bwd_dx.{op.name}",
                        cat=f"bwd_dx/{op.kind}",
                        pid=s,
                        tid=1,
                        ts_s=t,
                        dur_s=dx_dur,
                        args=_op_args(op, cost, "dx", s),
                    ))
                    t += dx_dur

                dw_dur = _op_phase_time(op, cost, "dw", model, system)
                if dw_dur > 0:
                    events.append(_event(
                        name=f"mb{mb}.bwd_dw.{op.name}",
                        cat=f"bwd_dw/{op.kind}",
                        pid=s,
                        tid=1,
                        ts_s=t,
                        dur_s=dw_dur,
                        args=_op_args(op, cost, "dw", s),
                    ))
                    t += dw_dur

            bwd_ready[s] = t
            bwd_done[mb][s] = t

    # Step tail: DP gradient sync / Muon comm if modeled by total_comm_time().
    tail_start = max(
        max(fwd_ready, default=0.0),
        max(bwd_ready, default=0.0),
        max(comm_ready, default=0.0),
    )
    tail_t = tail_start

    for name in ("dp_grad_reduce", "muon_ag", "muon_rs"):
        dur = comm_times.get(name, 0.0)
        if dur <= 0:
            continue

        events.append(_event(
            name=name,
            cat="step_tail/comm",
            pid=9999,
            tid=0,
            ts_s=tail_t,
            dur_s=dur,
            args={
                "name": name,
                "source": "total_comm_time",
            },
        ))
        tail_t += dur

    # Optional visual marker for estimator's final step time.
    estimated_step_s = getattr(report, "step_time_ms", 0.0) / 1000.0
    if estimated_step_s > 0:
        events.append(_event(
            name="estimated_step_time_from_report",
            cat="summary",
            pid=9999,
            tid=0,
            ts_s=0.0,
            dur_s=estimated_step_s,
            args={
                "step_time_ms": getattr(report, "step_time_ms", None),
                "note": (
                    "This event is the estimator-reported step time. "
                    "Per-op trace is a visualization schedule."
                ),
            },
        ))

    trace = {
        "displayTimeUnit": "ms",
        "metadata": {
            "format": "chrome_trace",
            "model": getattr(model, "name", "unknown"),
            "hardware": getattr(system.gpu, "name", "unknown"),
            "world_size": getattr(system, "world_size", None),
            "tp": strategy.tp,
            "cp": strategy.cp,
            "pp": strategy.pp,
            "ep": strategy.ep,
            "dp": strategy.dp,
            "microbatches": microbatches,
            "schedule": str(strategy.pp_schedule),
        },
        "traceEvents": events,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
