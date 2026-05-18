"""Training modeller: estimate training performance from captured computation graphs.

Usage::

    from python.zrt.transform.analysis import estimate_training_from_graphs
    report = estimate_training_from_graphs(
        forward_graph=fwd, backward_graph=bwd,
        hw_spec=hw, tp=8, pp=4, dp=2, ...
    )
    print(report.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

# Import shared TrainingReport type (canonical import path)
from zrt.training.spec.report import TrainingReport


def estimate_training_from_graphs(
    *,
    forward_graph: "OpGraph",
    backward_graph: "OpGraph | None" = None,
    output_dir: "str | Path | None" = None,
    hw_spec: "HardwareSpec | None" = None,
    total_params: int | None = None,
    hidden: int = 7168,
    num_layers: int = 4,
    num_layers_full: int | None = None,
    seq_len: int = 128,
    batch_size: int = 1,
    tp: int = 1, pp: int = 1, ep: int = 1, dp: int = 1, cp: int = 1,
    cp_kind: str = "ulysses",
    zero_stage: int = 1,
    optimizer: str = "adam",
    muon_rotation: bool = True,
    muon_ns_steps: int | None = None,
    model_type: str | None = None,
    micro_batch: int = 1,
    global_batch: int = 32,
    pp_schedule: str = "1f1b",
    vpp_chunks: int = 1,
    return_transformed: bool = False,
    quant: str | None = None,
    moe_total_experts: int = 0,
    moe_active_experts: int = 1,
    model_id: str = "",
    fusion_config: "FusionConfig | None" = None,
) -> "TrainingReport | tuple[TrainingReport, TransformContext, dict[str, OpGraph]]":
    """Estimate training performance from pre-built OpGraph instances.

    Takes already-captured forward and backward computation graphs and
    runs the training analysis pipeline. Use this when the graphs have
    already been captured by ``run_trace_phases``.

    Parameters
    ----------
    return_transformed : bool, default False
        If True, return (TrainingReport, TransformContext, transformed_graphs)
        where transformed_graphs contains the pipeline-processed graphs.
        This enables downstream Excel export via ``export_training_graphs``.
    output_dir : str or Path, optional
        If provided, export each transformed graph as a DOT file to this directory.
    """
    from python.zrt.transform.context import (
        FusionConfig, ParallelConfig, QuantConfig, TrainingConfig, TransformContext,
    )
    from python.zrt.transform.pipeline import build_default_pipeline

    metadata: dict = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "num_layers": num_layers_full or num_layers,
        "num_layers_traced": num_layers,
        "hidden": hidden,
    }
    if moe_total_experts > 0:
        metadata["moe_total_experts"] = moe_total_experts
    if moe_active_experts > 1:
        metadata["moe_active_experts"] = moe_active_experts
    if total_params is not None:
        metadata["total_params"] = int(total_params)
    if model_type is not None:
        metadata["model_type"] = model_type

    for key, val in metadata.items():
        if key not in forward_graph.metadata:
            forward_graph.metadata[key] = val
    if backward_graph is not None:
        for key, val in metadata.items():
            if key not in backward_graph.metadata:
                backward_graph.metadata[key] = val

    quant_cfg = QuantConfig(weight=quant, activation=quant) if quant else None
    ctx = TransformContext(
        hw_spec=hw_spec,
        model_id=model_id,
        parallel=ParallelConfig(tp=tp, pp=pp, ep=ep, dp=dp, cp=cp),
        training=TrainingConfig(
            optimizer=optimizer,
            zero_stage=zero_stage,
            muon_rotation=muon_rotation,
            muon_ns_steps=muon_ns_steps,
            micro_batch=micro_batch,
            global_batch=global_batch,
            pp_schedule=pp_schedule,
            vpp_chunks=vpp_chunks,
            seq_len=seq_len,
            hidden=hidden,
            cp_kind=cp_kind,
        ),
        fusion=fusion_config or FusionConfig(),
        quant=quant_cfg,
    )

    # Attach MoE profile to ctx so ExpertParallelPass and other MoE-aware
    # passes can read expert counts.
    if moe_total_experts > 0:
        from types import SimpleNamespace
        ctx.profile = SimpleNamespace(
            num_experts=moe_total_experts,
            moe_active=moe_active_experts,
        )

    pipe = build_default_pipeline()
    results: dict[str, "OpGraph"] = {}

    if backward_graph is not None:
        from python.zrt.ir.adapter import stitch_fwd_bwd
        unified = stitch_fwd_bwd(forward_graph, backward_graph)
        for key, val in metadata.items():
            if key not in unified.metadata:
                unified.metadata[key] = val
        results["unified"] = pipe.run(unified, ctx)
    else:
        results["train_forward"] = pipe.run(forward_graph, ctx)

    # DOT export.
    #
    # ``render_dot`` shells out to graphviz ``dot``; layout is super-linear
    # in node count (with ``splines=ortho`` it can take ~15s per 1k-node
    # graph, ~hours for 5k+).  Production runs only need the ``.dot`` text
    # — anyone wanting the SVG can render manually.  We skip rendering for
    # graphs above ``_RENDER_DOT_NODE_BUDGET`` so the e2e stays fast.
    _RENDER_DOT_NODE_BUDGET = 300
    if output_dir is not None:
        from python.zrt.report.dot_exporter import export_dot, render_dot
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_name = forward_graph.name or "model"

        def _maybe_render(graph, dot_path):
            if len(graph.nodes) <= _RENDER_DOT_NODE_BUDGET:
                render_dot(dot_path)  # no-op when graphviz absent

        # Export raw forward and backward graphs separately
        dot_path = export_dot(forward_graph, out / f"{model_name}_train_forward.dot")
        _maybe_render(forward_graph, dot_path)
        if backward_graph is not None:
            dot_path = export_dot(backward_graph, out / f"{model_name}_train_backward.dot")
            _maybe_render(backward_graph, dot_path)
        # Export transformed graphs (unified or forward-only)
        for tag, g in results.items():
            dot_path = export_dot(g, out / f"{model_name}_{tag}.dot")
            _maybe_render(g, dot_path)

    if "unified" in results:
        g = results["unified"]
        pipeline_metrics = g.metadata.get("pipeline_metrics")
        memory_breakdown = g.metadata.get("memory_breakdown")
        training_flops = g.metadata.get("training_flops", 0.0)
        forward_flops = g.metadata.get("forward_flops", 0.0)
        backward_flops = g.metadata.get("backward_flops", 0.0)
        total_params = g.metadata.get("total_params", 0)
    else:
        fwd = results["train_forward"]
        pipeline_metrics = fwd.metadata.get("pipeline_metrics")

        memory_breakdown = fwd.metadata.get("memory_breakdown")
        training_flops = fwd.metadata.get("training_flops", 0.0)
        forward_flops = fwd.metadata.get("forward_flops", 0.0)
        backward_flops = fwd.metadata.get("backward_flops", 0.0)
        total_params = fwd.metadata.get("total_params", 0)

    step_time_ms = pipeline_metrics.step_time_ms if pipeline_metrics else 0.0
    per_stage_ms = pipeline_metrics.per_stage_ms if pipeline_metrics else 0.0
    mfu = pipeline_metrics.mfu if pipeline_metrics else 0.0
    hfu = pipeline_metrics.hfu if pipeline_metrics else 0.0
    warmup_steps = pipeline_metrics.warmup_steps if pipeline_metrics else 0
    cooldown_steps = pipeline_metrics.cooldown_steps if pipeline_metrics else 0
    steady_steps = pipeline_metrics.steady_steps if pipeline_metrics else 0
    bubble_fraction = pipeline_metrics.bubble_fraction if pipeline_metrics else 0.0
    exposed_comm_ms = pipeline_metrics.exposed_comm_ms if pipeline_metrics else 0.0
    hidden_comm_ms = pipeline_metrics.hidden_comm_ms if pipeline_metrics else 0.0
    total_comm_ms = pipeline_metrics.total_comm_ms if pipeline_metrics else 0.0
    dp_exposed_from_metrics = pipeline_metrics.dp_exposed_ms if pipeline_metrics else 0.0
    dp_hidden_from_metrics = pipeline_metrics.dp_hidden_ms if pipeline_metrics else 0.0

    parallel = ctx.parallel
    training = ctx.training
    config_parts: list[str] = []
    if parallel.tp > 1:
        config_parts.append(f"TP{parallel.tp}")
    if parallel.pp > 1:
        config_parts.append(f"PP{parallel.pp}")
    if parallel.ep > 1:
        config_parts.append(f"EP{parallel.ep}")
    if parallel.dp > 1:
        config_parts.append(f"DP{parallel.dp}")
    if training:
        config_parts.append(f"ZeRO-{training.zero_stage}")
        config_parts.append(f"{training.optimizer}")
        config_parts.append(f"micro{training.micro_batch}")
    config_summary = "-".join(config_parts) if config_parts else "default"

    # ── Fused-operator summary ────────────────────────────────────────────────
    # Walk the transformed graph(s) and aggregate by op_type so the report
    # can show what fusion produced and how it scales.
    fused_ops_summary = _summarise_fused_ops(results)

    # Try to populate optimizer_time from metadata (set by TrainingPipelinePass)
    opt_us = 0.0
    if "unified" in results and results["unified"].metadata.get("optimizer_step_time_us"):
        opt_us = float(results["unified"].metadata.get("optimizer_step_time_us", 0.0))
    elif forward_graph.metadata.get("optimizer_step_time_us"):
        opt_us = float(forward_graph.metadata.get("optimizer_step_time_us", 0.0))

    optimizer_time_ms = opt_us / 1000.0 if opt_us else 0.0
    optimizer_comm_ms = 0.0

    # pipeline_time_ms: step minus optimizer time when optimizer time present
    pipeline_time_ms = max(0.0, step_time_ms - optimizer_time_ms) if step_time_ms > 0 else 0.0

    # Warmup/steady/cooldown durations (ms) — approximate from per_stage_ms
    warmup_ms = per_stage_ms * warmup_steps if per_stage_ms and warmup_steps else 0.0
    cooldown_ms = per_stage_ms * cooldown_steps if per_stage_ms and cooldown_steps else 0.0
    steady_ms = per_stage_ms * steady_steps if per_stage_ms and steady_steps else 0.0

    # Try to build a scheduler Timeline for more detailed timing if hw_spec available
    compute_time_ms = 0.0
    fwd_compute_ms = 0.0
    bwd_compute_ms = 0.0

    try:
        from python.zrt.executor.scheduler import DAGScheduler

        # Prefer transformed unified graph if present, else transformed forward-only
        schedule_graph = results.get("unified") if "unified" in results else results.get("train_forward")
        if schedule_graph is not None and ctx.hw_spec is not None:
            tl = DAGScheduler(ctx.hw_spec).schedule(schedule_graph)
            compute_time_ms = tl.compute_time_us / 1000.0

            # Per-phase compute split
            fwd_compute_us = sum(op.latency_us for op in tl.scheduled_ops if op.stream_type == "compute" and op.phase == "fwd")
            bwd_compute_us = sum(op.latency_us for op in tl.scheduled_ops if op.stream_type == "compute" and op.phase in ("bwd", "backward", "train_backward"))
            fwd_compute_ms = fwd_compute_us / 1000.0
            bwd_compute_ms = bwd_compute_us / 1000.0
            # Try to find optimizer node in the scheduled graph for split of optimizer compute vs comm
            opt_node = None
            if schedule_graph is not None:
                opt_node = schedule_graph.nodes.get("optimizer_step")
                if opt_node is None:
                    for n in schedule_graph.nodes.values():
                        if n.annotations.get("optimizer_step"):
                            opt_node = n
                            break
            # If optimizer node present, estimate optimizer compute/comm separately
            if opt_node is not None:
                try:
                    from python.zrt.ir.types import DType
                    # compute_time_us
                    optimizer = opt_node.attrs.get("optimizer", "adam")
                    step_flops = float(opt_node.attrs.get("step_flops", 0))
                    if optimizer == "muon":
                        peak_flops = ctx.hw_spec.peak_flops(DType.BF16)
                        compute_time_us_opt = (step_flops / peak_flops) * 1e6 if peak_flops > 0 else 0.0
                    else:
                        opt_state_bytes = float(opt_node.attrs.get("state_bytes", 0))
                        hbm_bw = ctx.hw_spec.memory.hbm_bandwidth_gbps * 1e9 / 8
                        compute_time_us_opt = (opt_state_bytes / hbm_bw) * 1e6 if hbm_bw > 0 else 0.0

                    # comm_time_us (Muon AG+RS)
                    comm_time_us_opt = 0.0
                    if optimizer == "muon":
                        ag_bytes = float(opt_node.attrs.get("muon_ag_bytes", 0))
                        ns_rotation = opt_node.attrs.get("ns_rotation", True)
                        if ag_bytes > 0:
                            dp = ctx.parallel.dp if ctx.parallel else 1
                            gpus_per_node = ctx.hw_spec.interconnect.intra_node.num_devices
                            link = ctx.hw_spec.interconnect.inter_node if dp > gpus_per_node else ctx.hw_spec.interconnect.intra_node
                            dp_bw = link.bandwidth_gbps * 1e9 / 8
                            if dp_bw > 0:
                                if ns_rotation:
                                    ring_factor = 2.0 * (dp - 1) / dp
                                else:
                                    ring_factor = 1.0 * (dp - 1) / dp
                                comm_time_us_opt = (ring_factor * ag_bytes / dp_bw) * 1e6

                    # override optimizer times if we computed them
                    optimizer_time_ms = compute_time_us_opt / 1000.0
                    optimizer_comm_ms = comm_time_us_opt / 1000.0
                except Exception:
                    # leave optimizer_time_ms as previously derived from metadata
                    optimizer_comm_ms = 0.0
    except Exception:
        # Scheduling is best-effort; fall back to zeros when it fails.
        pass

        # Best-effort: extract per-stage timelines if available to fill steady per-microbatch numbers
    schedule_graph = results.get("unified") if "unified" in results else results.get("train_forward")
    steady_fwd_per_mb_ms = 0.0
    steady_bwd_per_mb_ms = 0.0
    steady_per_mb_ms = 0.0
    if schedule_graph is not None:
        st_fwd = schedule_graph.metadata.get("stage_timelines_fwd")
        st_bwd = schedule_graph.metadata.get("stage_timelines_bwd")
        if st_fwd and st_bwd:
            try:
                steady_fwd_per_mb_ms = float(max(list(st_fwd.values()))) / 1000.0
                steady_bwd_per_mb_ms = float(max(list(st_bwd.values()))) / 1000.0
                steady_per_mb_ms = float(max((float(st_fwd[s]) + float(st_bwd.get(s, 0.0))) for s in list(st_fwd.keys()))) / 1000.0
            except Exception:
                steady_fwd_per_mb_ms = steady_bwd_per_mb_ms = steady_per_mb_ms = 0.0

    # Fill warmup/cooldown per-phase estimates (composer semantics: warmup is forward-heavy)
    warmup_fwd_ms = warmup_ms
    warmup_bwd_ms = 0.0
    cooldown_fwd_ms = 0.0
    cooldown_bwd_ms = cooldown_ms

    # Steady durations
    steady_fwd_ms = steady_fwd_per_mb_ms * steady_steps if steady_fwd_per_mb_ms and steady_steps else 0.0
    steady_bwd_ms = steady_bwd_per_mb_ms * steady_steps if steady_bwd_per_mb_ms and steady_steps else 0.0

    # Best-effort: compute DP exposed ms from dp_comm annotated nodes
    dp_exposed_ms = 0.0
    pp_exposed_ms = 0.0
    try:
        if schedule_graph is not None:
            for n in schedule_graph.nodes.values():
                if n.category == "communication":
                    lat = float(n.annotations.get("latency_us", 0.0))
                    if n.annotations.get("dp_comm"):
                        dp_exposed_ms += lat / 1000.0
                    if n.op_type == "comm.send_recv":
                        # stage-crossing P2P
                        src = n.attrs.get("src_stage")
                        dst = n.attrs.get("dst_stage")
                        if src is not None and dst is not None and src != dst:
                            pp_exposed_ms += lat / 1000.0
    except Exception:
        dp_exposed_ms = pp_exposed_ms = 0.0

    tp_hidden_ms = 0.0
    # Clamp sums
    try:
        remaining = max(0.0, exposed_comm_ms - dp_exposed_ms - pp_exposed_ms)
    except Exception:
        remaining = exposed_comm_ms

    # Do not attempt to split remaining across TP/CP/EP unless we have explicit metadata.
    tp_exposed_ms = 0.0
    cp_exposed_ms = 0.0
    ep_exposed_ms = 0.0

    # If we did not compute optimizer_time_ms earlier from opt_node, try to derive from metadata
    try:
        if optimizer_time_ms == 0.0:
            if "unified" in results and results["unified"].metadata.get("optimizer_step_time_us"):
                optimizer_time_ms = float(results["unified"].metadata.get("optimizer_step_time_us", 0.0)) / 1000.0
            elif forward_graph.metadata.get("optimizer_step_time_us"):
                optimizer_time_ms = float(forward_graph.metadata.get("optimizer_step_time_us", 0.0)) / 1000.0
    except Exception:
        pass

    # Derived metrics
    tokens = (ctx.training.global_batch if getattr(ctx, 'training', None) and getattr(ctx.training, 'global_batch', None) else global_batch) * seq_len
    use_time_s = (pipeline_time_ms if pipeline_time_ms > 0 else step_time_ms) / 1000.0
    tokens_per_sec = tokens / use_time_s if use_time_s > 0 else 0.0
    effective_params = getattr(ctx, 'profile', None).num_experts if getattr(ctx, 'profile', None) and getattr(ctx.profile, 'num_experts', 0) else total_params
    flops_per_token = (training_flops / tokens) if tokens > 0 else 0.0

    report = TrainingReport(
        # Core timing metrics
        config_summary=config_summary,
        step_time_ms=step_time_ms,
        per_stage=[],
        per_stage_ms=per_stage_ms,

        # Efficiency metrics
        mfu=mfu,
        hfu=hfu,

        # FLOPs breakdown
        total_flops=training_flops,
        training_flops=training_flops,
        forward_flops=forward_flops,
        backward_flops=backward_flops,

        # Memory metrics
        memory=None,
        memory_breakdown=memory_breakdown.to_dict() if memory_breakdown else {},

        # Pipeline metrics
        bubble_fraction=bubble_fraction,
        schedule_name=getattr(ctx.training, 'pp_schedule', '1f1b'),
        warmup_steps=warmup_steps,
        cooldown_steps=cooldown_steps,
        steady_steps=steady_steps,

        # Step time breakdown
        pipeline_time_ms=pipeline_time_ms,
        warmup_ms=warmup_ms,
        steady_ms=steady_ms,
        cooldown_ms=cooldown_ms,
        dp_exposed_ms=dp_exposed_from_metrics if dp_exposed_from_metrics > 0 else dp_exposed_ms,
        optimizer_time_ms=optimizer_time_ms,
        optimizer_comm_ms=optimizer_comm_ms,

        # Fwd/Bwd breakdown per phase
        warmup_fwd_ms=warmup_fwd_ms,
        warmup_bwd_ms=warmup_bwd_ms,
        steady_fwd_ms=steady_fwd_ms,
        steady_bwd_ms=steady_bwd_ms,
        cooldown_fwd_ms=cooldown_fwd_ms,
        cooldown_bwd_ms=cooldown_bwd_ms,

        # Per-microbatch time in steady phase
        steady_fwd_per_mb_ms=steady_fwd_per_mb_ms,
        steady_bwd_per_mb_ms=steady_bwd_per_mb_ms,
        steady_per_mb_ms=steady_per_mb_ms,

        # Compute / comm breakdown
        compute_time_ms=compute_time_ms,
        fwd_compute_ms=fwd_compute_ms,
        bwd_compute_ms=bwd_compute_ms,

        # Per-group exposed comm
        tp_exposed_ms=tp_exposed_ms,
        cp_exposed_ms=cp_exposed_ms,
        ep_exposed_ms=ep_exposed_ms,
        pp_exposed_ms=pp_exposed_ms,

        # Hidden comm
        dp_hidden_ms=dp_hidden_from_metrics if dp_hidden_from_metrics > 0 else max(0.0, hidden_comm_ms - (tp_hidden_ms if 'tp_hidden_ms' in locals() else 0.0)),
        tp_hidden_ms=0.0,

        # Config / model
        warnings=[],
        total_params=total_params,

        # Derived metrics
        tokens_per_sec=tokens_per_sec,
        effective_params=effective_params,
        flops_per_token=flops_per_token,

        # Fused ops summary
        fused_ops_summary=fused_ops_summary,
        exposed_comm_ms=exposed_comm_ms,
        hidden_comm_ms=hidden_comm_ms,
        total_comm_volume_ms=total_comm_ms,
    )

    if return_transformed:
        return report, ctx, results
    return report


# ── Fused-operator summary helper ────────────────────────────────────────────

def _summarise_fused_ops(graphs: dict) -> dict:
    """Aggregate fused-node statistics across all transformed graphs.

    Skips raw aten.* / comm.* nodes so the table focuses on what fusion
    actually produced — module-level units (Linear, RMSNorm, ...) and
    rich-rule outputs (mla_sparse_attn, kv_compressor, rms_norm, ...).

    Returns ``{op_type: {count, sample_names, total_flops, dtype, module_class}}``.
    """
    summary: dict[str, dict] = {}

    for g in graphs.values():
        for node in g.nodes.values():
            op_type = node.op_type or ""
            # Skip primitive aten / comm / optimizer nodes — those aren't
            # the "fused operators" the user wants to see.
            if op_type.startswith("aten.") or op_type.startswith("comm."):
                continue
            if op_type.startswith("optimizer."):
                continue

            entry = summary.setdefault(op_type, {
                "count": 0,
                "sample_names": [],
                "total_flops": 0.0,
                "dtype": None,
                "module_class": None,
            })
            entry["count"] += 1

            # Collect a friendly name from scope tail (e.g.
            # "transformer.layers.0.attn.wq_b" → "wq_b") or from the
            # leaf_attr stored on the node.  Keep up to 8 unique samples.
            name = node.name or (node.scope.rsplit(".", 1)[-1] if node.scope else "")
            if name and name not in entry["sample_names"] and len(entry["sample_names"]) < 8:
                entry["sample_names"].append(name)

            # Prefer the rule-derived sem_flops; fall back to the
            # downstream FlopsPass annotation.
            ann = node.annotations or {}
            flops = ann.get("sem_flops")
            if flops is None:
                flops = ann.get("flops")
            if isinstance(flops, (int, float)):
                entry["total_flops"] += float(flops)

            if entry["dtype"] is None:
                d = ann.get("sem_dtype")
                if d:
                    entry["dtype"] = d
                elif node.inputs:
                    entry["dtype"] = node.inputs[0].dtype.value
                elif node.outputs:
                    entry["dtype"] = node.outputs[0].dtype.value

            if entry["module_class"] is None and node.module_class:
                entry["module_class"] = node.module_class

    return summary