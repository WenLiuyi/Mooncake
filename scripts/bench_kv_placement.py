#!/usr/bin/env python3
"""Benchmark KV placement hint effect in Mooncake.

This script is intentionally focused on one question for a short-term thesis:

    Does setting ReplicateConfig.preferred_segment improve placement locality
    and reduce decode-side read latency?

Experiment design (kept simple on purpose):
1) Build two Mooncake clients: one "prefill" and one "decode".
2) Write the same number of keys in two modes:
   - baseline: no preferred segment hint
   - hint: preferred_segment = decode segment
3) Read all keys from decode client and record latency distribution.
4) Query replica descriptors to calculate placement hit rate.

The output CSV can be used directly for plotting or thesis tables.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from mooncake.mooncake_config import MooncakeConfig
from mooncake.store import MooncakeDistributedStore, ReplicateConfig


def _percentile(values: Sequence[float], q: float) -> float:
    """Return q percentile in milliseconds using linear interpolation.

    We avoid external dependencies (numpy/pandas) so that the script works in
    minimal environments.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@dataclass
class ExperimentResult:
    """One experiment cell result for (mode, workload)."""

    mode: str
    workload: str
    key_count: int
    placement_hit_rate: float
    put_failures: int
    get_failures: int
    get_p50_ms: float
    get_p99_ms: float
    get_avg_ms: float


def _apply_policy_env(policy: str, args: argparse.Namespace) -> Dict[str, str | None]:
    """Apply policy-specific env vars and return previous values for restore."""
    managed_keys = [
        "MC_MS_PREFERRED_SEGMENTS",
        "MC_MS_ONLINE_HINT_LEARNING",
        "MC_MS_ONLINE_HINT_LEARNING_CAPACITY",
        "MC_MS_ONLINE_HINT_MIN_HITS",
        "MC_MS_ONLINE_HINT_TTL_MS",
        "MC_MS_LAPS_STATIC_WEIGHT",
        "MC_MS_LAPS_LEARNED_WEIGHT",
    ]
    previous: Dict[str, str | None] = {k: os.environ.get(k) for k in managed_keys}

    # Start from a clean slate for every policy.
    for k in managed_keys:
        os.environ.pop(k, None)

    if policy == "baseline":
        return previous

    # static / laps / laps_ttl all use the same static candidate set.
    os.environ["MC_MS_PREFERRED_SEGMENTS"] = args.candidate_segments

    if policy in ("laps", "laps_ttl"):
        os.environ["MC_MS_ONLINE_HINT_LEARNING"] = "1"
        os.environ["MC_MS_ONLINE_HINT_LEARNING_CAPACITY"] = str(
            args.laps_learning_capacity
        )
        os.environ["MC_MS_ONLINE_HINT_MIN_HITS"] = str(args.laps_min_hits)
        os.environ["MC_MS_LAPS_STATIC_WEIGHT"] = str(args.laps_static_weight)
        os.environ["MC_MS_LAPS_LEARNED_WEIGHT"] = str(args.laps_learned_weight)
        if policy == "laps_ttl":
            os.environ["MC_MS_ONLINE_HINT_MIN_HITS"] = str(args.laps_ttl_min_hits)
            os.environ["MC_MS_ONLINE_HINT_TTL_MS"] = str(args.laps_ttl_ms)

    return previous


def _restore_env(previous: Dict[str, str | None]) -> None:
    """Restore managed environment vars after one policy run."""
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class PlacementBench:
    """Runs the placement benchmark end-to-end.

    We keep only two clients to match a 3~4 week project scope:
    - prefill_store: writes keys
    - decode_store: reads keys and represents the target locality
    """

    def __init__(self, base_cfg: MooncakeConfig, args: argparse.Namespace) -> None:
        self.args = args
        self.candidate_segments: List[str] = [
            s.strip() for s in args.candidate_segments.split(",") if s.strip()
        ]
        if args.decode_segment not in self.candidate_segments:
            self.candidate_segments.insert(0, args.decode_segment)
        if args.prefill_segment not in self.candidate_segments:
            self.candidate_segments.append(args.prefill_segment)

        # Build two configs with different logical hostnames (segment names).
        self.prefill_cfg = MooncakeConfig(
            local_hostname=args.prefill_segment,
            metadata_server=base_cfg.metadata_server,
            global_segment_size=args.global_segment_size,
            local_buffer_size=args.local_buffer_size,
            protocol=args.protocol,
            device_name=args.device_name,
            master_server_address=base_cfg.master_server_address,
        )
        self.decode_cfg = MooncakeConfig(
            local_hostname=args.decode_segment,
            metadata_server=base_cfg.metadata_server,
            global_segment_size=args.global_segment_size,
            local_buffer_size=args.local_buffer_size,
            protocol=args.protocol,
            device_name=args.device_name,
            master_server_address=base_cfg.master_server_address,
        )

        self.prefill_store = self._setup_store(self.prefill_cfg)
        self.decode_store = self._setup_store(self.decode_cfg)
        self.reader_stores: Dict[str, MooncakeDistributedStore] = {
            args.decode_segment: self.decode_store,
            args.prefill_segment: self.prefill_store,
        }
        for seg in self.candidate_segments:
            if seg in self.reader_stores:
                continue
            seg_cfg = MooncakeConfig(
                local_hostname=seg,
                metadata_server=base_cfg.metadata_server,
                global_segment_size=args.global_segment_size,
                local_buffer_size=args.local_buffer_size,
                protocol=args.protocol,
                device_name=args.device_name,
                master_server_address=base_cfg.master_server_address,
            )
            self.reader_stores[seg] = self._setup_store(seg_cfg)

    @staticmethod
    def _setup_store(cfg: MooncakeConfig) -> MooncakeDistributedStore:
        """Create and setup one MooncakeDistributedStore instance."""
        store = MooncakeDistributedStore()
        rc = store.setup(
            cfg.local_hostname,
            cfg.metadata_server,
            cfg.global_segment_size,
            cfg.local_buffer_size,
            cfg.protocol,
            cfg.device_name,
            cfg.master_server_address,
        )
        if rc != 0:
            raise RuntimeError(
                f"Failed to setup store for {cfg.local_hostname}, rc={rc}"
            )
        return store

    def _build_payload(self, key_idx: int) -> bytes:
        """Generate deterministic payload bytes for one key.

        Deterministic generation makes experiment reruns reproducible and keeps
        validation straightforward.
        """
        rng = random.Random(self.args.seed + key_idx)
        chars = [rng.choice(string.ascii_letters) for _ in range(self.args.value_bytes)]
        return ("".join(chars)).encode("utf-8")

    def _iter_decode_targets(self, key_count: int, workload: str) -> List[str]:
        """Return read target segment per key according to workload model.

        For this short project we keep two simple workloads:
        - uniform: all keys target decode segment (stable baseline)
        - sticky: sticky_ratio to decode segment, remainder to other segments
        - mixed: uniformly distributed over candidate segments

        Sticky mode emulates imperfect locality where some traffic still goes to
        non-primary nodes.
        """
        if workload == "uniform":
            return [self.args.decode_segment for _ in range(key_count)]

        if workload == "mixed":
            mixed_targets: List[str] = []
            for i in range(key_count):
                rng = random.Random(self.args.seed * 137 + i)
                mixed_targets.append(rng.choice(self.candidate_segments))
            return mixed_targets

        # sticky
        targets: List[str] = []
        sticky_ratio = self.args.sticky_ratio
        non_primary = [s for s in self.candidate_segments if s != self.args.decode_segment]
        if not non_primary:
            non_primary = [self.args.prefill_segment]
        for i in range(key_count):
            # Use deterministic pseudo-random branch for reproducibility.
            rng = random.Random(self.args.seed * 131 + i)
            if rng.random() < sticky_ratio:
                targets.append(self.args.decode_segment)
            else:
                targets.append(non_primary[i % len(non_primary)])
        return targets

    def _build_eval_targets(self, warmup_targets: List[str]) -> List[str]:
        """Build eval targets by optionally drifting from warmup targets."""
        drift_ratio = self.args.eval_drift_ratio
        if drift_ratio <= 0:
            return list(warmup_targets)

        eval_targets: List[str] = []
        for i, src in enumerate(warmup_targets):
            rng = random.Random(self.args.seed * 173 + i)
            if rng.random() >= drift_ratio:
                eval_targets.append(src)
                continue
            alternatives = [s for s in self.candidate_segments if s != src]
            if not alternatives:
                eval_targets.append(src)
                continue
            eval_targets.append(alternatives[i % len(alternatives)])
        return eval_targets

    def _build_eval_targets_for_workload(
        self, warmup_targets: List[str], workload: str
    ) -> List[str]:
        """Build eval targets with optional workload-specific drift override."""
        base_ratio = self.args.eval_drift_ratio
        if workload == "sticky" and self.args.eval_drift_ratio_sticky >= 0:
            base_ratio = self.args.eval_drift_ratio_sticky
        elif workload == "mixed" and self.args.eval_drift_ratio_mixed >= 0:
            base_ratio = self.args.eval_drift_ratio_mixed

        original_ratio = self.args.eval_drift_ratio
        try:
            # Reuse existing drift implementation with a temporary ratio override.
            self.args.eval_drift_ratio = base_ratio
            return self._build_eval_targets(warmup_targets)
        finally:
            self.args.eval_drift_ratio = original_ratio

    def _build_replicate_config(
        self, mode: str, target_segment: str, hint_mode_override: str | None = None
    ) -> ReplicateConfig:
        """Create ReplicateConfig for one policy mode."""
        cfg = ReplicateConfig()
        cfg.replica_num = 1

        if mode in ("hint", "static", "laps", "laps_ttl"):
            hint_mode = hint_mode_override or self.args.static_hint_mode
            if hint_mode == "decode_only":
                cfg.preferred_segment = self.args.decode_segment
            elif hint_mode == "oracle":
                cfg.preferred_segment = target_segment

        return cfg

    @staticmethod
    def _memory_endpoints(store: MooncakeDistributedStore, key: str) -> List[str]:
        """Extract memory replica endpoints for one key.

        Endpoint string corresponds to segment hostname used in setup().
        """
        descs = store.get_replica_desc(key)
        endpoints: List[str] = []
        for desc in descs:
            if getattr(desc, "is_memory_replica", False):
                mem_desc = desc.get_memory_descriptor()
                endpoints.append(mem_desc.buffer_descriptor.transport_endpoint)
        return endpoints

    def run_cell(self, mode: str, workload: str) -> Tuple[ExperimentResult, List[Dict[str, str]]]:
        """Run one experiment cell and return both summary and raw rows."""
        key_count = self.args.num_keys
        prefix = f"bench_{mode}_{workload}_{int(time.time())}"
        warmup_targets = self._iter_decode_targets(key_count, workload)
        eval_targets = self._build_eval_targets_for_workload(warmup_targets, workload)

        put_failures = 0
        get_failures = 0
        latency_ms: List[float] = []
        placement_hits = 0
        raw_rows: List[Dict[str, str]] = []

        # Build key/payload once, then optionally run two phases with key reuse.
        keys: List[str] = [f"{prefix}_k{i:06d}" for i in range(key_count)]
        payloads: List[bytes] = [self._build_payload(i) for i in range(key_count)]

        def write_phase(
            phase_targets: Sequence[str],
            write_mode: str,
            hint_mode_override: str | None = None,
        ) -> None:
            nonlocal put_failures
            for i, key in enumerate(keys):
                cfg = self._build_replicate_config(
                    mode, phase_targets[i], hint_mode_override=hint_mode_override
                )
                if write_mode == "upsert":
                    if hasattr(self.prefill_store, "upsert"):
                        rc = self.prefill_store.upsert(key, payloads[i], cfg)
                    else:
                        rc = self.prefill_store.put(key, payloads[i], cfg)
                elif write_mode == "put_recreate":
                    try:
                        self.prefill_store.remove(key, False)
                    except Exception:
                        pass
                    rc = self.prefill_store.put(key, payloads[i], cfg)
                else:
                    rc = self.prefill_store.put(key, payloads[i], cfg)
                if rc != 0:
                    put_failures += 1

        def read_phase(
            phase_targets: Sequence[str],
            record_metrics: bool,
            feedback_client: str = "reader",
        ) -> None:
            nonlocal placement_hits, get_failures

            # Placement snapshot for this phase.
            local_raw_rows: List[Dict[str, str]] = []
            for i, key in enumerate(keys):
                reader_segment = phase_targets[i]
                if feedback_client == "writer":
                    reader_store = self.prefill_store
                else:
                    reader_store = self.reader_stores.get(reader_segment, self.decode_store)
                endpoints = self._memory_endpoints(reader_store, key)
                expected = reader_segment
                hit = expected in endpoints
                if record_metrics:
                    placement_hits += int(hit)
                local_raw_rows.append(
                    {
                        "mode": mode,
                        "workload": workload,
                        "key": key,
                        "expected_segment": expected,
                        "replica_endpoints": "|".join(endpoints),
                        "placement_hit": "1" if hit else "0",
                        "get_latency_ms": "",
                        "get_rc": "",
                    }
                )

            for i, key in enumerate(keys):
                reader_segment = phase_targets[i]
                if feedback_client == "writer":
                    reader_store = self.prefill_store
                else:
                    reader_store = self.reader_stores.get(reader_segment, self.decode_store)
                t0 = time.perf_counter()
                ret = reader_store.get(key)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                # Python bindings differ across versions:
                # - some return bytes directly
                # - some return (bytes, rc)
                if isinstance(ret, tuple) and len(ret) == 2:
                    data, rc = ret
                else:
                    data = ret
                    rc = 0 if data is not None else -1

                if record_metrics:
                    if rc < 0 or data is None or data == b"\0" or len(data) == 0:
                        get_failures += 1
                    else:
                        # Optional lightweight correctness check:
                        # verify retrieved size only (keeps benchmark overhead low).
                        if len(data) != self.args.value_bytes:
                            get_failures += 1
                    latency_ms.append(elapsed_ms)
                    local_raw_rows[i]["get_latency_ms"] = f"{elapsed_ms:.6f}"
                    local_raw_rows[i]["get_rc"] = str(rc)

            if record_metrics:
                raw_rows.extend(local_raw_rows)

        # For learning policies, run warmup then eval on the same keys so
        # online feedback can influence subsequent writes.
        if mode in ("laps", "laps_ttl"):
            write_phase(
                warmup_targets,
                write_mode="put",
                hint_mode_override=self.args.laps_warmup_hint_mode,
            )  # warmup write
            warmup_rounds = max(1, self.args.warmup_read_rounds)
            for _ in range(warmup_rounds):
                read_phase(
                    warmup_targets,
                    record_metrics=False,
                    feedback_client=self.args.warmup_feedback_client,
                )  # warmup read: only train
            eval_delay_ms = (
                self.args.laps_ttl_eval_delay_ms if mode == "laps_ttl" else self.args.laps_eval_delay_ms
            )
            if eval_delay_ms > 0:
                time.sleep(eval_delay_ms / 1000.0)
            write_phase(eval_targets, write_mode=self.args.eval_write_mode)  # eval write with learned hints
            read_phase(eval_targets, record_metrics=True)      # eval read: record metrics
        else:
            write_phase(eval_targets, write_mode="put")
            read_phase(eval_targets, record_metrics=True)

        hit_rate = (placement_hits / key_count) if key_count else 0.0
        result = ExperimentResult(
            mode=mode,
            workload=workload,
            key_count=key_count,
            placement_hit_rate=hit_rate,
            put_failures=put_failures,
            get_failures=get_failures,
            get_p50_ms=_percentile(latency_ms, 0.50),
            get_p99_ms=_percentile(latency_ms, 0.99),
            get_avg_ms=statistics.mean(latency_ms) if latency_ms else 0.0,
        )

        # Best-effort cleanup so repeated runs do not collide on stale keys.
        for key in keys:
            try:
                self.prefill_store.remove(key, False)
            except Exception:
                pass

        return result, raw_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Mooncake KV placement optimization using preferred_segment hint"
        )
    )
    parser.add_argument("--num-keys", type=int, default=500, help="Number of keys per experiment cell")
    parser.add_argument("--value-bytes", type=int, default=4096, help="Payload bytes per key")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for deterministic payload/workload generation")
    parser.add_argument("--sticky-ratio", type=float, default=0.8, help="Sticky workload probability for decode segment")
    parser.add_argument(
        "--policies",
        default="baseline,static,laps,laps_ttl",
        help="Comma-separated policy list: baseline,static,laps,laps_ttl",
    )
    parser.add_argument(
        "--workloads",
        default="uniform,sticky,mixed",
        help="Comma-separated workload list: uniform,sticky,mixed",
    )

    # Segment identity for two logical roles.
    parser.add_argument("--prefill-segment", default="localhost:19101")
    parser.add_argument("--decode-segment", default="localhost:19102")

    # Setup configuration (override env-loaded defaults when needed).
    parser.add_argument("--protocol", default="tcp", choices=["tcp", "rdma"], help="Transport protocol used by Mooncake client")
    parser.add_argument("--device-name", default="", help="RDMA device name when protocol=rdma")
    parser.add_argument("--global-segment-size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--local-buffer-size", type=int, default=256 * 1024 * 1024)
    parser.add_argument(
        "--candidate-segments",
        default="localhost:19102,localhost:19103",
        help="Static candidate segments used by static/laps/laps_ttl policies",
    )
    parser.add_argument("--laps-learning-capacity", type=int, default=10000)
    parser.add_argument("--laps-static-weight", type=float, default=1.0)
    parser.add_argument("--laps-learned-weight", type=float, default=2.0)
    parser.add_argument(
        "--laps-min-hits",
        type=int,
        default=1,
        help="Min learned hits required for laps (non-ttl) to take effect",
    )
    parser.add_argument("--laps-ttl-ms", type=int, default=30000)
    parser.add_argument("--laps-ttl-min-hits", type=int, default=2)
    parser.add_argument(
        "--warmup-read-rounds",
        type=int,
        default=1,
        help="Number of warmup read rounds before eval for laps policies",
    )
    parser.add_argument(
        "--static-hint-mode",
        choices=["oracle", "decode_only", "none"],
        default="decode_only",
        help="oracle: per-key target hint; decode_only: fixed decode-segment hint",
    )
    parser.add_argument(
        "--laps-warmup-hint-mode",
        choices=["oracle", "decode_only", "none"],
        default="oracle",
        help="Hint mode used only for laps/laps_ttl warmup write phase",
    )
    parser.add_argument(
        "--warmup-feedback-client",
        choices=["reader", "writer"],
        default="writer",
        help="Client used for warmup get feedback updates: reader or writer client",
    )
    parser.add_argument(
        "--eval-write-mode",
        choices=["upsert", "put", "put_recreate"],
        default="put_recreate",
        help="Eval write behavior for laps/laps_ttl: upsert, put, or remove+put",
    )
    parser.add_argument(
        "--eval-drift-ratio",
        type=float,
        default=0.0,
        help="Probability that eval target drifts away from warmup target",
    )
    parser.add_argument(
        "--eval-drift-ratio-sticky",
        type=float,
        default=-1.0,
        help="Optional sticky-specific eval drift ratio; negative means use --eval-drift-ratio",
    )
    parser.add_argument(
        "--eval-drift-ratio-mixed",
        type=float,
        default=-1.0,
        help="Optional mixed-specific eval drift ratio; negative means use --eval-drift-ratio",
    )
    parser.add_argument(
        "--laps-eval-delay-ms",
        type=int,
        default=0,
        help="Delay between warmup and eval for laps (used to model staleness)",
    )
    parser.add_argument(
        "--laps-ttl-eval-delay-ms",
        type=int,
        default=0,
        help="Delay between warmup and eval for laps_ttl (set > ttl to force expiry)",
    )

    parser.add_argument(
        "--output-dir",
        default="scripts/results",
        help="Directory for summary/raw CSV outputs",
    )
    return parser.parse_args()


def write_outputs(
    output_dir: Path,
    summary: List[ExperimentResult],
    raw_rows: List[Dict[str, str]],
) -> None:
    """Write summary and per-key raw rows to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())

    summary_path = output_dir / f"placement_summary_{ts}.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "workload",
                "key_count",
                "placement_hit_rate",
                "put_failures",
                "get_failures",
                "get_avg_ms",
                "get_p50_ms",
                "get_p99_ms",
            ]
        )
        for row in summary:
            writer.writerow(
                [
                    row.mode,
                    row.workload,
                    row.key_count,
                    f"{row.placement_hit_rate:.6f}",
                    row.put_failures,
                    row.get_failures,
                    f"{row.get_avg_ms:.6f}",
                    f"{row.get_p50_ms:.6f}",
                    f"{row.get_p99_ms:.6f}",
                ]
            )

    raw_path = output_dir / f"placement_raw_{ts}.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "mode",
            "workload",
            "key",
            "expected_segment",
            "replica_endpoints",
            "placement_hit",
            "get_latency_ms",
            "get_rc",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row)

    print(f"[output] summary: {summary_path}")
    print(f"[output] raw:     {raw_path}")

    # Also generate a compact markdown comparison table for reports.
    by_workload: Dict[str, List[ExperimentResult]] = {}
    for row in summary:
        by_workload.setdefault(row.workload, []).append(row)

    md_lines: List[str] = []
    for workload, rows in by_workload.items():
        md_lines.append(f"## Workload: {workload}")
        md_lines.append("")
        md_lines.append(
            "| policy | key_count | placement_hit_rate | get_avg_ms | get_p50_ms | get_p99_ms | put_failures | get_failures |"
        )
        md_lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        )
        for r in rows:
            md_lines.append(
                f"| {r.mode} | {r.key_count} | {r.placement_hit_rate:.4f} | {r.get_avg_ms:.6f} | {r.get_p50_ms:.6f} | {r.get_p99_ms:.6f} | {r.put_failures} | {r.get_failures} |"
            )
        md_lines.append("")

    compare_md_path = output_dir / f"placement_compare_{ts}.md"
    compare_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[output] compare: {compare_md_path}")


def print_result(result: ExperimentResult) -> None:
    """Pretty-print one cell summary to stdout."""
    print(
        " | ".join(
            [
                f"mode={result.mode}",
                f"workload={result.workload}",
                f"keys={result.key_count}",
                f"hit_rate={result.placement_hit_rate:.2%}",
                f"put_fail={result.put_failures}",
                f"get_fail={result.get_failures}",
                f"avg={result.get_avg_ms:.3f}ms",
                f"p50={result.get_p50_ms:.3f}ms",
                f"p99={result.get_p99_ms:.3f}ms",
            ]
        )
    )


def main() -> None:
    args = parse_args()

    if args.protocol == "cxl":
        raise ValueError("This benchmark should not run with protocol=cxl.")

    base_cfg = MooncakeConfig.load_from_env()
    summary: List[ExperimentResult] = []
    raw_rows: List[Dict[str, str]] = []

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]

    for policy in policies:
        previous_env = _apply_policy_env(policy, args)
        try:
            bench = PlacementBench(base_cfg, args)
            for workload in workloads:
                result, rows = bench.run_cell(mode=policy, workload=workload)
                summary.append(result)
                raw_rows.extend(rows)
                print_result(result)
            # Ensure client objects are released before next policy.
            del bench
        finally:
            _restore_env(previous_env)

    write_outputs(Path(args.output_dir), summary, raw_rows)


if __name__ == "__main__":
    main()
