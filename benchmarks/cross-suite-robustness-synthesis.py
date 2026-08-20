#!/usr/bin/env python3
"""Deterministic P4-U06 cross-suite inventory and scenario-target coverage synthesis.

This program reads only the five stable Phase 4 benchmark artifact trees.  It
never reruns a method and never copies, normalizes, pools, averages, ranks, or
scores task-specific metric values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCHEMA_VERSION = 1
DATE = "2026-08-20"
ARTIFACT_FILES = ("manifest.json", "results.csv", "diagnostics.json", "summary.png")
CONTENT_FILES = ("results.csv", "diagnostics.json", "summary.png")
SCIENTIFIC_SOURCE_FILES = ("manifest.json", "results.csv", "diagnostics.json")
OPTIONAL_INTEGRITY_ATTACHMENTS = ("summary.png",)
STATUS_VOCABULARY = ("ok", "inapplicable", "invalid", "nonconvergence")
SOURCE_SPECS = (
    ("linear-gaussian-robustness", "linear_gaussian_inference"),
    ("static-linear-latent-robustness", "static_linear_latent_recovery"),
    ("temporal-covariance-inference", "temporal_covariance_inference"),
    ("demixing-rotational-robustness", None),
    ("nonlinear-graph-robustness", None),
)
ROW_FIELDS = (
    "row_key",
    "record_type",
    "benchmark",
    "task",
    "scenario",
    "scenario_class",
    "method",
    "dimension",
    "value_label",
    "numerator",
    "denominator",
    "rate",
    "status",
    "stress_class",
    "applicable_target_families",
    "source_reference",
    "notes",
)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    for key in ("files_sha256", "artifact_hashes_sha256"):
        value = manifest.get(key)
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        out: dict[str, str] = {}
        for name, metadata in artifacts.items():
            if isinstance(metadata, dict) and "sha256" in metadata:
                out[str(name)] = str(metadata["sha256"])
        if out:
            return out
    raise ValueError("manifest_missing_content_hashes")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing_csv_header:{path}")
        return list(reader.fieldnames), list(reader)


def benchmark_name(manifest: dict[str, Any], directory_name: str) -> str:
    for key in ("object_id", "benchmark"):
        value = manifest.get(key)
        if isinstance(value, str) and value:
            return value
    artifact = manifest.get("artifact")
    if isinstance(artifact, dict):
        for key in ("id", "name", "benchmark"):
            value = artifact.get(key)
            if isinstance(value, str) and value:
                return value
    return directory_name


def verify_source_tree(source_root: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for directory_name, default_task in SOURCE_SPECS:
        directory = source_root / directory_name
        missing = [name for name in SCIENTIFIC_SOURCE_FILES if not (directory / name).is_file()]
        if missing:
            raise ValueError(f"source_missing_files:{directory_name}:{','.join(missing)}")
        manifest_path = directory / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("maturity") != "L1" or manifest.get("status") != "stable":
            raise ValueError(
                f"source_not_stable:{directory_name}:maturity={manifest.get('maturity')}:status={manifest.get('status')}"
            )
        expected_hashes = manifest_hashes(manifest)
        actual_hashes: dict[str, str] = {}
        for name in ("results.csv", "diagnostics.json"):
            if name not in expected_hashes:
                raise ValueError(f"source_manifest_missing_hash:{directory_name}:{name}")
            actual = sha256_file(directory / name)
            actual_hashes[name] = actual
            if actual != expected_hashes[name]:
                raise ValueError(f"source_scientific_content_drift:{directory_name}:{name}")
        integrity_attachments: dict[str, dict[str, Any]] = {}
        for name in OPTIONAL_INTEGRITY_ATTACHMENTS:
            path = directory / name
            if not path.is_file():
                continue
            actual = sha256_file(path)
            declared = expected_hashes.get(name)
            if declared is not None and actual != declared:
                raise ValueError(f"source_integrity_attachment_drift:{directory_name}:{name}")
            integrity_attachments[name] = {
                "role": "integrity_only_attachment",
                "sha256": actual,
                "bytes": path.stat().st_size,
                "declared_hash_present": declared is not None,
            }
        fields, rows = read_csv(directory / "results.csv")
        required = {"scenario", "replicate", "method", "information_set", "target", "metric", "status"}
        absent = sorted(required - set(fields))
        if absent:
            raise ValueError(f"source_results_missing_fields:{directory_name}:{','.join(absent)}")
        unknown_statuses = sorted({row["status"] for row in rows} - set(STATUS_VOCABULARY))
        if unknown_statuses:
            raise ValueError(f"source_unknown_status:{directory_name}:{','.join(unknown_statuses)}")
        diagnostics_bytes = (directory / "diagnostics.json").read_bytes()
        diagnostics = json.loads(diagnostics_bytes)
        diagnostics_counts = diagnostics.get("status_counts")
        row_counts = Counter(row["status"] for row in rows)
        if isinstance(diagnostics_counts, dict):
            for status in STATUS_VOCABULARY:
                if int(diagnostics_counts.get(status, 0)) != int(row_counts.get(status, 0)):
                    raise ValueError(f"source_status_count_drift:{directory_name}:{status}")
        sources.append(
            {
                "directory": directory_name,
                "benchmark": benchmark_name(manifest, directory_name),
                "default_task": default_task,
                "manifest": manifest,
                "diagnostics": diagnostics,
                "fields": fields,
                "rows": rows,
                "digests": {
                    "manifest.json": sha256_bytes(manifest_bytes),
                    "results.csv": actual_hashes["results.csv"],
                    "diagnostics.json": actual_hashes["diagnostics.json"],
                },
                "bytes": {name: (directory / name).stat().st_size for name in SCIENTIFIC_SOURCE_FILES},
                "integrity_only_attachments": integrity_attachments,
            }
        )
    return sources


def source_task(source: dict[str, Any], row: dict[str, str]) -> str:
    task = row.get("task", "").strip()
    if task:
        return task
    default = source["default_task"]
    if not default:
        raise ValueError(f"source_task_missing:{source['directory']}")
    return default


def source_provenance(row: dict[str, str]) -> str:
    value = row.get("provenance", "").strip()
    if value:
        return value
    split = row.get("split", "").strip()
    return f"split:{split}" if split else "unspecified"


def make_row(
    *, record_type: str, benchmark: str, task: str, scenario: str = "", scenario_class: str = "",
    method: str = "", dimension: str = "", value_label: str = "", numerator: int | str = "",
    denominator: int | str = "", rate: float | str = "", status: str = "ok", stress_class: str = "",
    applicable_target_families: str = "", source_reference: str = "", notes: str = "",
) -> dict[str, str]:
    payload = [record_type, benchmark, task, scenario, method, dimension, value_label]
    key = "|".join(str(part) for part in payload)
    if isinstance(rate, float):
        rate_text = format(rate, ".12g")
    else:
        rate_text = str(rate)
    return {
        "row_key": key,
        "record_type": record_type,
        "benchmark": benchmark,
        "task": task,
        "scenario": scenario,
        "scenario_class": scenario_class,
        "method": method,
        "dimension": dimension,
        "value_label": value_label,
        "numerator": str(numerator),
        "denominator": str(denominator),
        "rate": rate_text,
        "status": status,
        "stress_class": stress_class,
        "applicable_target_families": applicable_target_families,
        "source_reference": source_reference,
        "notes": notes,
    }


def build_rows(sources: Iterable[dict[str, Any]], reverse: bool = False) -> list[dict[str, str]]:
    source_list = list(sources)
    if reverse:
        source_list.reverse()
    output: list[dict[str, str]] = []
    for source in source_list:
        benchmark = source["directory"]
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        source_rows = list(source["rows"])
        if reverse:
            source_rows.reverse()
        for row in source_rows:
            grouped[source_task(source, row)].append(row)
        task_items = list(grouped.items())
        if reverse:
            task_items.reverse()
        for task, task_rows in task_items:
            source_ref = f"{benchmark}/results.csv"
            scenarios = sorted({row["scenario"] for row in task_rows})
            replicates = sorted({row["replicate"] for row in task_rows})
            methods = sorted({row["method"] for row in task_rows})
            metrics = sorted({row["metric"] for row in task_rows})
            targets = sorted({row["target"] for row in task_rows})
            info_sets = sorted({row["information_set"] for row in task_rows})
            provenances = sorted({source_provenance(row) for row in task_rows})
            inventory = {
                "source_row_count": len(task_rows),
                "scenario_count": len(scenarios),
                "replicate_count": len(replicates),
                "method_count": len(methods),
                "metric_count": len(metrics),
                "target_count": len(targets),
                "information_set_count": len(info_sets),
                "provenance_count": len(provenances),
            }
            for dimension, count in inventory.items():
                output.append(make_row(
                    record_type="inventory", benchmark=benchmark, task=task,
                    dimension=dimension, numerator=count, denominator=count, rate=1.0,
                    source_reference=source_ref,
                    notes="task-specific inventory count; no source metric values imported",
                ))
            for dimension, values in (
                ("method_member", methods),
                ("metric_member", metrics),
                ("target_member", targets),
                ("information_set_member", info_sets),
                ("provenance_member", provenances),
            ):
                for value in values:
                    output.append(make_row(
                        record_type="inventory", benchmark=benchmark, task=task,
                        dimension=dimension, value_label=value, numerator=1, denominator=1, rate=1.0,
                        source_reference=source_ref, notes="namespaced task vocabulary member",
                    ))
            status_counts = Counter(row["status"] for row in task_rows)
            for source_status in STATUS_VOCABULARY:
                count = status_counts.get(source_status, 0)
                output.append(make_row(
                    record_type="inventory", benchmark=benchmark, task=task,
                    dimension="status_count", value_label=source_status, numerator=count,
                    denominator=len(task_rows), rate=count / len(task_rows), source_reference=source_ref,
                    notes="rate denominator is the task-specific source raw-row count",
                ))

            scenario_method: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
            scenario_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in task_rows:
                scenario_method[(row["scenario"], row["method"])].append(row)
                scenario_rows[row["scenario"]].append(row)
            pairs = list(scenario_method.items())
            if reverse:
                pairs.reverse()
            for (scenario, method), pair_rows in pairs:
                raw_denominator = len(pair_rows)
                applicable_denominator = sum(row["status"] != "inapplicable" for row in pair_rows)
                failures = sum(row["status"] in {"invalid", "nonconvergence"} for row in pair_rows)
                scenario_classes = sorted({row.get("scenario_class", "") or "unspecified" for row in pair_rows})
                scenario_class = ";".join(scenario_classes)
                output.append(make_row(
                    record_type="applicability_pattern", benchmark=benchmark, task=task,
                    scenario=scenario, scenario_class=scenario_class, method=method,
                    dimension="applicability_rate", numerator=applicable_denominator,
                    denominator=raw_denominator, rate=applicable_denominator / raw_denominator,
                    source_reference=source_ref,
                    notes="applicable means source status is not inapplicable; denominator is task/scenario/method raw Cartesian rows",
                ))
                output.append(make_row(
                    record_type="failure_pattern", benchmark=benchmark, task=task,
                    scenario=scenario, scenario_class=scenario_class, method=method,
                    dimension="raw_cartesian_failure_label_fraction", numerator=failures,
                    denominator=raw_denominator, rate=failures / raw_denominator,
                    source_reference=source_ref,
                    notes="failures are invalid plus nonconvergence; denominator is all task/scenario/method raw Cartesian rows",
                ))
                if applicable_denominator:
                    output.append(make_row(
                        record_type="failure_pattern", benchmark=benchmark, task=task,
                        scenario=scenario, scenario_class=scenario_class, method=method,
                        dimension="conditional_failure_rate", numerator=failures,
                        denominator=applicable_denominator, rate=failures / applicable_denominator,
                        source_reference=source_ref,
                        notes="failures are invalid plus nonconvergence; denominator is ok plus invalid plus nonconvergence rows",
                    ))
                else:
                    output.append(make_row(
                        record_type="failure_pattern", benchmark=benchmark, task=task,
                        scenario=scenario, scenario_class=scenario_class, method=method,
                        dimension="conditional_failure_rate", status="inapplicable",
                        source_reference=source_ref,
                        notes="conditional denominator is zero because every source row is inapplicable; numerator, denominator, and rate are blank",
                    ))
            scenario_items = list(scenario_rows.items())
            if reverse:
                scenario_items.reverse()
            for scenario, scoped in scenario_items:
                stress_classes = sorted({row.get("scenario_class", "") or "unspecified" for row in scoped})
                applicable_targets = sorted({row["target"] for row in scoped if row["status"] != "inapplicable"})
                coverage_status = "ok" if applicable_targets else "inapplicable"
                output.append(make_row(
                    record_type="scenario_target_coverage", benchmark=benchmark, task=task,
                    scenario=scenario, scenario_class=";".join(stress_classes),
                    dimension="applicable_target_family_inventory", status=coverage_status,
                    stress_class=";".join(stress_classes),
                    applicable_target_families=";".join(applicable_targets),
                    numerator=len(applicable_targets) if applicable_targets else "",
                    denominator=len(applicable_targets) if applicable_targets else "",
                    rate=1.0 if applicable_targets else "", source_reference=source_ref,
                    notes="scenario-level inventory of target families with at least one non-inapplicable source row; no direction or effect implication",
                ))
    output.sort(key=lambda row: row["row_key"])
    keys = [row["row_key"] for row in output]
    if len(keys) != len(set(keys)):
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        raise ValueError(f"duplicate_synthesis_keys:{duplicates[:3]}")
    return output

def csv_bytes(rows: list[dict[str, str]]) -> bytes:
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def task_inventory(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_task: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        if row["record_type"] != "inventory":
            continue
        if row["dimension"] in {"scenario_count", "replicate_count", "method_count", "metric_count", "target_count", "source_row_count"}:
            by_task[(row["benchmark"], row["task"])][row["dimension"]] = int(row["numerator"])
        if row["dimension"] == "status_count":
            by_task[(row["benchmark"], row["task"])].setdefault("status_counts", {})[row["value_label"]] = int(row["numerator"])
    return [dict(benchmark=k[0], task=k[1], **by_task[k]) for k in sorted(by_task)]


def draw_summary(rows: list[dict[str, str]]) -> bytes:
    inventory = task_inventory(rows)
    width, height = 1600, 940
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((45, 28), "P4-U06 Cross-suite coverage inventory", fill="black", font=font)
    draw.text((45, 48), "Counts and label coverage only — no pooled metrics, ranks, scores, or winners", fill=(70, 70, 70), font=font)
    labels = [f"{x['benchmark']} / {x['task']}" for x in inventory]
    colors = {"scenario_count": (65, 105, 225), "method_count": (46, 139, 87), "metric_count": (220, 120, 55), "target_count": (130, 80, 175)}
    panel_left, panel_top, panel_right, panel_bottom = 60, 100, 1540, 520
    draw.rectangle((panel_left, panel_top, panel_right, panel_bottom), outline=(180, 180, 180), width=1)
    max_count = max([1] + [x.get(k, 0) for x in inventory for k in colors])
    row_h = max(42, (panel_bottom - panel_top - 40) // max(1, len(inventory)))
    label_w = 470
    bar_w = panel_right - panel_left - label_w - 60
    for i, item in enumerate(inventory):
        y = panel_top + 28 + i * row_h
        draw.text((panel_left + 10, y), labels[i][:70], fill="black", font=font)
        for j, (key, color) in enumerate(colors.items()):
            val = item.get(key, 0)
            yy = y + 14 + j * 7
            length = int(bar_w * val / max_count)
            draw.rectangle((panel_left + label_w, yy, panel_left + label_w + length, yy + 4), fill=color)
            draw.text((panel_left + label_w + length + 5, yy - 3), f"{key.replace('_count','')}={val}", fill=color, font=font)
    draw.text((panel_left + 10, panel_bottom - 20), "Blue=scenarios  Green=methods  Orange=metrics  Purple=targets (task-specific inventories)", fill=(70, 70, 70), font=font)

    y0, y1 = 570, 875
    draw.rectangle((panel_left, y0, panel_right, y1), outline=(180, 180, 180), width=1)
    draw.text((panel_left + 10, y0 + 10), "Task-specific status composition (denominator = source raw rows for that task)", fill="black", font=font)
    status_colors = {"ok": (46, 139, 87), "inapplicable": (160, 160, 160), "invalid": (205, 70, 70), "nonconvergence": (230, 150, 45)}
    row_h2 = max(34, (y1 - y0 - 50) // max(1, len(inventory)))
    for i, item in enumerate(inventory):
        y = y0 + 35 + i * row_h2
        draw.text((panel_left + 10, y), labels[i][:70], fill="black", font=font)
        counts = item.get("status_counts", {})
        total = max(1, sum(counts.values()))
        x = panel_left + label_w
        usable = bar_w
        for status in STATUS_VOCABULARY:
            count = counts.get(status, 0)
            segment = int(round(usable * count / total))
            if segment:
                draw.rectangle((x, y, x + segment, y + 12), fill=status_colors[status])
            x += segment
        text_label = "  ".join(f"{s}={counts.get(s,0)}/{total}" for s in STATUS_VOCABULARY)
        draw.text((panel_left + label_w, y + 15), text_label, fill=(60, 60, 60), font=font)
    draw.text((45, 905), "Scenario-target coverage records list applicable target families only; no direction, magnitude, or source metric value is imported.", fill=(70, 70, 70), font=font)
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def diagnostics_document(rows: list[dict[str, str]], sources: list[dict[str, Any]]) -> dict[str, Any]:
    record_counts = Counter(row["record_type"] for row in rows)
    dimension_counts = Counter(row["dimension"] for row in rows)
    inventories = task_inventory(rows)
    conditional_rows = [row for row in rows if row["dimension"] == "conditional_failure_rate"]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "cross-suite-robustness-synthesis",
        "row_count": len(rows),
        "record_type_counts": dict(sorted(record_counts.items())),
        "dimension_counts": dict(sorted(dimension_counts.items())),
        "conditional_failure_rate_counts": {
            "ok": sum(row["status"] == "ok" for row in conditional_rows),
            "inapplicable_zero_denominator": sum(row["status"] == "inapplicable" for row in conditional_rows),
        },
        "task_inventory": inventories,
        "source_count": len(sources),
        "synthesis_limits": [
            "No source metric value is copied, normalized, averaged, ranked, or pooled across tasks.",
            "Applicability and failure rates retain explicit raw-row denominators and remain benchmark/task/scenario/method specific.",
            "Scenario-target coverage records inventory declared scenario classes and applicable target families without direction or magnitude implications.",
            "The synthesis does not establish mechanism, causality, biological validity, real-data transfer, or universal method superiority.",
        ],
        "verification_checks": {
            "stable_source_manifests_required": True,
            "source_scientific_content_hashes_verified": True,
            "source_digests_recorded": True,
            "summary_png_integrity_only_not_scientific_input": True,
            "canonical_unique_rows": True,
            "no_source_metric_values_imported": True,
            "rates_have_explicit_denominators": True,
            "task_specific_only": True,
            "no_rank_score_or_winner_fields": True,
        },
    }


def manifest_document(rows: list[dict[str, str]], sources: list[dict[str, Any]], content: dict[str, bytes]) -> dict[str, Any]:
    source_records = []
    for source in sources:
        source_records.append({
            "benchmark": source["directory"],
            "declared_benchmark": source["benchmark"],
            "maturity": source["manifest"]["maturity"],
            "status": source["manifest"]["status"],
            "files_sha256": dict(sorted(source["digests"].items())),
            "files_bytes": dict(sorted(source["bytes"].items())),
            "integrity_only_attachments": source["integrity_only_attachments"],
            "scientific_inputs": ["manifest.json", "results.csv", "diagnostics.json"],
            "row_count": len(source["rows"]),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "date": DATE,
        "benchmark": "cross-suite-robustness-synthesis",
        "maturity": "L1",
        "status": "stable",
        "scientific_question": "What task-specific coverage, applicability/failure patterns, and declared stress-to-target mappings are present across the five stable Phase 4 synthetic benchmark suites?",
        "contract": {
            "read_only_sources": True,
            "stable_sources_only": True,
            "source_method_reruns": False,
            "source_metric_values_imported": False,
            "rates_task_specific_with_denominators": True,
            "failure_rate_semantics": {
                "raw_cartesian_failure_label_fraction": "(invalid + nonconvergence) / all task-scenario-method raw Cartesian rows",
                "conditional_failure_rate": "(invalid + nonconvergence) / (ok + invalid + nonconvergence); blank and inapplicable when denominator is zero",
            },
            "source_boundary": {
                "scientific_inputs": ["manifest.json", "results.csv", "diagnostics.json"],
                "summary_png": "optional integrity_only_attachment; never parsed as scientific input",
            },
            "cross_task_metric_normalization": False,
            "cross_task_metric_averaging": False,
            "method_ranking": False,
            "score": False,
            "winner": False,
        },
        "sources": source_records,
        "row_fields": list(ROW_FIELDS),
        "row_key": "record_type|benchmark|task|scenario|method|dimension|value_label",
        "canonical_order": "lexicographic row_key",
        "row_count": len(rows),
        "record_types": ["inventory", "applicability_pattern", "failure_pattern", "scenario_target_coverage"],
        "status_vocabulary": list(STATUS_VOCABULARY),
        "artifact_files": list(ARTIFACT_FILES),
        "artifact_hashes_sha256": {name: sha256_bytes(content[name]) for name in CONTENT_FILES},
        "runtime": {
            "executable": sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": getattr(sys.modules.get("PIL"), "__version__", "unknown"),
            "platform": platform.platform(),
        },
        "commands": {
            "generate": f"{sys.executable} benchmarks/cross-suite-robustness-synthesis.py --generate",
            "verify": f"{sys.executable} benchmarks/cross-suite-robustness-synthesis.py --verify",
            "resume": f"{sys.executable} benchmarks/cross-suite-robustness-synthesis.py --generate --resume",
        },
        "interpretation_boundary": [
            "Inventory counts are not evidence of scientific quality or method superiority.",
            "Applicability and failure rates are conditional on each source task's own row design and must not be compared as normalized performance scores.",
            "Scenario-target coverage membership records applicability coverage only, not direction, magnitude, or robustness.",
            "No universal winner, mechanism, causal conclusion, or real-data claim is licensed.",
        ],
    }


def validate_synthesis_rows(rows: list[dict[str, str]]) -> None:
    keys = [row["row_key"] for row in rows]
    if keys != sorted(keys):
        raise ValueError("noncanonical_synthesis_order")
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate_synthesis_key")
    prohibited_columns = {"affected_target_families", "sensitivity", "value", "effect_size"}
    if prohibited_columns & set(ROW_FIELDS):
        raise ValueError("prohibited_synthesis_column")
    allowed_types = {"inventory", "applicability_pattern", "failure_pattern", "scenario_target_coverage"}
    for row in rows:
        if set(row) != set(ROW_FIELDS):
            raise ValueError("synthesis_row_schema_mismatch")
        if row["record_type"] not in allowed_types:
            raise ValueError("unknown_synthesis_record_type")
        if row["status"] not in {"ok", "inapplicable"}:
            raise ValueError("invalid_synthesis_status")
        if row["status"] == "inapplicable":
            if any(row[field] for field in ("numerator", "denominator", "rate")):
                raise ValueError("inapplicable_synthesis_value_not_blank")
        else:
            if not all(row[field] for field in ("numerator", "denominator", "rate")):
                raise ValueError("applicable_synthesis_value_missing")
            rate = float(row["rate"])
            if not math.isfinite(rate) or rate < 0 or rate > 1:
                raise ValueError("invalid_synthesis_rate")
            denominator = int(row["denominator"])
            numerator = int(row["numerator"])
            if denominator <= 0 or numerator < 0 or numerator > denominator:
                raise ValueError("invalid_synthesis_denominator")
            if abs(rate - numerator / denominator) > 1e-10:
                raise ValueError("synthesis_rate_mismatch")
        if row["record_type"] == "failure_pattern":
            if row["dimension"] not in {"raw_cartesian_failure_label_fraction", "conditional_failure_rate"}:
                raise ValueError("unknown_failure_rate_semantics")
        if row["record_type"] == "scenario_target_coverage" and row["dimension"] != "applicable_target_family_inventory":
            raise ValueError("unknown_scenario_target_coverage_semantics")
        forbidden = {"winner", "rank", "score", "normalized_metric", "effect_size", "affected_target_families", "sensitivity_map"}
        tokens = set(re.findall(r"[a-z_]+", " ".join(str(value).lower() for value in row.values())))
        if forbidden & tokens:
            raise ValueError("forbidden_synthesis_semantics")

def validate_resume_rows(existing: list[dict[str, str]], expected: list[dict[str, str]]) -> None:
    if any(set(row) != set(ROW_FIELDS) for row in existing):
        raise ValueError("resume_schema_mismatch")
    existing_keys = [row["row_key"] for row in existing]
    if len(existing_keys) != len(set(existing_keys)):
        raise ValueError("duplicate_resume_key")
    expected_map = {row["row_key"]: row for row in expected}
    for row in existing:
        key = row["row_key"]
        if key not in expected_map:
            raise ValueError("unknown_resume_key")
        if row != expected_map[key]:
            raise ValueError("stale_or_corrupted_resume_row")


def build_content(source_root: Path, reverse: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, bytes]]:
    sources = verify_source_tree(source_root)
    rows = build_rows(sources, reverse=reverse)
    validate_synthesis_rows(rows)
    results = csv_bytes(rows)
    diagnostics = canonical_json(diagnostics_document(rows, sources))
    summary = draw_summary(rows)
    content = {"results.csv": results, "diagnostics.json": diagnostics, "summary.png": summary}
    content["manifest.json"] = canonical_json(manifest_document(rows, sources, content))
    return sources, rows, content


def write_artifacts(output_dir: Path, source_root: Path, reverse: bool = False, resume: bool = False) -> None:
    _, rows, content = build_content(source_root, reverse=reverse)
    if resume and (output_dir / "results.csv").exists():
        _, existing = read_csv(output_dir / "results.csv")
        validate_resume_rows(existing, rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    extras = [path for path in output_dir.iterdir() if path.name not in ARTIFACT_FILES]
    if extras:
        raise ValueError("unexpected_existing_artifact:" + ",".join(sorted(path.name for path in extras)))
    for name in ARTIFACT_FILES:
        (output_dir / name).write_bytes(content[name])


def validate_artifacts(output_dir: Path, source_root: Path) -> list[str]:
    errors: list[str] = []
    actual_names = sorted(path.name for path in output_dir.iterdir()) if output_dir.is_dir() else []
    if actual_names != sorted(ARTIFACT_FILES):
        errors.append(f"artifact_inventory:{actual_names}")
        return errors
    try:
        _, expected_rows, expected = build_content(source_root)
        manifest = json.loads((output_dir / "manifest.json").read_bytes())
        actual_fields, actual_rows = read_csv(output_dir / "results.csv")
        if actual_fields != list(ROW_FIELDS):
            errors.append("results_header")
        validate_synthesis_rows(actual_rows)
        if actual_rows != expected_rows:
            errors.append("results_content")
        for name in ARTIFACT_FILES:
            if (output_dir / name).read_bytes() != expected[name]:
                errors.append(f"byte_mismatch:{name}")
        declared = manifest.get("artifact_hashes_sha256", {})
        for name in CONTENT_FILES:
            if declared.get(name) != sha256_file(output_dir / name):
                errors.append(f"artifact_hash:{name}")
    except Exception as exc:  # verifier returns bounded diagnostics
        errors.append(f"validation_exception:{type(exc).__name__}:{exc}")
    return errors


def adversarial_checks(source_root: Path, expected_rows: list[dict[str, str]], expected: dict[str, bytes]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="p4u06-verify-") as tmp:
        base = Path(tmp)
        clean = base / "clean"
        reverse = base / "reverse"
        resume = base / "resume"
        write_artifacts(clean, source_root)
        write_artifacts(reverse, source_root, reverse=True)
        checks["clean_byte_identity"] = all((clean / n).read_bytes() == expected[n] for n in ARTIFACT_FILES)
        checks["reverse_byte_identity"] = all((reverse / n).read_bytes() == expected[n] for n in ARTIFACT_FILES)

        resume.mkdir()
        partial = expected_rows[::3]
        (resume / "results.csv").write_bytes(csv_bytes(partial))
        write_artifacts(resume, source_root, resume=True)
        checks["valid_partial_resume_byte_identity"] = all((resume / n).read_bytes() == expected[n] for n in ARTIFACT_FILES)

        bad_cases: dict[str, list[dict[str, str]]] = {}
        corrupted = [dict(row) for row in expected_rows[:3]]
        corrupted[0]["numerator"] = str(int(corrupted[0]["numerator"]) + 1)
        bad_cases["corrupted_resume_rejected"] = corrupted
        duplicate = [dict(expected_rows[0]), dict(expected_rows[0])]
        bad_cases["duplicate_resume_rejected"] = duplicate
        unknown = [dict(expected_rows[0])]
        unknown[0]["row_key"] += "|unknown"
        bad_cases["unknown_resume_rejected"] = unknown
        incomplete = [dict(expected_rows[0])]
        incomplete[0].pop("notes")
        bad_cases["incomplete_resume_rejected"] = incomplete
        for label, bad in bad_cases.items():
            try:
                validate_resume_rows(bad, expected_rows)
            except ValueError:
                checks[label] = True
            else:
                checks[label] = False

        drift_root = base / "sources"
        shutil.copytree(source_root, drift_root)
        drift_file = drift_root / SOURCE_SPECS[0][0] / "diagnostics.json"
        drift_file.write_bytes(drift_file.read_bytes() + b"\n")
        try:
            verify_source_tree(drift_root)
        except ValueError as exc:
            checks["source_scientific_content_drift_rejected"] = "source_scientific_content_drift" in str(exc)
        else:
            checks["source_scientific_content_drift_rejected"] = False

        attachment_root = base / "attachment-drift"
        shutil.copytree(source_root, attachment_root)
        attachment = attachment_root / SOURCE_SPECS[0][0] / "summary.png"
        attachment.write_bytes(attachment.read_bytes() + b"drift")
        try:
            verify_source_tree(attachment_root)
        except ValueError as exc:
            checks["integrity_only_attachment_drift_rejected"] = "source_integrity_attachment_drift" in str(exc)
        else:
            checks["integrity_only_attachment_drift_rejected"] = False

        attachment_missing_root = base / "attachment-missing"
        shutil.copytree(source_root, attachment_missing_root)
        missing_attachment = attachment_missing_root / SOURCE_SPECS[0][0] / "summary.png"
        missing_attachment.unlink()
        try:
            verify_source_tree(attachment_missing_root)
        except ValueError:
            checks["optional_integrity_attachment_absence_accepted"] = False
        else:
            checks["optional_integrity_attachment_absence_accepted"] = True

        unstable_root = base / "unstable"
        shutil.copytree(source_root, unstable_root)
        manifest_path = unstable_root / SOURCE_SPECS[0][0] / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["status"] = "review"
        manifest_path.write_bytes(canonical_json(manifest))
        try:
            verify_source_tree(unstable_root)
        except ValueError as exc:
            checks["nonstable_source_rejected"] = "source_not_stable" in str(exc)
        else:
            checks["nonstable_source_rejected"] = False
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source_root = (args.source_root or repo / "benchmarks" / "artifacts").resolve()
    output_dir = (args.output_dir or source_root / "cross-suite-robustness-synthesis").resolve()
    try:
        if args.generate:
            write_artifacts(output_dir, source_root, reverse=args.reverse, resume=args.resume)
            print(f"generated {output_dir}")
            return 0
        sources, rows, expected = build_content(source_root)
        errors = validate_artifacts(output_dir, source_root)
        checks = adversarial_checks(source_root, rows, expected)
        failed = sorted(key for key, value in checks.items() if not value)
        if errors or failed:
            for error in errors:
                print(f"ERROR {error}", file=sys.stderr)
            for key in failed:
                print(f"ERROR adversarial_check:{key}", file=sys.stderr)
            return 1
        print(json.dumps({
            "status": "PASS",
            "source_count": len(sources),
            "row_count": len(rows),
            "artifact_inventory": list(ARTIFACT_FILES),
            "adversarial_checks": checks,
        }, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
