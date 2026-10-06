"""
General, domain-agnostic analytical experiment tools.

Twelve deterministic tools that operate on any tabular dataset described by a
:class:`~agentic.domain.manifest.DatasetManifest`. Each declares typed parameters,
required capabilities, and output schema, and emits structured observations,
evidence (with statistical backing where applicable), and reproducible artifacts.
No tool calls an LLM.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from agentic.domain.common import DomainModel
from agentic.domain.enums import (
    ColumnRole,
    EvidenceDirection,
    EvidenceType,
    Modality,
    ObservationType,
)
from agentic.domain.experiment import CostEstimate
from agentic.domain.manifest import DatasetManifest
from agentic.domain.statistics import Uncertainty

from .. import stats as st
from ..base import BaseExperimentTool, ExperimentOutcome
from ..capability import ExperimentCapability, ValidationIssue
from ..context import ExperimentContext
from ..descriptor import ArtifactType, ExperimentToolDescriptor, OutputField
from ..errors import ParameterError
from ._helpers import (
    default_entity_column,
    default_time_column,
    ensure_columns,
    make_evidence,
    make_observation,
    make_statistics,
    nonnull_coverage,
    numeric_array,
    require_frame,
    source_ref,
)

_TABULAR = [Modality.tabular, Modality.time_series]
_CHEAP = CostEstimate(compute_seconds=0.2)


def _period_order_index(series: pd.Series) -> np.ndarray:
    """Deterministic 0..n-1 ordinal for a sorted period column (string or numeric)."""
    return np.arange(len(series), dtype=float)


# ---------------------------------------------------------------------------
# 1. profile_dataset
# ---------------------------------------------------------------------------


class _ProfileParams(DomainModel):
    pass


class ProfileDatasetTool(BaseExperimentTool):
    params_model = _ProfileParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="profile_dataset",
            version="1.0",
            purpose="Summarize every column: dtype, role, missingness, distinct counts, numeric stats.",
            supported_input_modalities=_TABULAR,
            required_capabilities=ExperimentCapability(supported_modalities=_TABULAR, min_rows=1),
            parameter_schema=_ProfileParams.model_json_schema(),
            output_schema=[
                OutputField(name="column_profile", kind="artifact", description="per-column profile table"),
                OutputField(name="row_count", kind="metric"),
                OutputField(name="overall_missing_ratio", kind="metric"),
            ],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.table, ArtifactType.json],
            known_limitations=["Numeric summaries computed only on coercible columns."],
        )

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        rows = len(frame)
        role_by_name = {c.name: c.role for c in context.manifest.columns}
        records = []
        obs = []
        total_missing = 0
        for col in frame.columns:
            s = frame[col]
            missing = int(s.isna().sum())
            total_missing += missing
            distinct = int(s.nunique(dropna=True))
            rec = {
                "column": str(col),
                "role": role_by_name.get(str(col), ColumnRole.unknown).value,
                "dtype": str(s.dtype),
                "missing": missing,
                "missing_ratio": round(missing / rows, 6) if rows else 0.0,
                "distinct": distinct,
            }
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().any():
                d = st.describe(num.to_numpy(dtype=float))
                rec.update({k: round(v, 6) for k, v in d.items() if k in ("mean", "std", "min", "max")})
            records.append(rec)
            if rows and missing == rows:
                obs.append(make_observation(statement=f"Column '{col}' is entirely null.",
                                            provenance=prov, observation_type=ObservationType.gap, metric_ref=str(col)))
        profile_df = pd.DataFrame.from_records(records)
        overall_missing = total_missing / (rows * frame.shape[1]) if rows and frame.shape[1] else 0.0
        art_table = context.artifact_sink.emit_table("column_profile", profile_df)
        art_json = context.artifact_sink.emit_json(
            "dataset_profile",
            {"row_count": rows, "column_count": int(frame.shape[1]), "overall_missing_ratio": round(overall_missing, 6)},
        )
        statistics = make_statistics(sample_size=rows, coverage=round(1.0 - overall_missing, 6))
        evidence = make_evidence(
            evidence_type=EvidenceType.data_quality,
            claim=f"Dataset has {rows} rows across {frame.shape[1]} columns with {overall_missing:.1%} overall missingness.",
            direction=EvidenceDirection.neutral,
            provenance=prov,
            statistics=statistics,
            source_reference=source_ref(context.manifest),
            artifact_ids=[art_table.id, art_json.id],
        )
        return ExperimentOutcome(
            observations=obs,
            evidence=[evidence],
            metrics={"row_count": float(rows), "column_count": float(frame.shape[1]),
                     "overall_missing_ratio": round(overall_missing, 6)},
            statistics=[statistics],
            artifacts=[art_table, art_json],
            summary=f"Profiled {frame.shape[1]} columns over {rows} rows.",
        )


# ---------------------------------------------------------------------------
# 2. summarize_distribution
# ---------------------------------------------------------------------------


class _DistParams(DomainModel):
    column: str = Field(..., min_length=1)


class SummarizeDistributionTool(BaseExperimentTool):
    params_model = _DistParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="summarize_distribution",
            version="1.0",
            purpose="Descriptive statistics and shape of a numeric column.",
            supported_input_modalities=_TABULAR,
            required_capabilities=ExperimentCapability(supported_modalities=_TABULAR, min_rows=2),
            parameter_schema=_DistParams.model_json_schema(),
            output_schema=[
                OutputField(name="summary", kind="statistic", description="mean/std/quantiles/skew"),
                OutputField(name="histogram", kind="artifact"),
            ],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.json, ArtifactType.chart_spec],
            known_limitations=["Non-numeric values are coerced to NaN and ignored."],
        )

    def _check_params_against_manifest(self, params: BaseModel, manifest: DatasetManifest) -> list[ValidationIssue]:
        return ensure_columns([params.column], manifest)

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        arr = numeric_array(frame, params.column)
        d = st.describe(arr)
        n = int(d.get("count", 0))
        if n < 2:
            raise ParameterError(f"column '{params.column}' has fewer than 2 numeric values")
        skew = st.skewness(arr)
        cov = nonnull_coverage(frame, params.column)
        assumptions = ["Values treated as a single population; no grouping applied."]
        warnings = ["Distribution is skewed (|skew| > 1)."] if abs(skew) > 1 else []
        statistics = make_statistics(
            sample_size=n, coverage=round(cov, 6),
            diagnostics={"skew": round(skew, 6), "std": round(d["std"], 6)},
            assumptions=assumptions, warnings=warnings,
        )
        hist_counts, hist_edges = np.histogram(arr[np.isfinite(arr)], bins=min(10, max(1, n // 2)))
        chart = context.artifact_sink.emit_chart("distribution", {
            "chart_type": "histogram", "column": params.column,
            "bins": [round(float(e), 6) for e in hist_edges.tolist()],
            "counts": [int(c) for c in hist_counts.tolist()],
        })
        summary_art = context.artifact_sink.emit_json("distribution_summary", {k: round(v, 6) for k, v in d.items()})
        obs = make_observation(
            statement=f"'{params.column}' mean={d['mean']:.4g}, median={d['median']:.4g}, skew={skew:.3g}.",
            provenance=prov, observation_type=ObservationType.value, metric_ref=params.column, magnitude=d["mean"],
        )
        ev = make_evidence(
            evidence_type=EvidenceType.descriptive_stat,
            claim=f"'{params.column}' is centered near {d['median']:.4g} (n={n}).",
            direction=EvidenceDirection.neutral, provenance=prov, statistics=statistics,
            source_reference=source_ref(context.manifest, column=params.column),
            artifact_ids=[chart.id, summary_art.id],
        )
        return ExperimentOutcome(
            observations=[obs], evidence=[ev],
            metrics={"mean": round(d["mean"], 6), "std": round(d["std"], 6), "n": float(n)},
            statistics=[statistics], artifacts=[chart, summary_art],
            summary=f"Summarized distribution of '{params.column}' (n={n}).",
        )


# ---------------------------------------------------------------------------
# 3. analyze_missingness
# ---------------------------------------------------------------------------


class _MissingParams(DomainModel):
    columns: list[str] = Field(default_factory=list)


class AnalyzeMissingnessTool(BaseExperimentTool):
    params_model = _MissingParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="analyze_missingness",
            version="1.0",
            purpose="Per-column and overall missing-value analysis with warnings.",
            supported_input_modalities=_TABULAR,
            required_capabilities=ExperimentCapability(supported_modalities=_TABULAR, min_rows=1),
            parameter_schema=_MissingParams.model_json_schema(),
            output_schema=[OutputField(name="missingness", kind="artifact"),
                           OutputField(name="overall_missing_ratio", kind="metric")],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.table],
            known_limitations=["Treats empty strings as present unless the reader coerced them to NaN."],
        )

    def _check_params_against_manifest(self, params: BaseModel, manifest: DatasetManifest) -> list[ValidationIssue]:
        return ensure_columns(list(params.columns), manifest)

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        cols = list(params.columns) if params.columns else [str(c) for c in frame.columns]
        rows = len(frame)
        records, obs = [], []
        total_missing = 0
        for col in cols:
            missing = int(frame[col].isna().sum())
            total_missing += missing
            ratio = missing / rows if rows else 0.0
            records.append({"column": col, "missing": missing, "missing_ratio": round(ratio, 6)})
            if rows and ratio > 0.5:
                obs.append(make_observation(statement=f"'{col}' is {ratio:.0%} missing.",
                                            provenance=prov, observation_type=ObservationType.gap,
                                            metric_ref=col, magnitude=ratio))
        table = context.artifact_sink.emit_table("missingness", pd.DataFrame.from_records(records))
        overall = total_missing / (rows * len(cols)) if rows and cols else 0.0
        warnings = [f"{o.metric_ref} exceeds 50% missing" for o in obs]
        statistics = make_statistics(sample_size=rows, coverage=round(1.0 - overall, 6), warnings=warnings)
        ev = make_evidence(
            evidence_type=EvidenceType.data_quality,
            claim=f"Overall missingness across {len(cols)} column(s) is {overall:.1%}.",
            direction=EvidenceDirection.neutral, provenance=prov, statistics=statistics,
            source_reference=source_ref(context.manifest), artifact_ids=[table.id],
        )
        return ExperimentOutcome(
            observations=obs, evidence=[ev],
            metrics={"overall_missing_ratio": round(overall, 6), "columns_analyzed": float(len(cols))},
            statistics=[statistics], artifacts=[table], warnings=warnings,
            summary=f"Missingness over {len(cols)} column(s): {overall:.1%}.",
        )


# ---------------------------------------------------------------------------
# 4. detect_outliers
# ---------------------------------------------------------------------------


class _OutlierParams(DomainModel):
    column: str = Field(..., min_length=1)
    method: Literal["zscore", "iqr"] = "zscore"
    threshold: float = Field(default=3.0, gt=0.0)


class DetectOutliersTool(BaseExperimentTool):
    params_model = _OutlierParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="detect_outliers",
            version="1.0",
            purpose="Flag outliers in a numeric column via z-score or IQR fences.",
            supported_input_modalities=_TABULAR,
            required_capabilities=ExperimentCapability(supported_modalities=_TABULAR, min_rows=3),
            parameter_schema=_OutlierParams.model_json_schema(),
            output_schema=[OutputField(name="outliers", kind="artifact"),
                           OutputField(name="outlier_fraction", kind="metric"),
                           OutputField(name="effect_size", kind="statistic")],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.table, ArtifactType.chart_spec],
            known_limitations=["Z-score assumes approximate normality; IQR is distribution-free but coarse."],
        )

    def _check_params_against_manifest(self, params: BaseModel, manifest: DatasetManifest) -> list[ValidationIssue]:
        return ensure_columns([params.column], manifest)

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        arr = numeric_array(frame, params.column)
        finite = arr[np.isfinite(arr)]
        n = int(finite.size)
        if n < 3:
            raise ParameterError(f"column '{params.column}' has fewer than 3 numeric values")
        if params.method == "zscore":
            z = st.zscores(arr)
            mask = np.abs(z) >= params.threshold
            scores = np.where(np.isfinite(z), np.abs(z), 0.0)
            assumptions = ["z-score outlier rule assumes approximate normality."]
        else:
            lo, hi = st.iqr_bounds(arr, k=params.threshold)
            mask = (arr < lo) | (arr > hi)
            scores = np.where(np.isfinite(arr), np.maximum(lo - arr, arr - hi), 0.0)
            assumptions = ["IQR fences use k*IQR; distribution-free."]
        mask = np.where(np.isfinite(arr), mask, False)
        count = int(np.sum(mask))
        fraction = count / n
        max_score = float(np.nanmax(scores)) if scores.size else 0.0
        idx = np.where(mask)[0]
        out_rows = frame.iloc[idx].copy()
        out_rows.insert(0, "_row", idx)
        out_rows["_score"] = np.round(scores[idx], 6)
        table = context.artifact_sink.emit_table("outliers", out_rows.head(200))
        chart = context.artifact_sink.emit_chart("outliers", {
            "chart_type": "scatter", "column": params.column, "method": params.method,
            "threshold": params.threshold, "flagged_rows": [int(i) for i in idx[:200]],
        })
        statistics = make_statistics(
            sample_size=n, effect_size=round(fraction, 6), effect_size_kind="outlier_fraction",
            coverage=round(nonnull_coverage(frame, params.column), 6),
            diagnostics={"max_score": round(max_score, 6), "outlier_count": float(count)},
            assumptions=assumptions,
            warnings=["No outliers detected."] if count == 0 else [],
        )
        obs = [
            make_observation(statement=f"Row {int(i)} of '{params.column}' is an outlier (score={scores[i]:.2f}).",
                             provenance=prov, observation_type=ObservationType.outlier,
                             metric_ref=params.column, magnitude=float(scores[i]))
            for i in idx[:10]
        ]
        ev = make_evidence(
            evidence_type=EvidenceType.anomaly_flag,
            claim=f"{count} of {n} values in '{params.column}' are outliers ({fraction:.1%}).",
            direction=EvidenceDirection.supports if count else EvidenceDirection.neutral,
            provenance=prov, statistics=statistics,
            source_reference=source_ref(context.manifest, column=params.column), artifact_ids=[table.id, chart.id],
        )
        return ExperimentOutcome(
            observations=obs, evidence=[ev],
            metrics={"outlier_count": float(count), "outlier_fraction": round(fraction, 6),
                     "max_score": round(max_score, 6)},
            statistics=[statistics], artifacts=[table, chart],
            summary=f"{count} outlier(s) in '{params.column}' via {params.method}.",
        )


# ---------------------------------------------------------------------------
# 5. analyze_correlation
# ---------------------------------------------------------------------------


class _CorrParams(DomainModel):
    columns: list[str] = Field(default_factory=list)
    strong_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AnalyzeCorrelationTool(BaseExperimentTool):
    params_model = _CorrParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="analyze_correlation",
            version="1.0",
            purpose="Pairwise Pearson correlations among metric columns with CIs on strong pairs.",
            supported_input_modalities=_TABULAR,
            required_capabilities=ExperimentCapability(
                supported_modalities=_TABULAR, required_roles=[ColumnRole.metric], min_metric_columns=2, min_rows=4),
            parameter_schema=_CorrParams.model_json_schema(),
            output_schema=[OutputField(name="correlation_matrix", kind="artifact"),
                           OutputField(name="strong_pairs", kind="evidence")],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.table, ArtifactType.chart_spec],
            known_limitations=["Pearson only; captures linear association. p-values use a normal approximation."],
        )

    def _resolve_columns(self, params, manifest: DatasetManifest) -> list[str]:
        if params.columns:
            return list(params.columns)
        return manifest.metric_names()

    def _check_params_against_manifest(self, params: BaseModel, manifest: DatasetManifest) -> list[ValidationIssue]:
        return ensure_columns(list(params.columns), manifest)

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        cols = [c for c in self._resolve_columns(params, context.manifest) if c in frame.columns]
        if len(cols) < 2:
            raise ParameterError("need at least two numeric columns to correlate")
        data = {c: numeric_array(frame, c) for c in cols}
        matrix_rows, evidence, statistics, obs = [], [], [], []
        for i, a in enumerate(cols):
            row = {"column": a}
            for b in cols:
                r = st.pearson_r(data[a], data[b])
                row[b] = round(r, 6) if np.isfinite(r) else None
            matrix_rows.append(row)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                mask = np.isfinite(data[a]) & np.isfinite(data[b])
                n = int(mask.sum())
                r = st.pearson_r(data[a], data[b])
                if not np.isfinite(r) or abs(r) < params.strong_threshold:
                    continue
                ci = st.fisher_ci(r, n)
                z = r * np.sqrt(max(n - 1, 1))
                stat = make_statistics(
                    sample_size=n, effect_size=round(r, 6), effect_size_kind="pearson_r",
                    p_value=round(st.two_sided_p_from_z(z), 6),
                    uncertainty=Uncertainty(confidence_level=0.95, ci_low=round(ci[0], 6), ci_high=round(ci[1], 6)) if ci else None,
                    coverage=round(n / len(frame), 6) if len(frame) else None,
                    assumptions=["Linear association (Pearson).", "p-value via normal approximation."],
                )
                statistics.append(stat)
                evidence.append(make_evidence(
                    evidence_type=EvidenceType.statistical_test,
                    claim=f"'{a}' and '{b}' are correlated (r={r:.2f}, n={n}).",
                    direction=EvidenceDirection.supports, provenance=prov, statistics=stat,
                    source_reference=source_ref(context.manifest, column=f"{a},{b}"),
                ))
                obs.append(make_observation(statement=f"corr({a},{b})={r:.2f}", provenance=prov,
                                            observation_type=ObservationType.comparison, magnitude=float(r)))
        matrix = context.artifact_sink.emit_table("correlation_matrix", pd.DataFrame.from_records(matrix_rows))
        heatmap = context.artifact_sink.emit_chart("correlation_heatmap",
                                                   {"chart_type": "heatmap", "columns": cols,
                                                    "matrix": [[matrix_rows[i].get(c) for c in cols] for i in range(len(cols))]})
        for e in evidence:
            e.artifact_ids = [matrix.id, heatmap.id]
        return ExperimentOutcome(
            observations=obs, evidence=evidence,
            metrics={"columns": float(len(cols)), "strong_pairs": float(len(evidence))},
            statistics=statistics, artifacts=[matrix, heatmap],
            summary=f"Correlated {len(cols)} columns; {len(evidence)} strong pair(s).",
        )


# ---------------------------------------------------------------------------
# 6. compare_groups
# ---------------------------------------------------------------------------


class _CompareParams(DomainModel):
    value_column: str = Field(..., min_length=1)
    group_column: str = Field(..., min_length=1)
    max_groups: int = Field(default=10, ge=2)


class CompareGroupsTool(BaseExperimentTool):
    params_model = _CompareParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="compare_groups",
            version="1.0",
            purpose="Compare a numeric value across groups; effect size and t-test for two groups.",
            supported_input_modalities=_TABULAR,
            required_capabilities=ExperimentCapability(supported_modalities=_TABULAR, min_rows=4),
            parameter_schema=_CompareParams.model_json_schema(),
            output_schema=[OutputField(name="group_summary", kind="artifact"),
                           OutputField(name="effect_size", kind="statistic")],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.table, ArtifactType.chart_spec],
            known_limitations=["Two-group t-test uses a normal approximation for the p-value."],
        )

    def _check_params_against_manifest(self, params: BaseModel, manifest: DatasetManifest) -> list[ValidationIssue]:
        return ensure_columns([params.value_column, params.group_column], manifest)

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        values = pd.to_numeric(frame[params.value_column], errors="coerce")
        groups = frame[params.group_column].astype(str)
        grouped = {g: values[groups == g].dropna().to_numpy(dtype=float) for g in sorted(groups.dropna().unique())}
        grouped = {g: v for g, v in grouped.items() if v.size > 0}
        if len(grouped) < 2:
            raise ParameterError("need at least two non-empty groups to compare")
        rows = [{"group": g, "n": int(v.size), "mean": round(float(np.mean(v)), 6),
                 "std": round(float(np.std(v, ddof=1)) if v.size > 1 else 0.0, 6)} for g, v in grouped.items()]
        table = context.artifact_sink.emit_table("group_summary", pd.DataFrame.from_records(rows))
        chart = context.artifact_sink.emit_chart("group_means",
                                                 {"chart_type": "bar", "x": list(grouped.keys()),
                                                  "y": [round(float(np.mean(v)), 6) for v in grouped.values()]})
        keys = list(grouped.keys())
        warnings = [f"{len(grouped)} groups; effect size reported for the two largest only."] if len(grouped) > 2 else []
        if len(grouped) >= 2:
            ordered = sorted(keys, key=lambda g: grouped[g].size, reverse=True)[:2]
            a, b = grouped[ordered[0]], grouped[ordered[1]]
            cd = st.cohens_d(a, b)
            p = st.two_sided_p_from_z(cd["t"]) if np.isfinite(cd["t"]) else None
            stat = make_statistics(
                sample_size=int(a.size + b.size), effect_size=round(cd["d"], 6), effect_size_kind="cohens_d",
                p_value=round(p, 6) if p is not None else None,
                coverage=round(nonnull_coverage(frame, params.value_column), 6),
                diagnostics={"mean_diff": round(cd["mean_a"] - cd["mean_b"], 6), "t": round(cd["t"], 6)},
                assumptions=["Two-group comparison; p-value via normal approximation."], warnings=warnings,
            )
            claim = f"'{ordered[0]}' vs '{ordered[1]}' differ in {params.value_column} (d={cd['d']:.2f})."
            direction = EvidenceDirection.supports if abs(cd["d"]) >= 0.2 else EvidenceDirection.neutral
        ev = make_evidence(evidence_type=EvidenceType.peer_comparison, claim=claim, direction=direction,
                           provenance=prov, statistics=stat,
                           source_reference=source_ref(context.manifest, column=params.value_column),
                           artifact_ids=[table.id, chart.id])
        obs = [make_observation(statement=f"group '{r['group']}' mean {params.value_column}={r['mean']}",
                                provenance=prov, observation_type=ObservationType.comparison,
                                entity_ref=r["group"], magnitude=r["mean"]) for r in rows]
        return ExperimentOutcome(
            observations=obs, evidence=[ev],
            metrics={"groups": float(len(grouped)), "effect_size": round(stat.effect_size or 0.0, 6)},
            statistics=[stat], artifacts=[table, chart], warnings=warnings,
            summary=f"Compared {params.value_column} across {len(grouped)} groups.",
        )


# ---------------------------------------------------------------------------
# 7. analyze_time_series_trend
# ---------------------------------------------------------------------------


class _TrendParams(DomainModel):
    value_column: str = Field(..., min_length=1)
    time_column: str | None = Field(default=None)
    entity_column: str | None = Field(default=None)


class AnalyzeTimeSeriesTrendTool(BaseExperimentTool):
    params_model = _TrendParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="analyze_time_series_trend",
            version="1.0",
            purpose="Fit an ordinary-least-squares trend over ordered time and report slope, direction, R².",
            supported_input_modalities=[Modality.time_series, Modality.tabular],
            required_capabilities=ExperimentCapability(
                supported_modalities=[Modality.time_series, Modality.tabular], requires_temporal=True, min_rows=3),
            parameter_schema=_TrendParams.model_json_schema(),
            output_schema=[OutputField(name="trend", kind="statistic"), OutputField(name="trend_line", kind="artifact")],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.chart_spec, ArtifactType.table],
            known_limitations=["Uses period ordinal index (equal spacing assumed); linear trend only."],
        )

    def _check_params_against_manifest(self, params: BaseModel, manifest: DatasetManifest) -> list[ValidationIssue]:
        cols = [params.value_column]
        tcol = params.time_column or default_time_column(manifest)
        if tcol is None:
            return [ValidationIssue(code="NO_TIME_COLUMN", message="no time column provided or inferable")]
        cols.append(tcol)
        if params.entity_column:
            cols.append(params.entity_column)
        return ensure_columns(cols, manifest)

    def _series_for(self, frame, value_col, time_col):
        sub = frame[[time_col, value_col]].dropna()
        sub = sub.sort_values(time_col)
        y = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)
        x = _period_order_index(sub[time_col])
        mask = np.isfinite(y)
        return x[mask], y[mask], sub[time_col].to_numpy()

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        time_col = params.time_column or default_time_column(context.manifest)
        entity_col = params.entity_column or default_entity_column(context.manifest)

        segments: list[tuple[str | None, pd.DataFrame]] = []
        if entity_col and entity_col in frame.columns:
            for ent, sub in frame.groupby(entity_col):
                segments.append((str(ent), sub))
        else:
            segments.append((None, frame))

        obs, evidence, statistics, rows = [], [], [], []
        for ent, sub in segments:
            x, y, periods = self._series_for(sub, params.value_column, time_col)
            if x.size < 3:
                continue
            fit = st.ols_simple(x, y)
            slope, r2, se = fit["slope"], fit["r2"], fit["slope_se"]
            if not np.isfinite(slope):
                continue
            direction_word = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
            crit = 1.959963985
            unc = Uncertainty(confidence_level=0.95, ci_low=round(slope - crit * se, 6),
                              ci_high=round(slope + crit * se, 6), std_error=round(se, 6)) if np.isfinite(se) else None
            z = slope / se if (np.isfinite(se) and se > 0) else np.nan
            stat = make_statistics(
                sample_size=int(x.size), effect_size=round(r2, 6), effect_size_kind="r2",
                p_value=round(st.two_sided_p_from_z(z), 6) if np.isfinite(z) else None, uncertainty=unc,
                coverage=round(x.size / len(sub), 6) if len(sub) else None,
                diagnostics={"slope": round(slope, 6), "r2": round(r2, 6)},
                assumptions=["Equal-spaced periods.", "Linear trend; p-value via normal approximation."],
            )
            statistics.append(stat)
            ent_label = f" for {ent}" if ent else ""
            evidence.append(make_evidence(
                evidence_type=EvidenceType.trend_break,
                claim=f"'{params.value_column}'{ent_label} is {direction_word} (slope={slope:.3g}, R²={r2:.2f}).",
                direction=EvidenceDirection.supports if r2 >= 0.3 else EvidenceDirection.neutral,
                provenance=prov, statistics=stat,
                source_reference=source_ref(context.manifest, column=params.value_column)))
            obs.append(make_observation(statement=f"{params.value_column}{ent_label} slope={slope:.3g}",
                                        provenance=prov, observation_type=ObservationType.trend,
                                        entity_ref=ent, metric_ref=params.value_column, magnitude=float(slope)))
            rows.append({"entity": ent, "n": int(x.size), "slope": round(slope, 6), "r2": round(r2, 6)})
        if not rows:
            raise ParameterError("no series had enough points (>=3) to fit a trend")
        table = context.artifact_sink.emit_table("trend_fits", pd.DataFrame.from_records(rows))
        chart = context.artifact_sink.emit_chart("trend", {"chart_type": "line", "value": params.value_column,
                                                           "time": time_col, "fits": rows})
        for e in evidence:
            e.artifact_ids = [table.id, chart.id]
        return ExperimentOutcome(
            observations=obs, evidence=evidence,
            metrics={"series": float(len(rows)), "mean_slope": round(float(np.mean([r["slope"] for r in rows])), 6)},
            statistics=statistics, artifacts=[table, chart],
            summary=f"Fitted trend for {len(rows)} series of '{params.value_column}'.",
        )


# ---------------------------------------------------------------------------
# 8. detect_change_points
# ---------------------------------------------------------------------------


class _ChangeParams(DomainModel):
    value_column: str = Field(..., min_length=1)
    time_column: str | None = Field(default=None)
    min_segment: int = Field(default=2, ge=1)


#: Significance level for a change point. The test is of the *maximum* shift, so this is the
#: false-positive rate of the whole search, not of one split.
_CHANGE_POINT_ALPHA = 0.05


class DetectChangePointsTool(BaseExperimentTool):
    params_model = _ChangeParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="detect_change_points",
            version="1.1",
            purpose="Detect the single most significant mean shift via one-split binary segmentation.",
            supported_input_modalities=[Modality.time_series, Modality.tabular],
            required_capabilities=ExperimentCapability(
                supported_modalities=[Modality.time_series, Modality.tabular], requires_temporal=True, min_rows=4),
            parameter_schema=_ChangeParams.model_json_schema(),
            output_schema=[OutputField(name="change_point", kind="observation"),
                           OutputField(name="shift_effect_size", kind="statistic")],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.chart_spec],
            known_limitations=["Detects at most one change point (single split); no multi-segment search."],
        )

    def _check_params_against_manifest(self, params: BaseModel, manifest: DatasetManifest) -> list[ValidationIssue]:
        tcol = params.time_column or default_time_column(manifest)
        cols = [params.value_column] + ([tcol] if tcol else [])
        issues = ensure_columns(cols, manifest)
        if tcol is None:
            issues.append(ValidationIssue(code="NO_TIME_COLUMN", message="no time column provided or inferable"))
        return issues

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        time_col = params.time_column or default_time_column(context.manifest)
        sub = frame[[time_col, params.value_column]].dropna().sort_values(time_col)
        y = pd.to_numeric(sub[params.value_column], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        n = y.size
        min_segment = params.min_segment
        if n < 2 * min_segment:
            raise ParameterError(f"need >= {2 * min_segment} points; have {n}")
        best = st.max_mean_shift(y, min_segment)
        best_split, best_score, best_shift = int(best["split"]), best["score"], best["shift"]
        # The score is the largest of every split's standardised shift, so a large value is
        # what noise produces too. Version 1.0 treated it as a pre-chosen Cohen's d, and pure
        # noise reached evidence strength 1.0 often enough that the loop supported directional
        # trends at 0.95 on series whose trend fit explained 1% of the variance. Only a shift
        # the permutation test cannot attribute to the search counts as evidence.
        p_value = st.max_mean_shift_p_value(y, min_segment)
        significant = p_value < _CHANGE_POINT_ALPHA and best_score >= 0.5
        periods = sub[time_col].to_numpy()
        cp_period = str(periods[best_split]) if best_split is not None and best_split < len(periods) else None
        warnings = []
        if best_score < 0.5:
            warnings.append("Weak shift (score < 0.5).")
        if p_value >= _CHANGE_POINT_ALPHA:
            warnings.append(f"Not significant: noise produces a shift this large (permutation p={p_value:.3f}).")
        stat = make_statistics(
            sample_size=int(n), effect_size=round(best_score, 6), effect_size_kind="cohens_d",
            p_value=round(p_value, 6),
            coverage=round(nonnull_coverage(frame, params.value_column), 6),
            diagnostics={"shift": round(best_shift, 6), "split_index": float(best_split or 0)},
            assumptions=["Single change point; standardized mean-shift score.",
                         "Score is the maximum over all splits; p-value from a permutation test "
                         "of that maximum (exchangeable null, fixed seed)."],
            warnings=warnings,
        )
        chart = context.artifact_sink.emit_chart("change_point", {
            "chart_type": "line", "value": params.value_column, "time": time_col,
            "change_point_index": int(best_split or 0), "change_point_period": cp_period,
        })
        obs = make_observation(statement=f"Largest mean shift in '{params.value_column}' at {cp_period} (shift={best_shift:.3g}).",
                               provenance=prov, observation_type=ObservationType.trend,
                               metric_ref=params.value_column, magnitude=best_shift)
        ev = make_evidence(evidence_type=EvidenceType.trend_break,
                           claim=f"'{params.value_column}' shifts by {best_shift:.3g} around {cp_period}.",
                           direction=EvidenceDirection.supports if significant else EvidenceDirection.neutral,
                           provenance=prov, statistics=stat,
                           source_reference=source_ref(context.manifest, column=params.value_column),
                           artifact_ids=[chart.id])
        return ExperimentOutcome(
            observations=[obs], evidence=[ev],
            metrics={"shift": round(best_shift, 6), "shift_score": round(best_score, 6),
                     "split_index": float(best_split or 0)},
            statistics=[stat], artifacts=[chart],
            summary=f"Change point in '{params.value_column}' at {cp_period}.",
        )


# ---------------------------------------------------------------------------
# 9. fit_simple_regression
# ---------------------------------------------------------------------------


class _RegressionParams(DomainModel):
    x_column: str = Field(..., min_length=1)
    y_column: str = Field(..., min_length=1)


class FitSimpleRegressionTool(BaseExperimentTool):
    params_model = _RegressionParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="fit_simple_regression",
            version="1.0",
            purpose="Fit y ~ x by OLS; report slope, intercept, R², slope uncertainty, residual diagnostics.",
            supported_input_modalities=_TABULAR,
            required_capabilities=ExperimentCapability(supported_modalities=_TABULAR, min_rows=3),
            parameter_schema=_RegressionParams.model_json_schema(),
            output_schema=[OutputField(name="coefficients", kind="artifact"),
                           OutputField(name="r2", kind="statistic")],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.table, ArtifactType.chart_spec],
            known_limitations=["Simple (one predictor) linear model; p-value via normal approximation."],
        )

    def _check_params_against_manifest(self, params: BaseModel, manifest: DatasetManifest) -> list[ValidationIssue]:
        return ensure_columns([params.x_column, params.y_column], manifest)

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        x = numeric_array(frame, params.x_column)
        y = numeric_array(frame, params.y_column)
        fit = st.ols_simple(x, y)
        n = int(fit["n"])
        if n < 3 or not np.isfinite(fit["slope"]):
            raise ParameterError("insufficient finite (x, y) pairs or zero variance in x")
        se = fit["slope_se"]
        crit = 1.959963985
        unc = Uncertainty(confidence_level=0.95, ci_low=round(fit["slope"] - crit * se, 6),
                          ci_high=round(fit["slope"] + crit * se, 6), std_error=round(se, 6)) if np.isfinite(se) else None
        z = fit["slope"] / se if (np.isfinite(se) and se > 0) else np.nan
        stat = make_statistics(
            sample_size=n, effect_size=round(fit["r2"], 6), effect_size_kind="r2",
            p_value=round(st.two_sided_p_from_z(z), 6) if np.isfinite(z) else None, uncertainty=unc,
            coverage=round(n / len(frame), 6) if len(frame) else None,
            diagnostics={"slope": round(fit["slope"], 6), "intercept": round(fit["intercept"], 6),
                         "r2": round(fit["r2"], 6), "resid_std": round(fit["resid_std"], 6)},
            assumptions=["Linearity.", "Homoscedastic, independent residuals.", "p-value via normal approximation."],
        )
        table = context.artifact_sink.emit_table("coefficients", pd.DataFrame([
            {"term": "intercept", "estimate": round(fit["intercept"], 6)},
            {"term": params.x_column, "estimate": round(fit["slope"], 6), "std_error": round(se, 6) if np.isfinite(se) else None},
        ]))
        chart = context.artifact_sink.emit_chart("regression_fit", {"chart_type": "scatter",
                                                                    "x": params.x_column, "y": params.y_column,
                                                                    "slope": round(fit["slope"], 6),
                                                                    "intercept": round(fit["intercept"], 6)})
        obs = make_observation(statement=f"{params.y_column} ~ {params.x_column}: slope={fit['slope']:.3g}, R²={fit['r2']:.2f}",
                               provenance=prov, observation_type=ObservationType.comparison, magnitude=float(fit["slope"]))
        ev = make_evidence(evidence_type=EvidenceType.statistical_test,
                           claim=f"'{params.x_column}' predicts '{params.y_column}' (R²={fit['r2']:.2f}, n={n}).",
                           direction=EvidenceDirection.supports if fit["r2"] >= 0.3 else EvidenceDirection.neutral,
                           provenance=prov, statistics=stat,
                           source_reference=source_ref(context.manifest, column=f"{params.x_column},{params.y_column}"),
                           artifact_ids=[table.id, chart.id])
        return ExperimentOutcome(
            observations=[obs], evidence=[ev],
            metrics={"slope": round(fit["slope"], 6), "intercept": round(fit["intercept"], 6),
                     "r2": round(fit["r2"], 6), "n": float(n)},
            statistics=[stat], artifacts=[table, chart],
            summary=f"OLS {params.y_column} ~ {params.x_column}: R²={fit['r2']:.2f}.",
        )


# ---------------------------------------------------------------------------
# 10. test_association
# ---------------------------------------------------------------------------


class _AssocParams(DomainModel):
    column_a: str = Field(..., min_length=1)
    column_b: str = Field(..., min_length=1)
    max_levels: int = Field(default=20, ge=2)


class TestAssociationTool(BaseExperimentTool):
    params_model = _AssocParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="test_association",
            version="1.0",
            purpose="Test association between two categorical columns via chi-square and Cramér's V.",
            supported_input_modalities=_TABULAR,
            required_capabilities=ExperimentCapability(supported_modalities=_TABULAR, min_rows=4),
            parameter_schema=_AssocParams.model_json_schema(),
            output_schema=[OutputField(name="contingency", kind="artifact"),
                           OutputField(name="cramers_v", kind="statistic")],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.table],
            known_limitations=["p-value omitted (no chi-square distribution dependency); Cramér's V is the effect size."],
        )

    def _check_params_against_manifest(self, params: BaseModel, manifest: DatasetManifest) -> list[ValidationIssue]:
        return ensure_columns([params.column_a, params.column_b], manifest)

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        a = frame[params.column_a].astype(str)
        b = frame[params.column_b].astype(str)
        if a.nunique() > params.max_levels or b.nunique() > params.max_levels:
            raise ParameterError(f"a column exceeds max_levels={params.max_levels}; not categorical enough")
        ct = pd.crosstab(a, b)
        res = st.cramers_v(ct.to_numpy())
        n = int(res["n"])
        stat = make_statistics(
            sample_size=n, effect_size=round(res["v"], 6) if np.isfinite(res["v"]) else None, effect_size_kind="cramers_v",
            coverage=round(n / len(frame), 6) if len(frame) else None,
            diagnostics={"chi2": round(res["chi2"], 6), "dof": res.get("dof", 0.0)},
            assumptions=["Chi-square independence test; expected counts should exceed ~5 per cell."],
            warnings=["Some expected cell counts may be small."] if n < 5 * ct.size else [],
        )
        table = context.artifact_sink.emit_table("contingency", ct.reset_index())
        v = res["v"]
        direction = EvidenceDirection.supports if (np.isfinite(v) and v >= 0.1) else EvidenceDirection.neutral
        ev = make_evidence(evidence_type=EvidenceType.statistical_test,
                           claim=f"'{params.column_a}' and '{params.column_b}' association: Cramér's V={v:.2f} (n={n}).",
                           direction=direction, provenance=prov, statistics=stat,
                           source_reference=source_ref(context.manifest, column=f"{params.column_a},{params.column_b}"),
                           artifact_ids=[table.id])
        obs = make_observation(statement=f"Cramér's V({params.column_a},{params.column_b})={v:.2f}",
                               provenance=prov, observation_type=ObservationType.comparison,
                               magnitude=float(v) if np.isfinite(v) else None)
        return ExperimentOutcome(
            observations=[obs], evidence=[ev],
            metrics={"cramers_v": round(v, 6) if np.isfinite(v) else 0.0, "chi2": round(res["chi2"], 6), "n": float(n)},
            statistics=[stat], artifacts=[table],
            summary=f"Association {params.column_a}×{params.column_b}: V={v:.2f}.",
        )


# ---------------------------------------------------------------------------
# 11. rank_entities
# ---------------------------------------------------------------------------


class _RankParams(DomainModel):
    metric_column: str = Field(..., min_length=1)
    entity_column: str | None = Field(default=None)
    aggregation: Literal["mean", "sum", "median", "last"] = "mean"
    top_n: int = Field(default=5, ge=1)
    ascending: bool = False


class RankEntitiesTool(BaseExperimentTool):
    params_model = _RankParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="rank_entities",
            version="1.0",
            purpose="Rank entities by an aggregated metric and surface the extremes.",
            supported_input_modalities=_TABULAR,
            required_capabilities=ExperimentCapability(supported_modalities=_TABULAR, min_rows=1),
            parameter_schema=_RankParams.model_json_schema(),
            output_schema=[OutputField(name="ranking", kind="artifact"),
                           OutputField(name="top_entities", kind="observation")],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.table, ArtifactType.chart_spec],
            known_limitations=["Ranking ignores uncertainty in the per-entity aggregate."],
        )

    def _check_params_against_manifest(self, params: BaseModel, manifest: DatasetManifest) -> list[ValidationIssue]:
        cols = [params.metric_column]
        ecol = params.entity_column or default_entity_column(manifest)
        if ecol is None:
            return [ValidationIssue(code="NO_ENTITY_COLUMN", message="no entity column provided or inferable")]
        cols.append(ecol)
        return ensure_columns(cols, manifest)

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        ecol = params.entity_column or default_entity_column(context.manifest)
        vals = pd.to_numeric(frame[params.metric_column], errors="coerce")
        work = pd.DataFrame({ecol: frame[ecol].astype(str), "_v": vals}).dropna()
        if work.empty:
            raise ParameterError("no numeric values to rank")
        agg = "last" if params.aggregation == "last" else params.aggregation
        grouped = work.groupby(ecol)["_v"]
        series = grouped.last() if agg == "last" else grouped.agg(agg)
        ranked = series.sort_values(ascending=params.ascending)
        top = ranked.head(params.top_n)
        rows = [{"rank": i + 1, "entity": e, "value": round(float(v), 6)} for i, (e, v) in enumerate(ranked.items())]
        table = context.artifact_sink.emit_table("ranking", pd.DataFrame.from_records(rows))
        chart = context.artifact_sink.emit_chart("ranking", {"chart_type": "bar", "x": list(top.index),
                                                             "y": [round(float(v), 6) for v in top.values]})
        obs = [make_observation(statement=f"#{i+1} {e}: {params.metric_column}={float(v):.4g}",
                                provenance=prov, observation_type=ObservationType.comparison,
                                entity_ref=str(e), metric_ref=params.metric_column, magnitude=float(v))
               for i, (e, v) in enumerate(top.items())]
        stat = make_statistics(sample_size=int(series.size),
                               coverage=round(nonnull_coverage(frame, params.metric_column), 6),
                               diagnostics={"entities": float(series.size)})
        # The claim must name the end of the ranking it actually read. Saying "top" while
        # ranking ascending states something false about a correctly computed result, and the
        # claim is what the trace and the report quote.
        extreme = "Lowest" if params.ascending else "Top"
        ev = make_evidence(evidence_type=EvidenceType.descriptive_stat,
                           claim=f"{extreme} entity by {params.aggregation}({params.metric_column})"
                                 f" is '{top.index[0]}'.",
                           direction=EvidenceDirection.neutral, provenance=prov, statistics=stat,
                           source_reference=source_ref(context.manifest, column=params.metric_column),
                           artifact_ids=[table.id, chart.id])
        return ExperimentOutcome(
            observations=obs, evidence=[ev],
            metrics={"entities_ranked": float(series.size), "top_value": round(float(top.iloc[0]), 6)},
            statistics=[stat], artifacts=[table, chart],
            summary=f"Ranked {series.size} entities by {params.aggregation}({params.metric_column}).",
        )


# ---------------------------------------------------------------------------
# 12. generate_deterministic_chart
# ---------------------------------------------------------------------------


class _ChartParams(DomainModel):
    chart_type: Literal["bar", "line", "scatter", "histogram"] = "bar"
    x_column: str = Field(..., min_length=1)
    y_column: str | None = Field(default=None)
    series_column: str | None = Field(default=None)
    max_points: int = Field(default=500, ge=1)


class GenerateDeterministicChartTool(BaseExperimentTool):
    params_model = _ChartParams

    def descriptor(self) -> ExperimentToolDescriptor:
        return ExperimentToolDescriptor(
            name="generate_deterministic_chart",
            version="1.0",
            purpose="Emit a deterministic, self-contained chart specification (no rendering, no LLM).",
            supported_input_modalities=_TABULAR,
            required_capabilities=ExperimentCapability(supported_modalities=_TABULAR, min_rows=1),
            parameter_schema=_ChartParams.model_json_schema(),
            output_schema=[OutputField(name="chart", kind="artifact"), OutputField(name="point_count", kind="metric")],
            cost_estimate=_CHEAP,
            artifact_types=[ArtifactType.chart_spec],
            known_limitations=["Produces a chart spec (JSON), not a rendered image; truncates to max_points."],
        )

    def _check_params_against_manifest(self, params: BaseModel, manifest: DatasetManifest) -> list[ValidationIssue]:
        cols = [params.x_column] + [c for c in (params.y_column, params.series_column) if c]
        return ensure_columns(cols, manifest)

    def _compute(self, context: ExperimentContext, params: BaseModel) -> ExperimentOutcome:
        frame = require_frame(context)
        prov = context.tool_provenance(self.name, self.version)
        cols = [params.x_column] + [c for c in (params.y_column, params.series_column) if c]
        data = frame[cols].head(params.max_points)
        spec = {
            "chart_type": params.chart_type,
            "encoding": {"x": params.x_column, "y": params.y_column, "series": params.series_column},
            "data": [
                {k: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating)) else str(v)))
                 for k, v in rec.items()}
                for rec in data.to_dict(orient="records")
            ],
        }
        chart = context.artifact_sink.emit_chart(f"{params.chart_type}_chart", spec)
        obs = make_observation(statement=f"Generated a {params.chart_type} chart of {params.x_column}"
                                         + (f" vs {params.y_column}" if params.y_column else "") + ".",
                               provenance=prov, observation_type=ObservationType.value)
        return ExperimentOutcome(
            observations=[obs], evidence=[], metrics={"point_count": float(len(data))},
            artifacts=[chart], summary=f"Chart '{params.chart_type}' with {len(data)} point(s).",
        )


def general_tools() -> list[BaseExperimentTool]:
    """All twelve general analytical tools."""
    return [
        ProfileDatasetTool(),
        SummarizeDistributionTool(),
        AnalyzeMissingnessTool(),
        DetectOutliersTool(),
        AnalyzeCorrelationTool(),
        CompareGroupsTool(),
        AnalyzeTimeSeriesTrendTool(),
        DetectChangePointsTool(),
        FitSimpleRegressionTool(),
        TestAssociationTool(),
        RankEntitiesTool(),
        GenerateDeterministicChartTool(),
    ]
