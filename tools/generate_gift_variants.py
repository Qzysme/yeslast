#!/usr/bin/env python3
"""
Generate DREAMPlace configuration variants that cover every combination of
GiFt / GiFtPlus initialisation, pin-risk weighting, adaptive RASS weighting,
and PSO refinement.

Each source configuration under the --test-root directory is copied and the
GiFt-related knobs are overridden according to the selected variant.  The
resulting files are written to --output-root/<variant>/<relative path>.
"""

from __future__ import annotations

import argparse
import json
import shutil
from itertools import chain, combinations
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence


def powerset(items: Sequence[str]) -> Iterable[Sequence[str]]:
    """Return all subsets of `items` preserving lexicographic order."""
    for r in range(len(items) + 1):
        for subset in combinations(items, r):
            yield subset


# Common reset values applied before setting a variant.
RESET_DEFAULTS: Mapping[str, object] = {
    "gift_init_flag": 0,
    "giftplus_init_flag": 0,
    "gift_init_scale": 0.7,
    "gift_adapt_flag": 0,
    "gift_adapt_samples": 0,
    "gift_adapt_lambda_hpwl": 0.0,
    "gift_adapt_lambda_density": 0.0,
    "gift_adapt_lambda_risk": 0.0,
    "gift_adapt_risk_step": 0.0,
    "gift_adapt_seed": 2025,
    "gift_pso_flag": 0,
    "gift_pso_swarm": 12,
    "gift_pso_iters": 12,
    "gift_pso_inertia": 0.5,
    "gift_pso_cognitive": 1.4,
    "gift_pso_social": 1.4,
    "gift_pso_init_sigma": 0.02,
    "giftplus_sparsify_k": 16,
    "giftplus_dirichlet_iters": 3,
    "giftplus_refine_iters": 2,
    "pin_risk_weight": 0.0,
}


INIT_VARIANTS: Mapping[str, Mapping[str, object]] = {
    "baseline": {},
    "gift": {
        "gift_init_flag": 1,
        "giftplus_init_flag": 0,
    },
    "giftplus": {
        "gift_init_flag": 0,
        "giftplus_init_flag": 1,
        "giftplus_sparsify_k": 16,
        "giftplus_dirichlet_iters": 3,
        "giftplus_refine_iters": 2,
    },
}


FEATURE_OVERRIDES: Mapping[str, Mapping[str, Mapping[str, object]]] = {
    "gift": {
        "pin": {
            "pin_risk_weight": 0.1,
            "gift_adapt_lambda_risk": 0.8,
        },
        "rass": {
            "gift_adapt_flag": 1,
            "gift_adapt_samples": 48,
            "gift_adapt_lambda_hpwl": 1.0,
            "gift_adapt_lambda_density": 0.2,
            "gift_adapt_lambda_risk": 0.3,
            "gift_adapt_risk_step": 6.0,
        },
        "pso": {
            "gift_pso_flag": 1,
            "gift_pso_swarm": 12,
            "gift_pso_iters": 12,
            "gift_pso_inertia": 0.5,
            "gift_pso_cognitive": 1.4,
            "gift_pso_social": 1.4,
            "gift_pso_init_sigma": 0.02,
        },
    },
    "giftplus": {
        "pin": {
            "pin_risk_weight": 0.1,
            "gift_adapt_lambda_risk": 0.8,
        },
        "rass": {
            "gift_adapt_flag": 1,
            "gift_adapt_samples": 48,
            "gift_adapt_lambda_hpwl": 1.0,
            "gift_adapt_lambda_density": 0.3,
            "gift_adapt_lambda_risk": 0.5,
            "gift_adapt_risk_step": 8.0,
        },
        "pso": {
            "gift_pso_flag": 1,
            "gift_pso_swarm": 12,
            "gift_pso_iters": 12,
            "gift_pso_inertia": 0.5,
            "gift_pso_cognitive": 1.4,
            "gift_pso_social": 1.4,
            "gift_pso_init_sigma": 0.02,
        },
    },
}


def canonical_name(init: str, features: Sequence[str]) -> str:
    if not features:
        return init
    suffix = "_".join(features)
    return f"{init}_{suffix}"


def apply_overrides(data: MutableMapping[str, object], overrides: Mapping[str, object]) -> None:
    for key, value in overrides.items():
        data[key] = value


def build_variant_config(
    src: Mapping[str, object],
    init: str,
    features: Sequence[str],
) -> Mapping[str, object]:
    result = dict(src)
    for key, value in RESET_DEFAULTS.items():
        result[key] = value
    apply_overrides(result, INIT_VARIANTS[init])

    for feature in features:
        overrides = FEATURE_OVERRIDES.get(init, {}).get(feature, {})
        apply_overrides(result, overrides)
    return result


def iter_source_configs(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("*.json"))


def dump_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4)
        fh.write("\n")


def generate_variants(
    test_root: Path,
    output_root: Path,
    inits: Sequence[str],
    feature_order: Sequence[str],
) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for src_path in iter_source_configs(test_root):
        base = json.loads(src_path.read_text(encoding="utf-8"))
        relative = src_path.relative_to(test_root)
        for init in inits:
            allowed_features = FEATURE_OVERRIDES.get(init, {})
            for feature_subset in powerset([f for f in feature_order if f in allowed_features]):
                name = canonical_name(init, feature_subset)
                dst = output_root / name / relative
                config = build_variant_config(base, init, feature_subset)
                dump_json(dst, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GiFt configuration variants.")
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("test"),
        help="Directory containing baseline benchmark json files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("configs/generated"),
        help="Directory that will receive generated variant configs.",
    )
    parser.add_argument(
        "--inits",
        type=lambda s: [token.strip() for token in s.split(",") if token.strip()],
        default="baseline,gift,giftplus",
        help="Comma separated list of initialisation schemes to emit.",
    )
    parser.add_argument(
        "--feature-order",
        type=lambda s: [token.strip() for token in s.split(",") if token.strip()],
        default="rass,pin,pso",
        help="Comma separated list defining feature priority when composing overrides.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_variants(
        args.test_root,
        args.output_root,
        args.inits,
        args.feature_order,
    )


if __name__ == "__main__":
    main()

