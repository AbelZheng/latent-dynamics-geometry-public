# Latent Dynamics & Geometry

Latent Dynamics & Geometry (LDG) is a research-methodology knowledge system for latent states, dynamical systems, neural and behavioral dynamics, representation geometry, dimensionality reduction, topology, executable toy models, and synthetic robustness benchmarks.

This repository is a curated public snapshot of the LDG scientific layer. Internal planning, agent operations, private research data, and laboratory-translation work are not included.

## Browse the project

- Public website: https://abelzheng.github.io/latent-dynamics-geometry-public/
- Concepts: terminology, formal meaning, adjacent concepts, and non-implications
- Methods: assumptions, learned and fixed quantities, failure modes, baselines, and evaluation
- Math: reusable mathematical objects and derivations
- Toy Models: deterministic generation/recovery studies with known truth
- Benchmarks: bounded synthetic robustness studies and cross-suite synthesis

## Feedback

Page-specific comments are provided through GitHub Discussions. Comments are public and should be treated as review suggestions rather than accepted scientific conclusions. Substantive corrections are incorporated only after evidence review in the private canonical repository and a new audited public snapshot.

## Reproducibility boundary

The public snapshot includes the scientific Quarto sources, bounded Python implementations, and committed synthetic artifacts used by the published pages. It does not include private or laboratory data. Maturity labels record development depth and review state, not certainty or universal validity.

## Snapshot provenance

This snapshot was exported from canonical source commit `6c12ce644df7e48e5be3ecd8942ea7928599fdc5` using public-export schema version `1`. The export is a file-level snapshot, not a mirror of the private repository's Git history.

## Local rendering

With a compatible Quarto installation:

```bash
quarto render
```

The generated `_site/` directory is not committed to `main`; GitHub Actions publishes it to the repository's `gh-pages` branch.
