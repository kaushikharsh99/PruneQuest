# Benchmarks and Evaluation

Evaluation behavior is implemented in `src/prunequest/benchmarks/`. It includes native benchmark code, a BenchmarkSpec compiler, and an optional LM Evaluation Harness adapter. Available tasks and features depend on installed extras and dataset access.

## Evaluation integrity

- Use a development/acceptance split to make pruning decisions and reserve held-out test data for final assessment.
- Record the benchmark, split, sample limit, prompt format, scorer, and relevant model/tokenizer revisions when comparing results.
- Small smoke subsets are useful for checking execution, not for strong quality claims.
- Dataset sample ordering and split behavior are defined by each benchmark/dataset implementation; do not assume the same partition sizes across tasks.
- Retention values are meaningful only alongside the baseline metric and its direction. Review `benchmarks/constraints.py` and the selected benchmark implementation.

## Discover tasks

```bash
prunequest benchmarks --help
prunequest benchmarks list --limit 20
prunequest benchmarks search math
prunequest benchmarks info gsm8k
```

LM Evaluation Harness support is optional (`pip install -e '.[eval]'`). Consult command help for options and installed-task behavior.

## Custom BenchmarkSpec

The compiler and schema are in `src/prunequest/benchmarks/spec.py`. Use a checked-in example as the starting point, then validate and preview the spec before evaluation:

```bash
prunequest benchmarks validate --spec path/to/benchmark.yaml
prunequest benchmarks preview --spec path/to/benchmark.yaml --limit 3
```

Supported dataset sources and scorers are determined by the spec compiler and dataset library versions. Do not infer support from a YAML example alone.

## Interpreting results

Benchmark scores are estimates over sampled data and can vary with generation settings, prompt templates, and dataset revisions. A retention threshold is an optimization constraint, not a guarantee of generalization. Report sample counts and configuration with any result you publish.
