"""
Unit tests for benchmark parsing, mathematical normalization, and answer extraction.
"""

import pytest
from prunequest.benchmarks.gsm8k import (
    GSM8KBenchmark,
    extract_boxed_content,
    clean_math_answer,
    extract_gsm8k_gold_answer,
)
from prunequest.benchmarks.scoring import parse_numeric_value, verify_gsm8k_answer
from prunequest.exceptions import BenchmarkError


def test_boxed_parsing():
    assert extract_boxed_content(r"The answer is \boxed{42}.") == "42"
    assert extract_boxed_content(r"First \boxed{10} then \boxed{20}") == "20"
    assert extract_boxed_content(r"\boxed{\frac{3}{4}}") == r"\frac{3}{4}"
    assert extract_boxed_content(r"\boxed{\text{42}}") == r"\text{42}"
    assert extract_boxed_content("No boxed number") is None
    # Truncated generation with unclosed brace
    assert extract_boxed_content(r"Final result: \boxed{15") == "15"


def test_clean_math_answer():
    assert clean_math_answer("$1,250") == "1250"
    assert clean_math_answer("50%") == "50"
    assert clean_math_answer("12.5 kg") == "12.5"
    assert clean_math_answer(r"\text{10 dollars}") == "10"
    assert clean_math_answer("-15.0") == "-15.0"


def test_parse_numeric_value():
    assert parse_numeric_value("42") == 42.0
    assert parse_numeric_value("-15.5") == -15.5
    assert parse_numeric_value(r"\frac{3}{4}") == 0.75
    assert parse_numeric_value("1/2") == 0.5
    assert parse_numeric_value("invalid") is None


def test_extract_gold_answer():
    ans = "Natalia sold 48/2 = 24 clips.\n#### 72"
    assert extract_gsm8k_gold_answer(ans) == "72"
    ans_neg = "Net balance is negative.\n#### -15"
    assert extract_gsm8k_gold_answer(ans_neg) == "-15"
    ans_commas = "Population grew.\n#### 1,250"
    assert extract_gsm8k_gold_answer(ans_commas) == "1250"


def test_gsm8k_verification_logic():
    bench = GSM8KBenchmark()

    # Exact boxed match
    corr, pred, gold = bench.verify_answer(r"Reasoning... \boxed{72}", "#### 72")
    assert corr is True
    assert gold == "72"

    # Currency format
    corr, pred, gold = bench.verify_answer(r"\boxed{\$42}", "#### 42")
    assert corr is True

    # Decimal
    corr, pred, gold = bench.verify_answer(r"\boxed{3.5}", "#### 3.5")
    assert corr is True

    # Negative number
    corr, pred, gold = bench.verify_answer(r"\boxed{-15}", "#### -15")
    assert corr is True

    # Fraction equivalence
    corr, pred, gold = bench.verify_answer(r"\boxed{\frac{1}{2}}", "#### 0.5")
    assert corr is True
    corr, pred, gold = bench.verify_answer(r"\boxed{3/4}", "#### 0.75")
    assert corr is True

    # Wrong answer
    corr, pred, gold = bench.verify_answer(r"\boxed{73}", "#### 72")
    assert corr is False

    # Mentioning number in text but wrong final box
    corr, pred, gold = bench.verify_answer(r"We had 72 earlier but final is \boxed{10}", "#### 72")
    assert corr is False

    # Direct verify_gsm8k_answer tests
    corr, pred, gold = verify_gsm8k_answer("The result is 42.", "#### 42")
    assert corr is True
    assert gold == "42"


def test_gsm8k_unknown_split_raises():
    bench = GSM8KBenchmark()
    bench.data_splits = {"smoke": [], "dev": [], "calibration": []}
    dummy_model = None
    dummy_tok = None
    with pytest.raises(BenchmarkError, match="Unknown split 'invalid_split'"):
        bench.evaluate(dummy_model, dummy_tok, split_name="invalid_split")
