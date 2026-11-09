"""
Unit tests for benchmark parsing, mathematical normalization, and answer extraction.
"""

from prunequest.benchmarks.gsm8k import (
    GSM8KBenchmark,
    extract_boxed_content,
    clean_math_answer,
    extract_gsm8k_gold_answer,
)


def test_boxed_parsing():
    assert extract_boxed_content(r"The answer is \boxed{42}.") == "42"
    assert extract_boxed_content(r"First \boxed{10} then \boxed{20}") == "20"
    assert extract_boxed_content(r"\boxed{\frac{3}{4}}") == r"\frac{3}{4}"
    assert extract_boxed_content("No boxed number") is None


def test_clean_math_answer():
    assert clean_math_answer("$1,250") == "1250"
    assert clean_math_answer("50%") == "50"
    assert clean_math_answer("12.5 kg") == "12.5"
    assert clean_math_answer(r"\text{10 dollars}") == "10"


def test_extract_gold_answer():
    ans = "Natalia sold 48/2 = 24 clips.\n#### 72"
    assert extract_gsm8k_gold_answer(ans) == "72"


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

    # Fraction equivalence
    corr, pred, gold = bench.verify_answer(r"\boxed{\frac{1}{2}}", "#### 0.5")
    assert corr is True

    # Wrong answer
    corr, pred, gold = bench.verify_answer(r"\boxed{73}", "#### 72")
    assert corr is False

    # Mentioning number in text but wrong final box
    corr, pred, gold = bench.verify_answer(r"We had 72 earlier but final is \boxed{10}", "#### 72")
    assert corr is False
