"""
Shared mathematical answer extraction and verification utilities.
Consolidates boxed parsing, number cleaning, and answer verification
across all benchmark evaluators (native GSM8K and FLAP GSM8K).
"""

import re
import math
from typing import Optional, Tuple

try:
    from math_verify import parse as mv_parse, verify as mv_verify
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extract content from LaTeX \\boxed{...}, properly handling arbitrarily nested braces.
    Returns the last boxed expression if multiple exist.
    Handles unclosed/truncated braces gracefully.
    """
    if not text:
        return None

    results = []
    idx = 0
    text_len = len(text)

    while idx < text_len:
        pos = text.find(r"\boxed{", idx)
        if pos == -1:
            break
        brace_count = 1
        i = pos + len(r"\boxed{")
        start_content = i
        while i < text_len and brace_count > 0:
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
            i += 1
        if brace_count == 0:
            content = text[start_content : i - 1].strip()
            results.append(content)
            idx = i
        else:
            # Unclosed brace (e.g. truncated generation)
            content = text[start_content:].strip()
            if content:
                results.append(content)
            break

    return results[-1] if results else None


def clean_math_answer(raw_str: str) -> str:
    """
    Clean math string by stripping LaTeX text wrappers, currencies, percentages,
    commas, units, and trailing punctuation.
    """
    if not raw_str:
        return ""
    s = raw_str.strip().strip("$")
    # Remove text wrappers like \text{...}, \mathrm{...}, \mathbf{...}
    s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("%", "").replace(",", "")
    s = s.rstrip(".")
    # Remove common trailing measurement units
    s = re.sub(
        r"(?:kg|km|miles|meters|hours|mins|minutes|seconds|days|weeks|dollars|cents|points|degrees)$",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    return s


def extract_gsm8k_gold_answer(answer_str: str) -> str:
    """
    Extract the gold numerical answer from GSM8K format (after '####').
    """
    if not answer_str:
        return ""
    if "####" in answer_str:
        ans = answer_str.split("####")[-1].strip().replace(",", "")
        return clean_math_answer(ans)
    nums = re.findall(r"-?\d+(?:\.\d+)?", answer_str.replace(",", ""))
    return clean_math_answer(nums[-1]) if nums else clean_math_answer(answer_str.strip())


def parse_numeric_value(s: str) -> Optional[float]:
    """
    Parses a string into a float, supporting LaTeX fractions (\\frac{a}{b}),
    slash fractions (a/b), integers, and decimals.
    """
    if not s:
        return None
    s = s.strip()
    # Match LaTeX fraction: \frac{num}{denom}
    frac_match = re.match(r"^\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}$", s)
    if frac_match:
        num, denom = float(frac_match.group(1)), float(frac_match.group(2))
        return num / denom if denom != 0 else None

    # Match standard slash fraction: num / denom
    slash_frac = re.match(r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$", s)
    if slash_frac:
        num, denom = float(slash_frac.group(1)), float(slash_frac.group(2))
        return num / denom if denom != 0 else None

    try:
        return float(s)
    except ValueError:
        return None


def verify_gsm8k_answer(prediction_text: str, gold_str: str) -> Tuple[bool, str, str]:
    """
    Robust verification of model prediction against gold GSM8K answer.
    Returns: (is_correct, extracted_prediction, cleaned_gold).
    """
    gold_cleaned = extract_gsm8k_gold_answer(gold_str)
    if not prediction_text or not gold_cleaned:
        return False, "", gold_cleaned

    # 1. Try math-verify library if available
    if HAS_MATH_VERIFY:
        try:
            p_pred = mv_parse(prediction_text)
            p_gold = mv_parse(gold_cleaned)
            if p_pred and p_gold and mv_verify(p_gold, p_pred):
                extracted = str(p_pred[-1]) if p_pred else ""
                return True, extracted, gold_cleaned
        except Exception:
            pass

    # 2. Check LaTeX \boxed{...} content
    boxed = extract_boxed_content(prediction_text)
    if boxed is not None:
        cand = clean_math_answer(boxed)
        if HAS_MATH_VERIFY:
            try:
                p_cand = mv_parse(cand)
                p_gold = mv_parse(gold_cleaned)
                if p_cand and p_gold and mv_verify(p_gold, p_cand):
                    return True, cand, gold_cleaned
            except Exception:
                pass

        cand_val = parse_numeric_value(cand)
        gold_val = parse_numeric_value(gold_cleaned)
        if cand_val is not None and gold_val is not None:
            if math.isclose(cand_val, gold_val, rel_tol=1e-5, abs_tol=1e-5):
                return True, cand, gold_cleaned

        if cand.strip().lower() == gold_cleaned.strip().lower():
            return True, cand, gold_cleaned

        return False, cand, gold_cleaned

    # 3. Check for explicit '####' separator in prediction
    if "####" in prediction_text:
        pred_cand = prediction_text.split("####")[-1].strip()
        cand = clean_math_answer(pred_cand)
        cand_val = parse_numeric_value(cand)
        gold_val = parse_numeric_value(gold_cleaned)
        if cand_val is not None and gold_val is not None:
            if math.isclose(cand_val, gold_val, rel_tol=1e-5, abs_tol=1e-5):
                return True, pred_cand, gold_cleaned
        if cand.strip().lower() == gold_cleaned.strip().lower():
            return True, pred_cand, gold_cleaned
        return False, pred_cand, gold_cleaned

    # 4. Check for explicit final answer declarations (e.g. "The answer is X", "Final Answer: X")
    patterns = [
        r"(?:the\s+answer\s+is|final\s+answer\s+is|total\s+is|result\s+is)[\s:]*([\$]?-?\d+(?:,\d+)*(?:\.\d+)?|\$?-?\d+/\d+)",
        r"(?:equals|equal\s+to)[\s:]*([\$]?-?\d+(?:,\d+)*(?:\.\d+)?|\$?-?\d+/\d+)\s*\.?\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, prediction_text, flags=re.IGNORECASE)
        if matches:
            last_match = matches[-1]
            cand = clean_math_answer(last_match)
            cand_val = parse_numeric_value(cand)
            gold_val = parse_numeric_value(gold_cleaned)
            if cand_val is not None and gold_val is not None:
                if math.isclose(cand_val, gold_val, rel_tol=1e-5, abs_tol=1e-5):
                    return True, cand, gold_cleaned
            if cand.strip().lower() == gold_cleaned.strip().lower():
                return True, cand, gold_cleaned

    # Avoid broad regex matching that marks answers correct merely because a number appears in reasoning
    return False, "", gold_cleaned
