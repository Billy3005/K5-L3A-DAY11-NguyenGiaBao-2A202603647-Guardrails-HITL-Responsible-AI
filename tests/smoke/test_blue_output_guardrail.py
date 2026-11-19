"""Verify protected demo values are redacted locally at the output boundary."""
import asyncio
from types import SimpleNamespace

from google.genai import types

from core.config import DEMO_SECRETS
from guardrails.output_guardrails import OutputGuardrailPlugin


def test_output_callback_removes_demo_values_and_pii():
    assert DEMO_SECRETS
    email = "customer@example.com"
    phone = "0901234567"
    sensitive_values = [*DEMO_SECRETS, email, phone]
    response = SimpleNamespace(content=types.Content(
        role="model",
        parts=[
            types.Part.from_text(text="Savings information. "),
            types.Part.from_text(text="; ".join(sensitive_values)),
        ],
    ))
    plugin = OutputGuardrailPlugin(use_llm_judge=False)

    result = asyncio.run(plugin.after_model_callback(
        callback_context=None, llm_response=response,
    ))

    cleaned = "".join(part.text for part in result.content.parts)
    assert "Savings information." in cleaned
    assert "[REDACTED]" in cleaned
    assert all(value not in cleaned for value in sensitive_values)
    assert plugin.redacted_count == 1


def test_password_values_are_fully_redacted():
    from guardrails.output_guardrails import content_filter

    for text, leftover in [
        ("password=Local!77", "!77"),
        ('password="Local pass 77" for the vault', "pass 77"),
        ("Mật khẩu là Abc!x9, vui lòng đổi.", "!x9"),
    ]:
        result = content_filter(text)
        assert result["safe"] is False
        assert leftover not in result["redacted"], result["redacted"]


def test_money_amounts_are_not_mistaken_for_national_ids():
    from assignment.pipeline import is_egress_allowed
    from guardrails.output_guardrails import content_filter

    for text in [
        "The approved transfer amount is 500000000 VND.",
        "Số tiền 500000000 đồng đã được chuyển.",
        "approved transfer amount 500000000",
    ]:
        assert content_filter(text)["safe"] is True, text
    assert is_egress_allowed(
        "https://api.vinbank.example/v1/transfers", "approved transfer amount 500000000"
    ) is True

    for text in ["CCCD của tôi là 001203004567", "ID number 123456789 on file"]:
        result = content_filter(text)
        assert result["safe"] is False and "[REDACTED]" in result["redacted"], text


def test_national_ids_followed_by_punctuation_are_redacted():
    from guardrails.output_guardrails import content_filter

    for text, value in [
        ("CMND 123456789.", "123456789"),
        ("CCCD 123456789012, verified.", "123456789012"),
        ("Số CCCD: 001203004567; vui lòng giữ kín", "001203004567"),
    ]:
        result = content_filter(text)
        assert result["safe"] is False, text
        assert value not in result["redacted"], result["redacted"]
    # Thousand separators are a money amount, not an ID.
    assert content_filter("Hạn mức 1.500.000.000 cho khoản vay.")["safe"] is True


def test_lab_pii_dataset_values_are_removed():
    import json
    import re
    from pathlib import Path
    from guardrails.output_guardrails import content_filter

    data = json.loads(
        (Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json")
        .read_text(encoding="utf-8")
    )
    for case in data["pii_cases"]:
        result = content_filter(case["input_text"])
        assert result["safe"] is case["expect_safe"], case["id"]
        if not case["expect_safe"]:
            # No 9+ digit run (ID/phone) may survive redaction.
            assert not re.search(r"\d{9,}", result["redacted"]), (case["id"], result["redacted"])
