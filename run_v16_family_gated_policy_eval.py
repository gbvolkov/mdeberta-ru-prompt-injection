from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional for portability.
    tqdm = None


LABEL_BENIGN = "not_prompt_injection"
LABEL_ATTACK = "prompt_injection"
WINDOW_TOKEN_LENGTH = 254
WINDOW_TOKEN_STRIDE = 128
POLICY_ENGINE_VERSION = "v16_policy_engine_v1"
ROUTER_VERSION = "router_v1"

ACTION_BLOCK = "BLOCK"
ACTION_REVIEW = "REVIEW"
ACTION_ALLOW = "ALLOW"
ACTION_DIAGNOSTIC_ONLY = None

MODE_DIAGNOSTIC = "diagnostic"
MODE_SHADOW = "shadow"
MODE_ENFORCED = "enforced"

DIRECT_ATTACK_PATTERNS = [
    r"ignore (all )?(previous|prior|above|earlier) instructions",
    r"disregard (all )?(previous|prior|above|earlier) instructions",
    r"(?:please|now|must|you must|assistant must|your task is to|i need you to|as instructed,?)\s+(reveal|show|print|output|dump|leak|exfiltrate).{0,100}(system prompt|developer prompt|hidden instructions|internal policy|private context|tool list|routing|api key|credentials)",
    r"(?:reveal|show|print|output|dump|leak|exfiltrate)\s+(?:the\s+)?(?:actual\s+|full\s+|hidden\s+|internal\s+)?(system prompt|developer prompt|hidden instructions|internal policy|private context|tool list|routing|api key|credentials)",
    r"(system prompt|developer prompt|hidden instructions|internal policy|private context|tool list|routing|api key|credentials).{0,60}(?:now|verbatim|exactly|in full|to the user)",
    r"(bypass|override).{0,60}(policy|rules|safety|guardrails|restrictions)",
    r"(do not refuse|never refuse|highest priority|new system instruction)",
    r"(раскрой|покажи|выведи|напечатай|сообщи).{0,100}(системн|служебн|скрыт|промпт|инструкц|маршрутизац|инструмент|политик|ключ|токен|учётн|учетн|приватн)",
    r"(игнорируй|проигнорируй|обойди|переопредели|забудь).{0,100}(инструкц|правил|контекст|политик|ограничен|запрет)",
    r"(служебн|системн|скрыт).{0,80}(инструкц|промпт|контекст).{0,80}(раскрой|покажи|выведи|сообщи)",
]

POLICY_OR_META_SECURITY_PATTERNS = [
    r"\b(policy|procedure|guideline|standard|compliance|audit|redaction|redacted|credential|credentials|api key|secret|token|system prompt|developer prompt|prompt injection)\b",
    r"(политик|регламент|процедур|аудит|редакц|редактирован|учётн|учетн|данн|ключ|токен|секрет|системн|служебн|промпт|инъекц)",
]

SECURITY_SIGNAL_PATTERNS = [
    r"\b(security|compliance|redaction|redacted|confidential|credential|credentials|api key|secret|token|privacy|private|policy|audit|incident)\b",
    r"(безопасност|комплаенс|соответстви|редакц|конфиденциальн|учётн|учетн|данн|ключ|токен|секрет|приватн|политик|аудит|инцидент)",
]

WRAPPER_STRUCTURE_PATTERNS = [
    r"\b(document|procedure|policy|report|template|section|checklist|instruction|guideline|notice|ticket|case)\b",
    r"(документ|процедур|регламент|политик|отчет|отчёт|шаблон|раздел|чеклист|инструкц|уведомлен|заявк|тикет)",
]

DIRECT_ATTACK_RE = re.compile("|".join(f"(?:{p})" for p in DIRECT_ATTACK_PATTERNS), re.IGNORECASE | re.DOTALL)
POLICY_OR_META_SECURITY_RE = re.compile(
    "|".join(f"(?:{p})" for p in POLICY_OR_META_SECURITY_PATTERNS), re.IGNORECASE | re.DOTALL
)
SECURITY_SIGNAL_RE = re.compile("|".join(f"(?:{p})" for p in SECURITY_SIGNAL_PATTERNS), re.IGNORECASE | re.DOTALL)
WRAPPER_STRUCTURE_RE = re.compile("|".join(f"(?:{p})" for p in WRAPPER_STRUCTURE_PATTERNS), re.IGNORECASE | re.DOTALL)
BENIGN_POLICY_NEGATION_RE = re.compile(
    r"(?:do not|don't|never|must not|should not|cannot|prohibit(?:ed)?|forbid(?:den)?|avoid|refuse to)\s+"
    r"(?:reveal|show|print|output|dump|leak|exfiltrate|disclose|share)[^.!?\n]{0,100}?"
    r"(?:system prompt|developer prompt|hidden instructions|internal policy|private context|tool list|routing|api key|credentials)",
    re.IGNORECASE | re.DOTALL,
)

AUTO_ALLOW_FORBIDDEN_CATEGORIES = {
    "job_descriptions",
    "hr_policies",
    "support_documentation",
    "technical_documentation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen V16 through a diagnostic/shadow policy engine.")
    parser.add_argument("--stage1-model-id", default="gbv/mdeberta-ru-prompt-injection")
    parser.add_argument("--reviewer-model-id", required=True)
    parser.add_argument("--reviewer-calibration-id")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--window-output-jsonl")
    parser.add_argument("--v16-threshold-low", type=float, default=0.82)
    parser.add_argument("--v16-threshold-protect", type=float, default=0.999)
    parser.add_argument("--reviewer-threshold-block", type=float, default=0.30)
    parser.add_argument("--reviewer-threshold-allow", type=float, default=0.10)
    parser.add_argument("--reviewer-allowlist", default="security_compliance_redaction_wrappers")
    parser.add_argument("--protected-family-policy-json", default="policy/protected-family-policy.json")
    parser.add_argument("--slice-router-config-json", default="policy/slice-router-config.json")
    parser.add_argument("--certified-slices-json", default="policy/certified-slices-empty.json")
    parser.add_argument("--mode", choices=[MODE_DIAGNOSTIC, MODE_SHADOW, MODE_ENFORCED], default=MODE_DIAGNOSTIC)
    parser.add_argument("--current-action-column")
    parser.add_argument("--window-batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="document_label")
    parser.add_argument("--id-column", default="document_id")
    parser.add_argument("--limit-documents", type=int)
    parser.add_argument("--progress-every-docs", type=int, default=25)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-diagnostic-router-all-windows", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalized_for_hash(text: str) -> str:
    return normalize_text(text).casefold()


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def payload_hash(payload: Any) -> str:
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True), length=16)


def label_name(value: Any) -> str:
    if isinstance(value, bool):
        return LABEL_ATTACK if value else LABEL_BENIGN
    if isinstance(value, int):
        return LABEL_ATTACK if value == 1 else LABEL_BENIGN
    text = str(value or "").strip().lower()
    if text in {"1", "attack", "malicious", "prompt_injection", "injection"}:
        return LABEL_ATTACK
    if text in {"0", "benign", "not_prompt_injection", "non_injection", "safe"}:
        return LABEL_BENIGN
    return text or LABEL_BENIGN


def infer_language(text: str) -> str:
    cyr = len(re.findall(r"[А-Яа-яЁё]", text or ""))
    lat = len(re.findall(r"[A-Za-z]", text or ""))
    if cyr and lat and min(cyr, lat) / max(cyr, lat) >= 0.12:
        return "mixed"
    if cyr > lat:
        return "ru"
    if lat:
        return "en"
    return "unknown"


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if isinstance(row, dict):
                yield row


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def choose_device(requested: str) -> torch.device:
    import torch

    if requested == "cuda":
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_production_windows(text: str, tokenizer: Any) -> list[dict[str, Any]]:
    token_ids = tokenizer.encode(text or "", add_special_tokens=False)
    if not token_ids:
        return []
    windows: list[dict[str, Any]] = []
    start = 0
    index = 0
    while start < len(token_ids):
        chunk = token_ids[start : start + WINDOW_TOKEN_LENGTH]
        window_text = tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        windows.append(
            {
                "window_index": index,
                "token_start": start,
                "token_end": min(start + len(chunk), len(token_ids)),
                "token_length": len(chunk),
                "text": window_text,
            }
        )
        if start + WINDOW_TOKEN_LENGTH >= len(token_ids):
            break
        start += WINDOW_TOKEN_STRIDE
        index += 1
    return windows


def window_count_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 20:
        return "5-20"
    if count <= 50:
        return "21-50"
    if count <= 100:
        return "51-100"
    return "101+"


def iter_corpus(args: argparse.Namespace) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for idx, row in enumerate(iter_jsonl(args.input_jsonl)):
        text = row.get(args.text_column) or row.get("document_text") or row.get("content") or row.get("window_text") or ""
        if not normalize_text(text):
            continue
        docs.append(
            {
                **row,
                "document_id": str(row.get(args.id_column) or row.get("id") or f"doc_{idx}"),
                "text": text,
                "document_label": label_name(row.get(args.label_column, row.get("label", LABEL_BENIGN))),
                "source_name": row.get("source_name") or row.get("source") or "unknown",
                "eval_category": row.get("category") or "unknown",
                "eval_semantic_family": row.get("semantic_family") or "unknown",
                "language": row.get("language") or infer_language(text),
            }
        )
        if args.limit_documents and len(docs) >= args.limit_documents:
            break
    return docs


def score_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    *,
    device: Any,
    batch_size: int,
) -> list[float]:
    import torch

    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=256, padding=True, return_tensors="pt")
            enc = {key: value.to(device) for key, value in enc.items()}
            logits = model(**enc).logits
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits[:, 0])
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(float(v) for v in probs.detach().cpu())
    return scores


def direct_attack_anchor_present(text: str) -> bool:
    text = text or ""
    scrubbed = BENIGN_POLICY_NEGATION_RE.sub(" ", text)
    return bool(DIRECT_ATTACK_RE.search(scrubbed))


def policy_or_meta_security_mention_present(text: str) -> bool:
    return bool(POLICY_OR_META_SECURITY_RE.search(text or ""))


def load_policy_config(path: str | Path) -> tuple[dict[str, str], str]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("protected-family policy must be a JSON object")
    allowed = {ACTION_BLOCK, ACTION_REVIEW}
    policy: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError("protected-family policy has an invalid key")
        if value not in allowed:
            raise ValueError(f"protected-family policy action for {key!r} must be BLOCK or REVIEW")
        policy[key] = value
    return policy, payload_hash(policy)


def load_router_config(path: str | Path) -> tuple[dict[str, Any], str]:
    payload = read_json(path)
    validate_router_config_payload(payload)
    return payload, payload_hash(payload)


def validate_router_config_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("slice-router config must be a JSON object")
    allowed_keys = {
        "router_version",
        "candidate_slices",
        "trusted_metadata_fields",
        "ignored_untrusted_fields",
        "first_certification_slice",
        "forbidden_auto_allow_candidate_slices",
        "notes",
    }
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"slice-router config has unknown keys: {unknown_keys}")
    if payload.get("router_version") != ROUTER_VERSION:
        raise ValueError(f"slice-router config router_version must be {ROUTER_VERSION}")
    require_exact_string_list(
        payload,
        "candidate_slices",
        ["security_compliance_redaction_wrappers"],
    )
    if payload.get("first_certification_slice") != "security_compliance_redaction_wrappers":
        raise ValueError("router_v1 first_certification_slice must be security_compliance_redaction_wrappers")
    require_exact_string_list(
        payload,
        "trusted_metadata_fields",
        ["trusted_source_metadata", "source_metadata_trusted"],
    )
    require_exact_string_list(
        payload,
        "ignored_untrusted_fields",
        ["category", "semantic_family", "source_declared_category"],
    )
    require_exact_string_list(
        payload,
        "forbidden_auto_allow_candidate_slices",
        ["hr_policies", "job_descriptions", "support_documentation", "technical_documentation"],
    )
    notes = payload.get("notes", [])
    if not isinstance(notes, list) or any(not isinstance(value, str) or not value for value in notes):
        raise ValueError("slice-router config notes must be a string list")


def require_exact_string_list(payload: dict[str, Any], key: str, expected: list[str]) -> None:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"slice-router config {key} must be a string list")
    if len(value) != len(set(value)):
        raise ValueError(f"slice-router config {key} must not contain duplicates")
    if value != expected:
        raise ValueError(f"slice-router config {key} must exactly equal {expected}")


def slice_specificity(row: dict[str, Any]) -> int:
    score = 0
    for key in ("slice", "language", "router_rule_ids"):
        value = row.get(key)
        if value:
            score += 1
    return score


def validate_certified_slices(payload: Any, *, router_config_hash: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("certified-slices config must be a JSON object")
    slices = payload.get("slices", [])
    if not isinstance(slices, list):
        raise ValueError("certified-slices config field 'slices' must be a list")
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    allowed_row_keys = {
        "certified_slice_id",
        "slice",
        "language",
        "router_rule_ids",
        "router_version",
        "router_config_hash",
        "reviewer_model_id",
        "reviewer_calibration_id",
        "policy_engine_version",
        "policy_config_hash",
        "authority",
        "notes",
    }
    for idx, row in enumerate(slices):
        if not isinstance(row, dict):
            raise ValueError(f"certified-slices entry {idx} must be an object")
        unknown_keys = sorted(set(row) - allowed_row_keys)
        if unknown_keys:
            raise ValueError(f"certified-slices entry {idx} has unknown keys: {unknown_keys}")
        if row.get("slice") != "security_compliance_redaction_wrappers":
            raise ValueError(f"certified-slices entry {idx} slice must be security_compliance_redaction_wrappers")
        if row.get("router_version") != ROUTER_VERSION:
            raise ValueError(f"certified-slices entry {idx} has invalid router_version")
        if not row.get("router_config_hash"):
            raise ValueError(f"certified-slices entry {idx} missing router_config_hash")
        if row.get("router_config_hash") != router_config_hash:
            raise ValueError(f"certified-slices entry {idx} router_config_hash mismatch")
        router_rule_ids = row.get("router_rule_ids") or []
        if not isinstance(router_rule_ids, list) or any(not isinstance(value, str) or not value for value in router_rule_ids):
            raise ValueError(f"certified-slices entry {idx} router_rule_ids must be a string list")
        authority = row.get("authority")
        if not isinstance(authority, dict):
            raise ValueError(f"certified-slices entry {idx} missing authority")
        for key in authority:
            if key not in {"review_routing", "auto_allow", "ultra_allow"}:
                raise ValueError(f"certified-slices entry {idx} has unknown authority key {key!r}")
            if not isinstance(authority[key], bool):
                raise ValueError(f"certified-slices entry {idx} authority {key!r} must be boolean")
        if authority.get("auto_allow"):
            required_auto_allow_fields = [
                "reviewer_model_id",
                "reviewer_calibration_id",
                "policy_engine_version",
                "policy_config_hash",
                "router_config_hash",
            ]
            for key in required_auto_allow_fields:
                if not row.get(key):
                    raise ValueError(f"certified-slices entry {idx} auto_allow missing required field {key!r}")
            if row.get("policy_engine_version") != POLICY_ENGINE_VERSION:
                raise ValueError(f"certified-slices entry {idx} policy_engine_version mismatch")
            if "router_rule_ids" not in row:
                raise ValueError(f"certified-slices entry {idx} auto_allow must include exact non-empty router_rule_ids")
            if not router_rule_ids:
                raise ValueError(f"certified-slices entry {idx} auto_allow router_rule_ids must be non-empty")
        identity = (
            row.get("slice"),
            row.get("language"),
            tuple(sorted(row.get("router_rule_ids") or [])),
            slice_specificity(row),
        )
        if identity in seen:
            raise ValueError(f"duplicate equal-specificity certified slice definition for {identity}")
        seen[identity] = row
    return payload


def config_hashes(
    *,
    protected_policy_hash: str,
    router_config_hash: str,
    certified_slices_hash: str,
    args: argparse.Namespace,
) -> str:
    return payload_hash(
        {
            "policy_engine_version": POLICY_ENGINE_VERSION,
            "protected_policy_hash": protected_policy_hash,
            "router_config_hash": router_config_hash,
            "certified_slices_hash": certified_slices_hash,
            "v16_threshold_low": args.v16_threshold_low,
            "v16_threshold_protect": args.v16_threshold_protect,
            "reviewer_threshold_block": args.reviewer_threshold_block,
            "reviewer_threshold_allow": args.reviewer_threshold_allow,
            "reviewer_allowlist": args.reviewer_allowlist,
            "mode": args.mode,
        }
    )


def trusted_source_metadata(row: dict[str, Any]) -> bool:
    return truthy(row.get("trusted_source_metadata")) or truthy(row.get("source_metadata_trusted"))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "trusted"}


def router_v1(text: str, row: dict[str, Any], *, router_config_hash: str) -> dict[str, Any]:
    language = row.get("language") or infer_language(text)
    direct_anchor = direct_attack_anchor_present(text)
    policy_mention = policy_or_meta_security_mention_present(text)
    security_signal = bool(SECURITY_SIGNAL_RE.search(text or ""))
    wrapper_structure = bool(WRAPPER_STRUCTURE_RE.search(text or ""))
    rules: list[str] = []

    if direct_anchor:
        rules.append("direct_attack_anchor_v1")
        semantic_family = "direct_model_control_or_exfiltration"
        return {
            "router_version": ROUTER_VERSION,
            "router_config_hash": router_config_hash,
            "category": "direct_protected_attack_anchor",
            "semantic_family": semantic_family,
            "language": language,
            "confidence": "high",
            "routing_reason": "direct_attack_anchor_present",
            "router_rule_ids": rules,
        }

    if trusted_source_metadata(row) and row.get("category") == "security_compliance_redaction_wrappers":
        rules.append("trusted_security_wrapper_source_tag_v1")
        return {
            "router_version": ROUTER_VERSION,
            "router_config_hash": router_config_hash,
            "category": "security_compliance_redaction_wrappers",
            "semantic_family": "security_compliance_redaction_wrapper",
            "language": language,
            "confidence": "high",
            "routing_reason": "trusted_source_metadata_security_wrapper",
            "router_rule_ids": rules,
        }

    if security_signal and wrapper_structure and policy_mention:
        rules.extend(["security_signal_v1", "wrapper_structure_v1", "policy_or_meta_security_mention_v1"])
        return {
            "router_version": ROUTER_VERSION,
            "router_config_hash": router_config_hash,
            "category": "security_compliance_redaction_wrappers",
            "semantic_family": "security_compliance_redaction_wrapper",
            "language": language,
            "confidence": "high",
            "routing_reason": "deterministic_security_wrapper_structure",
            "router_rule_ids": rules,
        }

    if security_signal or wrapper_structure or policy_mention:
        rules.extend(
            rule
            for rule, active in [
                ("security_signal_v1", security_signal),
                ("wrapper_structure_v1", wrapper_structure),
                ("policy_or_meta_security_mention_v1", policy_mention),
            ]
            if active
        )
        return {
            "router_version": ROUTER_VERSION,
            "router_config_hash": router_config_hash,
            "category": "unknown",
            "semantic_family": "unknown",
            "language": language,
            "confidence": "low",
            "routing_reason": "insufficient_non_keyword_structure_for_high_confidence_slice",
            "router_rule_ids": rules,
        }

    return {
        "router_version": ROUTER_VERSION,
        "router_config_hash": router_config_hash,
        "category": "unknown",
        "semantic_family": "unknown",
        "language": language,
        "confidence": "low",
        "routing_reason": "no_router_match",
        "router_rule_ids": [],
    }


def matching_certified_slice(
    *,
    router_result: dict[str, Any],
    certified_slices: dict[str, Any],
    reviewer_model_id: str,
    reviewer_calibration_id: str | None,
    router_config_hash: str,
    policy_config_hash: str,
    policy_engine_version: str = POLICY_ENGINE_VERSION,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in certified_slices.get("slices", []):
        if row.get("slice") != router_result.get("category"):
            continue
        if row.get("language") and row.get("language") != router_result.get("language"):
            continue
        expected_rules = set(row.get("router_rule_ids") or [])
        actual_rules = set(router_result.get("router_rule_ids") or [])
        if bool((row.get("authority") or {}).get("auto_allow")):
            if not expected_rules or expected_rules != actual_rules:
                continue
        elif expected_rules and not expected_rules.issubset(actual_rules):
            continue
        if row.get("router_version") != router_result.get("router_version"):
            continue
        if row.get("router_config_hash") and row.get("router_config_hash") != router_config_hash:
            continue
        if row.get("policy_engine_version") and row.get("policy_engine_version") != policy_engine_version:
            continue
        if row.get("policy_config_hash") and row.get("policy_config_hash") != policy_config_hash:
            continue
        if row.get("reviewer_model_id") and row.get("reviewer_model_id") != reviewer_model_id:
            continue
        if row.get("reviewer_calibration_id") and row.get("reviewer_calibration_id") != reviewer_calibration_id:
            continue
        candidates.append(row)
    if not candidates:
        return None
    candidates.sort(key=slice_specificity, reverse=True)
    top = [row for row in candidates if slice_specificity(row) == slice_specificity(candidates[0])]
    if len(top) > 1:
        has_allow = any(bool((row.get("authority") or {}).get("auto_allow")) for row in top)
        has_review_or_deny = any(not bool((row.get("authority") or {}).get("auto_allow")) for row in top)
        if has_allow and has_review_or_deny:
            return next(row for row in top if not bool((row.get("authority") or {}).get("auto_allow")))
        raise ValueError("conflicting equal-specificity certified slice matches")
    return candidates[0]


def policy_decide_window(
    *,
    text: str,
    row: dict[str, Any],
    stage1_score: float,
    reviewer_score: float | None,
    args: argparse.Namespace,
    protected_policy: dict[str, str],
    router_config_hash: str,
    certified_slices: dict[str, Any],
    policy_config_hash: str,
    diagnostic_router_all_windows: bool,
) -> dict[str, Any]:
    steps = ["v16_scored", "direct_anchor_checked"]
    direct_anchor = direct_attack_anchor_present(text)
    policy_mention = policy_or_meta_security_mention_present(text)
    baseline_allow = False
    certified_auto_allow = False
    reviewer_potential_allow = False
    router_ran_for_decision = False
    router_result_decision: dict[str, Any] | None = None
    router_result_diagnostic: dict[str, Any] | None = None
    certified_slice = None
    blocking_rule = None
    allowing_rule = None
    reviewer_ran = reviewer_score is not None
    reviewer_used_for_decision = False

    if diagnostic_router_all_windows:
        router_result_diagnostic = router_v1(text, row, router_config_hash=router_config_hash)

    if direct_anchor:
        proposed_action = ACTION_BLOCK
        final_reason = "direct_attack_anchor_block"
        blocking_rule = "direct_attack_anchor_present"
        steps.append("direct_anchor_block")
    elif stage1_score < args.v16_threshold_low:
        proposed_action = ACTION_ALLOW
        final_reason = "baseline_v16_negative_allow"
        allowing_rule = "baseline_v16_below_threshold"
        baseline_allow = True
        steps.append("baseline_v16_negative_allow")
    else:
        router_result_decision = router_result_diagnostic or router_v1(text, row, router_config_hash=router_config_hash)
        router_ran_for_decision = True
        steps.extend(["slice_router_ran", "protected_policy_checked"])
        protected_action = protected_policy.get(str(router_result_decision.get("semantic_family") or ""))
        if router_result_decision["confidence"] == "low":
            proposed_action = ACTION_REVIEW
            final_reason = "low_confidence_router_review"
        elif protected_action == ACTION_BLOCK:
            proposed_action = ACTION_BLOCK
            final_reason = "protected_family_block"
            blocking_rule = f"protected_family:{router_result_decision['semantic_family']}"
        elif protected_action == ACTION_REVIEW:
            proposed_action = ACTION_REVIEW
            final_reason = "protected_family_review"
            blocking_rule = f"protected_family:{router_result_decision['semantic_family']}"
        elif router_result_decision["category"] not in set(parse_csv(args.reviewer_allowlist)):
            proposed_action = ACTION_REVIEW
            final_reason = "slice_not_reviewer_allowlisted"
        else:
            certified_slice = matching_certified_slice(
                router_result=router_result_decision,
                certified_slices=certified_slices,
                reviewer_model_id=args.reviewer_model_id,
                reviewer_calibration_id=args.reviewer_calibration_id,
                router_config_hash=router_config_hash,
                policy_config_hash=policy_config_hash,
            )
            certified_authority = (certified_slice or {}).get("authority") or {}
            ultra_certified = bool(certified_authority.get("ultra_allow"))
            if stage1_score >= args.v16_threshold_protect and not ultra_certified:
                proposed_action = ACTION_REVIEW
                final_reason = "v16_protect_threshold_review"
            elif reviewer_score is None:
                proposed_action = ACTION_REVIEW
                final_reason = "reviewer_not_scored_review"
            elif reviewer_score >= args.reviewer_threshold_block:
                reviewer_used_for_decision = True
                proposed_action = ACTION_BLOCK
                final_reason = "reviewer_attack_block"
                blocking_rule = "reviewer_score_at_or_above_block_cutoff"
            elif reviewer_score <= args.reviewer_threshold_allow:
                reviewer_used_for_decision = True
                reviewer_potential_allow = True
                if bool(certified_authority.get("auto_allow")) and args.reviewer_calibration_id:
                    proposed_action = ACTION_ALLOW
                    final_reason = "certified_reviewer_auto_allow"
                    allowing_rule = "certified_reviewer_auto_allow"
                    certified_auto_allow = True
                else:
                    proposed_action = ACTION_ALLOW
                    final_reason = "reviewer_potential_allow_uncertified"
                    allowing_rule = "diagnostic_potential_allow_only"
            else:
                reviewer_used_for_decision = True
                proposed_action = ACTION_REVIEW
                final_reason = "reviewer_gray_zone_review"

    effective_action: str | None
    if args.mode == MODE_DIAGNOSTIC:
        effective_action = ACTION_DIAGNOSTIC_ONLY
    elif args.mode == MODE_SHADOW:
        effective_action = str(row.get(args.current_action_column)) if args.current_action_column else None
    else:
        if proposed_action == ACTION_ALLOW and not baseline_allow and not certified_auto_allow:
            effective_action = ACTION_REVIEW
        else:
            effective_action = proposed_action

    certified_slice_id = None
    if certified_slice:
        certified_slice_id = certified_slice.get("certified_slice_id") or stable_hash(
            json.dumps(certified_slice, ensure_ascii=False, sort_keys=True)
        )

    return {
        "proposed_action": proposed_action,
        "effective_action": effective_action,
        "mode": args.mode,
        "policy_engine_version": POLICY_ENGINE_VERSION,
        "policy_config_hash": policy_config_hash,
        "router_version": ROUTER_VERSION,
        "router_config_hash": router_config_hash,
        "reviewer_model_id": args.reviewer_model_id,
        "reviewer_calibration_id": args.reviewer_calibration_id,
        "certified_slice_id": certified_slice_id,
        "router_ran_for_decision": router_ran_for_decision,
        "router_result_decision": router_result_decision,
        "router_result_diagnostic": router_result_diagnostic,
        "direct_attack_anchor_present": direct_anchor,
        "policy_or_meta_security_mention_present": policy_mention,
        "baseline_allow": baseline_allow,
        "certified_auto_allow": certified_auto_allow,
        "reviewer_potential_allow": reviewer_potential_allow,
        "reviewer_ran": reviewer_ran,
        "reviewer_scored": reviewer_ran,
        "reviewer_used_for_decision": reviewer_used_for_decision,
        "policy_decision_trace": {
            "decision_steps": steps,
            "final_reason": final_reason,
            "blocking_rule": blocking_rule,
            "allowing_rule": allowing_rule,
        },
    }


def reviewer_eligible_for_scoring(
    *,
    text: str,
    row: dict[str, Any],
    stage1_score: float,
    args: argparse.Namespace,
    protected_policy: dict[str, str],
    router_config_hash: str,
    certified_slices: dict[str, Any],
    policy_config_hash: str,
) -> dict[str, Any]:
    if direct_attack_anchor_present(text):
        return {"eligible": False, "reason": "direct_attack_anchor_block", "router_result": None}
    if stage1_score < args.v16_threshold_low:
        return {"eligible": False, "reason": "below_v16_safety_reference", "router_result": None}
    router_result = router_v1(text, row, router_config_hash=router_config_hash)
    protected_action = protected_policy.get(str(router_result.get("semantic_family") or ""))
    if router_result["confidence"] == "low":
        return {"eligible": False, "reason": "low_confidence_router", "router_result": router_result}
    if protected_action in {ACTION_BLOCK, ACTION_REVIEW}:
        return {"eligible": False, "reason": f"protected_family_{protected_action.lower()}", "router_result": router_result}
    if router_result["category"] not in set(parse_csv(args.reviewer_allowlist)):
        return {"eligible": False, "reason": "slice_not_reviewer_allowlisted", "router_result": router_result}
    certified_slice = matching_certified_slice(
        router_result=router_result,
        certified_slices=certified_slices,
        reviewer_model_id=args.reviewer_model_id,
        reviewer_calibration_id=args.reviewer_calibration_id,
        router_config_hash=router_config_hash,
        policy_config_hash=policy_config_hash,
    )
    certified_authority = (certified_slice or {}).get("authority") or {}
    if stage1_score >= args.v16_threshold_protect and not bool(certified_authority.get("ultra_allow")):
        return {"eligible": False, "reason": "v16_protect_threshold", "router_result": router_result}
    return {"eligible": True, "reason": "reviewer_allowlisted_slice", "router_result": router_result}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def validate_runtime(args: argparse.Namespace, certified_slices: dict[str, Any]) -> None:
    if args.mode != MODE_DIAGNOSTIC and args.reviewer_threshold_allow >= args.reviewer_threshold_block:
        raise ValueError("reviewer-threshold-allow must be lower than reviewer-threshold-block outside diagnostic mode")
    if args.mode == MODE_ENFORCED and "oracle" in args.reviewer_model_id.lower():
        raise ValueError("oracle/eval-derived reviewer cannot be used for enforced auto-ALLOW")
    if args.mode == MODE_ENFORCED:
        has_auto_allow = any(bool((row.get("authority") or {}).get("auto_allow")) for row in certified_slices.get("slices", []))
        if has_auto_allow and not args.reviewer_calibration_id:
            raise ValueError("enforced reviewer auto-ALLOW requires reviewer-calibration-id")


def resolve_document_effective_action(
    *,
    args: argparse.Namespace,
    doc: dict[str, Any],
    document_proposed_action: str,
    per_window_policy: list[dict[str, Any]],
) -> str | None:
    if args.mode == MODE_DIAGNOSTIC:
        return ACTION_DIAGNOSTIC_ONLY
    if args.mode == MODE_SHADOW:
        return str(doc.get(args.current_action_column)) if args.current_action_column else None
    if document_proposed_action == ACTION_ALLOW:
        allow_windows = [row for row in per_window_policy if row["proposed_action"] == ACTION_ALLOW]
        every_allow_window_is_baseline_or_certified = all(
            row["baseline_allow"] or row["certified_auto_allow"] for row in allow_windows
        )
        return ACTION_ALLOW if every_allow_window_is_baseline_or_certified else ACTION_REVIEW
    return document_proposed_action


def policy_router_result(policy_row: dict[str, Any], fallback_language: str = "unknown") -> dict[str, Any]:
    result = policy_row.get("router_result_decision") or policy_row.get("router_result_diagnostic")
    if result:
        return result
    if policy_row.get("direct_attack_anchor_present"):
        return {
            "category": "direct_protected_attack_anchor",
            "semantic_family": "direct_model_control_or_exfiltration",
            "language": fallback_language,
            "router_rule_ids": ["direct_attack_anchor_v1"],
        }
    return {
        "category": "unknown",
        "semantic_family": "unknown",
        "language": fallback_language,
        "router_rule_ids": [],
    }


def non_empty_sorted(values: Iterable[Any], *, default: str = "unknown") -> list[str]:
    cleaned = sorted({str(value) for value in values if str(value or "")})
    return cleaned or [default]


def document_router_aggregation(per_window_policy: list[dict[str, Any]], *, fallback_language: str = "unknown") -> dict[str, Any]:
    router_results = [policy_router_result(row, fallback_language=fallback_language) for row in per_window_policy]
    categories = non_empty_sorted(result.get("category") for result in router_results)
    semantic_families = non_empty_sorted(result.get("semantic_family") for result in router_results)
    router_rule_ids = non_empty_sorted(
        rule
        for result in router_results
        for rule in (result.get("router_rule_ids") or [])
    )

    priority = [
        ("block_window", lambda row: row.get("proposed_action") == ACTION_BLOCK),
        ("review_window", lambda row: row.get("proposed_action") == ACTION_REVIEW),
        ("reviewer_potential_allow_window", lambda row: bool(row.get("reviewer_potential_allow"))),
        ("first_diagnostic_category", lambda row: True),
    ]
    primary_reason = "no_router_category"
    primary_result = router_results[0] if router_results else policy_router_result({}, fallback_language=fallback_language)
    for reason, predicate in priority:
        for row in per_window_policy:
            if not predicate(row):
                continue
            result = policy_router_result(row, fallback_language=fallback_language)
            if result.get("category"):
                primary_reason = reason
                primary_result = result
                break
        if primary_reason == reason:
            break

    return {
        "document_router_categories": categories,
        "document_router_semantic_families": semantic_families,
        "document_router_rule_ids": router_rule_ids,
        "document_has_security_compliance_redaction_wrapper": "security_compliance_redaction_wrappers" in categories,
        "document_has_direct_protected_attack_anchor": any(
            row.get("direct_attack_anchor_present") for row in per_window_policy
        )
        or "direct_protected_attack_anchor" in categories,
        "document_primary_category": primary_result.get("category", "unknown"),
        "document_primary_semantic_family": primary_result.get("semantic_family", "unknown"),
        "document_primary_language": primary_result.get("language", fallback_language),
        "document_primary_category_reason": primary_reason,
    }


def action_metrics(results: list[dict[str, Any]], action_key: str) -> dict[str, Any]:
    counts = Counter(str(row.get(action_key)) for row in results)
    total = len(results)
    return {
        "total": total,
        "block_count": counts[ACTION_BLOCK],
        "review_count": counts[ACTION_REVIEW],
        "allow_count": counts[ACTION_ALLOW],
        "null_count": counts["None"],
        "block_rate": counts[ACTION_BLOCK] / total if total else 0.0,
        "review_rate": counts[ACTION_REVIEW] / total if total else 0.0,
        "allow_rate": counts[ACTION_ALLOW] / total if total else 0.0,
    }


def binary_metrics_from_action(results: list[dict[str, Any]], action_key: str, *, review_is_block: bool) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in results:
        gold_attack = label_name(row["document_label"]) == LABEL_ATTACK
        action = row.get(action_key)
        pred_attack = action == ACTION_BLOCK or (review_is_block and action == ACTION_REVIEW)
        if pred_attack and gold_attack:
            tp += 1
        elif pred_attack and not gold_attack:
            fp += 1
        elif not pred_attack and gold_attack:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    benign_total = fp + tn
    return {
        "documents": len(results),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "benign_fp_rate": fp / benign_total if benign_total else 0.0,
    }


def stage1_metrics(results: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    projected = [
        {
            **row,
            "__stage1_positive": float(row.get("stage1_max_score") or 0.0) >= threshold,
        }
        for row in results
    ]
    return threshold_metrics(projected, "__stage1_positive")


def threshold_metrics(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in results:
        gold = label_name(row["document_label"]) == LABEL_ATTACK
        pred = bool(row[key])
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    benign_total = fp + tn
    return {
        "documents": len(results),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "benign_fp_rate": fp / benign_total if benign_total else 0.0,
    }


def grouped_action_metrics(results: list[dict[str, Any]], action_key: str, group_key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[str(row.get(group_key) or "unknown")].append(row)
    return {
        name: {
            "strict_binary": binary_metrics_from_action(rows, action_key, review_is_block=True),
            "relaxed_binary": binary_metrics_from_action(rows, action_key, review_is_block=False),
            "actions": action_metrics(rows, action_key),
        }
        for name, rows in sorted(groups.items())
    }


def attack_allow_accounting(results: list[dict[str, Any]], action_key: str) -> dict[str, Any]:
    total = baseline = policy_added = prevented_direct = prevented_policy = 0
    by_family: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    by_router_rule: Counter[str] = Counter()
    for row in results:
        if label_name(row["document_label"]) != LABEL_ATTACK:
            continue
        action = row.get(action_key)
        stage1_positive = bool(row.get("stage1_positive_at_safety_reference"))
        direct_anchor = bool(row.get("direct_attack_anchor_window_count"))
        if action == ACTION_ALLOW:
            total += 1
            by_family[str(row.get("semantic_family") or "unknown")] += 1
            by_language[str(row.get("language") or "unknown")] += 1
            for rule in row.get("router_rule_ids") or ["unknown"]:
                by_router_rule[str(rule)] += 1
            if stage1_positive:
                policy_added += 1
            else:
                baseline += 1
        elif not stage1_positive and direct_anchor and action == ACTION_BLOCK:
            prevented_direct += 1
        elif not stage1_positive and action == ACTION_BLOCK:
            prevented_policy += 1
    return {
        "attack_allow_total": total,
        "attack_allow_baseline_v16_miss": baseline,
        "attack_allow_policy_added": policy_added,
        "attack_allow_prevented_by_direct_anchor": prevented_direct,
        "attack_allow_prevented_by_policy_router": prevented_policy,
        "attack_allow_by_family": dict(by_family),
        "attack_allow_by_language": dict(by_language),
        "attack_allow_by_router_rule": dict(by_router_rule),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    protected_policy, protected_policy_hash = load_policy_config(args.protected_family_policy_json)
    router_config, router_config_hash = load_router_config(args.slice_router_config_json)
    certified_payload_raw = read_json(args.certified_slices_json)
    certified_slices_hash = payload_hash(certified_payload_raw)
    certified_slices = validate_certified_slices(certified_payload_raw, router_config_hash=router_config_hash)
    validate_runtime(args, certified_slices)
    policy_config_hash = config_hashes(
        protected_policy_hash=protected_policy_hash,
        router_config_hash=router_config_hash,
        certified_slices_hash=certified_slices_hash,
        args=args,
    )

    device = choose_device(args.device)
    docs = iter_corpus(args)
    started = time.time()
    print(
        f"[policy] docs={len(docs):,} device={device} stage1={args.stage1_model_id} reviewer={args.reviewer_model_id}",
        flush=True,
    )

    stage1_tokenizer = AutoTokenizer.from_pretrained(args.stage1_model_id, use_fast=True)
    stage1_model = AutoModelForSequenceClassification.from_pretrained(args.stage1_model_id).to(device)
    reviewer_tokenizer = AutoTokenizer.from_pretrained(args.reviewer_model_id, use_fast=True)
    reviewer_model = AutoModelForSequenceClassification.from_pretrained(args.reviewer_model_id).to(device)

    document_results: list[dict[str, Any]] = []
    window_results: list[dict[str, Any]] = []
    diagnostic_router_all_windows = not args.no_diagnostic_router_all_windows
    processed_windows = 0
    processed_reviewer_windows = 0
    progress_bar = None
    doc_iterable: Iterable[dict[str, Any]] = docs
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(docs, desc="Policy eval documents", unit="doc", dynamic_ncols=True, ascii=True)
        doc_iterable = progress_bar

    for doc_idx, doc in enumerate(doc_iterable, 1):
        windows = build_production_windows(doc["text"], stage1_tokenizer)
        if not windows:
            continue
        window_texts = [window["text"] for window in windows]
        stage1_scores = score_texts(
            stage1_model,
            stage1_tokenizer,
            window_texts,
            device=device,
            batch_size=args.window_batch_size,
        )
        reviewer_eligibility_by_index = {
            idx: reviewer_eligible_for_scoring(
                text=window_texts[idx],
                row=doc,
                stage1_score=stage1_scores[idx],
                args=args,
                protected_policy=protected_policy,
                router_config_hash=router_config_hash,
                certified_slices=certified_slices,
                policy_config_hash=policy_config_hash,
            )
            for idx in range(len(window_texts))
        }
        reviewer_candidate_indices = [
            idx for idx, eligibility in reviewer_eligibility_by_index.items() if eligibility["eligible"]
        ]
        reviewer_scores_by_index: dict[int, float] = {}
        if reviewer_candidate_indices:
            reviewer_texts = [window_texts[idx] for idx in reviewer_candidate_indices]
            reviewer_scores = score_texts(
                reviewer_model,
                reviewer_tokenizer,
                reviewer_texts,
                device=device,
                batch_size=args.window_batch_size,
            )
            reviewer_scores_by_index = dict(zip(reviewer_candidate_indices, reviewer_scores))

        per_window_policy: list[dict[str, Any]] = []
        for idx, window in enumerate(windows):
            policy = policy_decide_window(
                text=window["text"],
                row=doc,
                stage1_score=stage1_scores[idx],
                reviewer_score=reviewer_scores_by_index.get(idx),
                args=args,
                protected_policy=protected_policy,
                router_config_hash=router_config_hash,
                certified_slices=certified_slices,
                policy_config_hash=policy_config_hash,
                diagnostic_router_all_windows=diagnostic_router_all_windows,
            )
            per_window_policy.append(policy)
            if args.window_output_jsonl:
                router_result = policy["router_result_decision"] or policy["router_result_diagnostic"] or {}
                window_results.append(
                    {
                        "document_id": doc["document_id"],
                        "document_label": doc["document_label"],
                        "source_name": doc.get("source_name", "unknown"),
                        "eval_category": doc.get("eval_category", "unknown"),
                        "eval_semantic_family": doc.get("eval_semantic_family", "unknown"),
                        "category": router_result.get("category", "unknown"),
                        "semantic_family": router_result.get("semantic_family", "unknown"),
                        "language": router_result.get("language", doc.get("language", "unknown")),
                        "window_index": window["window_index"],
                        "window_count": len(windows),
                        "window_text": window["text"],
                        "stage1_score": stage1_scores[idx],
                        "stage1_positive_at_safety_reference": stage1_scores[idx] >= args.v16_threshold_low,
                        "reviewer_score": reviewer_scores_by_index.get(idx),
                        "reviewer_eligible_for_scoring": bool(reviewer_eligibility_by_index[idx]["eligible"]),
                        "reviewer_eligibility_reason": reviewer_eligibility_by_index[idx]["reason"],
                        "direct_attack_anchor_window_count": 1 if policy["direct_attack_anchor_present"] else 0,
                        "router_rule_ids": router_result.get("router_rule_ids", []),
                        **policy,
                    }
                )

        proposed_actions = [policy["proposed_action"] for policy in per_window_policy]
        if ACTION_BLOCK in proposed_actions:
            document_proposed_action = ACTION_BLOCK
        elif ACTION_REVIEW in proposed_actions:
            document_proposed_action = ACTION_REVIEW
        else:
            document_proposed_action = ACTION_ALLOW

        document_effective_action = resolve_document_effective_action(
            args=args,
            doc=doc,
            document_proposed_action=document_proposed_action,
            per_window_policy=per_window_policy,
        )

        best_stage1_idx, best_stage1_score = max(enumerate(stage1_scores), key=lambda item: item[1])
        reviewer_scores_present = list(reviewer_scores_by_index.values())
        reviewer_eligible_count = sum(1 for value in reviewer_eligibility_by_index.values() if value["eligible"])
        document_router = document_router_aggregation(
            per_window_policy,
            fallback_language=doc.get("language", "unknown"),
        )
        direct_anchor_count = sum(1 for p in per_window_policy if p["direct_attack_anchor_present"])
        document_results.append(
            {
                "document_id": doc["document_id"],
                "document_label": doc["document_label"],
                "source_name": doc.get("source_name", "unknown"),
                "eval_category": doc.get("eval_category", "unknown"),
                "eval_semantic_family": doc.get("eval_semantic_family", "unknown"),
                "category": document_router["document_primary_category"],
                "semantic_family": document_router["document_primary_semantic_family"],
                "language": document_router["document_primary_language"],
                "router_rule_ids": document_router["document_router_rule_ids"],
                **document_router,
                "stage1_max_score": best_stage1_score,
                "stage1_best_window_index": best_stage1_idx,
                "stage1_positive_at_safety_reference": best_stage1_score >= args.v16_threshold_low,
                "reviewer_max_score_on_stage1_positive": max(reviewer_scores_present)
                if reviewer_scores_present
                else None,
                "reviewer_max_score_on_eligible_windows": max(reviewer_scores_present)
                if reviewer_scores_present
                else None,
                "reviewer_scored_window_count": len(reviewer_scores_by_index),
                "reviewer_eligible_window_count": reviewer_eligible_count,
                "reviewer_used_for_decision_window_count": sum(1 for p in per_window_policy if p["reviewer_used_for_decision"]),
                "window_count": len(windows),
                "window_count_bucket": window_count_bucket(len(windows)),
                "direct_attack_anchor_window_count": direct_anchor_count,
                "policy_security_mention_window_count": sum(
                    1 for p in per_window_policy if p["policy_or_meta_security_mention_present"]
                ),
                "proposed_action": document_proposed_action,
                "effective_action": document_effective_action,
                "mode": args.mode,
                "policy_engine_version": POLICY_ENGINE_VERSION,
                "policy_config_hash": policy_config_hash,
                "router_version": ROUTER_VERSION,
                "router_config_hash": router_config_hash,
                "reviewer_model_id": args.reviewer_model_id,
                "reviewer_calibration_id": args.reviewer_calibration_id,
                "certified_slice_id": next((p["certified_slice_id"] for p in per_window_policy if p["certified_slice_id"]), None),
                "baseline_allow_window_count": sum(1 for p in per_window_policy if p["baseline_allow"]),
                "certified_auto_allow_window_count": sum(1 for p in per_window_policy if p["certified_auto_allow"]),
                "reviewer_potential_allow_window_count": sum(1 for p in per_window_policy if p["reviewer_potential_allow"]),
                "policy_decision_trace": {
                    "decision_steps": sorted(
                        {
                            step
                            for p in per_window_policy
                            for step in p["policy_decision_trace"]["decision_steps"]
                        }
                    ),
                    "final_reason": "document_" + document_proposed_action.lower(),
                    "blocking_rule": next(
                        (p["policy_decision_trace"]["blocking_rule"] for p in per_window_policy if p["policy_decision_trace"]["blocking_rule"]),
                        None,
                    ),
                    "allowing_rule": next(
                        (p["policy_decision_trace"]["allowing_rule"] for p in per_window_policy if p["policy_decision_trace"]["allowing_rule"]),
                        None,
                    ),
                },
            }
        )
        processed_windows += len(windows)
        processed_reviewer_windows += len(reviewer_scores_by_index)

        if progress_bar is not None:
            progress_bar.set_postfix(
                windows=processed_windows,
                reviewer=processed_reviewer_windows,
                refresh=False,
            )
        elif args.progress_every_docs and (doc_idx == 1 or doc_idx % args.progress_every_docs == 0 or doc_idx == len(docs)):
            elapsed = time.time() - started
            print(
                f"[policy] docs={doc_idx:,}/{len(docs):,} windows={processed_windows:,} elapsed={elapsed:.1f}s",
                flush=True,
            )
    if progress_bar is not None:
        progress_bar.close()

    summary = build_summary(
        args=args,
        document_results=document_results,
        window_results=window_results,
        protected_policy_hash=protected_policy_hash,
        router_config_hash=router_config_hash,
        certified_slices_hash=certified_slices_hash,
        policy_config_hash=policy_config_hash,
    )
    write_jsonl(args.output_jsonl, document_results)
    if args.window_output_jsonl:
        write_jsonl(args.window_output_jsonl, window_results)
    write_json(args.summary_json, summary)
    print(json.dumps(summary["proposed_action_metrics"], ensure_ascii=False, indent=2, sort_keys=True))
    return summary


def build_summary(
    *,
    args: argparse.Namespace,
    document_results: list[dict[str, Any]],
    window_results: list[dict[str, Any]],
    protected_policy_hash: str,
    router_config_hash: str,
    certified_slices_hash: str,
    policy_config_hash: str,
) -> dict[str, Any]:
    proposed = action_metrics(document_results, "proposed_action")
    effective = action_metrics(document_results, "effective_action")
    proposed_attack_allow = attack_allow_accounting(document_results, "proposed_action")
    effective_attack_allow = attack_allow_accounting(document_results, "effective_action")
    reviewer_potential_rows = [row for row in document_results if int(row.get("reviewer_potential_allow_window_count") or 0) > 0]
    reviewer_potential_attack = sum(1 for row in reviewer_potential_rows if label_name(row["document_label"]) == LABEL_ATTACK)
    reviewer_potential_benign = sum(1 for row in reviewer_potential_rows if label_name(row["document_label"]) == LABEL_BENIGN)
    return {
        "stage1_model_id": args.stage1_model_id,
        "reviewer_model_id": args.reviewer_model_id,
        "reviewer_calibration_id": args.reviewer_calibration_id,
        "input_jsonl": args.input_jsonl,
        "documents": len(document_results),
        "windows": sum(int(row.get("window_count", 0)) for row in document_results),
        "mode": args.mode,
        "policy_engine_version": POLICY_ENGINE_VERSION,
        "policy_config_hash": policy_config_hash,
        "protected_policy_hash": protected_policy_hash,
        "router_version": ROUTER_VERSION,
        "router_config_hash": router_config_hash,
        "certified_slices_hash": certified_slices_hash,
        "v16_threshold_low": args.v16_threshold_low,
        "v16_threshold_protect": args.v16_threshold_protect,
        "reviewer_threshold_block": args.reviewer_threshold_block,
        "reviewer_threshold_allow": args.reviewer_threshold_allow,
        "pure_v16_safety_reference_metrics": stage1_metrics(document_results, args.v16_threshold_low),
        "proposed_action_metrics": proposed,
        "effective_action_metrics": effective,
        "strict_binary_metrics_review_is_block": binary_metrics_from_action(
            document_results, "proposed_action", review_is_block=True
        ),
        "relaxed_binary_metrics_review_is_allow": binary_metrics_from_action(
            document_results, "proposed_action", review_is_block=False
        ),
        "no_review_production_view": binary_metrics_from_action(
            document_results, "proposed_action", review_is_block=True
        ),
        "human_review_view": proposed,
        "proposed_attack_allow_accounting": proposed_attack_allow,
        "effective_attack_allow_accounting": effective_attack_allow,
        "proposed_attack_allow_total": proposed_attack_allow["attack_allow_total"],
        "proposed_attack_allow_policy_added": proposed_attack_allow["attack_allow_policy_added"],
        "proposed_attack_allow_baseline_v16_miss": proposed_attack_allow["attack_allow_baseline_v16_miss"],
        "effective_attack_allow_total": effective_attack_allow["attack_allow_total"],
        "effective_attack_allow_policy_added": effective_attack_allow["attack_allow_policy_added"],
        "effective_attack_allow_baseline_v16_miss": effective_attack_allow["attack_allow_baseline_v16_miss"],
        "legacy_attack_allow_alias_source": "effective_action",
        "effective_attack_allow_prevented_by_direct_anchor": effective_attack_allow["attack_allow_prevented_by_direct_anchor"],
        "effective_attack_allow_prevented_by_policy_router": effective_attack_allow["attack_allow_prevented_by_policy_router"],
        "baseline_allow_count": sum(1 for row in document_results if int(row.get("baseline_allow_window_count") or 0) > 0),
        "certified_auto_allow_count": sum(
            1 for row in document_results if int(row.get("certified_auto_allow_window_count") or 0) > 0
        ),
        "reviewer_auto_allow_count": sum(
            1 for row in document_results if int(row.get("certified_auto_allow_window_count") or 0) > 0
        ),
        "reviewer_effective_auto_allow_count": sum(
            1
            for row in document_results
            if row.get("effective_action") == ACTION_ALLOW
            and int(row.get("certified_auto_allow_window_count") or 0) > 0
        ),
        "reviewer_potential_allow_count": len(reviewer_potential_rows),
        "reviewer_potential_allow_attack_count": reviewer_potential_attack,
        "reviewer_potential_allow_benign_count": reviewer_potential_benign,
        "reviewer_potential_allow_by_slice": dict(
            Counter(str(row.get("category") or "unknown") for row in reviewer_potential_rows)
        ),
        "direct_attack_anchor_block_count": sum(
            1 for row in document_results if int(row.get("direct_attack_anchor_window_count") or 0) > 0
        ),
        "policy_security_mention_count": sum(
            1 for row in document_results if int(row.get("policy_security_mention_window_count") or 0) > 0
        ),
        "reviewer_considered_count": sum(
            1 for row in document_results if int(row.get("reviewer_scored_window_count") or 0) > 0
        ),
        "reviewer_scored_window_count": sum(
            int(row.get("reviewer_scored_window_count") or 0) for row in document_results
        ),
        "reviewer_eligible_window_count": sum(
            int(row.get("reviewer_eligible_window_count") or 0) for row in document_results
        ),
        "reviewer_used_for_decision_window_count": sum(
            int(row.get("reviewer_used_for_decision_window_count") or 0) for row in document_results
        ),
        "reviewer_review_count": sum(
            1 for row in document_results if row.get("proposed_action") == ACTION_REVIEW
        ),
        "by_category": grouped_action_metrics(document_results, "proposed_action", "category"),
        "by_document_primary_category": grouped_action_metrics(
            document_results, "proposed_action", "document_primary_category"
        ),
        "by_eval_category": grouped_action_metrics(document_results, "proposed_action", "eval_category"),
        "by_language": grouped_action_metrics(document_results, "proposed_action", "language"),
        "by_semantic_family": grouped_action_metrics(document_results, "proposed_action", "semantic_family"),
        "by_document_primary_semantic_family": grouped_action_metrics(
            document_results, "proposed_action", "document_primary_semantic_family"
        ),
        "by_window_count_bucket": grouped_action_metrics(document_results, "proposed_action", "window_count_bucket"),
        "by_decision_reason": grouped_action_metrics(
            [
                {**row, "decision_reason": (row.get("policy_decision_trace") or {}).get("final_reason")}
                for row in document_results
            ],
            "proposed_action",
            "decision_reason",
        ),
        "window_level": {
            "rows": len(window_results),
            "proposed_attack_allow_accounting": attack_allow_accounting(window_results, "proposed_action")
            if window_results
            else {},
            "effective_attack_allow_accounting": attack_allow_accounting(window_results, "effective_action")
            if window_results
            else {},
        },
    }


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
