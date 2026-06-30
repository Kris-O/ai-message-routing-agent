import re

# Deterministic prompt-injection detector. Conservative patterns (PL + EN) — must NOT fire on ordinary
# employee messages (false positives would wrongly quarantine real requests). Tighten over broaden.
_INJECTION_PATTERNS = [
    r"\bzignoruj\b.{0,40}\b(instrukcj|polece|powy[żz]sz|wcze[śs]niejsz)",
    r"\bignore\b.{0,40}\b(instruction|previous|above|prior|all)\b",
    r"\bdisregard\b.{0,40}\b(instruction|previous|above|prompt)\b",
    r"\bforget\b.{0,40}\b(instruction|rule|previous|above|all|everything|prompt)\b",
    r"\bzapomnij\b.{0,40}\b(instrukcj|polece|regu[łl]|wszystk|powy[żz]sz|wcze[śs]niejsz)",
    r"\bodpowiedz\b.{0,20}\b(wy[łl][ąa]cznie|tylko|jedynie)\b",
    r"\b(answer|reply|respond)\b.{0,20}\b(only|with)\b.{0,20}\b(KADRY|HR|HELPDESK|IT|INNE)\b",
    r"\bmasz\b.{0,20}\b(napisa[ćc]|odpowiedzie[ćc]|wypisa[ćc]|zwr[óo]ci[ćc])\b",
    r"(^|\n)\s*(system|assistant)\s*[:>]",
    r"</?\s*(system|instrukcj\w*)\s*>",
    # framing the message AS an instruction to the classifier/model (legit mail never says this)
    r"\binstrukcj\w*\b.{0,30}\b(klasyfikator\w*|model\w*|asystent\w*|system\w*)",
    # imperative "pick/assign <LABEL>" — a directive to choose a specific department
    r"\b(wybierz|wska[żz]|zaklasyfikuj|sklasyfikuj)\b.{0,30}\b(KADRY|HR|HELPDESK|IT|INNE)\b",
]
_RX = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]


def looks_like_injection(text: str) -> bool:
    """True if the message appears to try to manipulate the classifier (deterministic, no LLM)."""
    return any(rx.search(text or "") for rx in _RX)
