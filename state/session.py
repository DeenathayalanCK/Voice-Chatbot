# state/session.py — slot-filling, validation, retry logic
#
# RESPONSE_MODE awareness
# ───────────────────────
# When RESPONSE_MODE=deterministic, get_next_prompt() returns a short *key*
# (e.g. "ask.name", "ack.name|Ravi Kumar") that DeterministicClient resolves
# to a scripted string — no LLM call, zero latency.
#
# When RESPONSE_MODE=llm, get_next_prompt() returns a natural-language
# instruction that the configured LLM paraphrases into a spoken response
# (original behaviour — fully backward compatible).
#
# Adding a new slot (both modes)
# ──────────────────────────────
# 1. Append the slot name to SLOTS below.
# 2. Add a validator to VALIDATORS (or set None for free-form).
# 3. For LLM mode: add entries to SLOT_PROMPTS, SLOT_ACK, SLOT_RETRY_PROMPTS.
# 4. For deterministic mode: add matching keys to responses/responses.yaml.
#    Required keys:  ask.<slot>  ack.<slot>  retry.<slot>  exceeded.<slot>

from __future__ import annotations

import os
import re

# ── Ordered slot definitions ─────────────────────────────────────────────────
# Add new slots here. Order determines collection sequence.
SLOTS = ["name", "phone"]

MAX_RETRIES = 3   # attempts per field before session is abandoned

# ── Response-mode detection ──────────────────────────────────────────────────
def _is_deterministic() -> bool:
    return os.getenv("RESPONSE_MODE", "llm").strip().lower() == "deterministic"

# ── LLM-mode prompt strings ──────────────────────────────────────────────────
# Used only when RESPONSE_MODE=llm.
_SHORT = " Reply in 10 words or fewer."

SLOT_PROMPTS_LLM = {
    "name":  "Ask the user for their full name." + _SHORT,
    "phone": "Ask the user for their 10-digit phone number, digits one by one." + _SHORT,
}
SLOT_ACK_LLM = {
    "name":  "Briefly confirm you got the name '{value}'." + _SHORT,
    "phone": "Briefly confirm you got the number '{value}'. Say goodbye." + _SHORT,
}
SLOT_RETRY_LLM = {
    "name":  "Ask for their name only — no extra words." + _SHORT,
    "phone": "Say the number was invalid. Ask for 10 digits again." + _SHORT,
}
SLOT_EXCEEDED_LLM = (
    "Apologise briefly. Say you could not collect the number. End the call." + _SHORT
)

# ── Deterministic-mode prompt keys ───────────────────────────────────────────
# Keys must exist in responses/responses.yaml.
# Value substitution: "ack.<slot>|<value>" → script["ack.<slot>"].format(value=...)
SLOT_PROMPTS_DET = {
    "name":  "ask.name",
    "phone": "ask.phone",
}
SLOT_ACK_DET = {
    "name":  "ack.name",   # pipe-joined with value at runtime: "ack.name|Ravi"
    "phone": "ack.phone",
}
SLOT_RETRY_DET = {
    "name":  "retry.name",
    "phone": "retry.phone",
}
SLOT_EXCEEDED_DET = "exceeded.phone"   # generic exceeded key


# ── Shared normalization / validation ─────────────────────────────────────────

NUMBER_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "won": "1",
    "two": "2", "to": "2", "too": "2",
    "three": "3",
    "four": "4", "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8", "ate": "8",
    "nine": "9",
}

NAME_BANNED_TOKENS = {
    "do", "you", "want", "no", "yes", "phone", "number", "hello",
    "maybe", "or", "what", "whats", "what's", "please", "provide",
}


def _normalize_name(value: str) -> str:
    text = value.strip()
    lower = text.lower()
    for prefix in ("my name is ", "this is ", "i am ", "i'm ", "call me "):
        if lower.startswith(prefix):
            text = text[len(prefix):]
            break
    text = re.sub(r"[^a-zA-Z'\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,-")
    parts = [p for p in text.split() if p]
    return " ".join(p.capitalize() for p in parts)


def _validate_name(value: str) -> bool:
    if not value:
        return False
    if any(ch.isdigit() for ch in value):
        return False
    parts = value.lower().split()
    if not (1 <= len(parts) <= 4):
        return False
    for token in parts:
        if token in NAME_BANNED_TOKENS:
            return False
        if not re.fullmatch(r"[a-z][a-z'\-]*", token):
            return False
        if len(token.strip("'-")) < 2:
            return False
    return True


def _normalize_phone(value: str) -> str:
    """
    Normalize spoken or formatted phone input to a digit-only string.
    Caps result at 10 digits — ignores trailing hallucinated content.
    """
    # Fast path: pure digit string (or digit+punctuation like dashes)
    raw_digits = re.sub(r"\D", "", value)
    if raw_digits:
        # Strip known country-code prefixes first
        if len(raw_digits) >= 11 and raw_digits.startswith("0"):
            raw_digits = raw_digits[1:]
        if len(raw_digits) >= 12 and raw_digits.startswith("91"):
            raw_digits = raw_digits[2:]
        # Cap at 10 — any extra digits are hallucinated prose (e.g. "10 digit")
        return raw_digits[:10]

    # Slow path: parse number words
    tokens = re.findall(r"[a-zA-Z0-9]+", value.lower())
    converted: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {"double", "triple"} and i + 1 < len(tokens):
            next_digit = NUMBER_WORDS.get(tokens[i + 1])
            if next_digit is not None:
                count = 2 if token == "double" else 3
                converted.extend([next_digit] * count)
                i += 2
                continue
        if token.isdigit():
            converted.extend(list(token))
        else:
            mapped = NUMBER_WORDS.get(token)
            if mapped is not None:
                converted.append(mapped)
        i += 1

    normalized = "".join(converted)
    if len(normalized) >= 11 and normalized.startswith("0"):
        normalized = normalized[1:]
    if len(normalized) >= 12 and normalized.startswith("91"):
        normalized = normalized[2:]
    return normalized[:10]


def _validate_phone(value: str) -> bool:
    digits = _normalize_phone(value)
    return digits.isdigit() and len(digits) == 10


VALIDATORS = {
    "name":  _validate_name,
    "phone": _validate_phone,
}

NORMALIZERS = {
    "name":  _normalize_name,
    "phone": _normalize_phone,
}


# ── SessionManager ────────────────────────────────────────────────────────────

class SessionManager:
    """
    Manages slot-filling state: collection order, validation, retries.

    get_next_prompt() returns:
      • a YAML key string  when RESPONSE_MODE=deterministic
      • an LLM instruction when RESPONSE_MODE=llm
    Both forms are passed directly into the client's generate_response().
    """

    def __init__(self) -> None:
        self.state:    dict[str, str | None] = {s: None for s in SLOTS}
        self._retries: dict[str, int]        = {s: 0    for s in SLOTS}
        self._just_filled:         str | None = None
        self._failed:              bool       = False
        self._pending_instruction: str | None = None
        self._det = _is_deterministic()

    # ── Public API ──────────────────────────────────────────────────────────

    def update(self, user_text: str) -> None:
        slot = self._current_slot()
        if not slot:
            return

        normalizer = NORMALIZERS.get(slot)
        value = normalizer(user_text) if normalizer else user_text.strip()

        validator = VALIDATORS.get(slot)
        if validator and not validator(value):
            self._retries[slot] += 1
            if self._retries[slot] >= MAX_RETRIES:
                self._failed = True
                self._pending_instruction = (
                    f"exceeded.{slot}" if self._det else SLOT_EXCEEDED_LLM
                )
            else:
                self._pending_instruction = (
                    SLOT_RETRY_DET.get(slot, "retry.fallback")
                    if self._det
                    else SLOT_RETRY_LLM.get(slot, "")
                )
            print(f"[state] Invalid {slot}: '{user_text}' → '{value}'")
            return

        self.state[slot] = value
        print(f"[state] Captured {slot}: {value}")
        self._just_filled = slot
        self._pending_instruction = None

    def get_next_prompt(self) -> str:
        """
        Returns a prompt key (deterministic) or LLM instruction (llm mode).
        Priority: pending override → ack+next → ask next slot → complete.
        """
        # 1. Pending override (retry / exceeded)
        if self._pending_instruction:
            out = self._pending_instruction
            self._pending_instruction = None
            return out

        # 2. Acknowledge just-filled slot, then chain next question
        if self._just_filled:
            filled_slot  = self._just_filled
            filled_value = self.state[filled_slot]
            self._just_filled = None
            next_slot = self._current_slot()

            if self._det:
                # Return "ack.<slot>|<value>" — DeterministicClient splits on "|"
                ack_key = SLOT_ACK_DET.get(filled_slot, f"ack.{filled_slot}")
                ack = f"{ack_key}|{filled_value}"
                # If there's a next slot, chain it in a second call naturally —
                # deterministic mode returns one crisp sentence so no chaining needed.
                # The next loop iteration will call get_next_prompt() → ask.next_slot.
                return ack
            else:
                # LLM mode: chain ack + next question into one instruction
                ack_tmpl = SLOT_ACK_LLM.get(filled_slot, "")
                ack = ack_tmpl.format(value=filled_value)
                if next_slot:
                    next_q = SLOT_PROMPTS_LLM.get(next_slot, "")
                    return f"{ack} Then immediately ask: {next_q}"
                return ack

        # 3. Ask for the next unfilled slot
        slot = self._current_slot()
        if slot:
            return (
                SLOT_PROMPTS_DET.get(slot, f"ask.{slot}")
                if self._det
                else SLOT_PROMPTS_LLM.get(slot, "")
            )

        # 4. All done
        return "complete" if self._det else "Thank the user — conversation is complete."

    def current_slot(self) -> str | None:
        return self._current_slot()

    def is_complete(self) -> bool:
        return all(self.state[s] is not None for s in SLOTS)

    def is_failed(self) -> bool:
        return self._failed

    def summary(self) -> str:
        lines = [f"  {k}: {v}" for k, v in self.state.items()]
        return "Collected details:\n" + "\n".join(lines)

    # ── Internal ────────────────────────────────────────────────────────────

    def _current_slot(self) -> str | None:
        for slot in SLOTS:
            if self.state[slot] is None:
                return slot
        return None