# state/session.py — Phase 4: slot-filling + validation + retry logic
# Phase 3 flow unchanged. Adds validators, retry counters, and failure handling.

import re

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

# Ordered list of fields to collect
SLOTS = ["name", "phone"]

# Max attempts per field before the session is abandoned
MAX_RETRIES = 3

# FIX 3 — Keep all LLM responses under ~10 words to minimize TTS duration.
# The instruction suffix "Reply in 10 words or fewer." is appended to every prompt.

_SHORT = " Reply in 10 words or fewer."

# LLM instructions to ask for each field
SLOT_PROMPTS = {
    "name":  "Ask the user for their full name." + _SHORT,
    "phone": "Ask the user for their 10-digit phone number, digits one by one." + _SHORT,
}

# LLM instructions to acknowledge a successfully filled field
SLOT_ACK = {
    "name":  "Briefly confirm you got the name '{value}'." + _SHORT,
    "phone": "Briefly confirm you got the number '{value}'. Say goodbye." + _SHORT,
}

# LLM instructions to re-ask after a validation failure
SLOT_RETRY_PROMPTS = {
    "name":  "Ask for their name only — no extra words." + _SHORT,
    "phone": "Say the number was invalid. Ask for 10 digits again." + _SHORT,
}

# LLM instruction when a user exhausts all retries for a field
SLOT_EXCEEDED_PROMPT = (
    "Apologise briefly. Say you could not collect the number. End the call." + _SHORT
)


# ---------------------------------------------------------------------------
# Validators — return True if the value is acceptable
# ---------------------------------------------------------------------------

def _validate_phone(value: str) -> bool:
    """10 digits, numeric only (spaces/dashes stripped before check)."""
    digits = _normalize_phone(value)
    return digits.isdigit() and len(digits) == 10


def _normalize_name(value: str) -> str:
    """Normalize name text by removing conversational prefixes and punctuation noise."""
    text = value.strip()
    lower = text.lower()

    prefixes = [
        "my name is ",
        "this is ",
        "i am ",
        "i'm ",
        "call me ",
    ]
    for prefix in prefixes:
        if lower.startswith(prefix):
            text = text[len(prefix):]
            break

    text = re.sub(r"[^a-zA-Z'\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,-")

    parts = [p for p in text.split(" ") if p]
    return " ".join(p.capitalize() for p in parts)


def _validate_name(value: str) -> bool:
    """Accept 1-4 realistic name tokens; reject conversational phrases."""
    if not value:
        return False
    if any(ch.isdigit() for ch in value):
        return False

    parts = value.lower().split()
    if len(parts) < 1 or len(parts) > 4:
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
    """Normalize spoken or formatted phone input to a digit-only string."""
    raw_digits = re.sub(r"\D", "", value)
    if len(raw_digits) == 10:
        return raw_digits

    if len(raw_digits) == 11 and raw_digits.startswith("0"):
        return raw_digits[1:]

    if len(raw_digits) == 12 and raw_digits.startswith("91"):
        return raw_digits[-10:]

    tokens = re.findall(r"[a-zA-Z0-9]+", value.lower())
    converted: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in {"double", "triple"} and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            next_digit = NUMBER_WORDS.get(next_token)
            if next_digit is not None:
                count = 2 if token == "double" else 3
                converted.extend([next_digit] * count)
                i += 2
                continue

        if token.isdigit():
            converted.append(token)
        else:
            mapped = NUMBER_WORDS.get(token)
            if mapped is not None:
                converted.append(mapped)

        i += 1

    normalized = "".join(converted)
    if len(normalized) == 11 and normalized.startswith("0"):
        return normalized[1:]
    if len(normalized) == 12 and normalized.startswith("91"):
        return normalized[-10:]
    return normalized


VALIDATORS = {
    "name": _validate_name,
    "phone": _validate_phone,
}


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Manages slot-filling state with per-field validation and retry limits.

    Public API (unchanged from Phase 3):
        update(user_text)       — try to store text into the current slot
        get_next_prompt() -> str — LLM instruction for the next turn
        is_complete()     -> bool
        is_failed()       -> bool — True if retries were exhausted
        summary()         -> str
    """

    def __init__(self):
        self.state: dict[str, str | None] = {slot: None for slot in SLOTS}
        self._retries: dict[str, int]     = {slot: 0 for slot in SLOTS}
        self._just_filled: str | None     = None   # slot acked on next prompt
        self._failed: bool                = False  # True if MAX_RETRIES exceeded
        self._pending_instruction: str | None = None  # override next prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, user_text: str) -> None:
        """
        Validate and store user_text into the current empty slot.
        If validation fails, increments retry counter.
        If retries are exhausted, marks session as failed.
        """
        slot = self._current_slot()
        if not slot:
            return

        value = user_text.strip()
        if slot == "name":
            value = _normalize_name(value)
        if slot == "phone":
            value = _normalize_phone(value)

        # Run validator if one exists for this slot
        validator = VALIDATORS.get(slot)
        if validator and not validator(value):
            self._retries[slot] += 1

            if self._retries[slot] >= MAX_RETRIES:
                # Retries exhausted — mark session failed
                self._failed = True
                self._pending_instruction = SLOT_EXCEEDED_PROMPT
            else:
                # Ask again with a validation-specific retry prompt
                retry_prompt = SLOT_RETRY_PROMPTS.get(slot)
                if retry_prompt:
                    self._pending_instruction = retry_prompt
            print(f"[state] Invalid {slot} input: {user_text}")
            return  # do NOT fill the slot

        # Valid (or no validator) — store and queue acknowledgement
        self.state[slot] = value
        print(f"[state] Captured {slot}: {value}")
        self._just_filled = slot
        self._pending_instruction = None   # clear any pending retry message

    def get_next_prompt(self) -> str:
        """
        Return an LLM instruction string for the current turn.
        Priority: pending override -> ack (+ chained next-slot) -> next slot prompt.

        FIX: When acknowledging a filled slot, we immediately chain the next-slot
        question into the same instruction. Without this, the recorder starts
        listening *before* the user ever hears what they are supposed to say next,
        causing a silent gap and confused captures.
        """
        # Pending override (retry message or failure message)
        if self._pending_instruction:
            instruction = self._pending_instruction
            self._pending_instruction = None
            return instruction

        # Acknowledge the last successfully filled slot, then immediately ask next
        if self._just_filled:
            ack_template = SLOT_ACK[self._just_filled]
            ack = ack_template.format(value=self.state[self._just_filled])
            self._just_filled = None

            # Chain the next-slot question so the user hears it before recording starts
            next_slot = self._current_slot()
            if next_slot:
                return (
                    f"{ack} "
                    f"Then immediately ask: {SLOT_PROMPTS[next_slot]}"
                )
            return ack  # last slot filled — ack only (session will complete)

        # Ask for the next empty slot
        slot = self._current_slot()
        if slot:
            return SLOT_PROMPTS[slot]

        return "Thank the user and let them know the conversation is complete."

    def current_slot(self) -> str | None:
        """Expose the active slot so upstream components can adapt behavior."""
        return self._current_slot()

    def is_complete(self) -> bool:
        """True when all slots are filled successfully."""
        return all(self.state[slot] is not None for slot in SLOTS)

    def is_failed(self) -> bool:
        """True if a field exhausted its retry limit."""
        return self._failed

    def summary(self) -> str:
        lines = [f"  {k}: {v}" for k, v in self.state.items()]
        return "Collected details:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_slot(self) -> str | None:
        for slot in SLOTS:
            if self.state[slot] is None:
                return slot
        return None