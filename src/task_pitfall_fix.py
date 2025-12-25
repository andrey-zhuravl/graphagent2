from dataclasses import dataclass


@dataclass
class PitfallFix:
    symptom: str = ""  # exact error / log pattern if possible
    cause: str = ""
    fix: str = ""
    verify: str = ""
