"""Built-in demo scenarios. These are inputs only, not expected answers."""

from __future__ import annotations

from api_client import Question

SCENARIOS: dict[str, dict] = {
    "Customer support triage": {
        "state": (
            "My business has not received payouts for three days. "
            "I have 25 employees waiting for payroll."
        ),
        "questions": [
            Question("is_urgent", "noul", "Is this urgent?"),
            Question(
                "department",
                "choice",
                "Which department should handle this?",
                {
                    "billing": "Payment and money issues",
                    "technical": "Software and infrastructure problems",
                    "sales": "Pricing and new customer enquiries",
                },
            ),
            Question(
                "severity", "score", "How severe is the situation?",
                ["low", "medium", "high", "critical"],
            ),
        ],
    },
    "Coding agent": {
        "state": "The user asks to modify a Python file and execute its unit tests.",
        "questions": [
            Question("is_coding_task", "noul", "Is this a coding task?"),
            Question(
                "tool",
                "choice",
                "Which tool should be selected?",
                {"terminal": None, "browser": None, "editor": None},
            ),
            Question(
                "execution_risk", "score", "How risky is automatic execution?",
                ["low", "medium", "high"],
            ),
        ],
    },
    "Security triage": {
        "state": (
            "An employee receives an email asking them to urgently reset their password "
            "using an unfamiliar external link."
        ),
        "questions": [
            Question("is_suspicious", "noul", "Is this potentially suspicious?"),
            Question(
                "action",
                "choice",
                "What should happen?",
                {"allow": None, "quarantine": None, "investigate": None},
            ),
            Question(
                "risk", "score", "How severe is the risk?",
                ["low", "medium", "high", "critical"],
            ),
        ],
    },
    "Smart home": {
        "state": (
            "The house is empty. The thermostat is set to 72 degrees and the outside "
            "temperature is 95 degrees."
        ),
        "questions": [
            Question("energy_saving", "noul", "Should energy-saving mode activate?"),
            Question(
                "action",
                "choice",
                "Which action is preferable?",
                {
                    "maintain temperature": None,
                    "reduce cooling": None,
                    "activate away mode": None,
                },
            ),
            Question(
                "conservation_importance", "score", "How important is energy conservation?",
                ["low", "medium", "high"],
            ),
        ],
    },
}
