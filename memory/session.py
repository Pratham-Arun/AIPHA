"""
Session Module
──────────────
Represents a single conversation session.
Stores user profile information gathered during the conversation.
"""


class Session:
    """
    Represents one conversation session.

    Attributes:
        session_id  – unique identifier for the session
        user_info   – dictionary storing user profile data
                      (name, age, weight, height, conditions, allergies, diet)
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.user_info: dict = {
            "name": None,
            "age": None,
            "weight": None,
            "height": None,
            "medical_conditions": [],
            "allergies": [],
            "dietary_preference": None,
        }

    def update_info(self, key: str, value) -> None:
        """Update a specific user info field."""
        if key in self.user_info:
            self.user_info[key] = value

    def get_info(self) -> dict:
        """Return a copy of the user info dictionary."""
        return self.user_info.copy()

    def get_summary(self) -> str:
        """
        Return a readable summary of known user info.
        Only includes fields that have been set.
        """
        lines = []
        for key, value in self.user_info.items():
            if value and value != []:
                label = key.replace("_", " ").title()
                if isinstance(value, list):
                    lines.append(f"- {label}: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"- {label}: {value}")
        return "\n".join(lines) if lines else "No user information collected yet."

    def clear(self) -> None:
        """Reset all user info to defaults."""
        for key in self.user_info:
            if isinstance(self.user_info[key], list):
                self.user_info[key] = []
            else:
                self.user_info[key] = None
