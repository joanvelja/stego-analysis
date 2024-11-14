import regex as re


class Redactor:
    """Supposed to be used as a .apply() function on Datasets/Pandas dfs"""

    def __init__(self) -> None:
        # Init university pattern (easy)
        self.university_pattern = r"\bThe (Good|Bad) University\b"

        # Replacement rules for various categories
        self.gender_replacements = {
            r"\b(he|him|his|He|Him)\b": "they",
            r"\b(she|her|hers|She|Her)\b": "they",
            r"\b(his|hers|His|Hers)\b": "their",
            r" \b(male|female|man|woman|boy|girl|gentleman|lady|mrs\.|ms\.|mr\.|sir|ma'am|madam)\b": "",
        }

    def clean_university(self, input: str) -> str:
        """
        Example: Studied Genetics at The Good University to enhance skills.
        Expected output: Studied Genetics at university to enhance skills.
        """

        # Apply replacement pattern
        text = re.sub(self.university_pattern, "university", input, flags=re.IGNORECASE)
        return text

    def gender_neutralize(self, text):
        # Dictionary for replacements
        replacements = {
            r"\b(male|female)\b": "person",
            r"\b(he|she)\b": "they",
            r"\b(his|her)\b": "their",
            r"\b(him)\b": "them",
        }

        # Apply replacements
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Handle specific cases like "A male/female applicant" or "A Female applicant"
        text = re.sub(
            r"\bA\s+(?:male|female)\s+applicant",
            "An applicant",
            text,
            flags=re.IGNORECASE,
        )

        # Handle cases where gender is followed by a comma
        text = re.sub(r"\b(male|female),\s*", "person, ", text, flags=re.IGNORECASE)

        # Handle cases where gender is at the beginning of the sentence
        text = re.sub(r"^(?:male|female)\s*", "person ", text, flags=re.IGNORECASE)

        return text
