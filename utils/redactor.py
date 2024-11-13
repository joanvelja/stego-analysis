import regex as re


class Redactor:
    """Supposed to be used as a .apply() function on Datasets/Pandas dfs"""

    def __init__(self) -> None:
        # Init university pattern (easy)
        self.university_pattern = r"\bThe (Good|Bad) University\b"

        # Replacement rules for various categories
        self.gender_replacements = {
            r"\bhe\b|\bhim\b|\bhis\b": "they",
            r"\bshe\b|\bher\b|\bhers\b": "they",
            r"\b(male|female|man|woman|boy|girl|gentleman|lady|mrs\.|ms\.|mr\.|sir|ma’am|madam|)\b": "person",
        }

    def clean_university(self, input: str) -> str:
        """
        Example: Studied Genetics at The Good University to enhance skills.
        Expected output: Studied Genetics at university to enhance skills.
        """

        # Apply replacement pattern
        text = re.sub(self.university_pattern, "university", input, flags=re.IGNORECASE)
        return text

    def clean_gender(self, input: str) -> str:
        """
        Example: This male has studied Astronomy and is a good communicator.
        Expected output: This person has studied Astronomy and is a good communicator.

        2-fold pattern:
        1) Removes male/female/boy/girl/... with 'person'
        2) Removes pronouns he/she with 'them'
        """

        # Apply replacement patterns
        for pattern, replacement in self.gender_replacements.items():
            text = re.sub(pattern, replacement, input, flags=re.IGNORECASE)

        return text
