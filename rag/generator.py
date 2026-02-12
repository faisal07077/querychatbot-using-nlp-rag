import re


class RAGGenerator:
    """
    Extractive response generator using retrieved official content.
    """

    def generate(self, retrieved_chunks, question: str) -> str:
        if not retrieved_chunks:
            return "Information not available in official records."

        answers = []

        for chunk in retrieved_chunks:
            text = chunk.get("text", "")
            match = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
            if match:
                answers.append(match.group(1).strip())

        if not answers:
            return "Information not available in official records."

        unique_answers = list(dict.fromkeys(answers))

        if len(unique_answers) == 1:
            return unique_answers[0]

        response = "Here is the relevant information:\n\n"
        for i, ans in enumerate(unique_answers, 1):
            response += f"{i}. {ans}\n\n"

        return response.strip()
