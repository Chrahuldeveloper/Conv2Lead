from config import GEMINI_CLIENT, GEMINI_MODEL


def generate_answer(question, context):
    prompt = f"""
    You are a helpful assistant.
    Answer ONLY using the context below.
    If the answer is not present, say "I don't know".

    Context:
    {context}

    Question:
    {question}
    """

    response = GEMINI_CLIENT.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text
