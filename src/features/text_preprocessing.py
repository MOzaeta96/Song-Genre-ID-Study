import re


def clean_lyrics(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"[^a-záéíóúñü'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
