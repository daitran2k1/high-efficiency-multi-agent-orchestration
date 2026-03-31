def extract_content(response) -> str:
    """
    Robustly extracts text content from provider responses.
    Handles plain strings and list-based content blocks.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            else:
                text_parts.append(str(part))
        return " ".join(text_parts)
    return str(content)
