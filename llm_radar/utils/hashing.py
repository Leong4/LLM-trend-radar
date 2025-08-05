import hashlib
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()