import os
import sys
import argparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}

SYSTEM_PROMPT = (
    "You are an assistant that analyzes the contents of a website and provides "
    "a short summary, ignoring text that might be navigation related. "
    "Respond in markdown."
)

MAX_TEXT_CHARS = 20_000  # ~5k tokens — keeps costs down and avoids token limits

OLLAMA_APPROACHES = ["local-call", "python-package", "openai-compatible"]


# ---------------------------------------------------------------------------
# Website scraper
# ---------------------------------------------------------------------------

class Website:
    def __init__(self, url: str):
        self.url = url
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        self.title = soup.title.string if soup.title else "No title found"
        for tag in soup.body(["script", "style", "img", "input"]):
            tag.decompose()
        raw_text = soup.body.get_text(separator="\n", strip=True)
        self.text = raw_text[:MAX_TEXT_CHARS]
        self.truncated = len(raw_text) > MAX_TEXT_CHARS


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def build_messages(website: Website) -> list[dict]:
    user_prompt = (
        f"You are looking at a website titled {website.title}\n"
        "The contents of this website is as follows; please provide a short "
        "summary in markdown. If it includes news or announcements, summarize "
        "these too.\n\n"
        + website.text
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
# Backend callers
# ---------------------------------------------------------------------------

def summarize_openai(website: Website, model: str = "gpt-4o-mini") -> str:
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY not found. Add it to your .env file.")
    if not api_key.startswith("sk-proj-"):
        print("Warning: API key doesn't start with 'sk-proj-'; double-check your key.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(model=model, messages=build_messages(website))
    return response.choices[0].message.content


def summarize_ollama(website: Website, model: str = "llama3.2", approach: str = "openai-compatible") -> str:
    messages = build_messages(website)

    if approach == "local-call":
        response = requests.post(
            url="http://localhost:11434/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    elif approach == "python-package":
        import ollama
        response = ollama.chat(model=model, messages=messages, stream=False)
        return response["message"]["content"]

    elif approach == "openai-compatible":
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown Ollama approach '{approach}'. Choose from: {OLLAMA_APPROACHES}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a website using OpenAI or a local Ollama model."
    )
    parser.add_argument("url", help="URL of the website to summarize")
    parser.add_argument(
        "--backend", choices=["openai", "ollama"], default="openai",
        help="LLM backend to use (default: openai)"
    )
    parser.add_argument(
        "--model",
        help="Model name (default: gpt-4o-mini for OpenAI, llama3.2 for Ollama)"
    )
    parser.add_argument(
        "--ollama-approach", choices=OLLAMA_APPROACHES, default="openai-compatible",
        dest="ollama_approach",
        help="How to call Ollama (default: openai-compatible)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Fetching: {args.url}")
    website = Website(args.url)
    print(f"Title: {website.title}")
    if website.truncated:
        print(f"Note: content truncated to {MAX_TEXT_CHARS:,} characters")
    print()

    if args.backend == "openai":
        model = args.model or "gpt-4o-mini"
        print(f"Backend: OpenAI ({model})\n{'─' * 60}")
        summary = summarize_openai(website, model=model)
    else:
        model = args.model or "llama3.2"
        print(f"Backend: Ollama ({model}, approach={args.ollama_approach})\n{'─' * 60}")
        summary = summarize_ollama(website, model=model, approach=args.ollama_approach)

    print(summary)


if __name__ == "__main__":
    main()
