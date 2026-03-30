# Website Summarizer

A command-line tool that scrapes any website and generates a concise markdown summary using an LLM. Supports both **OpenAI** (cloud) and **Ollama** (local) backends.

## Features

- Scrapes and cleans website content using BeautifulSoup (strips nav, footer, header, aside, and form elements for cleaner output)
- Summarizes content via OpenAI (`gpt-4o-mini` by default) or a local Ollama model
- Three Ollama calling approaches: direct HTTP, Python package, OpenAI-compatible client
- Automatically truncates large pages to 20,000 characters (~5k tokens) to control costs and avoid token limits
- Simple CLI interface with configurable model and backend

## Requirements

- Python 3.10+
- An OpenAI API key (for the OpenAI backend)
- [Ollama](https://ollama.com) installed and running locally (for the Ollama backend)

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/website-summarizer.git
   cd website-summarizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your OpenAI API key** *(only needed for the OpenAI backend)*
   ```bash
   cp .env.example .env
   # then edit .env and paste your key
   ```
   `.env` file format:
   ```
   OPENAI_API_KEY=sk-proj-...
   ```

## Usage

```bash
# Summarize using OpenAI (default)
python website_summarizer.py https://example.com

# Summarize using Ollama (local)
python website_summarizer.py https://example.com --backend ollama

# Specify a different model
python website_summarizer.py https://example.com --model gpt-4o
python website_summarizer.py https://example.com --backend ollama --model mistral

# Choose an Ollama calling approach
python website_summarizer.py https://example.com --backend ollama --ollama-approach local-call
python website_summarizer.py https://example.com --backend ollama --ollama-approach python-package
python website_summarizer.py https://example.com --backend ollama --ollama-approach openai-compatible
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `url` | *(required)* | Website URL to summarize |
| `--backend` | `openai` | `openai` or `ollama` |
| `--model` | `gpt-4o-mini` / `llama3.2` | Model name |
| `--ollama-approach` | `openai-compatible` | Ollama API calling method |

## Project Structure

```
website-summarizer/
├── website_summarizer.py   # Main script
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore
└── README.md
```

## Notes

- JavaScript-rendered pages (React, Vue, etc.) won't be scraped correctly — only static HTML is supported.
- Some sites protected by Cloudflare or similar services may return a 403 error.
- Ollama must be running locally (`ollama serve`) before using the `--backend ollama` option.
- Pages exceeding 20,000 characters are automatically truncated. The tool will print a note when this happens.
