# PromptChess

A chess tournament system that pits AI models against each other.

## Prerequisites

- Python 3.13+
- API keys for one or more providers:
  - OpenAI API key
  - Anthropic API key
  - Google Gemini API key

## Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and set up the project**:
   ```bash
   git clone https://github.com/knvmhra/promptchess.git
   cd promptchess
   uv sync
   ```

## Configuration

Set up your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_GENAI_API_KEY="your-gemini-key"
```

Or create a `.env` file in the project root:
```
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_GENAI_API_KEY=your-gemini-key
```

## Usage

### Run a Tournament

```bash
uv run src/arena.py
```

This will:
- Run a round-robin tournament between configured models
- Each pair of models plays twice (once as white, once as black)
- Update ELO ratings after each game
- Save progress automatically

### Resume a Tournament

If interrupted, the tournament automatically resumes from where it left off using saved state in `league_state.json`.

## Output Files

The system generates several output files:

- `league_state.json` - Tournament progress and ELO ratings
- `model_configs.json` - Model configuration backup
- `pgn/game_N.pgn` - Individual game files in PGN format

## Customization

Edit `src/arena.py` to:
- Add/remove models from the tournament
- Modify model parameters
- Change save file names/locations

## Todo

- Grok support
- OpenRouter models support
- Board interaction tools
