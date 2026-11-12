"""Shared configuration: model endpoints and experiment defaults."""
from os import getenv

from dotenv import find_dotenv, load_dotenv

# Looks for .env in the current directory and its parents
load_dotenv(find_dotenv(usecwd=True))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")

DEFAULT_MODEL = getenv("DEFAULT_MODEL", "openai/gpt-4o-mini-2024-07-18")
GEMMA_MODEL = "google/gemma-4-31b-it"

# Models evaluated in the paper
LLAMA_MODELS = [
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
]
STRONGEST_MODEL = "meta-llama/llama-3.3-70b-instruct"
# Reasoning-type classifier (paper, Section V-D)
CLASSIFIER_MODEL = "google/gemini-2.0-flash-001"

# Experiments in the paper run for 25 rounds, 50 games per matchup
MAX_ROUNDS = 25
GAMES_PER_MATCHUP = 50
RESULTS_DIR = getenv("RESULTS_DIR", "results")
LOGS_DIR = getenv("LOGS_DIR", "logs")
