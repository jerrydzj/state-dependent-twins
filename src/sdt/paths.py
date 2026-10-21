"""Repository paths. Raw data is read-only; everything under PROCESSED is rebuilt by code and not in git."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
RAW_CSV = DATA / 'data.csv'
PROCESSED = DATA / 'processed'
