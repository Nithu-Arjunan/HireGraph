import os

from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXTRACT_MODEL = os.getenv("HIREGRAPH_EXTRACT_MODEL", "gpt-4o")
SCORING_MODEL = os.getenv("HIREGRAPH_SCORING_MODEL", "gpt-4o-mini")
EMAIL_MODEL = os.getenv("HIREGRAPH_EMAIL_MODEL", "gpt-4o-mini")
CRITIC_MODEL = os.getenv("HIREGRAPH_CRITIC_MODEL", "gpt-4o-mini")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL")
