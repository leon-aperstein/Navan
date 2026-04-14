from dotenv import load_dotenv
import os

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

MAX_HISTORY_MESSAGES = 20

OPENWEATHERMAP_BASE_URL = "https://api.openweathermap.org/data/2.5"
RESTCOUNTRIES_BASE_URL = "https://restcountries.com/v3.1"
