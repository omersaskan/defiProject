import requests
import logging

class TelegramAlerter:
    """
    Sends markdown-formatted alerts to a Telegram bot.
    """
    def __init__(self, token: str = None, chat_id: str = None):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage" if token else None
        self.logger = logging.getLogger(__name__)

    def send_message(self, text: str):
        """
        Sends a message via Telegram Bot API.
        """
        if not self.token or not self.chat_id:
            self.logger.warning("Telegram Alert skipped: Missing token or chat_id.")
            return False

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")
            return False
