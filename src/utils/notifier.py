import requests
import json
import logging

class FeishuNotifier:
    def __init__(self, webhook_url: str):
        """
        Initializes the Feishu Notifier.
        :param webhook_url: The Webhook URL for the Feishu bot.
        """
        if not webhook_url or "open.feishu.cn" not in webhook_url:
            logging.warning("Feishu Webhook URL is not configured or invalid, push notifications will be unavailable.")
            self.webhook_url = None
        else:
            self.webhook_url = webhook_url
            logging.info("Feishu Notifier initialized.")

    def send_text(self, message: str) -> bool:
        """
        Sends a plain text message.
        :param message: The text message to send.
        :return: True if the message was sent successfully, False otherwise.
        """
        if not self.webhook_url:
            logging.error("Cannot send Feishu message: Webhook URL is not configured.")
            return False

        headers = {"Content-Type": "application/json"}
        payload = {
            "msg_type": "text",
            "content": {"text": message}
        }
        
        try:
            response = requests.post(self.webhook_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            result = response.json()
            
            if result.get("StatusCode") == 0:
                logging.info("Feishu message sent successfully.")
                return True
            else:
                logging.error(f"Failed to send Feishu message: {result}")
                return False
        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred while requesting the Feishu API: {e}")
            return False

# --- Singleton Instance ---
# The user provided the Webhook URL directly.
# For better security and flexibility, it's recommended to move this URL to a configuration file (e.g., config.yaml)
# and load it from there instead of hardcoding.
FEISHU_WEBHOOK_URL = "https://open.feishu.cn/open-apis/bot/v2/hook/e6b154f3-dca6-4566-9b49-5767d3ec7437"

# Create a singleton instance for easy import and use across the project.
feishu_notifier = FeishuNotifier(FEISHU_WEBHOOK_URL)

if __name__ == '__main__':
    # Example usage and test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("Sending a test message to Feishu...")
    success = feishu_notifier.send_text("Hello from the new Notifier! This is a test message.")
    if success:
        print("Test message sent successfully.")
    else:
        print("Failed to send the test message.")
