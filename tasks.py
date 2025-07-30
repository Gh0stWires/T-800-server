import time
from celery import Celery
from ai_processor import ask_t800  # Import your AI processing function

# Configure Celery with Redis
celery = Celery("tasks", broker="redis://localhost:6379", backend="redis://localhost:6379")

@celery.task(bind=True)
def process_chat_task(self, user_id, question):
    """Run AI chat processing as a background task."""
    time.sleep(1)  # Simulate slight delay
    response = ask_t800(user_id, question)
    return response
