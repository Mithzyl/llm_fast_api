import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from utils.notifier import feishu_notifier
# Note: We need to import the actual workflow runner function.
# This might require adjusting the import path depending on where the LlmGraph instance is created.
# For now, we'll use a placeholder.
# from main import run_assistant_workflow  # Placeholder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def trigger_summary_workflow():
    """
    This is the function that will be executed by the scheduler.
    It defines the proactive task for the personal assistant.
    """
    logging.info("Scheduler triggered: Starting hourly summary workflow...")
    
    try:
        # 1. Define the proactive task/prompt for the assistant
        task_prompt = "Based on my recent memories and conversations, what is the single most important thing I should focus on right now? Provide a concise summary and one actionable suggestion."
        
        # --- Placeholder for Workflow Invocation ---
        # In a real implementation, you would call your LangGraph workflow here.
        # This requires having access to the LlmGraph instance and a user ID to run the workflow for.
        # For example:
        # llm_graph = get_llm_graph_instance() # This function would need to be created
        # user_id = "default_user" # The user for whom the summary is generated
        # conversation_id = f"scheduled-summary-{int(time.time())}"
        # final_state = await llm_graph.run_integrated_workflow(
        #     conversation_id=conversation_id,
        #     user_message=task_prompt,
        #     history_messages=[],
        #     user_id=user_id
        # )
        # summary_result = final_state.get("result", ["No summary generated."])[-1].content
        
        # For this example, we will use a mock result.
        summary_result = "Mock Result: The most important task is to follow up on the project proposal discussed yesterday. Suggestion: Draft an email to the team with the key action points."
        logging.info(f"Workflow finished. Generated summary: {summary_result}")

        # 2. Send the result to Feishu
        success = feishu_notifier.send_text(f" hourly Summary:\n\n{summary_result}")
        
        if success:
            logging.info("Successfully sent summary to Feishu.")
        else:
            logging.error("Failed to send summary to Feishu.")
            
    except Exception as e:
        logging.error(f"An error occurred during the scheduled workflow: {e}", exc_info=True)


class SchedulerManager:
    def __init__(self):
        self.scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")

    def start(self):
        """
        Adds jobs to the scheduler and starts it.
        """
        logging.info("Starting scheduler...")
        # Schedule the job to run every hour
        self.scheduler.add_job(trigger_summary_workflow, 'interval', hours=1)
        self.scheduler.start()
        logging.info("Scheduler started. Hourly summary job is scheduled.")

    def stop(self):
        """
        Stops the scheduler.
        """
        if self.scheduler.running:
            logging.info("Stopping scheduler...")
            self.scheduler.shutdown()
            logging.info("Scheduler stopped.")

# Singleton instance to be used in the main application
scheduler_manager = SchedulerManager()
