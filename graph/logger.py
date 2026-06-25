import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LangGraph")

def log_node_execution(node_name: str, execution_time: float, status: str, details: str = ""):
    """Logs the execution of a LangGraph node."""
    msg = f"Node: {node_name} | Status: {status} | Execution Time: {execution_time:.2f}s"
    if details:
        msg += f" | {details}"
    logger.info(msg)

def log_error(node_name: str, error: Exception):
    """Logs an error from a LangGraph node."""
    logger.error(f"Node: {node_name} | ERROR: {str(error)}", exc_info=True)
