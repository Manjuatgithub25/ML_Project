import logging, os, sys
from datetime import datetime

log_format = "[%(asctime)s] %(lineno)d %(levelname)s %(name)s %(message)s"
log_dir = "logs"
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(log_dir, f"running_logs_{current_time}.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]   
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
