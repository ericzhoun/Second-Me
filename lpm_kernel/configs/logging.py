import os
import sys
import logging
import logging.config
import logging.handlers
import datetime
import shutil
import time

# Get project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    A RotatingFileHandler that handles Windows file locking issues gracefully.
    If rotation fails due to file being locked, it continues logging to the current file.
    """

    def doRollover(self):
        """
        Override doRollover to handle Windows file locking issues.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        # Try rotation with retries
        for attempt in range(3):
            try:
                # Standard rotation logic
                if self.backupCount > 0:
                    for i in range(self.backupCount - 1, 0, -1):
                        sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
                        dfn = self.rotation_filename("%s.%d" % (self.baseFilename, i + 1))
                        if os.path.exists(sfn):
                            if os.path.exists(dfn):
                                os.remove(dfn)
                            os.rename(sfn, dfn)
                    dfn = self.rotation_filename(self.baseFilename + ".1")
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    self.rotate(self.baseFilename, dfn)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.1)  # Brief wait before retry
                else:
                    # If rotation fails after retries, just truncate the file
                    # or continue appending - don't crash
                    pass
            except Exception:
                # For any other error, just continue
                break

        # Reopen the file
        if not self.delay:
            self.stream = self._open()

# Define log directories
LOG_BASE_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_DIR = os.path.join(LOG_BASE_DIR, "train")

# Define log file paths
APP_LOG_FILE = os.path.join(LOG_BASE_DIR, "app.log")
TRAIN_LOG_FILE = os.path.join(TRAIN_LOG_DIR, "train.log")

# Function to rename log file if it exists
def rename_existing_log_file():
    if os.path.exists(TRAIN_LOG_FILE):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"train_{timestamp}.log"
        backup_path = os.path.join(TRAIN_LOG_DIR, backup_filename)
        
        shutil.move(TRAIN_LOG_FILE, backup_path)
        print(f"Existing train.log renamed to {backup_filename}")
        return True
    return False

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": sys.stdout,
        },
        "file": {
            "()": SafeRotatingFileHandler,
            "level": "INFO",
            "formatter": "standard",
            "filename": APP_LOG_FILE,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "train_process_file": {
            "()": SafeRotatingFileHandler,
            "level": "INFO",
            "formatter": "standard",
            "filename": TRAIN_LOG_FILE,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "train_process": {
            "level": "INFO",
            "handlers": ["train_process_file", "console"],
            "propagate": False,
        },
    },
    "root": {  # root logger configuration
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}

# Initialize logging configuration
def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)

# Get train process logger
def get_train_process_logger():
    return logging.getLogger("train_process")
