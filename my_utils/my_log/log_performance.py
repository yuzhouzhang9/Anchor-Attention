import datetime
import os
import time
from functools import wraps
from contextlib import contextmanager

class Logger:
    # Default path for logging application messages
    log_file_path = "log/application.log"

    @classmethod
    def set_log_file_path(cls, path: str = None):
        """
        Set a custom log file path under the 'log' directory.

        Args:
            path (str): Filename (or timestamp) for the log file.
        """
        # Prepend the 'log/' directory to the given filename
        filename = path or str(time.time())
        full_path = os.path.join("log", filename)
        print(f"Creating log file at {full_path}... Done\n")
        cls.log_file_path = full_path
        cls._ensure_log_file_exists()

    @classmethod
    def _ensure_log_file_exists(cls):
        """
        Ensure that the log file and its parent directory exist; create them if needed.
        """
        if not os.path.exists(cls.log_file_path):
            directory = os.path.dirname(cls.log_file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            # Initialize the log file with a creation timestamp
            with open(cls.log_file_path, 'w') as f:
                f.write(f"Log file created at {datetime.datetime.now()}\n")

    @classmethod
    def log(cls, message: str):
        """
        Append a timestamped message to the log file.

        Args:
            message (str): The message content to write.
        """
        cls._ensure_log_file_exists()
        # Format timestamp to milliseconds precision
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        entry = f"{timestamp} - {message}"
        try:
            with open(cls.log_file_path, 'a') as log_file:
                log_file.write(entry + "\n")
        except Exception as e:
            # Fallback to console if file write fails
            print(f"Unable to write to log file: {e}")

    @classmethod
    def log_error(cls, error_message: str):
        """
        Log an error-level message.

        Args:
            error_message (str): Description of the error.
        """
        cls.log(f"ERROR: {error_message}")

    @classmethod
    def log_warning(cls, warning_message: str):
        """
        Log a warning-level message.

        Args:
            warning_message (str): Description of the warning.
        """
        cls.log(f"WARNING: {warning_message}")

    @classmethod
    def log_info(cls, info_message: str):
        """
        Log an informational message.

        Args:
            info_message (str): Information to be logged.
        """
        cls.log(f"INFO: {info_message}")

    @classmethod
    def measure_performance(cls, func=None, *, log_file: str = None):
        """
        Decorator to measure and log a function's execution time.

        Args:
            func (callable): Function to wrap.
            log_file (str): Optional override for performance log file.
        """
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = f(*args, **kwargs)
                end = time.perf_counter()
                duration = end - start
                cls.log(f"Function {f.__name__} executed in {duration:.9f}s")
                return result
            return wrapper

        # Support using as @measure_performance or @measure_performance()
        if func:
            return decorator(func)
        return decorator

    @classmethod
    @contextmanager
    def measure_block_time(cls, block_name: str = "Code block"):
        """
        Context manager to measure and log execution time of a code block.

        Args:
            block_name (str): Label for the timed block.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration = end - start
            cls.log(f"{block_name} took {duration:.9f}s")


# Example usage when run as a script
def main():
    # Set a custom log filename
    Logger.set_log_file_path("test_log.log")

    # Measure a function's performance
    @Logger.measure_performance
    def example_function():
        time.sleep(1)
        return "Done"

    result = example_function()
    print(f"Result: {result}")

    # Measure a code block
    with Logger.measure_block_time("Test block"):
        time.sleep(2)


if __name__ == "__main__":
    main()
