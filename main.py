import logging
import os
import signal
import subprocess
import sys
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_database():
    """Initialize the database with sample data"""
    logger.info("Initializing database...")
    from src.utils.db_init import initialize_database as init_db

    init_db()


def start_fastapi_backend():
    """Start the FastAPI backend server"""
    logger.info("Starting FastAPI backend...")

    # Import inside function to ensure database is initialized first
    import uvicorn

    from src.backend.api import app
    from src.config import API_HOST, API_PORT

    # Run the backend in a separate thread
    def run_backend():
        uvicorn.run(app, host=API_HOST, port=API_PORT)

    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    # Wait for backend to start
    logger.info("Waiting for backend to start...")
    time.sleep(2)

    return backend_thread


def start_streamlit_frontend():
    """Start the Streamlit frontend"""
    logger.info("Starting Streamlit frontend...")

    from src.config import FRONTEND_HOST, FRONTEND_PORT

    # Create .streamlit directory and config if they don't exist
    os.makedirs(".streamlit", exist_ok=True)

    # Create or update Streamlit config for proper server settings
    config_path = ".streamlit/config.toml"
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(
                f"""
[server]
headless = true
address = "{FRONTEND_HOST}"
port = {FRONTEND_PORT}
"""
            )

    # Start Streamlit
    streamlit_cmd = [
        "streamlit",
        "run",
        "src/frontend/app.py",
        "--server.port",
        str(FRONTEND_PORT),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--server.enableCORS",
        "false",
        "--server.enableWebsocketCompression",
        "false",
    ]

    # Use Popen to start process without waiting for it to finish
    streamlit_process = subprocess.Popen(
        streamlit_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    return streamlit_process


def check_openai_key():
    """Check if OpenAI API key is set"""
    from src.config import OPENAI_API_KEY

    if not OPENAI_API_KEY:
        logger.warning(
            "OPENAI_API_KEY environment variable not set. You'll need to set this to use the NLP features."
        )
        print("\n" + "=" * 80)
        print("WARNING: OPENAI_API_KEY environment variable not set.")
        print("Natural language processing will not work without an API key.")
        print(
            "Set the OPENAI_API_KEY environment variable before running the application."
        )
        print("=" * 80 + "\n")


def main():
    """Main entry point for the application"""
    # Initialize the database first
    initialize_database()

    # Check for OpenAI API key
    check_openai_key()

    # Start the backend
    _ = start_fastapi_backend()

    # Start the Streamlit frontend
    streamlit_process = start_streamlit_frontend()

    # Keep the main thread alive to allow the backend and frontend to run
    try:
        # Monitor the Streamlit process
        while True:
            output_line = streamlit_process.stdout.readline()
            if output_line == "" and streamlit_process.poll() is not None:
                break
            if output_line:
                print(output_line.strip())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        # Clean up processes
        if streamlit_process.poll() is None:
            streamlit_process.terminate()
        sys.exit(0)


if __name__ == "__main__":
    main()
