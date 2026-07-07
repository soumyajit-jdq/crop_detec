import json
import os
import sys
from functools import wraps
from typing import Any, Dict
from datetime import datetime

from fastapi import Request
from loguru import logger

# Constants
LOG_DIR = "logs"
SENSITIVE_HEADERS = {"authorization", "x-api-key", "cookie", "password"}
MAX_BODY_LENGTH = 1000  # Maximum length for request body logging


def setup_logging(
    log_level: str = "DEBUG",
    retention: str = "1 week",
    rotation: str = "500 MB",
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
) -> None:
    """
    Configure logging settings and handlers.

    Args:
        log_level (str): Minimum log level to capture
        retention (str): How long to keep log files
        rotation (str): When to rotate log files
        log_format (str): Custom format for log messages
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    # Remove default logger
    logger.remove()

    # Add console handler with custom format
    logger.add(
        sys.stdout, level=log_level, format=log_format, backtrace=True, diagnose=True
    )

    # Add JSON file handler for structured logging
    logger.add(
        f"{LOG_DIR}/app.log",
        level="INFO",
        serialize=True,
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    # Add error file handler
    logger.add(
        f"{LOG_DIR}/error.log",
        level="ERROR",
        rotation=rotation,
        retention=retention,
        compression="zip",
        backtrace=True,
        diagnose=True,
    )


def sanitize_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Sanitize sensitive information from headers."""
    sanitized = headers.copy()
    for header in SENSITIVE_HEADERS:
        if header.lower() in sanitized:
            sanitized[header.lower()] = "[REDACTED]"
    return sanitized


def truncate_body(body: Any) -> Any:
    """Truncate large request bodies to prevent log flooding."""
    if isinstance(body, dict):
        body_str = json.dumps(body)
        if len(body_str) > MAX_BODY_LENGTH:
            return {"truncated_content": f"{body_str[:MAX_BODY_LENGTH]}... (truncated)"}
    return body


async def get_request_body(request: Request) -> Dict[str, Any]:
    """
    Extract body content from request, handling both JSON and form data.

    Args:
        request (Request): FastAPI request object

    Returns:
        Dict[str, Any]: Dictionary containing either body or form data
    """
    try:
        body = await request.json()
        return {"body": truncate_body(body)}
    except Exception:
        try:
            form_data = await request.form()
            form_info = {}
            for key, value in form_data.items():
                if isinstance(value, str):
                    form_info[key] = value
                else:
                    # Log only file metadata
                    form_info[key] = {
                        "filename": value.filename,
                        "content_type": value.content_type,
                        "size": value.spool_max_size,
                    }
            return {"form": truncate_body(form_info)}
        except Exception as e:
            logger.debug(f"Could not parse request body: {str(e)}")
            return {"body": None}


def log_request(func):
    """
    Decorator to log incoming request details.

    Args:
        func: The route handler function to wrap

    Returns:
        Wrapped function that logs request details before execution
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        request = kwargs.get("request") or next(
            (arg for arg in args if isinstance(arg, Request)), None
        )

        if not isinstance(request, Request):
            logger.error("Request object not found in args or kwargs")
            raise ValueError("Request object not found in args or kwargs")

        # Generate request ID for tracking
        request_id = request.headers.get("X-Request-ID", os.urandom(8).hex())

        request_info = {
            "request_id": request_id,
            "timestamp": start_time.isoformat(),
            "method": request.method,
            "url": str(request.url),
            "headers": sanitize_headers(dict(request.headers)),
            "query_params": dict(request.query_params),
            "client_ip": request.client.host if request.client else None,
        }

        # Log request details
        logger.bind(request_id=request_id).info(
            f"Incoming {request.method} request to {request.url.path}"
        )

        try:
            request_body = await get_request_body(request)
            request_info.update(request_body)

            # Execute the route handler
            response = await func(*args, **kwargs)

            # Calculate request duration
            duration = (datetime.now() - start_time).total_seconds()

            # Log successful response
            logger.bind(request_id=request_id).debug(
                json.dumps(
                    {**request_info, "duration": duration, "status": "success"},
                    indent=4,
                )
            )

            return response

        except Exception as e:
            # Log error with full context
            duration = (datetime.now() - start_time).total_seconds()
            logger.bind(request_id=request_id).error(
                json.dumps(
                    {
                        **request_info,
                        "duration": duration,
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                    indent=4,
                )
            )
            raise

    return wrapper


# Initialize logging configuration when module is imported
setup_logging()
