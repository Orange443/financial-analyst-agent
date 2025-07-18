# google_api_handler.py
import logging
import time
import re
from typing import Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

class GoogleAPIErrorType(Enum):
    QUOTA_EXCEEDED = "quota_exceeded"
    RATE_LIMITED = "rate_limited"
    SERVICE_UNAVAILABLE = "service_unavailable"
    AUTHENTICATION_ERROR = "auth_error"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class APIErrorInfo:
    error_type: GoogleAPIErrorType
    message: str
    retry_after: str
    suggestion: str
    is_retryable: bool
    backoff_seconds: int

class GoogleAPIErrorHandler:
    def __init__(self):
        self.error_patterns = {
            GoogleAPIErrorType.QUOTA_EXCEEDED: [
                r'quota.*exceeded',
                r'daily.*limit.*exceeded',
                r'rate limit exceeded',
                r'too many requests',
                r'429',
                r'quota.*exhausted'
            ],
            GoogleAPIErrorType.RATE_LIMITED: [
                r'rate.*limit',
                r'too many.*requests',
                r'throttled',
                r'429'
            ],
            GoogleAPIErrorType.SERVICE_UNAVAILABLE: [
                r'503',
                r'service.*unavailable',
                r'temporarily.*unavailable',
                r'server.*overloaded',
                r'internal.*server.*error',
                r'500'
            ],
            GoogleAPIErrorType.AUTHENTICATION_ERROR: [
                r'401',
                r'unauthorized',
                r'invalid.*api.*key',
                r'authentication.*failed',
                r'forbidden',
                r'403'
            ],
            GoogleAPIErrorType.INVALID_REQUEST: [
                r'400',
                r'bad.*request',
                r'invalid.*parameter',
                r'malformed.*request'
            ]
        }
    
    def analyze_error(self, error: Exception) -> APIErrorInfo:
        """Analyze error and return structured information"""
        error_str = str(error).lower()
        
        # Check for specific error types
        for error_type, patterns in self.error_patterns.items():
            if any(re.search(pattern, error_str) for pattern in patterns):
                return self._create_error_info(error_type, error_str)
        
        # Default to unknown error
        return self._create_error_info(GoogleAPIErrorType.UNKNOWN_ERROR, error_str)
    
    def _create_error_info(self, error_type: GoogleAPIErrorType, error_str: str) -> APIErrorInfo:
        """Create error info based on error type"""
        
        error_configs = {
            GoogleAPIErrorType.QUOTA_EXCEEDED: {
                'message': 'Google API daily quota exceeded',
                'retry_after': 'Until quota reset (midnight UTC)',
                'suggestion': 'Wait for quota reset or upgrade API plan',
                'is_retryable': False,
                'backoff_seconds': 0
            },
            GoogleAPIErrorType.RATE_LIMITED: {
                'message': 'API rate limit exceeded',
                'retry_after': '1-2 minutes',
                'suggestion': 'Implement exponential backoff',
                'is_retryable': True,
                'backoff_seconds': 60
            },
            GoogleAPIErrorType.SERVICE_UNAVAILABLE: {
                'message': 'Google API service temporarily unavailable',
                'retry_after': '5-10 minutes',
                'suggestion': 'Retry with exponential backoff',
                'is_retryable': True,
                'backoff_seconds': 300
            },
            GoogleAPIErrorType.AUTHENTICATION_ERROR: {
                'message': 'API authentication failed',
                'retry_after': 'Immediate (after fixing)',
                'suggestion': 'Check API key configuration',
                'is_retryable': False,
                'backoff_seconds': 0
            },
            GoogleAPIErrorType.INVALID_REQUEST: {
                'message': 'Invalid API request format',
                'retry_after': 'Immediate (after fixing)',
                'suggestion': 'Check request parameters',
                'is_retryable': False,
                'backoff_seconds': 0
            },
            GoogleAPIErrorType.UNKNOWN_ERROR: {
                'message': f'Unknown Google API error: {error_str[:100]}',
                'retry_after': '2-5 minutes',
                'suggestion': 'Check Google Cloud Console logs',
                'is_retryable': True,
                'backoff_seconds': 120
            }
        }
        
        config = error_configs[error_type]
        return APIErrorInfo(
            error_type=error_type,
            message=config['message'],
            retry_after=config['retry_after'],
            suggestion=config['suggestion'],
            is_retryable=config['is_retryable'],
            backoff_seconds=config['backoff_seconds']
        )
    
    def should_retry(self, error_info: APIErrorInfo, attempt_count: int, max_attempts: int = 3) -> Tuple[bool, int]:
        """Determine if should retry and calculate backoff time"""
        
        if not error_info.is_retryable:
            return False, 0
        
        if attempt_count >= max_attempts:
            return False, 0
        
        # Calculate exponential backoff
        base_backoff = error_info.backoff_seconds
        backoff_time = min(base_backoff * (2 ** (attempt_count - 1)), 600)  # Max 10 minutes
        
        return True, backoff_time
    
    def log_error_details(self, error: Exception, error_info: APIErrorInfo):
        """Log detailed error information"""
        logging.error(f"Google API Error Detected:")
        logging.error(f"  Type: {error_info.error_type.value}")
        logging.error(f"  Message: {error_info.message}")
        logging.error(f"  Retry After: {error_info.retry_after}")
        logging.error(f"  Suggestion: {error_info.suggestion}")
        logging.error(f"  Is Retryable: {error_info.is_retryable}")
        logging.error(f"  Original Error: {str(error)}")

# Global error handler instance
google_error_handler = GoogleAPIErrorHandler()
