from typing import List, Dict, Optional, Tuple
import re
from datetime import datetime
import logging
import json
from config.settings import get_gemini_model

logger = logging.getLogger(__name__)

class MessageParser:
    """
    Universal chat parser that can handle ANY chat format using AI-powered pattern detection.
    Supports WhatsApp, Discord, Slack, Telegram, iMessage, SMS exports, and custom formats.
    """
    
    # Common patterns for fallback regex matching
    PATTERNS = {
        'whatsapp_24h': re.compile(r"^(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - (.*?): (.*)$"),
        'whatsapp_12h': re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*[AP]M) - (.*?): (.*)$", re.IGNORECASE),
        'discord': re.compile(r"^\[(.*?)\]\s+(.*?):\s+(.*)$"),
        'slack': re.compile(r"^(.*?)\s+\[(.*?)\]\s+(.*)$"),
        'telegram': re.compile(r"^\[(.*?)\]\s+(.*?):\n(.*)$", re.MULTILINE),
        'timestamp_first': re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\|\s+(.*?):\s+(.*)$"),
        'generic_colon': re.compile(r"^(.*?)\s*-\s*(.*?):\s*(.*)$"),
    }
    
    def __init__(self, use_ai: bool = True):
        """
        Initialize parser with AI capability.
        
        Args:
            use_ai: Whether to use LLM for intelligent parsing (default: True)
        """
        self.use_ai = use_ai
        self.detected_format = None
        self.custom_pattern = None
    
    @classmethod
    def parse_messages(cls, lines: List[str], use_ai: bool = True) -> List[Dict[str, str]]:
        """
        Universal parser that auto-detects format and extracts messages.
        
        Args:
            lines: Raw chat lines from any platform
            use_ai: Use AI for format detection (default: True)
            
        Returns:
            List of parsed messages with timestamp, sender, and message
        """
        parser = cls(use_ai=use_ai)
        
        # Step 1: Detect format using AI or pattern matching
        sample = lines[:min(20, len(lines))]
        format_info = parser._detect_format(sample)
        
        if not format_info:
            logger.error("Could not detect chat format")
            return []
        
        logger.info(f"Detected format: {format_info.get('format_name', 'custom')}")
        
        # Step 2: Parse messages based on detected format
        if parser.use_ai and format_info.get('use_ai_parsing'):
            return parser._ai_parse_messages(lines, format_info)
        else:
            return parser._regex_parse_messages(lines, format_info)
    
    def _detect_format(self, sample_lines: List[str]) -> Optional[Dict]:
        """
        Detect chat format using AI or pattern matching.
        
        Args:
            sample_lines: Sample of chat lines for analysis
            
        Returns:
            Dict with format information and parsing instructions
        """
        if self.use_ai:
            return self._ai_detect_format(sample_lines)
        else:
            return self._regex_detect_format(sample_lines)
    
    def _ai_detect_format(self, sample_lines: List[str]) -> Optional[Dict]:
        """
        Use LLM to intelligently detect chat format and create parsing rules.
        
        Args:
            sample_lines: Sample chat lines
            
        Returns:
            Format detection result with parsing instructions
        """
        try:
            model = get_gemini_model()
            
            prompt = f"""Analyze these chat log samples and determine the format:

{chr(10).join(sample_lines[:15])}

Your task:
1. Identify the chat format (WhatsApp, Discord, Slack, Telegram, SMS, or custom)
2. Detect the timestamp format
3. Identify how senders are indicated
4. Determine message structure

Respond in JSON format ONLY:
{{
    "format_name": "detected format name",
    "timestamp_pattern": "describe timestamp pattern",
    "sender_pattern": "describe how sender is indicated",
    "message_pattern": "describe message structure",
    "regex_pattern": "regex pattern to extract (timestamp)(sender)(message) groups or null if complex",
    "use_ai_parsing": true if format is too complex for regex, false otherwise,
    "sample_parse": {{
        "timestamp": "example extracted timestamp",
        "sender": "example extracted sender",
        "message": "example extracted message"
    }}
}}

CRITICAL: Output ONLY valid JSON, no markdown, no explanation."""

            response = model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up any markdown formatting
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            format_info = json.loads(result_text)
            self.detected_format = format_info
            
            logger.info(f"AI detected format: {format_info.get('format_name')}")
            return format_info
            
        except Exception as e:
            logger.error(f"AI format detection failed: {e}. Falling back to regex.")
            return self._regex_detect_format(sample_lines)
    
    def _regex_detect_format(self, sample_lines: List[str]) -> Optional[Dict]:
        """
        Fallback pattern matching for common formats.
        
        Args:
            sample_lines: Sample chat lines
            
        Returns:
            Format info with regex pattern
        """
        for format_name, pattern in self.PATTERNS.items():
            matches = sum(1 for line in sample_lines if pattern.match(line))
            
            # If >50% of lines match, we found the format
            if matches > len(sample_lines) * 0.5:
                logger.info(f"Regex detected format: {format_name}")
                return {
                    'format_name': format_name,
                    'regex_pattern': pattern,
                    'use_ai_parsing': False
                }
        
        logger.warning("No standard format detected, trying AI parsing")
        return {'format_name': 'unknown', 'use_ai_parsing': True}
    
    def _regex_parse_messages(self, lines: List[str], format_info: Dict) -> List[Dict[str, str]]:
        """
        Parse messages using regex pattern.
        
        Args:
            lines: All chat lines
            format_info: Format detection result
            
        Returns:
            Parsed messages
        """
        messages = []
        pattern = format_info.get('regex_pattern')
        
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        
        for line in lines:
            try:
                if match := pattern.match(line):
                    groups = match.groups()
                    if len(groups) >= 3:
                        messages.append({
                            "timestamp": self._normalize_timestamp(groups[0]),
                            "sender": groups[1].strip(),
                            "message": groups[2].strip()
                        })
            except Exception as e:
                logger.debug(f"Failed to parse line: {line[:50]}... Error: {e}")
                continue
        
        return messages
    
    def _ai_parse_messages(self, lines: List[str], format_info: Dict) -> List[Dict[str, str]]:
        """
        Use AI to parse complex or non-standard chat formats.
        Processes in batches for efficiency.
        
        Args:
            lines: All chat lines
            format_info: Format detection result
            
        Returns:
            Parsed messages
        """
        messages = []
        batch_size = 50  # Process 50 lines at a time
        
        try:
            model = get_gemini_model()
            
            for i in range(0, len(lines), batch_size):
                batch = lines[i:i + batch_size]
                
                prompt = f"""Parse these chat messages based on detected format: {format_info.get('format_name')}

Chat lines:
{chr(10).join(batch)}

Extract each message as JSON array. Each entry must have:
- timestamp: extracted timestamp
- sender: sender name/username
- message: message content

Output format (JSON array ONLY, no markdown):
[
  {{"timestamp": "...", "sender": "...", "message": "..."}},
  ...
]

CRITICAL: Output ONLY valid JSON array, no explanation."""

                response = model.generate_content(prompt)
                result_text = response.text.strip()
                
                # Clean markdown
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0].strip()
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0].strip()
                
                batch_messages = json.loads(result_text)
                
                # Normalize timestamps
                for msg in batch_messages:
                    msg['timestamp'] = self._normalize_timestamp(msg['timestamp'])
                
                messages.extend(batch_messages)
                logger.info(f"AI parsed batch {i//batch_size + 1}, extracted {len(batch_messages)} messages")
            
        except Exception as e:
            logger.error(f"AI parsing failed: {e}. Trying fallback regex.")
            return self._regex_parse_messages(lines, format_info)
        
        return messages
    
    @staticmethod
    def _normalize_timestamp(timestamp_str: str) -> str:
        """
        Convert any timestamp format to ISO format using intelligent parsing.
        
        Args:
            timestamp_str: Timestamp in any format
            
        Returns:
            ISO formatted timestamp or original if parsing fails
        """
        # Try common formats
        formats = [
            "%d/%m/%Y, %H:%M",           # WhatsApp 24h
            "%m/%d/%Y, %I:%M %p",         # WhatsApp 12h US
            "%d/%m/%y, %H:%M",            # Short year
            "%Y-%m-%d %H:%M:%S",          # ISO-like
            "%Y-%m-%d %H:%M",             # ISO without seconds
            "%d.%m.%Y, %H:%M",            # EU format
            "%d-%m-%Y %H:%M",             # Dash format
            "%b %d, %Y %I:%M %p",         # Discord-like
            "%Y/%m/%d %H:%M:%S",          # Slash ISO
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str.strip(), fmt)
                return dt.isoformat()
            except ValueError:
                continue
        
        # If all fail, return original
        logger.debug(f"Could not parse timestamp: {timestamp_str}")
        return timestamp_str