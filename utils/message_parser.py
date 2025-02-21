from typing import List, Dict
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MessageParser:
    """
    Utility class for parsing WhatsApp chat messages.
    """
    
    MESSAGE_PATTERN = re.compile(r"^(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - (.*?): (.*)$")
    
    @classmethod
    def parse_messages(cls, lines: List[str]) -> List[Dict[str, str]]:
        """
        Parse raw chat lines into structured message objects.
        
        Args:
            lines (List[str]): Raw chat lines from the input file
            
        Returns:
            List[Dict[str, str]]: List of parsed messages
        """
        messages = []
        for line in lines:
            try:
                if match := cls.MESSAGE_PATTERN.match(line):
                    timestamp, sender, content = match.groups()
                    messages.append({
                        "timestamp": cls._parse_timestamp(timestamp),
                        "sender": sender.strip(),
                        "message": content.strip()
                    })
            except Exception as e:
                logger.warning(f"Failed to parse line: {line}. Error: {str(e)}")
                continue
        return messages
    
    @staticmethod
    def _parse_timestamp(timestamp_str: str) -> str:
        """
        Convert WhatsApp timestamp to ISO format.
        
        Args:
            timestamp_str (str): Timestamp in WhatsApp format (DD/MM/YYYY, HH:MM)
            
        Returns:
            str: ISO formatted timestamp
        """
        try:
            dt = datetime.strptime(timestamp_str, "%d/%m/%Y, %H:%M")
            return dt.isoformat()
        except ValueError as e:
            logger.error(f"Failed to parse timestamp: {timestamp_str}. Error: {str(e)}")
            return timestamp_str