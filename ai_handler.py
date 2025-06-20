"""
ai_handler.py - AI Handler Module for B2B Report Analyzer
Supports Claude and OpenAI for SKU and Reason extraction
This file must be in the same directory as app.py
"""

import os
import logging
import streamlit as st
import requests
import json
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class AIHandler:
    """Handle AI interactions for SKU and Reason extraction"""
    
    def __init__(self):
        self.claude_key = self._get_api_key('claude')
        self.openai_key = self._get_api_key('openai')
        self.provider = self._determine_provider()
        self.api_calls = 0
        self.last_error = None
        
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from Streamlit secrets"""
        try:
            if provider == 'claude':
                # Try different key names
                for key_name in ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key']:
                    if key_name in st.secrets:
                        return st.secrets[key_name]
            elif provider == 'openai':
                # Try different key names
                for key_name in ['OPENAI_API_KEY', 'openai_api_key', 'openai']:
                    if key_name in st.secrets:
                        return st.secrets[key_name]
        except Exception as e:
            logger.warning(f"Error accessing secrets for {provider}: {e}")
        
        # Try environment variables as fallback
        if provider == 'claude':
            return os.getenv('ANTHROPIC_API_KEY')
        elif provider == 'openai':
            return os.getenv('OPENAI_API_KEY')
        
        return None
    
    def _determine_provider(self) -> str:
        """Determine which AI provider to use"""
        if self.claude_key:
            return 'claude'
        elif self.openai_key:
            return 'openai'
        else:
            return 'none'
    
    def get_status(self) -> Dict[str, Any]:
        """Get AI handler status"""
        return {
            'available': self.provider != 'none',
            'provider': self.provider,
            'api_calls': self.api_calls,
            'last_error': self.last_error
        }
    
    def extract_sku(self, description: str) -> str:
        """Extract SKU from description using AI"""
        if self.provider == 'none':
            return ""
        
        prompt = f"""Extract the SKU (product code) from this support ticket description. 
Common SKU patterns include: LVA1004-UPC, SUP1001, MOB2003, RHB3002, or similar alphanumeric codes.
If no SKU is found, respond with "NOT_FOUND".

Description: {description}

SKU:"""
        
        try:
            if self.provider == 'claude':
                return self._call_claude(prompt, max_tokens=20)
            elif self.provider == 'openai':
                return self._call_openai(prompt, max_tokens=20)
        except Exception as e:
            logger.error(f"AI SKU extraction error: {e}")
            self.last_error = str(e)
        
        return ""
    
    def extract_reason(self, description: str) -> str:
        """Extract return/refund reason from description using AI"""
        if self.provider == 'none':
            return ""
        
        prompt = f"""Extract and categorize the return/refund reason from this support ticket description.
Provide a brief, clear reason (2-5 words). Common categories include:
- Defective/Broken
- Wrong Item
- Not Needed
- Quality Issue
- Size Issue
- Missing Parts
- Damaged in Shipping
- Other

If no clear reason is found, respond with "Other".

Description: {description}

Reason:"""
        
        try:
            if self.provider == 'claude':
                return self._call_claude(prompt, max_tokens=30)
            elif self.provider == 'openai':
                return self._call_openai(prompt, max_tokens=30)
        except Exception as e:
            logger.error(f"AI reason extraction error: {e}")
            self.last_error = str(e)
        
        return "Other"
    
    def _call_claude(self, prompt: str, max_tokens: int = 50) -> str:
        """Call Claude API"""
        if not self.claude_key:
            return ""
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.claude_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-haiku-20240307",  # Fast and efficient
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.api_calls += 1
                result = response.json()
                return result['content'][0]['text'].strip()
            else:
                logger.error(f"Claude API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
        
        return ""
    
    def _call_openai(self, prompt: str, max_tokens: int = 50) -> str:
        """Call OpenAI API"""
        if not self.openai_key:
            return ""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }
        
        data = {
            "model": "gpt-3.5-turbo",  # Fast and cost-effective
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts specific information from text. Be concise and accurate."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                self.api_calls += 1
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
        
        return ""
    
    def batch_extract(self, descriptions: list, extract_type: str = 'both') -> list:
        """Batch extract SKUs and/or Reasons from multiple descriptions"""
        results = []
        
        for desc in descriptions:
            result = {'description': desc}
            
            if extract_type in ['sku', 'both']:
                result['sku'] = self.extract_sku(desc)
            
            if extract_type in ['reason', 'both']:
                result['reason'] = self.extract_reason(desc)
            
            results.append(result)
        
        return results

# Example patterns for fallback (if needed elsewhere)
SKU_PATTERNS = [
    r'\b(LVA\d{4}[A-Z0-9\-]*)\b',
    r'\b(SUP\d{4}[A-Z0-9\-]*)\b',
    r'\b(MOB\d{4}[A-Z0-9\-]*)\b',
    r'\b(RHB\d{4}[A-Z0-9\-]*)\b',
    r'\b([A-Z]{3}\d{4}[A-Z0-9\-]*)\b',
]

REASON_KEYWORDS = {
    'Defective': ['defect', 'broken', 'not working', 'malfunction', 'damaged', 'faulty'],
    'Wrong Item': ['wrong', 'incorrect', 'not what ordered', 'different'],
    'Not Needed': ['no longer need', "don't need", 'changed mind', 'cancel'],
    'Quality Issue': ['poor quality', 'cheap', 'flimsy', 'low quality'],
    'Size Issue': ['too small', 'too large', 'wrong size', "doesn't fit", 'too big'],
    'Missing Parts': ['missing', 'incomplete', 'not all parts', 'parts missing'],
    'Damaged in Shipping': ['damaged in shipping', 'arrived damaged', 'shipping damage'],
    'Not Compatible': ['not compatible', 'incompatible', "doesn't work with"],
}
