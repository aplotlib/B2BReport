"""
ai_handler.py - Optimized AI Handler Module for B2B Report Analyzer
Supports Claude and OpenAI with batch processing for faster extraction
Includes caching and parallel processing capabilities
"""

import os
import logging
import streamlit as st
import requests
import json
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time

logger = logging.getLogger(__name__)

class AIHandler:
    """Optimized AI handler with batch processing and caching"""
    
    def __init__(self):
        self.claude_key = self._get_api_key('claude')
        self.openai_key = self._get_api_key('openai')
        self.provider = self._determine_provider()
        self.api_calls = 0
        self.last_error = None
        
        # Caching for repeated descriptions
        self.sku_cache = {}
        self.reason_cache = {}
        self.cache_hits = 0
        
        # Batch processing settings
        self.batch_size = 10
        self.max_workers = 3
        
        # Pattern compilation for faster regex
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for faster matching"""
        self.sku_patterns = [
            re.compile(r'\b(LVA\d{4}[A-Z0-9\-]*)\b', re.IGNORECASE),
            re.compile(r'\b(SUP\d{4}[A-Z0-9\-]*)\b', re.IGNORECASE),
            re.compile(r'\b(MOB\d{4}[A-Z0-9\-]*)\b', re.IGNORECASE),
            re.compile(r'\b(RHB\d{4}[A-Z0-9\-]*)\b', re.IGNORECASE),
            re.compile(r'\b([A-Z]{3}\d{4}[A-Z0-9\-]*)\b', re.IGNORECASE),
            re.compile(r'SKU[:\s]+([A-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'Item[:\s]+([A-Z0-9\-]+)', re.IGNORECASE),
        ]
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from Streamlit secrets"""
        try:
            if provider == 'claude':
                for key_name in ['ANTHROPIC_API_KEY', 'anthropic_api_key', 'claude_api_key']:
                    if key_name in st.secrets:
                        return st.secrets[key_name]
            elif provider == 'openai':
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
        """Get AI handler status with cache stats"""
        return {
            'available': self.provider != 'none',
            'provider': self.provider,
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.sku_cache) + len(self.reason_cache),
            'last_error': self.last_error
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def extract_sku_fast(self, description: str) -> str:
        """Fast SKU extraction using patterns first, then AI"""
        if not description or pd.isna(description):
            return ""
        
        description = str(description)
        
        # Check cache first
        cache_key = self._get_cache_key(description)
        if cache_key in self.sku_cache:
            self.cache_hits += 1
            return self.sku_cache[cache_key]
        
        # Try pattern matching first (faster)
        for pattern in self.sku_patterns:
            match = pattern.search(description)
            if match:
                sku = match.group(1).upper()
                self.sku_cache[cache_key] = sku
                return sku
        
        # Fall back to AI if no pattern match
        if self.provider != 'none':
            sku = self.extract_sku(description)
            self.sku_cache[cache_key] = sku
            return sku
        
        self.sku_cache[cache_key] = ""
        return ""
    
    def extract_sku(self, description: str) -> str:
        """Extract SKU from description using AI"""
        if self.provider == 'none':
            return ""
        
        prompt = f"""Extract the SKU (product code) from this support ticket description. 
Common SKU patterns include: LVA1004-UPC, SUP1001, MOB2003, RHB3002, or similar alphanumeric codes.
If no SKU is found, respond with "NOT_FOUND".

Description: {description[:500]}  # Limit description length for speed

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
    
    def extract_reason_fast(self, description: str) -> str:
        """Fast reason extraction with caching"""
        if not description or pd.isna(description):
            return "Other"
        
        description = str(description)
        
        # Check cache first
        cache_key = self._get_cache_key(description)
        if cache_key in self.reason_cache:
            self.cache_hits += 1
            return self.reason_cache[cache_key]
        
        # Quick keyword check first
        description_lower = description.lower()
        quick_matches = {
            'defective': ['defect', 'broken', 'not working', 'malfunction'],
            'wrong item': ['wrong', 'incorrect', 'not what ordered'],
            'not needed': ['no longer need', "don't need", 'changed mind'],
            'quality issue': ['poor quality', 'cheap', 'flimsy'],
            'size issue': ['too small', 'too large', 'wrong size'],
            'missing parts': ['missing', 'incomplete', 'not all parts'],
        }
        
        for reason, keywords in quick_matches.items():
            if any(keyword in description_lower for keyword in keywords):
                self.reason_cache[cache_key] = reason
                return reason
        
        # Use AI for ambiguous cases
        if self.provider != 'none':
            reason = self.extract_reason(description)
            self.reason_cache[cache_key] = reason
            return reason
        
        self.reason_cache[cache_key] = "Other"
        return "Other"
    
    def extract_reason(self, description: str) -> str:
        """Extract return/refund reason from description using AI"""
        if self.provider == 'none':
            return "Other"
        
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

Description: {description[:500]}  # Limit for speed

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
    
    def batch_extract_parallel(self, descriptions: List[Tuple[int, str]], extract_type: str = 'both') -> Dict[int, Dict[str, str]]:
        """Extract SKUs and/or Reasons from multiple descriptions in parallel"""
        results = {}
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {}
            
            for idx, desc in descriptions:
                if extract_type == 'sku':
                    future = executor.submit(self.extract_sku_fast, desc)
                    futures[future] = (idx, 'sku')
                elif extract_type == 'reason':
                    future = executor.submit(self.extract_reason_fast, desc)
                    futures[future] = (idx, 'reason')
                else:  # both
                    future_sku = executor.submit(self.extract_sku_fast, desc)
                    future_reason = executor.submit(self.extract_reason_fast, desc)
                    futures[future_sku] = (idx, 'sku')
                    futures[future_reason] = (idx, 'reason')
            
            # Collect results
            for future in as_completed(futures):
                idx, field_type = futures[future]
                try:
                    result = future.result(timeout=5)
                    if idx not in results:
                        results[idx] = {}
                    results[idx][field_type] = result
                except Exception as e:
                    logger.error(f"Batch extraction error for {idx}: {e}")
                    if idx not in results:
                        results[idx] = {}
                    results[idx][field_type] = "" if field_type == 'sku' else "Other"
        
        return results
    
    def batch_extract_ai(self, items: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
        """Batch process multiple items with AI for efficiency"""
        if self.provider == 'none':
            return items
        
        # Process in smaller batches to avoid token limits
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Create combined prompt for batch
            batch_prompt = self._create_batch_prompt(batch)
            
            try:
                if self.provider == 'claude':
                    response = self._call_claude(batch_prompt, max_tokens=200)
                elif self.provider == 'openai':
                    response = self._call_openai(batch_prompt, max_tokens=200)
                
                # Parse batch response
                self._parse_batch_response(batch, response)
                
            except Exception as e:
                logger.error(f"Batch AI processing error: {e}")
        
        return items
    
    def _create_batch_prompt(self, batch: List[Dict[str, Any]]) -> str:
        """Create a combined prompt for batch processing"""
        prompt = """Extract SKU and Reason for each item below. Format each response as:
Item X: SKU: [sku_code], Reason: [brief_reason]

"""
        for i, item in enumerate(batch):
            prompt += f"Item {i+1}: {item.get('description', '')[:200]}\n\n"
        
        return prompt
    
    def _parse_batch_response(self, batch: List[Dict[str, Any]], response: str):
        """Parse AI response for batch items"""
        lines = response.strip().split('\n')
        
        for i, line in enumerate(lines):
            if i < len(batch) and 'Item' in line:
                # Extract SKU and Reason from line
                sku_match = re.search(r'SKU:\s*([A-Z0-9\-]+)', line, re.IGNORECASE)
                reason_match = re.search(r'Reason:\s*([^,]+)', line, re.IGNORECASE)
                
                if sku_match:
                    batch[i]['sku'] = sku_match.group(1).strip()
                if reason_match:
                    batch[i]['reason'] = reason_match.group(1).strip()
    
    def _call_claude(self, prompt: str, max_tokens: int = 50) -> str:
        """Call Claude API with retry logic"""
        if not self.claude_key:
            return ""
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.claude_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        for attempt in range(2):  # Retry once
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=5  # Shorter timeout for speed
                )
                
                if response.status_code == 200:
                    self.api_calls += 1
                    result = response.json()
                    return result['content'][0]['text'].strip()
                elif response.status_code == 429:  # Rate limited
                    time.sleep(1)  # Brief pause before retry
                    continue
                else:
                    logger.error(f"Claude API error: {response.status_code}")
                    break
                    
            except requests.Timeout:
                logger.warning("Claude API timeout, retrying...")
                continue
            except Exception as e:
                logger.error(f"Claude API call failed: {e}")
                break
        
        return ""
    
    def _call_openai(self, prompt: str, max_tokens: int = 50) -> str:
        """Call OpenAI API with retry logic"""
        if not self.openai_key:
            return ""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts specific information from text. Be concise and accurate."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }
        
        for attempt in range(2):  # Retry once
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=5  # Shorter timeout
                )
                
                if response.status_code == 200:
                    self.api_calls += 1
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                elif response.status_code == 429:  # Rate limited
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"OpenAI API error: {response.status_code}")
                    break
                    
            except requests.Timeout:
                logger.warning("OpenAI API timeout, retrying...")
                continue
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
                break
        
        return ""
    
    def clear_cache(self):
        """Clear the cache to free memory"""
        self.sku_cache.clear()
        self.reason_cache.clear()
        self.cache_hits = 0
        logger.info("AI cache cleared")

# Add missing import at the top
import re

# Example patterns for reference
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
