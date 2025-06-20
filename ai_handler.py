"""
ai_handler.py - Multi-Provider AI Handler Module for B2B Report Analyzer
Supports Claude and OpenAI with automatic fallback and rate limit handling
User can select preferred provider or use auto-failover
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
import re

logger = logging.getLogger(__name__)

class AIHandler:
    """Multi-provider AI handler with rate limit handling and fallback"""
    
    def __init__(self, preferred_provider='auto'):
        """
        Initialize AI handler with provider preference
        
        Args:
            preferred_provider: 'claude', 'openai', or 'auto' (tries both)
        """
        self.claude_key = self._get_api_key('claude')
        self.openai_key = self._get_api_key('openai')
        self.preferred_provider = preferred_provider
        
        # Track rate limits
        self.claude_rate_limited_until = 0
        self.openai_rate_limited_until = 0
        
        # API call tracking
        self.api_calls = {'claude': 0, 'openai': 0, 'total': 0}
        self.api_errors = {'claude': 0, 'openai': 0}
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
        
        # Log available providers
        self._log_provider_status()
    
    def _log_provider_status(self):
        """Log which providers are available"""
        providers = []
        if self.claude_key:
            providers.append("Claude")
        if self.openai_key:
            providers.append("OpenAI")
        
        if providers:
            logger.info(f"AI Providers available: {', '.join(providers)}")
            logger.info(f"Preferred provider: {self.preferred_provider}")
        else:
            logger.warning("No AI providers configured")
    
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
        """Get API key from Streamlit secrets or environment"""
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
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        providers = []
        if self.claude_key and not self._is_rate_limited('claude'):
            providers.append('claude')
        if self.openai_key and not self._is_rate_limited('openai'):
            providers.append('openai')
        return providers
    
    def _is_rate_limited(self, provider: str) -> bool:
        """Check if provider is currently rate limited"""
        current_time = time.time()
        if provider == 'claude':
            return current_time < self.claude_rate_limited_until
        elif provider == 'openai':
            return current_time < self.openai_rate_limited_until
        return False
    
    def _set_rate_limit(self, provider: str, duration: int = 60):
        """Set rate limit for provider"""
        current_time = time.time()
        if provider == 'claude':
            self.claude_rate_limited_until = current_time + duration
            logger.warning(f"Claude rate limited for {duration} seconds")
        elif provider == 'openai':
            self.openai_rate_limited_until = current_time + duration
            logger.warning(f"OpenAI rate limited for {duration} seconds")
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed AI handler status"""
        available_providers = self.get_available_providers()
        
        # Determine active provider
        if self.preferred_provider == 'auto':
            active_provider = available_providers[0] if available_providers else 'none'
        elif self.preferred_provider in available_providers:
            active_provider = self.preferred_provider
        else:
            active_provider = available_providers[0] if available_providers else 'none'
        
        return {
            'available': len(available_providers) > 0,
            'providers': available_providers,
            'active_provider': active_provider,
            'preferred_provider': self.preferred_provider,
            'api_calls': self.api_calls,
            'api_errors': self.api_errors,
            'cache_hits': self.cache_hits,
            'cache_size': len(self.sku_cache) + len(self.reason_cache),
            'last_error': self.last_error,
            'rate_limits': {
                'claude': self._is_rate_limited('claude'),
                'openai': self._is_rate_limited('openai')
            }
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def _select_provider(self) -> Optional[str]:
        """Select best available provider based on preference and availability"""
        available = self.get_available_providers()
        
        if not available:
            return None
        
        if self.preferred_provider == 'auto':
            # Auto mode: prefer OpenAI for speed, Claude for quality
            if 'openai' in available:
                return 'openai'
            elif 'claude' in available:
                return 'claude'
        elif self.preferred_provider in available:
            return self.preferred_provider
        else:
            # Fallback to any available
            return available[0]
        
        return None
    
    def extract_sku(self, description: str) -> str:
        """Extract SKU with intelligent provider selection"""
        if not description or pd.isna(description):
            return ""
        
        description = str(description)
        
        # Check cache first
        cache_key = self._get_cache_key(description)
        if cache_key in self.sku_cache:
            self.cache_hits += 1
            return self.sku_cache[cache_key]
        
        # Try pattern matching first (fastest)
        for pattern in self.sku_patterns:
            match = pattern.search(description)
            if match:
                sku = match.group(1).upper()
                self.sku_cache[cache_key] = sku
                return sku
        
        # Use AI with provider selection
        provider = self._select_provider()
        if not provider:
            self.sku_cache[cache_key] = ""
            return ""
        
        prompt = f"""Extract the SKU (product code) from this support ticket description. 
Common SKU patterns include: LVA1004-UPC, SUP1001, MOB2003, RHB3002, or similar alphanumeric codes.
If no SKU is found, respond with "NOT_FOUND".

Description: {description[:500]}

SKU:"""
        
        try:
            if provider == 'claude':
                result = self._call_claude(prompt, max_tokens=20)
            else:  # openai
                result = self._call_openai(prompt, max_tokens=20)
            
            if result and result != "NOT_FOUND":
                self.sku_cache[cache_key] = result
                return result
                
        except Exception as e:
            logger.error(f"AI SKU extraction error ({provider}): {e}")
            self.last_error = str(e)
            
            # Try fallback provider
            fallback = 'openai' if provider == 'claude' else 'claude'
            if fallback in self.get_available_providers():
                try:
                    if fallback == 'claude':
                        result = self._call_claude(prompt, max_tokens=20)
                    else:
                        result = self._call_openai(prompt, max_tokens=20)
                    
                    if result and result != "NOT_FOUND":
                        self.sku_cache[cache_key] = result
                        return result
                except:
                    pass
        
        self.sku_cache[cache_key] = ""
        return ""
    
    def extract_reason(self, description: str) -> str:
        """Extract reason with intelligent provider selection"""
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
            'Defective': ['defect', 'broken', 'not working', 'malfunction'],
            'Wrong Item': ['wrong', 'incorrect', 'not what ordered'],
            'Not Needed': ['no longer need', "don't need", 'changed mind'],
            'Quality Issue': ['poor quality', 'cheap', 'flimsy'],
            'Size Issue': ['too small', 'too large', 'wrong size'],
            'Missing Parts': ['missing', 'incomplete', 'not all parts'],
        }
        
        for reason, keywords in quick_matches.items():
            if any(keyword in description_lower for keyword in keywords):
                self.reason_cache[cache_key] = reason
                return reason
        
        # Use AI with provider selection
        provider = self._select_provider()
        if not provider:
            self.reason_cache[cache_key] = "Other"
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

Description: {description[:500]}

Reason:"""
        
        try:
            if provider == 'claude':
                result = self._call_claude(prompt, max_tokens=30)
            else:  # openai
                result = self._call_openai(prompt, max_tokens=30)
            
            if result:
                self.reason_cache[cache_key] = result
                return result
                
        except Exception as e:
            logger.error(f"AI reason extraction error ({provider}): {e}")
            self.last_error = str(e)
            
            # Try fallback provider
            fallback = 'openai' if provider == 'claude' else 'claude'
            if fallback in self.get_available_providers():
                try:
                    if fallback == 'claude':
                        result = self._call_claude(prompt, max_tokens=30)
                    else:
                        result = self._call_openai(prompt, max_tokens=30)
                    
                    if result:
                        self.reason_cache[cache_key] = result
                        return result
                except:
                    pass
        
        self.reason_cache[cache_key] = "Other"
        return "Other"
    
    def _call_claude(self, prompt: str, max_tokens: int = 50) -> str:
        """Call Claude API with rate limit handling"""
        if not self.claude_key or self._is_rate_limited('claude'):
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
        
        for attempt in range(2):
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.api_calls['claude'] += 1
                    self.api_calls['total'] += 1
                    result = response.json()
                    return result['content'][0]['text'].strip()
                    
                elif response.status_code == 429:
                    # Rate limited
                    self.api_errors['claude'] += 1
                    retry_after = int(response.headers.get('retry-after', 60))
                    self._set_rate_limit('claude', retry_after)
                    logger.warning(f"Claude rate limited. Retry after {retry_after}s")
                    break
                    
                else:
                    self.api_errors['claude'] += 1
                    logger.error(f"Claude API error {response.status_code}: {response.text}")
                    break
                    
            except requests.Timeout:
                if attempt < 1:
                    logger.warning("Claude API timeout, retrying...")
                    continue
                else:
                    self.api_errors['claude'] += 1
                    break
                    
            except Exception as e:
                self.api_errors['claude'] += 1
                logger.error(f"Claude API call failed: {e}")
                break
        
        return ""
    
    def _call_openai(self, prompt: str, max_tokens: int = 50) -> str:
        """Call OpenAI API with rate limit handling"""
        if not self.openai_key or self._is_rate_limited('openai'):
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
        
        for attempt in range(2):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=5
                )
                
                if response.status_code == 200:
                    self.api_calls['openai'] += 1
                    self.api_calls['total'] += 1
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                    
                elif response.status_code == 429:
                    # Rate limited
                    self.api_errors['openai'] += 1
                    retry_after = int(response.headers.get('retry-after', 60))
                    self._set_rate_limit('openai', retry_after)
                    logger.warning(f"OpenAI rate limited. Retry after {retry_after}s")
                    break
                    
                else:
                    self.api_errors['openai'] += 1
                    logger.error(f"OpenAI API error {response.status_code}: {response.text}")
                    break
                    
            except requests.Timeout:
                if attempt < 1:
                    logger.warning("OpenAI API timeout, retrying...")
                    continue
                else:
                    self.api_errors['openai'] += 1
                    break
                    
            except Exception as e:
                self.api_errors['openai'] += 1
                logger.error(f"OpenAI API call failed: {e}")
                break
        
        return ""
    
    def set_preferred_provider(self, provider: str):
        """Update preferred provider"""
        if provider in ['claude', 'openai', 'auto']:
            self.preferred_provider = provider
            logger.info(f"AI provider preference set to: {provider}")
    
    def clear_cache(self):
        """Clear the cache to free memory"""
        self.sku_cache.clear()
        self.reason_cache.clear()
        self.cache_hits = 0
        logger.info("AI cache cleared")
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for each provider"""
        stats = {}
        
        for provider in ['claude', 'openai']:
            success_rate = 0
            if self.api_calls.get(provider, 0) > 0:
                success_rate = ((self.api_calls[provider] - self.api_errors.get(provider, 0)) / 
                               self.api_calls[provider] * 100)
            
            stats[provider] = {
                'calls': self.api_calls.get(provider, 0),
                'errors': self.api_errors.get(provider, 0),
                'success_rate': success_rate,
                'rate_limited': self._is_rate_limited(provider),
                'available': (self.claude_key if provider == 'claude' else self.openai_key) is not None
            }
        
        return stats

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
