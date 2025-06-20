"""
ai_handler.py - Enhanced AI Handler with Improved Accuracy
Multi-provider support with advanced pattern matching and validation
Optimized for Vive Health medical device SKUs
"""

import os
import logging
import streamlit as st
import requests
import json
from typing import Optional, Dict, Any, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import time
import re
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)

class AIHandler:
    """Enhanced AI handler with improved accuracy and validation"""
    
    def __init__(self, preferred_provider='auto'):
        """Initialize with enhanced pattern matching"""
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
        
        # Enhanced caching
        self.sku_cache = {}
        self.reason_cache = {}
        self.cache_hits = 0
        
        # Known SKUs for validation
        self.known_skus = set()
        self.sku_prefixes = set()
        
        # Batch processing settings
        self.batch_size = 10
        self.max_workers = 3
        
        # Compile patterns
        self._compile_enhanced_patterns()
        
        # Log status
        self._log_provider_status()
    
    def _compile_enhanced_patterns(self):
        """Compile comprehensive patterns for Vive Health products"""
        # Enhanced SKU patterns with better accuracy
        self.sku_patterns = [
            # Vive Health specific patterns
            re.compile(r'\b(LVA\d{4}(?:-[A-Z0-9]+)?)\b', re.IGNORECASE),  # LVA series
            re.compile(r'\b(SUP\d{4}(?:-[A-Z0-9]+)?)\b', re.IGNORECASE),  # Support series
            re.compile(r'\b(MOB\d{4}(?:-[A-Z0-9]+)?)\b', re.IGNORECASE),  # Mobility series
            re.compile(r'\b(RHB\d{4}(?:-[A-Z0-9]+)?)\b', re.IGNORECASE),  # Rehab series
            re.compile(r'\b(INS\d{4}(?:-[A-Z0-9]+)?)\b', re.IGNORECASE),  # Insole series
            re.compile(r'\b(BAT\d{4}(?:-[A-Z0-9]+)?)\b', re.IGNORECASE),  # Bath series
            re.compile(r'\b(BRA\d{4}(?:-[A-Z0-9]+)?)\b', re.IGNORECASE),  # Brace series
            re.compile(r'\b(CAN\d{4}(?:-[A-Z0-9]+)?)\b', re.IGNORECASE),  # Cane series
            re.compile(r'\b(WAL\d{4}(?:-[A-Z0-9]+)?)\b', re.IGNORECASE),  # Walker series
            re.compile(r'\b(WHL\d{4}(?:-[A-Z0-9]+)?)\b', re.IGNORECASE),  # Wheelchair series
            
            # Generic patterns with validation
            re.compile(r'\b([A-Z]{3}\d{4}(?:-[A-Z0-9]+)?)\b'),  # XXX####-SUFFIX
            re.compile(r'(?:SKU|Item)[\s:#]+([A-Z0-9\-]+)', re.IGNORECASE),
            re.compile(r'(?:Model|Part)[\s:#]+([A-Z0-9\-]+)', re.IGNORECASE),
            
            # With common suffixes
            re.compile(r'\b([A-Z]{3}\d{4}-(?:UPC|BLK|GRY|BLU|RED|SML|MED|LRG|XL))\b', re.IGNORECASE),
        ]
        
        # Extract prefixes for validation
        self.sku_prefixes = {
            'LVA', 'SUP', 'MOB', 'RHB', 'INS', 'BAT', 'BRA', 'CAN', 'WAL', 'WHL',
            'CSH', 'ELC', 'MED', 'ORT', 'PAD', 'SLG', 'SPT', 'THR', 'TOI'
        }
        
        # Enhanced reason patterns with medical device focus
        self.reason_patterns = {
            'Defective/Quality': {
                'keywords': ['defect', 'broken', 'break', 'not working', 'stopped working', 
                            'malfunction', 'faulty', 'failed', 'failure', 'poor quality',
                            'fell apart', 'came apart', 'ripped', 'torn', 'cracked', 'split'],
                'priority': 1
            },
            'Size/Fit Issue': {
                'keywords': ['too small', 'too large', 'too big', 'wrong size', "doesn't fit",
                            'size issue', 'fit issue', 'too tight', 'too loose', 'incorrect size'],
                'priority': 2
            },
            'Comfort Issue': {
                'keywords': ['uncomfortable', 'not comfortable', 'hurts', 'painful', 'discomfort',
                            'irritating', 'irritation', 'pressure', 'too hard', 'too soft'],
                'priority': 3
            },
            'Wrong Item': {
                'keywords': ['wrong item', 'wrong product', 'incorrect', 'not what ordered',
                            'different than', 'not as described', 'wrong model', 'wrong color'],
                'priority': 4
            },
            'Missing Parts': {
                'keywords': ['missing', 'incomplete', 'not all parts', 'parts missing',
                            'missing pieces', 'missing component', 'missing accessory'],
                'priority': 5
            },
            'Not Compatible': {
                'keywords': ['not compatible', 'incompatible', "doesn't work with", "won't fit",
                            "doesn't fit toilet", "doesn't fit chair", 'wrong type'],
                'priority': 6
            },
            'Not Needed': {
                'keywords': ['no longer need', "don't need", 'changed mind', 'ordered by mistake',
                            'duplicate order', 'accidentally ordered', 'not needed anymore'],
                'priority': 7
            },
            'Damaged in Shipping': {
                'keywords': ['damaged in shipping', 'arrived damaged', 'shipping damage',
                            'damaged box', 'damaged package', 'crushed in transit'],
                'priority': 8
            },
            'Difficulty Using': {
                'keywords': ['too difficult', 'hard to use', 'complicated', 'confusing',
                            'difficult to assemble', "can't figure out", 'instructions unclear'],
                'priority': 9
            }
        }
    
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
        
        # Try environment variables
        if provider == 'claude':
            return os.getenv('ANTHROPIC_API_KEY')
        elif provider == 'openai':
            return os.getenv('OPENAI_API_KEY')
        
        return None
    
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
    
    def extract_sku_enhanced(self, description: str, context: Dict[str, Any] = None) -> Tuple[str, float]:
        """
        Enhanced SKU extraction with validation and confidence scoring
        
        Returns:
            Tuple of (sku, confidence_score)
        """
        if not description or pd.isna(description):
            return "", 0.0
        
        description = str(description)
        
        # Check cache first
        cache_key = self._get_cache_key(description)
        if cache_key in self.sku_cache:
            self.cache_hits += 1
            cached = self.sku_cache[cache_key]
            return cached if isinstance(cached, tuple) else (cached, 0.9)
        
        # Try pattern matching with validation
        best_sku = ""
        best_confidence = 0.0
        
        for pattern in self.sku_patterns:
            matches = pattern.findall(description)
            for match in matches:
                sku = match.upper() if isinstance(match, str) else match[0].upper()
                confidence = self._validate_sku(sku)
                
                if confidence > best_confidence:
                    best_sku = sku
                    best_confidence = confidence
        
        # If high confidence from pattern, return it
        if best_confidence >= 0.8:
            self.sku_cache[cache_key] = (best_sku, best_confidence)
            return best_sku, best_confidence
        
        # Try AI extraction for better accuracy
        if self._select_provider():
            ai_sku = self._extract_sku_with_ai(description, context)
            if ai_sku and ai_sku != "NOT_FOUND":
                ai_confidence = self._validate_sku(ai_sku)
                if ai_confidence > best_confidence:
                    best_sku = ai_sku
                    best_confidence = ai_confidence
        
        # Cache and return result
        result = (best_sku, best_confidence)
        self.sku_cache[cache_key] = result
        return result
    
    def _validate_sku(self, sku: str) -> float:
        """Validate SKU and return confidence score"""
        if not sku:
            return 0.0
        
        confidence = 0.0
        
        # Check if in known SKUs (highest confidence)
        if sku in self.known_skus:
            return 1.0
        
        # Check prefix
        prefix = sku[:3] if len(sku) >= 3 else ""
        if prefix in self.sku_prefixes:
            confidence += 0.5
        
        # Check format (XXX#### or XXX####-SUFFIX)
        if re.match(r'^[A-Z]{3}\d{4}(?:-[A-Z0-9]+)?$', sku):
            confidence += 0.3
        
        # Length check
        if 7 <= len(sku) <= 15:
            confidence += 0.1
        
        # Has both letters and numbers
        if any(c.isalpha() for c in sku) and any(c.isdigit() for c in sku):
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def extract_reason_enhanced(self, description: str, context: Dict[str, Any] = None) -> Tuple[str, float]:
        """
        Enhanced reason extraction with medical device focus
        
        Returns:
            Tuple of (reason, confidence_score)
        """
        if not description or pd.isna(description):
            return "Other", 0.1
        
        description = str(description)
        
        # Check cache
        cache_key = self._get_cache_key(description)
        if cache_key in self.reason_cache:
            self.cache_hits += 1
            cached = self.reason_cache[cache_key]
            return cached if isinstance(cached, tuple) else (cached, 0.9)
        
        # Pattern-based extraction with priority
        description_lower = description.lower()
        best_reason = "Other"
        best_priority = 999
        best_confidence = 0.1
        
        for reason, data in self.reason_patterns.items():
            keywords_found = sum(1 for keyword in data['keywords'] if keyword in description_lower)
            
            if keywords_found > 0:
                confidence = min(0.6 + (keywords_found * 0.2), 0.95)
                
                if data['priority'] < best_priority or (data['priority'] == best_priority and confidence > best_confidence):
                    best_reason = reason
                    best_priority = data['priority']
                    best_confidence = confidence
        
        # Use AI for ambiguous cases
        if best_confidence < 0.7 and self._select_provider():
            ai_reason = self._extract_reason_with_ai(description, context)
            if ai_reason and ai_reason != "Other":
                # Validate AI response against our categories
                ai_reason_normalized = self._normalize_reason(ai_reason)
                if ai_reason_normalized != "Other":
                    best_reason = ai_reason_normalized
                    best_confidence = 0.8
        
        # Cache and return
        result = (best_reason, best_confidence)
        self.reason_cache[cache_key] = result
        return result
    
    def _normalize_reason(self, reason: str) -> str:
        """Normalize AI reason to standard categories"""
        reason_lower = reason.lower()
        
        # Map variations to standard categories
        mapping = {
            'defective': 'Defective/Quality',
            'broken': 'Defective/Quality',
            'quality': 'Defective/Quality',
            'size': 'Size/Fit Issue',
            'fit': 'Size/Fit Issue',
            'comfort': 'Comfort Issue',
            'uncomfortable': 'Comfort Issue',
            'wrong': 'Wrong Item',
            'incorrect': 'Wrong Item',
            'missing': 'Missing Parts',
            'incomplete': 'Missing Parts',
            'compatible': 'Not Compatible',
            'compatibility': 'Not Compatible',
            'needed': 'Not Needed',
            'shipping': 'Damaged in Shipping',
            'difficult': 'Difficulty Using',
            'hard to use': 'Difficulty Using'
        }
        
        for key, category in mapping.items():
            if key in reason_lower:
                return category
        
        # Check against our categories
        for category in self.reason_patterns.keys():
            if category.lower() in reason_lower or reason_lower in category.lower():
                return category
        
        return "Other"
    
    def _extract_sku_with_ai(self, description: str, context: Dict[str, Any] = None) -> str:
        """Extract SKU using AI with enhanced context"""
        prompt = f"""Extract the SKU (product code) from this medical device support ticket.

Vive Health SKU patterns:
- LVA#### (Lavatory/Toilet products)
- SUP#### (Support products)
- MOB#### (Mobility aids)
- RHB#### (Rehab equipment)
- Common suffixes: -UPC, -BLK, -GRY, etc.

Look for patterns like XXX#### or XXX####-SUFFIX.

Description: {description[:600]}

Return ONLY the SKU code, nothing else. If no SKU found, return NOT_FOUND.

SKU:"""
        
        provider = self._select_provider()
        if not provider:
            return ""
        
        try:
            if provider == 'claude':
                result = self._call_claude(prompt, max_tokens=20)
            else:
                result = self._call_openai(prompt, max_tokens=20)
            
            # Clean and validate result
            if result:
                result = result.strip().upper()
                if result != "NOT_FOUND" and self._validate_sku(result) > 0.5:
                    return result
        except Exception as e:
            logger.error(f"AI SKU extraction error: {e}")
        
        return ""
    
    def _extract_reason_with_ai(self, description: str, context: Dict[str, Any] = None) -> str:
        """Extract reason using AI with medical device categories"""
        prompt = f"""Categorize this medical device return into ONE category:

Categories:
- Defective/Quality (broken, not working, poor quality)
- Size/Fit Issue (too small/large, doesn't fit)
- Comfort Issue (uncomfortable, painful, irritating)
- Wrong Item (incorrect product, not as described)
- Missing Parts (incomplete, missing components)
- Not Compatible (doesn't fit equipment)
- Not Needed (changed mind, ordered by mistake)
- Damaged in Shipping (arrived damaged)
- Difficulty Using (hard to use, complicated)
- Other

Description: {description[:600]}

Return ONLY the category name, nothing else.

Category:"""
        
        provider = self._select_provider()
        if not provider:
            return "Other"
        
        try:
            if provider == 'claude':
                result = self._call_claude(prompt, max_tokens=30)
            else:
                result = self._call_openai(prompt, max_tokens=30)
            
            if result:
                return result.strip()
        except Exception as e:
            logger.error(f"AI reason extraction error: {e}")
        
        return "Other"
    
    def add_known_sku(self, sku: str):
        """Add a known valid SKU for validation"""
        if sku and len(sku) >= 6:
            self.known_skus.add(sku.upper())
            # Also add prefix
            prefix = sku[:3].upper()
            if prefix.isalpha():
                self.sku_prefixes.add(prefix)
    
    def extract_sku(self, description: str) -> str:
        """Legacy method for compatibility"""
        sku, confidence = self.extract_sku_enhanced(description)
        return sku
    
    def extract_reason(self, description: str) -> str:
        """Legacy method for compatibility"""
        reason, confidence = self.extract_reason_enhanced(description)
        return reason
    
    def batch_extract_enhanced(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch extract with enhanced accuracy"""
        results = []
        
        for item in items:
            description = item.get('description', '')
            
            # Extract SKU with confidence
            sku, sku_confidence = self.extract_sku_enhanced(description, item)
            
            # Extract reason with confidence
            reason, reason_confidence = self.extract_reason_enhanced(description, item)
            
            results.append({
                'index': item.get('index'),
                'sku': sku,
                'sku_confidence': sku_confidence,
                'reason': reason,
                'reason_confidence': reason_confidence,
                'description': description
            })
        
        return results
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get detailed extraction statistics"""
        total_extractions = len(self.sku_cache) + len(self.reason_cache)
        
        # Calculate confidence distribution
        confidence_dist = {'high': 0, 'medium': 0, 'low': 0}
        
        for cached in list(self.sku_cache.values()) + list(self.reason_cache.values()):
            if isinstance(cached, tuple) and len(cached) > 1:
                conf = cached[1]
                if conf >= 0.8:
                    confidence_dist['high'] += 1
                elif conf >= 0.6:
                    confidence_dist['medium'] += 1
                else:
                    confidence_dist['low'] += 1
        
        return {
            'total_extractions': total_extractions,
            'cache_hits': self.cache_hits,
            'known_skus': len(self.known_skus),
            'confidence_distribution': confidence_dist,
            'api_efficiency': (self.cache_hits / max(1, total_extractions)) * 100
        }
    
    # Keep all the existing methods from the previous version...
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def _select_provider(self) -> Optional[str]:
        """Select best available provider"""
        available = self.get_available_providers()
        
        if not available:
            return None
        
        if self.preferred_provider == 'auto':
            if 'openai' in available:
                return 'openai'
            elif 'claude' in available:
                return 'claude'
        elif self.preferred_provider in available:
            return self.preferred_provider
        else:
            return available[0]
        
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
        """Check if provider is rate limited"""
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
    
    def _call_claude(self, prompt: str, max_tokens: int = 50) -> str:
        """Call Claude API"""
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
                    self.api_errors['claude'] += 1
                    retry_after = int(response.headers.get('retry-after', 60))
                    self._set_rate_limit('claude', retry_after)
                    break
                else:
                    self.api_errors['claude'] += 1
                    logger.error(f"Claude API error {response.status_code}")
                    break
            except Exception as e:
                self.api_errors['claude'] += 1
                logger.error(f"Claude API call failed: {e}")
                break
        
        return ""
    
    def _call_openai(self, prompt: str, max_tokens: int = 50) -> str:
        """Call OpenAI API"""
        if not self.openai_key or self._is_rate_limited('openai'):
            return ""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are an expert at extracting product SKUs and categorizing return reasons for medical devices."},
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
                    self.api_errors['openai'] += 1
                    retry_after = int(response.headers.get('retry-after', 60))
                    self._set_rate_limit('openai', retry_after)
                    break
                else:
                    self.api_errors['openai'] += 1
                    logger.error(f"OpenAI API error {response.status_code}")
                    break
            except Exception as e:
                self.api_errors['openai'] += 1
                logger.error(f"OpenAI API call failed: {e}")
                break
        
        return ""
    
    def get_status(self) -> Dict[str, Any]:
        """Get handler status"""
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
            },
            'extraction_stats': self.get_extraction_statistics()
        }
    
    def set_preferred_provider(self, provider: str):
        """Update preferred provider"""
        if provider in ['claude', 'openai', 'auto']:
            self.preferred_provider = provider
            logger.info(f"AI provider preference set to: {provider}")
    
    def clear_cache(self):
        """Clear the cache"""
        self.sku_cache.clear()
        self.reason_cache.clear()
        self.cache_hits = 0
        logger.info("AI cache cleared")
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
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

# Medical device specific patterns
VIVE_SKU_PATTERNS = {
    'toilet_safety': ['LVA', 'TOI'],
    'mobility': ['MOB', 'WAL', 'CAN', 'WHL'],
    'support': ['SUP', 'CSH', 'PAD'],
    'rehab': ['RHB', 'THR', 'ELC'],
    'bath_safety': ['BAT', 'SHW'],
    'braces': ['BRA', 'ORT', 'SPT'],
    'insoles': ['INS', 'SHO'],
    'medical': ['MED', 'HLT']
}

RETURN_CATEGORIES_MEDICAL = [
    'Defective/Quality',
    'Size/Fit Issue',
    'Comfort Issue',
    'Wrong Item',
    'Missing Parts',
    'Not Compatible',
    'Not Needed',
    'Damaged in Shipping',
    'Difficulty Using',
    'Other'
]
