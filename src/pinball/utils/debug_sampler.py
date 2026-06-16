# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class DebugSampler:
    """
    A debug-oriented token sampler that logs predictions and helps diagnose generation issues.
    """
    
    def __init__(self, tokenizer, strategy="diverse"):
        """
        Initialize the debug sampler.
        
        Args:
            tokenizer: Tokenizer for decoding tokens
            strategy: Sampling strategy ('diverse', 'focused', 'greedy', 'diagnostic')
        """
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.last_tokens = []
        self.sample_counter = 0
        
    def sample_token(self, logits, input_ids, temperature=1.0, top_k=40, top_p=0.95, repetition_penalty=1.8):
        """
        Sample a token with debugging information.
        
        Args:
            logits: Token logits [batch_size, vocab_size]
            input_ids: Current input ids [batch_size, seq_len]
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Top-p filtering parameter
            repetition_penalty: Repetition penalty
            
        Returns:
            next_token: Sampled token
        """
        batch_size = logits.size(0)
        next_token_logits = logits.clone() / temperature
        
        # Apply repetition penalty
        token_counts = {}
        for token in input_ids[0, -50:]:
            token_id = token.item()
            token_counts[token_id] = token_counts.get(token_id, 0) + 1
        
        # Apply scaled penalty based on frequency and recency
        for i, token_id in enumerate(input_ids[0, -50:].tolist()):
            # Higher penalty for recent tokens
            recency_factor = 1.0 + 0.05 * (50 - i)  # More recent = higher penalty
            count = token_counts.get(token_id, 0)
            if count > 1:
                penalty = repetition_penalty * recency_factor * (1 + 0.2 * min(count-1, 5))
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= penalty
                else:
                    next_token_logits[0, token_id] *= penalty
        
        # Store original logits for logging
        orig_logits = next_token_logits.clone()
        
        # Apply different sampling strategies
        if self.strategy == "diverse":
            # More diverse sampling - higher temperature, lower top_k
            next_token_logits /= temperature * 1.2
            top_k = max(20, top_k - 10)
            top_p = min(0.98, top_p + 0.05)
        elif self.strategy == "focused":
            # More focused sampling - lower temperature, higher top_k
            next_token_logits /= temperature * 0.8
            top_k = min(100, top_k + 10)
            top_p = max(0.9, top_p - 0.05)
        elif self.strategy == "diagnostic":
            # Rotating sampling approach
            self.sample_counter += 1
            if self.sample_counter % 3 == 0:
                logger.info("Using diverse sampling")
                next_token_logits /= temperature * 1.3
                top_k = 20
                top_p = 0.98
            elif self.sample_counter % 3 == 1:
                logger.info("Using focused sampling")
                next_token_logits /= temperature * 0.7
                top_k = 50
                top_p = 0.9
            else:
                logger.info("Using balanced sampling")
                next_token_logits /= temperature
                top_k = 40
                top_p = 0.95
        
        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
            next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            for batch_idx in range(batch_size):
                indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                next_token_logits[batch_idx, indices_to_remove] = float('-inf')
        
        # Sample from the distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Log top token candidates for debugging
        self._log_top_tokens(orig_logits, next_token)
        
        return next_token
    
    def _log_top_tokens(self, logits, sampled_token):
        """Log top token candidates for debugging."""
        # Get top 10 tokens
        top_logits, top_indices = torch.topk(logits[0], 10)
        top_probs = F.softmax(top_logits, dim=-1)
        
        # Decode tokens
        top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices]
        sampled_token_text = self.tokenizer.decode([sampled_token[0].item()])
        
        # Log information
        log_str = "\nTop token candidates:\n"
        for i, (token, prob, idx) in enumerate(zip(top_tokens, top_probs, top_indices)):
            marker = " (SELECTED)" if idx.item() == sampled_token[0].item() else ""
            log_str += f"{i+1}. '{token}' - {prob.item():.4f}{marker}\n"
        
        log_str += f"Selected token: '{sampled_token_text}'\n"
        
        # Keep track of recent tokens
        self.last_tokens.append(sampled_token_text)
        if len(self.last_tokens) > 20:
            self.last_tokens.pop(0)
        
        # Show recent token history
        log_str += f"Recent tokens: {''.join(self.last_tokens[-20:])}\n"
        
        logger.info(log_str)
        
        return top_tokens, top_probs