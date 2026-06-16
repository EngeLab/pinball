# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 David van Bruggen
# Part of Pinball — a hierarchical graph transformer for efficient long-context sequence modeling.
# Licensed under the GNU GPL v3.0 (see LICENSE). Please cite via CITATION.cff.
import math
import logging

logger = logging.getLogger(__name__)

class CycleWarmupScheduler:
    """
    Scheduler for gradually warming up the number of refinement cycles during training.
    
    This implements a curriculum learning approach where model complexity
    increases gradually over epochs.
    """
    
    def __init__(self, min_cycles, max_cycles, warmup_epochs, mode='linear'):
        """
        Initialize the cycle warmup scheduler.
        
        Args:
            min_cycles: Minimum number of cycles to start with
            max_cycles: Maximum number of cycles to reach after warmup
            warmup_epochs: Number of epochs for warmup
            mode: Warmup schedule mode ('linear', 'cosine', or 'step')
        """
        self.min_cycles = min_cycles
        self.max_cycles = max_cycles
        self.warmup_epochs = warmup_epochs
        self.mode = mode
        self.current_epoch = 0
        
        logger.info(f"Initialized CycleWarmupScheduler with {min_cycles} -> {max_cycles} cycles "
                   f"over {warmup_epochs} epochs using {mode} schedule")
    
    def step(self):
        """Increment the current epoch count."""
        self.current_epoch += 1
        current = self.get_current_cycles()
        logger.info(f"Cycle warmup epoch {self.current_epoch}/{self.warmup_epochs}, "
                   f"cycles: {current}")
        return current
    
    def get_current_cycles(self):
        """
        Get the current number of cycles based on warmup schedule.
        
        Returns:
            current_cycles: Integer number of cycles for current epoch
        """
        if self.current_epoch >= self.warmup_epochs:
            return self.max_cycles
        
        progress = self.current_epoch / self.warmup_epochs
        
        if self.mode == 'linear':
            # Linear warmup from min_cycles to max_cycles
            cycles = self.min_cycles + (self.max_cycles - self.min_cycles) * progress
        elif self.mode == 'cosine':
            # Cosine warmup (slower at start and end, faster in middle)
            cycles = self.min_cycles + (self.max_cycles - self.min_cycles) * (1 - math.cos(progress * math.pi)) / 2
        elif self.mode == 'step':
            # Step increase (thirds)
            if progress < 0.33:
                cycles = self.min_cycles
            elif progress < 0.67:
                cycles = (self.min_cycles + self.max_cycles) // 2
            else:
                cycles = self.max_cycles
        else:
            # Default to linear
            cycles = self.min_cycles + (self.max_cycles - self.min_cycles) * progress
        
        return max(1, int(cycles))


class MultipleCycleScheduler:
    """
    Scheduler for managing multiple cycle types (refinement, L0, and internal).
    
    This scheduler coordinates the warmup of all cycle types.
    """
    
    def __init__(
        self,
        refinement_min=1,
        refinement_max=3,
        l0_min=2,
        l0_max=8,
        internal_min=[1, 1, 1],
        internal_max=[4, 3, 2],
        warmup_epochs=3,
        mode='linear'
    ):
        """
        Initialize multiple cycle schedulers.
        
        Args:
            refinement_min: Minimum refinement cycles
            refinement_max: Maximum refinement cycles
            l0_min: Minimum L0 cycles
            l0_max: Maximum L0 cycles
            internal_min: Minimum internal cycles per level
            internal_max: Maximum internal cycles per level
            warmup_epochs: Number of epochs for warmup
            mode: Warmup schedule mode
        """
        self.refinement_scheduler = CycleWarmupScheduler(
            refinement_min, refinement_max, warmup_epochs, mode
        )
        
        self.l0_scheduler = CycleWarmupScheduler(
            l0_min, l0_max, warmup_epochs, mode
        )
        
        # Ensure internal min/max lists are of same length
        min_len = min(len(internal_min), len(internal_max))
        self.internal_schedulers = []
        
        for i in range(min_len):
            self.internal_schedulers.append(
                CycleWarmupScheduler(
                    internal_min[i], internal_max[i], warmup_epochs, mode
                )
            )
        
        self.current_epoch = 0
        
        logger.info(f"Initialized MultipleCycleScheduler with warmup over {warmup_epochs} epochs")
    
    def step(self):
        """
        Step all cycle schedulers and return current cycle counts.
        
        Returns:
            cycles: Dictionary with current cycle counts
        """
        self.current_epoch += 1
        
        # Step all schedulers
        refinement_cycles = self.refinement_scheduler.step()
        l0_cycles = self.l0_scheduler.step()
        
        internal_cycles = []
        for scheduler in self.internal_schedulers:
            internal_cycles.append(scheduler.step())
        
        return {
            'refinement': refinement_cycles,
            'l0': l0_cycles,
            'internal': internal_cycles
        }
    
    def get_current_cycles(self):
        """
        Get current cycle counts without stepping.
        
        Returns:
            cycles: Dictionary with current cycle counts
        """
        refinement_cycles = self.refinement_scheduler.get_current_cycles()
        l0_cycles = self.l0_scheduler.get_current_cycles()
        
        internal_cycles = []
        for scheduler in self.internal_schedulers:
            internal_cycles.append(scheduler.get_current_cycles())
        
        return {
            'refinement': refinement_cycles,
            'l0': l0_cycles,
            'internal': internal_cycles
        }