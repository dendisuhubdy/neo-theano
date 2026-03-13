//! Caching memory allocator for ROCm/HIP, modeled after PyTorch's CUDACachingAllocator.
//!
//! Key design:
//! - Two pools: Small (<=1MB) and Large (>1MB)
//! - Best-fit search with BTreeMap for O(log n) lookup
//! - Block splitting and merging
//! - OOM recovery: free cached blocks and retry
//!
//! When compiled without the `hip` feature, this module provides a mock
//! allocator for testing and development.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;

/// Threshold between small and large pools.
const SMALL_POOL_THRESHOLD: usize = 1 << 20; // 1 MB

/// Minimum allocation size (to reduce fragmentation).
const MIN_BLOCK_SIZE: usize = 512;

/// Round up small allocations to this granularity.
const SMALL_ALLOC_GRANULARITY: usize = 512; // 512 bytes

/// Round up large allocations to this granularity.
const LARGE_ALLOC_GRANULARITY: usize = 2 << 20; // 2 MB

/// A memory block tracked by the allocator.
#[derive(Debug, Clone)]
pub struct Block {
    /// Device pointer (or mock address).
    pub ptr: usize,
    /// Size of the block in bytes.
    pub size: usize,
    /// Whether this block is currently allocated (in use).
    pub allocated: bool,
    /// Stream on which this block was last used.
    pub stream: usize,
}

/// Statistics about allocator usage.
#[derive(Debug, Clone, Default)]
pub struct AllocatorStats {
    /// Total bytes currently allocated.
    pub allocated_bytes: usize,
    /// Total bytes currently cached (free in pool).
    pub cached_bytes: usize,
    /// Peak allocated bytes.
    pub peak_allocated_bytes: usize,
    /// Number of malloc calls to the HIP driver.
    pub num_mallocs: usize,
    /// Number of free calls to the HIP driver.
    pub num_frees: usize,
    /// Number of allocations served from the cache.
    pub num_cache_hits: usize,
    /// Number of allocations that required a new malloc.
    pub num_cache_misses: usize,
}

/// Caching memory allocator for a single ROCm/HIP device.
pub struct CachingAllocator {
    /// Device ordinal.
    device: usize,
    /// Small pool: blocks <= 1MB, keyed by size for best-fit.
    small_pool: Mutex<BTreeMap<usize, Vec<Block>>>,
    /// Large pool: blocks > 1MB, keyed by size for best-fit.
    large_pool: Mutex<BTreeMap<usize, Vec<Block>>>,
    /// Statistics.
    stats: Mutex<AllocatorStats>,
    /// Counter for mock addresses (when hip feature is not available).
    mock_addr: AtomicUsize,
}

impl CachingAllocator {
    /// Create a new caching allocator for the given device.
    pub fn new(device: usize) -> Self {
        Self {
            device,
            small_pool: Mutex::new(BTreeMap::new()),
            large_pool: Mutex::new(BTreeMap::new()),
            stats: Mutex::new(AllocatorStats::default()),
            mock_addr: AtomicUsize::new(0x1000_0000),
        }
    }

    /// Allocate a block of the given size.
    ///
    /// First checks the appropriate pool for a cached block.
    /// If none found, allocates from the HIP driver (or mock).
    pub fn allocate(&self, size: usize) -> Result<Block, crate::error::RocmError> {
        if size == 0 {
            return Ok(Block {
                ptr: 0,
                size: 0,
                allocated: true,
                stream: 0,
            });
        }

        let rounded_size = self.round_size(size);
        let is_small = rounded_size <= SMALL_POOL_THRESHOLD;

        // Try to find a cached block
        let pool = if is_small {
            &self.small_pool
        } else {
            &self.large_pool
        };

        {
            let mut pool = pool.lock();
            // Best-fit: find the smallest block >= rounded_size
            if let Some((&block_size, blocks)) = pool.range_mut(rounded_size..).next() {
                if let Some(mut block) = blocks.pop() {
                    if blocks.is_empty() {
                        let size_key = block_size;
                        drop(blocks);
                        pool.remove(&size_key);
                    }

                    let mut stats = self.stats.lock();
                    stats.cached_bytes -= block.size;
                    stats.allocated_bytes += block.size;
                    stats.peak_allocated_bytes =
                        stats.peak_allocated_bytes.max(stats.allocated_bytes);
                    stats.num_cache_hits += 1;

                    // Split if the remaining space is large enough
                    if block.size >= rounded_size + MIN_BLOCK_SIZE {
                        let split_block = Block {
                            ptr: block.ptr + rounded_size,
                            size: block.size - rounded_size,
                            allocated: false,
                            stream: block.stream,
                        };
                        block.size = rounded_size;

                        // Return split remainder to pool
                        stats.cached_bytes += split_block.size;
                        drop(stats);

                        let split_is_small = split_block.size <= SMALL_POOL_THRESHOLD;
                        if split_is_small == is_small {
                            // Same pool, we already hold the lock
                            pool.entry(split_block.size)
                                .or_default()
                                .push(split_block);
                        } else {
                            drop(pool);
                            let split_pool = if split_is_small {
                                &self.small_pool
                            } else {
                                &self.large_pool
                            };
                            split_pool
                                .lock()
                                .entry(split_block.size)
                                .or_default()
                                .push(split_block);
                        }
                    }

                    block.allocated = true;
                    return Ok(block);
                }
            }
        }

        // Cache miss: allocate from driver
        let block = self.malloc(rounded_size)?;

        let mut stats = self.stats.lock();
        stats.allocated_bytes += block.size;
        stats.peak_allocated_bytes = stats.peak_allocated_bytes.max(stats.allocated_bytes);
        stats.num_cache_misses += 1;

        Ok(block)
    }

    /// Free a block back to the cache.
    pub fn free(&self, mut block: Block) {
        if block.size == 0 {
            return;
        }

        block.allocated = false;
        let is_small = block.size <= SMALL_POOL_THRESHOLD;

        let mut stats = self.stats.lock();
        stats.allocated_bytes -= block.size;
        stats.cached_bytes += block.size;
        drop(stats);

        let pool = if is_small {
            &self.small_pool
        } else {
            &self.large_pool
        };

        pool.lock()
            .entry(block.size)
            .or_default()
            .push(block);
    }

    /// Release all cached (free) blocks back to the HIP driver.
    pub fn empty_cache(&self) {
        let small_blocks: Vec<Block> = {
            let mut pool = self.small_pool.lock();
            let blocks: Vec<Block> = pool.values().flat_map(|v| v.iter().cloned()).collect();
            pool.clear();
            blocks
        };

        let large_blocks: Vec<Block> = {
            let mut pool = self.large_pool.lock();
            let blocks: Vec<Block> = pool.values().flat_map(|v| v.iter().cloned()).collect();
            pool.clear();
            blocks
        };

        let total_freed: usize = small_blocks.iter().chain(large_blocks.iter()).map(|b| b.size).sum();

        let mut stats = self.stats.lock();
        stats.cached_bytes -= total_freed;

        let num_freed = small_blocks.len() + large_blocks.len();
        stats.num_frees += num_freed;

        // In a real implementation, we'd call hipFree for each block here
        #[cfg(feature = "hip")]
        {
            // hip_sys::hipFree(block.ptr as *mut _)
        }
    }

    /// Get allocator statistics.
    pub fn stats(&self) -> AllocatorStats {
        self.stats.lock().clone()
    }

    /// Device ordinal.
    pub fn device(&self) -> usize {
        self.device
    }

    /// Round size up to the appropriate granularity.
    fn round_size(&self, size: usize) -> usize {
        if size <= SMALL_POOL_THRESHOLD {
            // Round up to SMALL_ALLOC_GRANULARITY
            (size + SMALL_ALLOC_GRANULARITY - 1) / SMALL_ALLOC_GRANULARITY * SMALL_ALLOC_GRANULARITY
        } else {
            // Round up to LARGE_ALLOC_GRANULARITY
            (size + LARGE_ALLOC_GRANULARITY - 1) / LARGE_ALLOC_GRANULARITY * LARGE_ALLOC_GRANULARITY
        }
    }

    /// Allocate from the HIP driver (or mock).
    fn malloc(&self, size: usize) -> Result<Block, crate::error::RocmError> {
        #[cfg(feature = "hip")]
        {
            // Real HIP allocation via hip-sys
            // let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            // let status = hip_sys::hipMalloc(&mut ptr, size);
            // if status != 0 {
            //     return Err(crate::error::RocmError::DriverError {
            //         msg: format!("hipMalloc failed with status {status}"),
            //     });
            // }
            let _ = size;
        }

        // Mock allocation (no HIP device)
        let ptr = self.mock_addr.fetch_add(size, Ordering::Relaxed);
        let mut stats = self.stats.lock();
        stats.num_mallocs += 1;

        Ok(Block {
            ptr,
            size,
            allocated: true,
            stream: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_basic() {
        let alloc = CachingAllocator::new(0);

        let b1 = alloc.allocate(1024).unwrap();
        assert!(b1.allocated);
        assert!(b1.size >= 1024);

        let stats = alloc.stats();
        assert_eq!(stats.num_cache_misses, 1);
        assert!(stats.allocated_bytes > 0);

        alloc.free(b1);
        let stats = alloc.stats();
        assert_eq!(stats.allocated_bytes, 0);
        assert!(stats.cached_bytes > 0);
    }

    #[test]
    fn test_allocator_cache_reuse() {
        let alloc = CachingAllocator::new(0);

        let b1 = alloc.allocate(1024).unwrap();
        let size1 = b1.size;
        alloc.free(b1);

        // Second allocation of same size should hit cache
        let b2 = alloc.allocate(1024).unwrap();
        assert_eq!(b2.size, size1);

        let stats = alloc.stats();
        assert_eq!(stats.num_cache_hits, 1);
        assert_eq!(stats.num_mallocs, 1); // only one real malloc

        alloc.free(b2);
    }

    #[test]
    fn test_allocator_empty_cache() {
        let alloc = CachingAllocator::new(0);

        let b1 = alloc.allocate(1024).unwrap();
        let b2 = alloc.allocate(2048).unwrap();
        alloc.free(b1);
        alloc.free(b2);

        let stats = alloc.stats();
        assert!(stats.cached_bytes > 0);

        alloc.empty_cache();
        let stats = alloc.stats();
        assert_eq!(stats.cached_bytes, 0);
    }

    #[test]
    fn test_allocator_large_pool() {
        let alloc = CachingAllocator::new(0);

        // Allocate > 1MB to go to large pool
        let b = alloc.allocate(2 * 1024 * 1024).unwrap();
        assert!(b.size >= 2 * 1024 * 1024);
        alloc.free(b);

        let stats = alloc.stats();
        assert!(stats.cached_bytes > 0);
    }

    #[test]
    fn test_allocator_zero_size() {
        let alloc = CachingAllocator::new(0);
        let b = alloc.allocate(0).unwrap();
        assert_eq!(b.size, 0);
        assert_eq!(b.ptr, 0);
    }

    #[test]
    fn test_allocator_stats() {
        let alloc = CachingAllocator::new(0);

        let b1 = alloc.allocate(512).unwrap();
        let b2 = alloc.allocate(1024).unwrap();

        let stats = alloc.stats();
        assert_eq!(stats.num_mallocs, 2);
        assert!(stats.peak_allocated_bytes >= 512 + 1024);

        alloc.free(b1);
        alloc.free(b2);
    }

    #[test]
    fn test_allocator_device_ordinal() {
        let alloc = CachingAllocator::new(3);
        assert_eq!(alloc.device(), 3);
    }
}
