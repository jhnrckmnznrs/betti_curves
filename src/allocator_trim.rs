//! Narrow, audited wrapper around glibc `malloc_trim(0)`.
//!
//! This module is the only place in the crate allowed to use `unsafe` code.
//! The operation is opt-in and is used only at an explicit phase boundary,
//! after local H2 slab state has been dropped and before global reduction
//! allocates its union-find state.

use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrimOutcome {
    /// glibc returns non-zero when it released memory back to the OS.
    pub(crate) released: bool,
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
unsafe extern "C" {
    #[link_name = "malloc_trim"]
    fn c_malloc_trim(pad: usize) -> i32;
}

pub(crate) fn trim_process_heap() -> Result<TrimOutcome> {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: `malloc_trim` takes a plain size_t and has no pointer
        // preconditions. Calling it at this phase boundary is equivalent to
        // the documented C call `malloc_trim(0)`. It may release unused heap
        // pages but does not invalidate live allocations.
        let result = unsafe { c_malloc_trim(0) };
        Ok(TrimOutcome {
            released: result != 0,
        })
    }

    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    {
        anyhow::bail!(
            "--phase-trim before-reduce requires Linux with the GNU/glibc target environment"
        )
    }
}
