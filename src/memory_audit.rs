use std::fs;

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ProcessMemorySnapshot {
    pub(crate) rss_kb: u64,
    pub(crate) hwm_kb: u64,
    pub(crate) rss_anon_kb: u64,
    pub(crate) rss_file_kb: u64,
    pub(crate) rss_shmem_kb: u64,
    pub(crate) pss_kb: u64,
    pub(crate) pss_anon_kb: u64,
    pub(crate) pss_file_kb: u64,
    pub(crate) private_clean_kb: u64,
    pub(crate) private_dirty_kb: u64,
    pub(crate) anonymous_kb: u64,
}

fn parse_kb(contents: &str, key: &str) -> u64 {
    contents
        .lines()
        .find_map(|line| {
            let (name, rest) = line.split_once(':')?;
            if name != key {
                return None;
            }
            rest.split_whitespace().next()?.parse::<u64>().ok()
        })
        .unwrap_or(0)
}

pub(crate) fn snapshot() -> ProcessMemorySnapshot {
    let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
    let rollup = fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
    ProcessMemorySnapshot {
        rss_kb: parse_kb(&status, "VmRSS"),
        hwm_kb: parse_kb(&status, "VmHWM"),
        rss_anon_kb: parse_kb(&status, "RssAnon"),
        rss_file_kb: parse_kb(&status, "RssFile"),
        rss_shmem_kb: parse_kb(&status, "RssShmem"),
        pss_kb: parse_kb(&rollup, "Pss"),
        pss_anon_kb: parse_kb(&rollup, "Pss_Anon"),
        pss_file_kb: parse_kb(&rollup, "Pss_File"),
        private_clean_kb: parse_kb(&rollup, "Private_Clean"),
        private_dirty_kb: parse_kb(&rollup, "Private_Dirty"),
        anonymous_kb: parse_kb(&rollup, "Anonymous"),
    }
}

pub(crate) fn emit(scope: &str, stage: &str, slab_id: Option<usize>) {
    let m = snapshot();
    let slab = slab_id.map_or_else(|| "none".to_owned(), |value| value.to_string());
    println!(
        "PROFILE_MEM {scope} stage={stage} slab={slab} rss_kb={} hwm_kb={} \
rss_anon_kb={} rss_file_kb={} rss_shmem_kb={} pss_kb={} pss_anon_kb={} \
pss_file_kb={} private_clean_kb={} private_dirty_kb={} anonymous_kb={}",
        m.rss_kb,
        m.hwm_kb,
        m.rss_anon_kb,
        m.rss_file_kb,
        m.rss_shmem_kb,
        m.pss_kb,
        m.pss_anon_kb,
        m.pss_file_kb,
        m.private_clean_kb,
        m.private_dirty_kb,
        m.anonymous_kb,
    );
}

#[inline]
pub(crate) fn vec_capacity_bytes<T>(values: &Vec<T>) -> u64 {
    u64::try_from(values.capacity())
        .unwrap_or(u64::MAX)
        .saturating_mul(core::mem::size_of::<T>() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_kilobyte_fields() {
        let text = "VmRSS:\t12345 kB\nVmHWM:\t23456 kB\n";
        assert_eq!(parse_kb(text, "VmRSS"), 12345);
        assert_eq!(parse_kb(text, "VmHWM"), 23456);
        assert_eq!(parse_kb(text, "Missing"), 0);
    }

    #[test]
    fn vector_capacity_helper_uses_allocated_capacity() {
        let mut values = Vec::<u32>::with_capacity(32);
        values.extend(0..7);
        assert!(vec_capacity_bytes(&values) >= 32 * 4);
    }
}
