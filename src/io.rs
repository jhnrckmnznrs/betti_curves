use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};

use crate::tiff_paths::list_tiff_slices;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerPixelType {
    U8,
    U16,
}

impl std::fmt::Display for IntegerPixelType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::U8 => "U8",
            Self::U16 => "U16",
        })
    }
}

impl IntegerPixelType {
    fn bits_per_sample(self) -> u8 {
        match self {
            Self::U8 => 8,
            Self::U16 => 16,
        }
    }
}

#[derive(Debug)]
pub struct TiffStackReader {
    paths: Vec<PathBuf>,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub pixel_type: IntegerPixelType,
}

#[derive(Debug)]
pub struct Block {
    /// Global starting z-index of this slab.
    pub z0: usize,

    /// `shape = [width, height, depth]` of this block/slab.
    pub shape: [usize; 3],

    /// Layout:
    /// `values[z_local * width * height + y_local * width + x_local]`
    pub values: Vec<u16>,
}

impl Block {
    pub fn voxel_count(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }
}

impl TiffStackReader {
    pub fn open(dir: &Path) -> Result<Self> {
        let paths = list_tiff_slices(dir)?;

        let (width, height, pixel_type, _) = read_tiff_u16(&paths[0])
            .with_context(|| format!("Failed to read first slice {:?}", paths[0]))?;

        println!("Found {} TIFF slices", paths.len());
        println!("First slice dimensions: {} x {}", width, height);

        for path in &paths {
            let (w, h, color_type) = read_tiff_dimensions(path)
                .with_context(|| format!("Failed to read dimensions for {:?}", path))?;

            if w != width || h != height {
                bail!(
                    "Dimension mismatch: {:?} has {} x {}, expected {} x {}",
                    path,
                    w,
                    h,
                    width,
                    height
                );
            }
            if color_type != ColorType::Gray(pixel_type.bits_per_sample()) {
                bail!(
                    "mixed TIFF pixel types are not supported: {:?} is {color_type:?}, expected grayscale {}",
                    path,
                    pixel_type
                );
            }
        }

        let depth = paths.len();
        width
            .checked_mul(height)
            .and_then(|slice| slice.checked_mul(depth))
            .ok_or_else(|| anyhow::anyhow!("TIFF stack dimensions overflow usize"))?;

        Ok(Self {
            paths,
            width,
            height,
            depth,
            pixel_type,
        })
    }

    pub fn shape(&self) -> [usize; 3] {
        [self.width, self.height, self.depth]
    }

    pub fn voxel_count(&self) -> usize {
        self.width * self.height * self.depth
    }

    pub fn raw_u16_bytes(&self) -> usize {
        self.voxel_count() * std::mem::size_of::<u16>()
    }

    /// Read a full XY slab: x = full width, y = full height, z = z0..z1.
    pub fn read_z_slab(&self, z0: usize, z1: usize) -> Result<Block> {
        if z0 >= z1 || z1 > self.depth {
            bail!("Invalid z range {}..{} for depth {}", z0, z1, self.depth);
        }

        let slab_depth = z1 - z0;
        let slice_size = self
            .width
            .checked_mul(self.height)
            .ok_or_else(|| anyhow::anyhow!("TIFF slice size overflow"))?;
        let capacity = slice_size
            .checked_mul(slab_depth)
            .ok_or_else(|| anyhow::anyhow!("TIFF slab size overflow"))?;
        let mut values = Vec::with_capacity(capacity);

        for z in z0..z1 {
            let (w, h, pixel_type, slice) = read_tiff_u16(&self.paths[z])
                .with_context(|| format!("Failed to read slice {:?}", self.paths[z]))?;

            if w != self.width || h != self.height {
                bail!("Slice {:?} has wrong shape", self.paths[z]);
            }
            if pixel_type != self.pixel_type {
                bail!(
                    "mixed TIFF pixel types are not supported: {:?} is {pixel_type}, expected {}",
                    self.paths[z],
                    self.pixel_type
                );
            }

            values.extend_from_slice(&slice);
        }

        Ok(Block {
            z0,
            shape: [self.width, self.height, slab_depth],
            values,
        })
    }
}

pub fn collect_unique_values_by_slabs(
    volume: &TiffStackReader,
    slab_depth: usize,
    verbose: bool,
) -> Result<Vec<u16>> {
    let mut present = vec![false; 65536];

    let mut z0 = 0usize;

    while z0 < volume.depth {
        let z1 = z0.saturating_add(slab_depth).min(volume.depth);

        if verbose {
            println!("Scanning values in slab z={}..{}", z0, z1);
        }

        let block = volume.read_z_slab(z0, z1)?;

        for &v in &block.values {
            present[v as usize] = true;
        }

        z0 = z1;
    }

    let values: Vec<u16> = present
        .iter()
        .enumerate()
        .filter_map(|(i, &seen)| if seen { Some(i as u16) } else { None })
        .collect();

    Ok(values)
}

pub fn print_volume_info(volume: &TiffStackReader) {
    let [w, h, d] = volume.shape();

    println!("=== Virtual 3D volume ===");
    println!("shape: {} x {} x {}", w, h, d);
    println!("voxel count: {}", volume.voxel_count());
    println!("source pixel type: {}", volume.pixel_type);
    println!(
        "raw u16 volume size: {}",
        human_bytes(volume.raw_u16_bytes() as u64)
    );

    let parent_u32_bytes = volume.voxel_count() as u64 * std::mem::size_of::<u32>() as u64;
    let parent_u64_bytes = volume.voxel_count() as u64 * std::mem::size_of::<u64>() as u64;
    let active_u8_bytes = volume.voxel_count() as u64;
    let active_bitset_bytes = (volume.voxel_count() as u64).div_ceil(8);

    println!(
        "if global parent Vec<u32>: {}",
        human_bytes(parent_u32_bytes)
    );
    println!(
        "if global parent Vec<u64>: {}",
        human_bytes(parent_u64_bytes)
    );
    println!("if global active Vec<u8>: {}", human_bytes(active_u8_bytes));
    println!(
        "if global active bitset: {}",
        human_bytes(active_bitset_bytes)
    );
    println!();
}

fn read_tiff_dimensions(path: &Path) -> Result<(usize, usize, ColorType)> {
    let file = File::open(path).with_context(|| format!("Could not open TIFF file {:?}", path))?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .with_context(|| format!("Could not create TIFF decoder for {:?}", path))?;

    let (width, height) = decoder
        .dimensions()
        .with_context(|| format!("Could not read dimensions for {:?}", path))?;
    let color_type = decoder
        .colortype()
        .with_context(|| format!("Could not read TIFF color type for {path:?}"))?;
    if !matches!(color_type, ColorType::Gray(_)) {
        bail!("{path:?} is {color_type:?}; only one-channel grayscale TIFF slices are supported");
    }
    if decoder.more_images() {
        bail!("{path:?} contains more than one TIFF page; supply one single-page file per z-slice");
    }

    Ok((width as usize, height as usize, color_type))
}

fn read_tiff_u16(path: &Path) -> Result<(usize, usize, IntegerPixelType, Vec<u16>)> {
    let file = File::open(path).with_context(|| format!("Could not open TIFF file {:?}", path))?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .with_context(|| format!("Could not create TIFF decoder for {:?}", path))?;

    let (width, height) = decoder
        .dimensions()
        .with_context(|| format!("Could not read dimensions for {:?}", path))?;
    let color_type = decoder
        .colortype()
        .with_context(|| format!("Could not read TIFF color type for {path:?}"))?;
    if !matches!(color_type, ColorType::Gray(_)) {
        bail!("{path:?} is {color_type:?}; only one-channel grayscale TIFF slices are supported");
    }

    let image = decoder
        .read_image()
        .with_context(|| format!("Could not decode TIFF image {:?}", path))?;

    let result = match image {
        DecodingResult::U16(data) => {
            let expected = width as usize * height as usize;
            if data.len() != expected {
                bail!(
                    "Unexpected U16 data length in {:?}: got {}, expected {}",
                    path,
                    data.len(),
                    expected
                );
            }
            (IntegerPixelType::U16, data)
        }
        DecodingResult::U8(data) => {
            let expected = width as usize * height as usize;
            if data.len() != expected {
                bail!(
                    "Unexpected U8 data length in {:?}: got {}, expected {}",
                    path,
                    data.len(),
                    expected
                );
            }

            let data_u16: Vec<u16> = data.into_iter().map(u16::from).collect();
            (IntegerPixelType::U8, data_u16)
        }
        other => {
            bail!(
                "unsupported TIFF pixel type in {:?}: {:?}. Integer modes support grayscale U8 and U16",
                path,
                other
            );
        }
    };

    if decoder.more_images() {
        bail!("{path:?} contains more than one TIFF page; supply one single-page file per z-slice");
    }
    Ok((width as usize, height as usize, result.0, result.1))
}

fn human_bytes(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit = 0;

    while size >= 1024.0 && unit < units.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }

    format!("{:.2} {}", size, units[unit])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tiff::encoder::{TiffEncoder, colortype};

    static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn test_directory(label: &str) -> PathBuf {
        let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "betti_io_{label}_{}_{}",
            std::process::id(),
            sequence
        ));
        if path.exists() {
            std::fs::remove_dir_all(&path).unwrap();
        }
        std::fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn rejects_rgb_slices() {
        let directory = test_directory("rgb");
        let path = directory.join("slice_0.tif");
        let file = File::create(&path).unwrap();
        let mut encoder = TiffEncoder::new(file).unwrap();
        encoder
            .write_image::<colortype::RGB8>(1, 1, &[1, 2, 3])
            .unwrap();

        let error = TiffStackReader::open(&directory).unwrap_err();
        assert!(format!("{error:#}").contains("one-channel grayscale"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_multi_page_slices() {
        let directory = test_directory("multipage");
        let path = directory.join("slice_0.tif");
        let file = File::create(&path).unwrap();
        let mut encoder = TiffEncoder::new(file).unwrap();
        encoder.write_image::<colortype::Gray8>(1, 1, &[1]).unwrap();
        encoder.write_image::<colortype::Gray8>(1, 1, &[2]).unwrap();

        let error = TiffStackReader::open(&directory).unwrap_err();
        assert!(format!("{error:#}").contains("more than one TIFF page"));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_mixed_integer_pixel_types_before_slab_processing() {
        let directory = test_directory("mixed");
        {
            let file = File::create(directory.join("slice_0.tif")).unwrap();
            let mut encoder = TiffEncoder::new(file).unwrap();
            encoder.write_image::<colortype::Gray8>(1, 1, &[1]).unwrap();
        }
        {
            let file = File::create(directory.join("slice_1.tif")).unwrap();
            let mut encoder = TiffEncoder::new(file).unwrap();
            encoder
                .write_image::<colortype::Gray16>(1, 1, &[2])
                .unwrap();
        }

        let error = TiffStackReader::open(&directory).unwrap_err();
        assert!(error.to_string().contains("mixed TIFF pixel types"));
        std::fs::remove_dir_all(directory).unwrap();
    }
}
