use anyhow::{Context, Result, bail};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use tiff::decoder::{Decoder, DecodingResult};

#[derive(Debug)]
pub struct TiffStackReader {
    paths: Vec<PathBuf>,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

#[derive(Debug)]
pub struct Block {
    /// Global starting z-index of this slab.
    pub z0: usize,

    /// shape = [width, height, depth] of this block/slab.
    pub shape: [usize; 3],

    /// Layout:
    /// values[z_local * width * height + y_local * width + x_local]
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

        let (width, height, _) = read_tiff_u16(&paths[0])
            .with_context(|| format!("Failed to read first slice {:?}", paths[0]))?;

        println!("Found {} TIFF slices", paths.len());
        println!("First slice dimensions: {} x {}", width, height);

        for path in &paths {
            let (w, h) = read_tiff_dimensions(path)
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
        }

        let depth = paths.len();

        Ok(Self {
            paths,
            width,
            height,
            depth,
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
        let slice_size = self.width * self.height;
        let mut values = Vec::with_capacity(slice_size * slab_depth);

        for z in z0..z1 {
            let (w, h, slice) = read_tiff_u16(&self.paths[z])
                .with_context(|| format!("Failed to read slice {:?}", self.paths[z]))?;

            if w != self.width || h != self.height {
                bail!("Slice {:?} has wrong shape", self.paths[z]);
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
        let z1 = usize::min(z0 + slab_depth, volume.depth);

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

fn list_tiff_slices(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .with_context(|| format!("Could not read directory {:?}", dir))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .map(|ext| {
                    let ext = ext.to_ascii_lowercase();
                    ext == "tif" || ext == "tiff"
                })
                .unwrap_or(false)
        })
        .collect();

    files.sort();

    if files.is_empty() {
        bail!("No .tif or .tiff files found in {:?}", dir);
    }

    Ok(files)
}

fn read_tiff_dimensions(path: &Path) -> Result<(usize, usize)> {
    let file = File::open(path).with_context(|| format!("Could not open TIFF file {:?}", path))?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .with_context(|| format!("Could not create TIFF decoder for {:?}", path))?;

    let (width, height) = decoder
        .dimensions()
        .with_context(|| format!("Could not read dimensions for {:?}", path))?;

    Ok((width as usize, height as usize))
}

fn read_tiff_u16(path: &Path) -> Result<(usize, usize, Vec<u16>)> {
    let file = File::open(path).with_context(|| format!("Could not open TIFF file {:?}", path))?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .with_context(|| format!("Could not create TIFF decoder for {:?}", path))?;

    let (width, height) = decoder
        .dimensions()
        .with_context(|| format!("Could not read dimensions for {:?}", path))?;

    let image = decoder
        .read_image()
        .with_context(|| format!("Could not decode TIFF image {:?}", path))?;

    match image {
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
            Ok((width as usize, height as usize, data))
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
            Ok((width as usize, height as usize, data_u16))
        }
        other => {
            bail!(
                "Unsupported TIFF pixel type in {:?}: {:?}. This demo supports grayscale U8 and U16.",
                path,
                other
            );
        }
    }
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
