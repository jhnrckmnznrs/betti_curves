use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};

use crate::scalar::{F32Key, ScalarKey};
use crate::tiff_paths::list_tiff_slices;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarPixelType {
    U8,
    U16,
    F32,
    F64,
}

impl std::fmt::Display for ScalarPixelType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::F32 => "F32",
            Self::F64 => "F64",
        };
        formatter.write_str(name)
    }
}

impl ScalarPixelType {
    fn bits_per_sample(self) -> u8 {
        match self {
            Self::U8 => 8,
            Self::U16 => 16,
            Self::F32 => 32,
            Self::F64 => 64,
        }
    }
}

#[derive(Debug)]
pub struct ScalarTiffStackReader {
    paths: Vec<PathBuf>,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub pixel_type: ScalarPixelType,
}

#[derive(Debug)]
pub struct ScalarBlock {
    pub z0: usize,
    pub shape: [usize; 3],
    pub values: Vec<ScalarKey>,
    pub pixel_type: ScalarPixelType,
}

impl ScalarBlock {
    pub fn voxel_count(&self) -> usize {
        self.shape[0] * self.shape[1] * self.shape[2]
    }
}

/// Native-width F32 slab used by the optimized scalar persistence streams.
#[derive(Debug)]
pub struct F32ScalarBlock {
    pub z0: usize,
    pub shape: [usize; 3],
    pub values: Vec<F32Key>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ScalarReadProfile {
    pub(crate) decode_seconds: f64,
    pub(crate) key_conversion_seconds: f64,
    pub(crate) slab_copy_seconds: f64,
}

impl ScalarReadProfile {
    fn add_assign(&mut self, other: Self) {
        self.decode_seconds += other.decode_seconds;
        self.key_conversion_seconds += other.key_conversion_seconds;
        self.slab_copy_seconds += other.slab_copy_seconds;
    }
}

impl ScalarTiffStackReader {
    pub fn open(dir: &Path) -> Result<Self> {
        let paths = list_tiff_slices(dir)?;
        let (width, height, pixel_type, _) = read_tiff_scalar(&paths[0])
            .with_context(|| format!("failed to read first slice {:?}", paths[0]))?;

        for path in &paths {
            let (current_width, current_height, color_type) = read_tiff_dimensions(path)
                .with_context(|| format!("failed to read dimensions for {path:?}"))?;

            if current_width != width || current_height != height {
                bail!(
                    "dimension mismatch: {:?} has {} x {}, expected {} x {}",
                    path,
                    current_width,
                    current_height,
                    width,
                    height
                );
            }
            if color_type != ColorType::Gray(pixel_type.bits_per_sample()) {
                bail!(
                    "mixed TIFF pixel types are not supported: {path:?} is {color_type:?}, expected grayscale {pixel_type}"
                );
            }
        }

        let depth = paths.len();
        width
            .checked_mul(height)
            .and_then(|slice| slice.checked_mul(depth))
            .ok_or_else(|| anyhow::anyhow!("scalar TIFF stack dimensions overflow usize"))?;
        println!("Found {depth} scalar TIFF slices");
        println!("First slice dimensions: {width} x {height}");

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

    pub fn read_z_slab(&self, z0: usize, z1: usize) -> Result<ScalarBlock> {
        self.read_z_slab_profiled(z0, z1).map(|(block, _)| block)
    }

    pub(crate) fn read_z_slab_profiled(
        &self,
        z0: usize,
        z1: usize,
    ) -> Result<(ScalarBlock, ScalarReadProfile)> {
        if z0 >= z1 || z1 > self.depth {
            bail!("invalid z range {z0}..{z1} for depth {}", self.depth);
        }

        let slab_depth = z1 - z0;
        let slice_size = self
            .width
            .checked_mul(self.height)
            .ok_or_else(|| anyhow::anyhow!("scalar TIFF slice size overflow"))?;
        let capacity = slice_size
            .checked_mul(slab_depth)
            .ok_or_else(|| anyhow::anyhow!("scalar TIFF slab size overflow"))?;
        let mut values = Vec::with_capacity(capacity);
        let mut profile = ScalarReadProfile::default();

        for z in z0..z1 {
            let (width, height, pixel_type, slice, slice_profile) =
                read_tiff_scalar_profiled(&self.paths[z])
                    .with_context(|| format!("failed to read slice {:?}", self.paths[z]))?;
            profile.add_assign(slice_profile);

            if width != self.width || height != self.height {
                bail!("slice {:?} has the wrong shape", self.paths[z]);
            }
            if pixel_type != self.pixel_type {
                bail!(
                    "mixed TIFF pixel types are not supported: {:?} is {pixel_type}, expected {}",
                    self.paths[z],
                    self.pixel_type
                );
            }

            let copy_start = Instant::now();
            values.extend_from_slice(&slice);
            profile.slab_copy_seconds += copy_start.elapsed().as_secs_f64();
        }

        Ok((
            ScalarBlock {
                z0,
                shape: [self.width, self.height, slab_depth],
                values,
                pixel_type: self.pixel_type,
            },
            profile,
        ))
    }

    /// Read an F32 slab directly into four-byte order-preserving keys.
    ///
    /// This is intentionally separate from `read_z_slab`: the in-memory
    /// scalar implementations remain the independent legacy-wide oracle, while
    /// the disk-backed persistence streams can opt into native-width F32
    /// storage without changing U8/U16/F64 behavior.
    #[cfg(test)]
    pub fn read_z_slab_f32_native(&self, z0: usize, z1: usize) -> Result<F32ScalarBlock> {
        self.read_z_slab_f32_native_profiled(z0, z1)
            .map(|(block, _)| block)
    }

    pub(crate) fn read_z_slab_f32_native_profiled(
        &self,
        z0: usize,
        z1: usize,
    ) -> Result<(F32ScalarBlock, ScalarReadProfile)> {
        if self.pixel_type != ScalarPixelType::F32 {
            bail!(
                "native F32 slab requested for {} TIFF stack",
                self.pixel_type
            );
        }
        if z0 >= z1 || z1 > self.depth {
            bail!("invalid z range {z0}..{z1} for depth {}", self.depth);
        }

        let slab_depth = z1 - z0;
        let slice_size = self
            .width
            .checked_mul(self.height)
            .ok_or_else(|| anyhow::anyhow!("scalar TIFF slice size overflow"))?;
        let capacity = slice_size
            .checked_mul(slab_depth)
            .ok_or_else(|| anyhow::anyhow!("scalar TIFF slab size overflow"))?;
        let mut values = Vec::with_capacity(capacity);
        let mut profile = ScalarReadProfile::default();

        for z in z0..z1 {
            let slice_profile = append_tiff_f32_native_profiled(
                &self.paths[z],
                self.width,
                self.height,
                &mut values,
            )
            .with_context(|| format!("failed to read F32 slice {:?}", self.paths[z]))?;
            profile.add_assign(slice_profile);
        }

        Ok((
            F32ScalarBlock {
                z0,
                shape: [self.width, self.height, slab_depth],
                values,
            },
            profile,
        ))
    }
}

pub fn print_scalar_volume_info(volume: &ScalarTiffStackReader) {
    let [width, height, depth] = volume.shape();
    println!("=== Scalar 3D volume ===");
    println!("shape: {width} x {height} x {depth}");
    println!("voxel count: {}", volume.voxel_count());
    println!("source pixel type: {}", volume.pixel_type);
    println!(
        "legacy-wide ScalarKey storage for a full volume: {:.3} GiB",
        volume.voxel_count() as f64 * std::mem::size_of::<ScalarKey>() as f64 / 1024_f64.powi(3)
    );
    println!();
}

fn read_tiff_dimensions(path: &Path) -> Result<(usize, usize, ColorType)> {
    let file = File::open(path).with_context(|| format!("could not open TIFF file {path:?}"))?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .with_context(|| format!("could not create TIFF decoder for {path:?}"))?;
    let (width, height) = decoder
        .dimensions()
        .with_context(|| format!("could not read dimensions for {path:?}"))?;
    let color_type = decoder
        .colortype()
        .with_context(|| format!("could not read TIFF color type for {path:?}"))?;
    if !matches!(color_type, ColorType::Gray(_)) {
        bail!("{path:?} is {color_type:?}; only one-channel grayscale TIFF slices are supported");
    }
    if decoder.more_images() {
        bail!("{path:?} contains more than one TIFF page; supply one single-page file per z-slice");
    }
    Ok((width as usize, height as usize, color_type))
}

fn read_tiff_scalar(path: &Path) -> Result<(usize, usize, ScalarPixelType, Vec<ScalarKey>)> {
    read_tiff_scalar_profiled(path)
        .map(|(width, height, pixel_type, values, _)| (width, height, pixel_type, values))
}

fn read_tiff_scalar_profiled(
    path: &Path,
) -> Result<(
    usize,
    usize,
    ScalarPixelType,
    Vec<ScalarKey>,
    ScalarReadProfile,
)> {
    let file = File::open(path).with_context(|| format!("could not open TIFF file {path:?}"))?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .with_context(|| format!("could not create TIFF decoder for {path:?}"))?;

    let (width, height) = decoder
        .dimensions()
        .with_context(|| format!("could not read dimensions for {path:?}"))?;
    let color_type = decoder
        .colortype()
        .with_context(|| format!("could not read TIFF color type for {path:?}"))?;
    if !matches!(color_type, ColorType::Gray(_)) {
        bail!("{path:?} is {color_type:?}; only one-channel grayscale TIFF slices are supported");
    }
    let expected = width as usize * height as usize;
    let decode_start = Instant::now();
    let image = decoder
        .read_image()
        .with_context(|| format!("could not decode TIFF image {path:?}"))?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    let conversion_start = Instant::now();
    let (pixel_type, values) = match image {
        DecodingResult::U8(data) => {
            validate_length(path, data.len(), expected, "U8")?;
            (
                ScalarPixelType::U8,
                data.into_iter().map(ScalarKey::from_u8).collect(),
            )
        }
        DecodingResult::U16(data) => {
            validate_length(path, data.len(), expected, "U16")?;
            (
                ScalarPixelType::U16,
                data.into_iter().map(ScalarKey::from_u16).collect(),
            )
        }
        DecodingResult::F32(data) => {
            validate_length(path, data.len(), expected, "F32")?;
            (
                ScalarPixelType::F32,
                data.into_iter()
                    .map(ScalarKey::from_f32)
                    .collect::<Result<Vec<_>>>()?,
            )
        }
        DecodingResult::F64(data) => {
            validate_length(path, data.len(), expected, "F64")?;
            (
                ScalarPixelType::F64,
                data.into_iter()
                    .map(ScalarKey::from_f64)
                    .collect::<Result<Vec<_>>>()?,
            )
        }
        other => {
            bail!(
                "unsupported TIFF pixel type in {:?}: {:?}. Supported scalar types are U8, U16, F32, and F64",
                path,
                other
            );
        }
    };
    let key_conversion_seconds = conversion_start.elapsed().as_secs_f64();

    if decoder.more_images() {
        bail!("{path:?} contains more than one TIFF page; supply one single-page file per z-slice");
    }

    Ok((
        width as usize,
        height as usize,
        pixel_type,
        values,
        ScalarReadProfile {
            decode_seconds,
            key_conversion_seconds,
            slab_copy_seconds: 0.0,
        },
    ))
}

fn append_tiff_f32_native_profiled(
    path: &Path,
    expected_width: usize,
    expected_height: usize,
    output: &mut Vec<F32Key>,
) -> Result<ScalarReadProfile> {
    let file = File::open(path).with_context(|| format!("could not open TIFF file {path:?}"))?;
    let reader = BufReader::new(file);
    let mut decoder = Decoder::new(reader)
        .with_context(|| format!("could not create TIFF decoder for {path:?}"))?;
    let (width, height) = decoder
        .dimensions()
        .with_context(|| format!("could not read dimensions for {path:?}"))?;
    if width as usize != expected_width || height as usize != expected_height {
        bail!(
            "slice {path:?} has {} x {}, expected {} x {}",
            width,
            height,
            expected_width,
            expected_height
        );
    }
    let color_type = decoder
        .colortype()
        .with_context(|| format!("could not read TIFF color type for {path:?}"))?;
    if color_type != ColorType::Gray(32) {
        bail!("{path:?} is {color_type:?}; native F32 path requires grayscale F32 TIFF");
    }

    let expected = expected_width * expected_height;
    let decode_start = Instant::now();
    let image = decoder
        .read_image()
        .with_context(|| format!("could not decode TIFF image {path:?}"))?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    let conversion_start = Instant::now();
    match image {
        DecodingResult::F32(data) => {
            validate_length(path, data.len(), expected, "F32")?;
            output.reserve(data.len());
            for value in data {
                output.push(F32Key::from_f32(value)?);
            }
        }
        other => bail!(
            "TIFF decoder returned {other:?} for {path:?}; native F32 path requires F32 samples"
        ),
    }
    let key_conversion_seconds = conversion_start.elapsed().as_secs_f64();

    if decoder.more_images() {
        bail!("{path:?} contains more than one TIFF page; supply one single-page file per z-slice");
    }
    Ok(ScalarReadProfile {
        decode_seconds,
        key_conversion_seconds,
        slab_copy_seconds: 0.0,
    })
}

fn validate_length(path: &Path, actual: usize, expected: usize, pixel_type: &str) -> Result<()> {
    if actual != expected {
        bail!(
            "unexpected {pixel_type} data length in {:?}: got {actual}, expected {expected}",
            path
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tiff::encoder::{TiffEncoder, colortype};

    static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temporary_tiff_path(label: &str) -> PathBuf {
        let sequence = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "betti_curves_{label}_{}_{}.tif",
            std::process::id(),
            sequence
        ))
    }

    #[test]
    fn decodes_f32_and_f64_without_changing_scalar_order() {
        let f32_path = temporary_tiff_path("f32");
        let f64_path = temporary_tiff_path("f64");

        {
            let file = File::create(&f32_path).unwrap();
            let mut encoder = TiffEncoder::new(file).unwrap();
            encoder
                .write_image::<colortype::Gray32Float>(2, 2, &[-3.5, -0.0, 0.25, 8.0])
                .unwrap();
        }
        {
            let file = File::create(&f64_path).unwrap();
            let mut encoder = TiffEncoder::new(file).unwrap();
            encoder
                .write_image::<colortype::Gray64Float>(
                    2,
                    2,
                    &[-9.25, -0.0, f64::MIN_POSITIVE, 1.0e100],
                )
                .unwrap();
        }

        let (_, _, _, f32_values) = read_tiff_scalar(&f32_path).unwrap();
        let (_, _, _, f64_values) = read_tiff_scalar(&f64_path).unwrap();

        assert_eq!(
            f32_values,
            [-3.5f32, 0.0, 0.25, 8.0].map(|value| ScalarKey::from_f32(value).unwrap())
        );
        assert_eq!(
            f64_values,
            [-9.25, 0.0, f64::MIN_POSITIVE, 1.0e100]
                .map(|value| ScalarKey::from_f64(value).unwrap())
        );

        std::fs::remove_file(f32_path).unwrap();
        std::fs::remove_file(f64_path).unwrap();
    }

    #[test]
    fn native_f32_slab_matches_legacy_wide_values() {
        let directory = std::env::temp_dir().join(format!(
            "betti_curves_native_f32_stack_{}_{}",
            std::process::id(),
            TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        for (z, values) in [
            [-3.5f32, -0.0, 0.25, 8.0],
            [1.0f32, 1.0, -2.0, f32::MIN_POSITIVE],
        ]
        .into_iter()
        .enumerate()
        {
            let path = directory.join(format!("slice_{z:03}.tif"));
            let file = File::create(path).unwrap();
            let mut encoder = TiffEncoder::new(file).unwrap();
            encoder
                .write_image::<colortype::Gray32Float>(2, 2, &values)
                .unwrap();
        }

        let reader = ScalarTiffStackReader::open(&directory).unwrap();
        let legacy = reader.read_z_slab(0, 2).unwrap();
        let native = reader.read_z_slab_f32_native(0, 2).unwrap();
        assert_eq!(legacy.values.len(), native.values.len());
        for (legacy_value, native_value) in legacy.values.iter().zip(&native.values) {
            assert_eq!(*legacy_value, native_value.to_scalar_key());
        }
        assert_eq!(std::mem::size_of::<F32Key>(), 4);
        assert_eq!(std::mem::size_of::<ScalarKey>(), 8);

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn rejects_non_finite_float_pixels() {
        let path = temporary_tiff_path("nan");
        {
            let file = File::create(&path).unwrap();
            let mut encoder = TiffEncoder::new(file).unwrap();
            encoder
                .write_image::<colortype::Gray32Float>(1, 1, &[f32::NAN])
                .unwrap();
        }

        let error = read_tiff_scalar(&path).unwrap_err();
        assert!(error.to_string().contains("non-finite"));
        std::fs::remove_file(path).unwrap();
    }
}
