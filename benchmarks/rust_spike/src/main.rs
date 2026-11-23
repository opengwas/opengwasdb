use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use memmap2::Mmap;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use zarrs::array::{Array, ArrayShardedExt, ArraySubset};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableStorage;

const Z_MISSING: i16 = -32_768;
const Z_OVERFLOW: i16 = -32_767;
const RESIDUAL_MISSING: i8 = -128;
const RESIDUAL_EXCEPTION: i8 = -127;

#[derive(Deserialize)]
struct Request {
    store_path: PathBuf,
    shape: String,
    stage: String,
    alids: Vec<String>,
    analysis_ids: Vec<String>,
    rows: Vec<u64>,
    cols: Vec<u64>,
    z_scale: f32,
    se_step: f64,
    eaf_step: f64,
    #[serde(default = "default_reps")]
    reps: usize,
}

fn default_reps() -> usize {
    5
}

#[derive(Serialize)]
struct Measurement {
    candidate: &'static str,
    shape: String,
    stage: String,
    cache: String,
    repetitions: usize,
    times_ms: Vec<f64>,
    result_count: Option<usize>,
    arrays: BTreeMap<String, EncodedArray>,
    zarrs_crate: &'static str,
}

#[derive(Serialize)]
struct EncodedArray {
    dtype: &'static str,
    base64: String,
}

#[derive(Clone, Copy)]
enum DType {
    Int32,
    Float32,
    Int8,
}

trait Cells: Copy {
    const DTYPE: DType;
    fn append_le(self, output: &mut Vec<u8>);
}

macro_rules! cells {
    ($type:ty, $dtype:expr) => {
        impl Cells for $type {
            const DTYPE: DType = $dtype;
            fn append_le(self, output: &mut Vec<u8>) {
                output.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}
cells!(i32, DType::Int32);
cells!(f32, DType::Float32);
cells!(i8, DType::Int8);

fn encoded<T: Cells>(values: &[T]) -> EncodedArray {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        value.append_le(&mut bytes);
    }
    let dtype = match T::DTYPE {
        DType::Int32 => "int32",
        DType::Float32 => "float32",
        DType::Int8 => "int8",
    };
    EncodedArray {
        dtype,
        base64: BASE64.encode(bytes),
    }
}

struct VariantResolver {
    keys: Mmap,
    keys_offset: usize,
    rows: Mmap,
    rows_offset: usize,
    count: usize,
}

impl VariantResolver {
    fn open(store: &Path) -> Result<Self, Box<dyn Error>> {
        let key_file = File::open(store.join("variant_alid_bytes.npy"))?;
        let keys = unsafe { Mmap::map(&key_file)? };
        let (keys_offset, key_dtype, count) = npy_layout(&keys)?;
        if key_dtype != "|S32" {
            return Err(format!("ALID index dtype is {key_dtype}, expected |S32").into());
        }
        let row_file = File::open(store.join("variant_alid_rows.npy"))?;
        let rows = unsafe { Mmap::map(&row_file)? };
        let (rows_offset, row_dtype, row_count) = npy_layout(&rows)?;
        if row_dtype != "<i4" || row_count != count {
            return Err("ALID row index dtype/length is inconsistent".into());
        }
        Ok(Self {
            keys,
            keys_offset,
            rows,
            rows_offset,
            count,
        })
    }

    fn row(&self, alid: &str) -> Result<u64, Box<dyn Error>> {
        if alid.len() > 32 {
            return Err("this spike accepts only indexable ALIDs (at most 32 bytes)".into());
        }
        let mut query = [0u8; 32];
        query[..alid.len()].copy_from_slice(alid.as_bytes());
        let keys = &self.keys[self.keys_offset..self.keys_offset + self.count * 32];
        let (mut lo, mut hi) = (0usize, self.count);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if keys[mid * 32..(mid + 1) * 32] < query[..] {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let slot = lo;
        if slot == self.count || keys[slot * 32..(slot + 1) * 32] != query {
            return Err(format!("ALID {alid} is absent from this Store Release").into());
        }
        let offset = self.rows_offset + slot * 4;
        let row = i32::from_le_bytes(self.rows[offset..offset + 4].try_into()?);
        Ok(u64::try_from(row)?)
    }
}

fn npy_layout(bytes: &[u8]) -> Result<(usize, String, usize), Box<dyn Error>> {
    if bytes.get(..6) != Some(b"\x93NUMPY") {
        return Err("not an NPY file".into());
    }
    let major = bytes[6];
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes(bytes[8..10].try_into()?) as usize, 10),
        2 | 3 => (u32::from_le_bytes(bytes[8..12].try_into()?) as usize, 12),
        _ => return Err(format!("unsupported NPY version {major}").into()),
    };
    let header = std::str::from_utf8(&bytes[header_start..header_start + header_len])?;
    let descr_marker = "'descr':";
    let descr_start = header.find(descr_marker).ok_or("NPY has no descr")? + descr_marker.len();
    let descr_tail = header[descr_start..].trim_start();
    let quote = descr_tail.chars().next().ok_or("empty NPY descr")?;
    let descr = descr_tail[1..]
        .split(quote)
        .next()
        .ok_or("unterminated NPY descr")?
        .to_string();
    let shape_start = header.find("'shape':").ok_or("NPY has no shape")?;
    let open = header[shape_start..].find('(').ok_or("invalid NPY shape")? + shape_start;
    let close = header[open..].find(')').ok_or("invalid NPY shape")? + open;
    let count = header[open + 1..close]
        .split(',')
        .next()
        .ok_or("empty NPY shape")?
        .trim()
        .parse::<usize>()?;
    Ok((header_start + header_len, descr, count))
}

struct AnalysisResolver {
    by_id: HashMap<String, u64>,
}

impl AnalysisResolver {
    fn open(store: &Path) -> Result<Self, Box<dyn Error>> {
        let text = fs::read_to_string(store.join("analyses.tsv"))?;
        let mut by_id = HashMap::new();
        for line in text.lines().skip(1) {
            let mut fields = line.split('\t');
            let index = fields.next().ok_or("missing analysis_index")?;
            let id = fields.next().ok_or("missing analysis_id")?;
            by_id.insert(id.to_string(), index.parse()?);
        }
        Ok(Self { by_id })
    }
}

type ZarrArray = Array<dyn zarrs_storage::ReadableStorageTraits>;

struct ExactTable {
    positions: Vec<i64>,
    values: Vec<f32>,
}

impl ExactTable {
    fn lookup(&self, position: i64) -> Result<f32, Box<dyn Error>> {
        let slot = self.positions.binary_search(&position).map_err(|_| {
            format!("exact table has no value for flat position {position}")
        })?;
        Ok(self.values[slot])
    }
}

struct Reader {
    root: PathBuf,
    z: ZarrArray,
    se: ZarrArray,
    eaf: ZarrArray,
    imputed: ZarrArray,
    baseline: ZarrArray,
    reference: ZarrArray,
    se_coefficients: ZarrArray,
    z_overflow_index: ZarrArray,
    z_overflow_value: ZarrArray,
    eaf_exception_index: ZarrArray,
    eaf_exception_value: ZarrArray,
    se_exception_index: ZarrArray,
    se_exception_value: ZarrArray,
    variants: VariantResolver,
    analyses: AnalysisResolver,
    n_analyses: u64,
}

#[derive(Default)]
struct Decoded {
    variant_index: Vec<i32>,
    analysis_index: Vec<i32>,
    z: Vec<f32>,
    se: Vec<f32>,
    eaf: Vec<f32>,
    status: Vec<i8>,
}

impl Reader {
    fn open(request: &Request) -> Result<Self, Box<dyn Error>> {
        let storage: ReadableStorage = Arc::new(FilesystemStore::new(
            request.store_path.join("data.zarr"),
        )?);
        let z = Array::open(storage.clone(), "/z")?;
        let n_analyses = z.shape()[1];
        Ok(Self {
            root: request.store_path.clone(),
            se: Array::open(storage.clone(), "/se")?,
            eaf: Array::open(storage.clone(), "/eaf")?,
            imputed: Array::open(storage.clone(), "/imputed")?,
            baseline: Array::open(storage.clone(), "/eaf_baseline")?,
            reference: Array::open(storage.clone(), "/eaf_reference")?,
            se_coefficients: Array::open(storage.clone(), "/se_coefficients")?,
            z_overflow_index: Array::open(storage.clone(), "/z_overflow_index")?,
            z_overflow_value: Array::open(storage.clone(), "/z_overflow_value")?,
            eaf_exception_index: Array::open(storage.clone(), "/eaf_exception_index")?,
            eaf_exception_value: Array::open(storage.clone(), "/eaf_exception_value")?,
            se_exception_index: Array::open(storage.clone(), "/se_exception_index")?,
            se_exception_value: Array::open(storage.clone(), "/se_exception_value")?,
            variants: VariantResolver::open(&request.store_path)?,
            analyses: AnalysisResolver::open(&request.store_path)?,
            z,
            n_analyses,
        })
    }

    fn resolve(&self, request: &Request) -> Result<(Vec<u64>, Vec<u64>), Box<dyn Error>> {
        let mut rows = Vec::with_capacity(request.alids.len());
        for alid in &request.alids {
            rows.push(self.variants.row(alid)?);
        }
        let mut cols = Vec::with_capacity(request.analysis_ids.len());
        for id in &request.analysis_ids {
            cols.push(*self.analyses.by_id.get(id).ok_or("unknown Analysis")?);
        }
        Ok((rows, cols))
    }

    fn decode(&self, request: &Request, rows: &[u64], cols: &[u64]) -> Result<Decoded, Box<dyn Error>> {
        if rows.is_empty() || cols.is_empty() {
            return Ok(Decoded::default());
        }
        let raw_z = self.read_cross::<i16>(&self.z, rows, cols)?;
        let raw_eaf = self.read_cross::<i8>(&self.eaf, rows, cols)?;
        let raw_imputed = self.read_cross::<u8>(&self.imputed, rows, cols)?;
        let raw_se = self.read_cross::<i8>(&self.se, rows, cols)?;
        let baselines = self.read_rows::<f32>(&self.baseline, rows)?;
        let references = self.read_rows::<f32>(&self.reference, rows)?;
        let coefficients = self.read_all::<f32>(&self.se_coefficients)?;
        let z_overflow = self.read_exact(&self.z_overflow_index, &self.z_overflow_value)?;
        let eaf_exceptions = self.read_exact(&self.eaf_exception_index, &self.eaf_exception_value)?;
        let se_exceptions = self.read_exact(&self.se_exception_index, &self.se_exception_value)?;

        let n = rows.len() * cols.len();
        let mut z = vec![0.0f32; n];
        let mut se = vec![0.0f32; n];
        let mut eaf = vec![0.0f32; n];
        let mut states = vec![0i8; n];
        for (i, row) in rows.iter().enumerate() {
            for (j, col) in cols.iter().enumerate() {
                let cell = i * cols.len() + j;
                let position = i64::try_from(row * self.n_analyses + col)?;
                let z_code = raw_z[cell];
                z[cell] = match z_code {
                    Z_MISSING => f32::NAN,
                    Z_OVERFLOW => z_overflow.lookup(position)?,
                    _ => z_code as f32 / request.z_scale,
                };
                let eaf_code = raw_eaf[cell];
                eaf[cell] = match eaf_code {
                    RESIDUAL_MISSING => f32::NAN,
                    RESIDUAL_EXCEPTION => eaf_exceptions.lookup(position)?,
                    _ => {
                        let baseline = baselines[i] as f64;
                        let logit = baseline.ln() - (-baseline).ln_1p();
                        expit(logit + f64::from(eaf_code) * request.eaf_step) as f32
                    }
                };
                if raw_imputed[cell] == 1 {
                    eaf[cell] = references[i];
                }
                let se_code = raw_se[cell];
                se[cell] = match se_code {
                    RESIDUAL_MISSING => f32::NAN,
                    RESIDUAL_EXCEPTION => se_exceptions.lookup(position)?,
                    _ => {
                        let f = eaf[cell] as f64;
                        if !f.is_finite() {
                            return Err("present residual SE has no finite EAF".into());
                        }
                        let coefficient = usize::try_from(*col)? * 2;
                        let intercept = coefficients[coefficient] as f64;
                        let slope = coefficients[coefficient + 1] as f64;
                        let prediction = intercept + slope * (2.0 * f * (1.0 - f)).ln();
                        (prediction + f64::from(se_code) * request.se_step).exp() as f32
                    }
                };
                let z_missing = z_code == Z_MISSING;
                let se_missing = se_code == RESIDUAL_MISSING;
                states[cell] = if z_missing != se_missing {
                    return Err(format!("partial association at flat position {position}").into());
                } else if raw_imputed[cell] > 1 {
                    return Err(format!("invalid imputed flag at flat position {position}").into());
                } else if z_missing && raw_imputed[cell] == 1 {
                    return Err(format!("missing association flagged imputed at {position}").into());
                } else if z_missing {
                    0
                } else if raw_imputed[cell] == 1 {
                    2
                } else {
                    1
                };
            }
        }

        let mut decoded = Decoded::default();
        // Dense `lookup` emits Analysis-major rows: columns outer, variants inner.
        for (j, col) in cols.iter().enumerate() {
            for (i, row) in rows.iter().enumerate() {
                let cell = i * cols.len() + j;
                if states[cell] == 0 {
                    continue;
                }
                decoded.variant_index.push(i32::try_from(*row)?);
                decoded.analysis_index.push(i32::try_from(*col)?);
                decoded.z.push(z[cell]);
                decoded.se.push(se[cell]);
                decoded.eaf.push(eaf[cell]);
                decoded.status.push(states[cell]);
            }
        }
        Ok(decoded)
    }

    fn read_exact(&self, index: &ZarrArray, values: &ZarrArray) -> Result<ExactTable, Box<dyn Error>> {
        Ok(ExactTable {
            positions: self.read_all(index)?,
            values: self.read_all(values)?,
        })
    }

    fn read_all<T: zarrs::array::ElementOwned + Copy>(&self, array: &ZarrArray) -> Result<Vec<T>, Box<dyn Error>> {
        let values: ArrayD<T> = array.retrieve_array_subset(&array.subset_all())?;
        Ok(values.into_raw_vec_and_offset().0)
    }

    fn read_rows<T: zarrs::array::ElementOwned + Copy>(
        &self,
        array: &ZarrArray,
        rows: &[u64],
    ) -> Result<Vec<T>, Box<dyn Error>> {
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let values: Vec<T> = array.retrieve_array_subset(&[*row..*row + 1])?;
            out.push(values[0]);
        }
        Ok(out)
    }

    fn read_cross<T: zarrs::array::ElementOwned + Copy + Default>(
        &self,
        array: &ZarrArray,
        rows: &[u64],
        cols: &[u64],
    ) -> Result<Vec<T>, Box<dyn Error>> {
        let inner = array
            .subchunk_shape()
            .ok_or("lookup planes must be sharded")?
            .iter()
            .map(|value| value.get())
            .collect::<Vec<_>>();
        let mut out = vec![T::default(); rows.len() * cols.len()];
        let mut groups: BTreeMap<(u64, u64), Vec<(usize, usize)>> = BTreeMap::new();
        for (i, row) in rows.iter().enumerate() {
            for (j, col) in cols.iter().enumerate() {
                groups
                    .entry((row / inner[0], col / inner[1]))
                    .or_default()
                    .push((i, j));
            }
        }
        for cells in groups.values() {
            let r0 = cells.iter().map(|(i, _)| rows[*i]).min().unwrap();
            let r1 = cells.iter().map(|(i, _)| rows[*i]).max().unwrap();
            let c0 = cells.iter().map(|(_, j)| cols[*j]).min().unwrap();
            let c1 = cells.iter().map(|(_, j)| cols[*j]).max().unwrap();
            let subset = ArraySubset::new_with_ranges(&[r0..r1 + 1, c0..c1 + 1]);
            let block: ArrayD<T> = array.retrieve_array_subset(&subset)?;
            for (i, j) in cells {
                out[*i * cols.len() + *j] = block[[(rows[*i] - r0) as usize, (cols[*j] - c0) as usize]];
            }
        }
        Ok(out)
    }
}

fn expit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let tail = x.exp();
        tail / (1.0 + tail)
    }
}

fn timed<T>(mut call: impl FnMut() -> Result<T, Box<dyn Error>>) -> Result<(f64, T), Box<dyn Error>> {
    let started = Instant::now();
    let result = call()?;
    Ok((started.elapsed().as_secs_f64() * 1000.0, result))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let request_path = args.next().ok_or("usage: rust-spike REQUEST.json [cold|warm]")?;
    let cache = args.next().unwrap_or_else(|| "warm".to_string());
    let request: Request = serde_json::from_slice(&fs::read(request_path)?)?;
    let reader = Reader::open(&request)?;
    let mut times_ms = Vec::new();
    let mut result_count = None;
    let mut arrays = BTreeMap::new();
    let run_decode = |reader: &Reader, rows: &[u64], cols: &[u64]| -> Result<Decoded, Box<dyn Error>> {
        reader.decode(&request, rows, cols)
    };
    match request.stage.as_str() {
        "axis_resolution" => {
            let (rows, cols) = reader.resolve(&request)?;
            assert_eq!(rows, request.rows);
            assert_eq!(cols, request.cols);
            if cache == "warm" {
                let _ = reader.resolve(&request)?;
            }
            let reps = if cache == "cold" { 1 } else { request.reps };
            for _ in 0..reps {
                let (elapsed, (rows, cols)) = timed(|| reader.resolve(&request))?;
                assert_eq!(rows, request.rows);
                assert_eq!(cols, request.cols);
                times_ms.push(elapsed);
            }
        }
        "decode" | "end_to_end" => {
            let execute = || -> Result<Decoded, Box<dyn Error>> {
                if request.stage == "decode" {
                    run_decode(&reader, &request.rows, &request.cols)
                } else {
                    let (rows, cols) = reader.resolve(&request)?;
                    run_decode(&reader, &rows, &cols)
                }
            };
            if cache == "warm" {
                let warmup = execute()?;
                result_count = Some(warmup.variant_index.len());
                arrays.insert("variant_index".into(), encoded(&warmup.variant_index));
                arrays.insert("analysis_index".into(), encoded(&warmup.analysis_index));
                arrays.insert("z".into(), encoded(&warmup.z));
                arrays.insert("se".into(), encoded(&warmup.se));
                arrays.insert("eaf".into(), encoded(&warmup.eaf));
                arrays.insert("association_status_int8".into(), encoded(&warmup.status));
            }
            let reps = if cache == "cold" { 1 } else { request.reps };
            for _ in 0..reps {
                let (elapsed, decoded) = timed(&execute)?;
                if cache == "cold" {
                    result_count = Some(decoded.variant_index.len());
                    arrays.insert("variant_index".into(), encoded(&decoded.variant_index));
                    arrays.insert("analysis_index".into(), encoded(&decoded.analysis_index));
                    arrays.insert("z".into(), encoded(&decoded.z));
                    arrays.insert("se".into(), encoded(&decoded.se));
                    arrays.insert("eaf".into(), encoded(&decoded.eaf));
                    arrays.insert("association_status_int8".into(), encoded(&decoded.status));
                }
                times_ms.push(elapsed);
            }
        }
        _ => return Err(format!("unknown stage {}", request.stage).into()),
    }
    let measurement = Measurement {
        candidate: "rust-zarrs",
        shape: request.shape,
        stage: request.stage,
        cache,
        repetitions: times_ms.len(),
        times_ms,
        result_count,
        arrays,
        zarrs_crate: "0.23.14",
    };
    println!("{}", serde_json::to_string(&measurement)?);
    Ok(())
}
