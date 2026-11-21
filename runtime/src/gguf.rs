//! Just enough of a GGUF reader to size the model's private memory before it loads:
//! the metadata key/values (the tensor list and the weights are never read).

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

/// A metadata value. Big arrays (the tokenizer vocab) are kept only as a length.
#[derive(Debug, Clone, PartialEq)]
pub enum Meta {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    /// Numeric or bool array (up to `MAX_KEPT` items are kept; `len` is the real length).
    Arr { len: u64, items: Vec<f64> },
    /// Array of strings, or a big array: only the length.
    Opaque { len: u64 },
}

const MAX_KEPT: u64 = 4096;

pub struct Gguf {
    pub kv: HashMap<String, Meta>,
}

impl Gguf {
    pub fn read(path: &Path) -> Result<Gguf> {
        let f = std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        Gguf::parse(&mut BufReader::with_capacity(1 << 20, f)).with_context(|| format!("cannot read GGUF metadata of {}", path.display()))
    }

    pub fn parse<R: Read + Seek>(r: &mut R) -> Result<Gguf> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            bail!("not a GGUF file");
        }
        let version = u32_(r)?;
        if !(2..=3).contains(&version) {
            bail!("GGUF version {version} is not supported");
        }
        let _tensors = u64_(r)?;
        let n_kv = u64_(r)?;
        let mut kv = HashMap::new();
        for _ in 0..n_kv {
            let key = string(r)?;
            let ty = u32_(r)?;
            let v = value(r, ty)?;
            kv.insert(key, v);
        }
        Ok(Gguf { kv })
    }

    pub fn arch(&self) -> Option<&str> {
        match self.kv.get("general.architecture") {
            Some(Meta::Str(s)) => Some(s),
            _ => None,
        }
    }

    /// `{arch}.{key}` as a number.
    pub fn num(&self, key: &str) -> Option<f64> {
        match self.kv.get(&format!("{}.{key}", self.arch()?))? {
            Meta::Int(i) => Some(*i as f64),
            Meta::Float(f) => Some(*f),
            Meta::Bool(b) => Some(*b as u8 as f64),
            _ => None,
        }
    }

    /// `{arch}.{key}` for one layer: a scalar, or the layer's item of a per-layer array.
    pub fn per_layer(&self, key: &str, layer: usize) -> Option<f64> {
        match self.kv.get(&format!("{}.{key}", self.arch()?))? {
            Meta::Int(i) => Some(*i as f64),
            Meta::Float(f) => Some(*f),
            Meta::Arr { items, .. } => items.get(layer).copied(),
            _ => None,
        }
    }

    /// `{arch}.{key}` as a per-layer array (bool arrays become 0/1).
    pub fn array(&self, key: &str) -> Option<&[f64]> {
        match self.kv.get(&format!("{}.{key}", self.arch()?))? {
            Meta::Arr { items, .. } => Some(items),
            _ => None,
        }
    }

    pub fn len_of(&self, full_key: &str) -> Option<u64> {
        match self.kv.get(full_key)? {
            Meta::Arr { len, .. } | Meta::Opaque { len } => Some(*len),
            _ => None,
        }
    }
}

fn u32_<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn u64_<R: Read>(r: &mut R) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn string<R: Read>(r: &mut R) -> Result<String> {
    let n = u64_(r)?;
    if n > 1 << 24 {
        bail!("metadata string of {n} bytes is too long");
    }
    let mut b = vec![0u8; n as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

/// Size in bytes of a fixed-size GGUF scalar type.
fn scalar_size(ty: u32) -> Option<u64> {
    Some(match ty {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

fn scalar<R: Read>(r: &mut R, ty: u32) -> Result<Meta> {
    let size = scalar_size(ty).with_context(|| format!("unknown GGUF value type {ty}"))?;
    let mut b = [0u8; 8];
    r.read_exact(&mut b[..size as usize])?;
    Ok(match ty {
        0 => Meta::Int(b[0] as i64),
        1 => Meta::Int(b[0] as i8 as i64),
        2 => Meta::Int(u16::from_le_bytes([b[0], b[1]]) as i64),
        3 => Meta::Int(i16::from_le_bytes([b[0], b[1]]) as i64),
        4 => Meta::Int(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64),
        5 => Meta::Int(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64),
        6 => Meta::Float(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64),
        7 => Meta::Bool(b[0] != 0),
        10 => Meta::Int(u64::from_le_bytes(b) as i64),
        11 => Meta::Int(i64::from_le_bytes(b)),
        12 => Meta::Float(f64::from_le_bytes(b)),
        _ => unreachable!(),
    })
}

fn value<R: Read + Seek>(r: &mut R, ty: u32) -> Result<Meta> {
    match ty {
        8 => Ok(Meta::Str(string(r)?)),
        9 => {
            let et = u32_(r)?;
            let len = u64_(r)?;
            if et == 8 {
                for _ in 0..len {
                    let n = u64_(r)?;
                    r.seek_relative(n as i64)?;
                }
                return Ok(Meta::Opaque { len });
            }
            if et == 9 {
                bail!("nested GGUF arrays are not supported");
            }
            let size = scalar_size(et).with_context(|| format!("unknown GGUF array type {et}"))?;
            if len > MAX_KEPT {
                r.seek_relative((len * size) as i64)?;
                return Ok(Meta::Opaque { len });
            }
            let mut items = Vec::with_capacity(len as usize);
            for _ in 0..len {
                items.push(match scalar(r, et)? {
                    Meta::Int(i) => i as f64,
                    Meta::Float(f) => f,
                    Meta::Bool(b) => b as u8 as f64,
                    _ => unreachable!(),
                });
            }
            Ok(Meta::Arr { len, items })
        }
        t => scalar(r, t),
    }
}

/// A tiny GGUF writer for tests.
#[cfg(test)]
pub mod testfile {
    pub enum V<'a> {
        U32(u32),
        F32(f32),
        Str(&'a str),
        ArrU32(&'a [u32]),
        ArrBool(&'a [bool]),
        ArrStr(&'a [&'a str]),
    }

    pub fn build(kvs: &[(&str, V)]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        let s = |b: &mut Vec<u8>, t: &str| {
            b.extend_from_slice(&(t.len() as u64).to_le_bytes());
            b.extend_from_slice(t.as_bytes());
        };
        for (k, v) in kvs {
            s(&mut b, k);
            match v {
                V::U32(x) => {
                    b.extend_from_slice(&4u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::F32(x) => {
                    b.extend_from_slice(&6u32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::Str(t) => {
                    b.extend_from_slice(&8u32.to_le_bytes());
                    s(&mut b, t);
                }
                V::ArrU32(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&4u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in *xs {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                V::ArrBool(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&7u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in *xs {
                        b.push(*x as u8);
                    }
                }
                V::ArrStr(xs) => {
                    b.extend_from_slice(&9u32.to_le_bytes());
                    b.extend_from_slice(&8u32.to_le_bytes());
                    b.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                    for x in *xs {
                        s(&mut b, x);
                    }
                }
            }
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::testfile::{build, V};
    use super::*;

    #[test]
    fn metadata_is_read_and_big_arrays_are_skipped() {
        let bytes = build(&[
            ("general.architecture", V::Str("qwen3")),
            ("tokenizer.ggml.tokens", V::ArrStr(&["a", "bb", "ccc"])),
            ("qwen3.block_count", V::U32(36)),
            ("qwen3.rope.freq_base", V::F32(5e6)),
            ("qwen3.feed_forward_length", V::ArrU32(&[10, 20])),
            ("qwen3.attention.sliding_window_pattern", V::ArrBool(&[true, false])),
        ]);
        let g = Gguf::parse(&mut std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(g.arch(), Some("qwen3"));
        assert_eq!(g.num("block_count"), Some(36.0));
        assert_eq!(g.num("rope.freq_base"), Some(5e6));
        assert_eq!(g.per_layer("feed_forward_length", 1), Some(20.0));
        assert_eq!(g.per_layer("block_count", 7), Some(36.0));
        assert_eq!(g.array("attention.sliding_window_pattern"), Some(&[1.0, 0.0][..]));
        assert_eq!(g.len_of("tokenizer.ggml.tokens"), Some(3));
        assert!(Gguf::parse(&mut std::io::Cursor::new(b"GGML....".to_vec())).is_err());
    }
}
