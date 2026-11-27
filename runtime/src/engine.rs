//! In-process engine (P0, see docs/08-inproc-engine.md): llama.cpp loaded as a library from a
//! prebuilt build folder (CPU or Vulkan), with our own generation loop.
//!
//! The FFI below mirrors `llama.h` of llama.cpp **b12606**. Structs passed by value must match
//! that build exactly; `Lib::load` checks the default params look sane before anything runs.
#![allow(non_camel_case_types, dead_code)]

use anyhow::{anyhow, bail, Context as _, Result};
use libloading::os::windows::{Library, LOAD_WITH_ALTERED_SEARCH_PATH};
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::ptr::null;
use std::time::{Duration, Instant};

pub const PINNED_BUILD: &str = "b12606";

type llama_token = i32;
type Ptr = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ModelParams {
    devices: Ptr,
    tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    split_mode: i32,
    load_mode: i32,
    lazy_mode: i32,
    main_gpu: i32,
    tensor_split: *const f32,
    progress_callback: Ptr,
    progress_callback_user_data: Ptr,
    kv_overrides: *const c_void,
    vocab_only: bool,
    check_tensors: bool,
    pub use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
    load_mtp: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    n_rs_seq: u32,
    n_outputs_max: u32,
    n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    ctx_type: i32,
    rope_scaling_type: i32,
    pooling_type: i32,
    attention_type: i32,
    pub flash_attn_type: i32,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    cb_eval: Ptr,
    cb_eval_user_data: Ptr,
    pub type_k: i32,
    pub type_v: i32,
    moe_cache_size: usize,
    moe_cache_cpu_share: f32,
    abort_callback: Ptr,
    abort_callback_data: Ptr,
    embeddings: bool,
    offload_kqv: bool,
    pub no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: Ptr,
    n_samplers: usize,
    ctx_other: Ptr,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Batch {
    n_tokens: i32,
    token: *mut llama_token,
    embd: *mut f32,
    pos: *mut i32,
    n_seq_id: *mut i32,
    seq_id: *mut *mut i32,
    logits: *mut i8,
}

#[repr(C)]
struct ChatMessage {
    role: *const c_char,
    content: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SamplerChainParams {
    no_perf: bool,
}

/// The llama.cpp functions we use, resolved once from llama.dll / ggml.dll.
pub struct Lib {
    backend_load_all_from_path: unsafe extern "C" fn(*const c_char),
    backend_init: unsafe extern "C" fn(),
    model_default_params: unsafe extern "C" fn() -> ModelParams,
    context_default_params: unsafe extern "C" fn() -> ContextParams,
    sampler_chain_default_params: unsafe extern "C" fn() -> SamplerChainParams,
    model_load_from_file: unsafe extern "C" fn(*const c_char, ModelParams) -> Ptr,
    model_free: unsafe extern "C" fn(Ptr),
    init_from_model: unsafe extern "C" fn(Ptr, ContextParams) -> Ptr,
    free: unsafe extern "C" fn(Ptr),
    model_get_vocab: unsafe extern "C" fn(Ptr) -> Ptr,
    model_chat_template: unsafe extern "C" fn(Ptr, *const c_char) -> *const c_char,
    chat_apply_template: unsafe extern "C" fn(*const c_char, *const ChatMessage, usize, bool, *mut c_char, i32) -> i32,
    tokenize: unsafe extern "C" fn(Ptr, *const c_char, i32, *mut llama_token, i32, bool, bool) -> i32,
    token_to_piece: unsafe extern "C" fn(Ptr, llama_token, *mut c_char, i32, i32, bool) -> i32,
    vocab_is_eog: unsafe extern "C" fn(Ptr, llama_token) -> bool,
    batch_get_one: unsafe extern "C" fn(*mut llama_token, i32) -> Batch,
    decode: unsafe extern "C" fn(Ptr, Batch) -> i32,
    get_memory: unsafe extern "C" fn(Ptr) -> Ptr,
    memory_clear: unsafe extern "C" fn(Ptr, bool),
    memory_seq_rm: unsafe extern "C" fn(Ptr, i32, i32, i32) -> bool,
    n_ctx: unsafe extern "C" fn(Ptr) -> u32,
    adapter_lora_init: unsafe extern "C" fn(Ptr, *const c_char) -> Ptr,
    set_adapters_lora: unsafe extern "C" fn(Ptr, *mut Ptr, usize, *mut f32) -> i32,
    state_seq_get_size: unsafe extern "C" fn(Ptr, i32) -> usize,
    state_seq_get_data: unsafe extern "C" fn(Ptr, *mut u8, usize, i32) -> usize,
    state_seq_set_data: unsafe extern "C" fn(Ptr, *const u8, usize, i32) -> usize,
    set_n_threads: unsafe extern "C" fn(Ptr, i32, i32),
    synchronize: unsafe extern "C" fn(Ptr),
    sampler_chain_init: unsafe extern "C" fn(SamplerChainParams) -> Ptr,
    sampler_chain_add: unsafe extern "C" fn(Ptr, Ptr),
    sampler_init_greedy: unsafe extern "C" fn() -> Ptr,
    sampler_init_grammar: unsafe extern "C" fn(Ptr, *const c_char, *const c_char) -> Ptr,
    sampler_init_top_k: unsafe extern "C" fn(i32) -> Ptr,
    sampler_init_top_p: unsafe extern "C" fn(f32, usize) -> Ptr,
    sampler_init_min_p: unsafe extern "C" fn(f32, usize) -> Ptr,
    sampler_init_temp: unsafe extern "C" fn(f32) -> Ptr,
    sampler_init_dist: unsafe extern "C" fn(u32) -> Ptr,
    sampler_sample: unsafe extern "C" fn(Ptr, Ptr, i32) -> llama_token,
    sampler_free: unsafe extern "C" fn(Ptr),
    print_system_info: unsafe extern "C" fn() -> *const c_char,
    _ggml: Library,
    _llama: Library,
}

unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

macro_rules! sym {
    ($lib:expr, $name:literal) => {
        *$lib.get(concat!($name, "\0").as_bytes()).with_context(|| format!("{} not found", $name))?
    };
}

impl Lib {
    /// Load llama.dll and ggml.dll from a llama.cpp build folder (CPU or Vulkan release zip,
    /// build `PINNED_BUILD`) and its backends (`ggml-cpu-*.dll`, `ggml-vulkan.dll`).
    pub fn load(dir: &Path) -> Result<Lib> {
        let ggml_path = dir.join("ggml.dll");
        let llama_path = dir.join("llama.dll");
        // SAFETY: loading a DLL runs its init code; these are llama.cpp's own release DLLs.
        // The altered search path makes Windows resolve their dependencies in `dir`.
        let (ggml, llama) = unsafe {
            (
                Library::load_with_flags(&ggml_path, LOAD_WITH_ALTERED_SEARCH_PATH).with_context(|| format!("cannot load {}", ggml_path.display()))?,
                Library::load_with_flags(&llama_path, LOAD_WITH_ALTERED_SEARCH_PATH).with_context(|| format!("cannot load {}", llama_path.display()))?,
            )
        };
        // SAFETY: the signatures mirror llama.h / ggml-backend.h of PINNED_BUILD.
        let lib = unsafe {
            Lib {
                backend_load_all_from_path: sym!(ggml, "ggml_backend_load_all_from_path"),
                backend_init: sym!(llama, "llama_backend_init"),
                model_default_params: sym!(llama, "llama_model_default_params"),
                context_default_params: sym!(llama, "llama_context_default_params"),
                sampler_chain_default_params: sym!(llama, "llama_sampler_chain_default_params"),
                model_load_from_file: sym!(llama, "llama_model_load_from_file"),
                model_free: sym!(llama, "llama_model_free"),
                init_from_model: sym!(llama, "llama_init_from_model"),
                free: sym!(llama, "llama_free"),
                model_get_vocab: sym!(llama, "llama_model_get_vocab"),
                model_chat_template: sym!(llama, "llama_model_chat_template"),
                chat_apply_template: sym!(llama, "llama_chat_apply_template"),
                tokenize: sym!(llama, "llama_tokenize"),
                token_to_piece: sym!(llama, "llama_token_to_piece"),
                vocab_is_eog: sym!(llama, "llama_vocab_is_eog"),
                batch_get_one: sym!(llama, "llama_batch_get_one"),
                decode: sym!(llama, "llama_decode"),
                get_memory: sym!(llama, "llama_get_memory"),
                memory_clear: sym!(llama, "llama_memory_clear"),
                memory_seq_rm: sym!(llama, "llama_memory_seq_rm"),
                n_ctx: sym!(llama, "llama_n_ctx"),
                adapter_lora_init: sym!(llama, "llama_adapter_lora_init"),
                set_adapters_lora: sym!(llama, "llama_set_adapters_lora"),
                state_seq_get_size: sym!(llama, "llama_state_seq_get_size"),
                state_seq_get_data: sym!(llama, "llama_state_seq_get_data"),
                state_seq_set_data: sym!(llama, "llama_state_seq_set_data"),
                set_n_threads: sym!(llama, "llama_set_n_threads"),
                synchronize: sym!(llama, "llama_synchronize"),
                sampler_chain_init: sym!(llama, "llama_sampler_chain_init"),
                sampler_chain_add: sym!(llama, "llama_sampler_chain_add"),
                sampler_init_greedy: sym!(llama, "llama_sampler_init_greedy"),
                sampler_init_grammar: sym!(llama, "llama_sampler_init_grammar"),
                sampler_init_top_k: sym!(llama, "llama_sampler_init_top_k"),
                sampler_init_top_p: sym!(llama, "llama_sampler_init_top_p"),
                sampler_init_min_p: sym!(llama, "llama_sampler_init_min_p"),
                sampler_init_temp: sym!(llama, "llama_sampler_init_temp"),
                sampler_init_dist: sym!(llama, "llama_sampler_init_dist"),
                sampler_sample: sym!(llama, "llama_sampler_sample"),
                sampler_free: sym!(llama, "llama_sampler_free"),
                print_system_info: sym!(llama, "llama_print_system_info"),
                _ggml: ggml,
                _llama: llama,
            }
        };
        let d = CString::new(dir.to_string_lossy().as_bytes())?;
        // SAFETY: valid C string; plain init calls.
        unsafe {
            (lib.backend_load_all_from_path)(d.as_ptr());
            (lib.backend_init)();
        }
        lib.check_abi()?;
        Ok(lib)
    }

    /// The default params of b12606 have known values; garbage here means the struct layout
    /// does not match this DLL (another llama.cpp build).
    fn check_abi(&self) -> Result<()> {
        // SAFETY: plain calls returning structs by value.
        let (m, c) = unsafe { ((self.model_default_params)(), (self.context_default_params)()) };
        let ok = c.n_ctx == 512 && c.n_batch == 2048 && c.n_ubatch == 512 && c.n_seq_max == 1
            && c.flash_attn_type == -1 && c.type_k == 1 && c.type_v == 1 && c.offload_kqv && !c.embeddings
            && m.main_gpu == 0 && !m.vocab_only && !m.no_alloc;
        if !ok {
            bail!(
                "llama.dll does not look like llama.cpp {PINNED_BUILD} (default params: n_ctx {} n_batch {} n_ubatch {} seq_max {} fa {} type_k/v {}/{} offload_kqv {} embeddings {} main_gpu {} vocab_only {} no_alloc {}); use the {PINNED_BUILD} release",
                c.n_ctx, c.n_batch, c.n_ubatch, c.n_seq_max, c.flash_attn_type, c.type_k, c.type_v, c.offload_kqv, c.embeddings, m.main_gpu, m.vocab_only, m.no_alloc
            );
        }
        Ok(())
    }

    pub fn system_info(&self) -> String {
        // SAFETY: returns a static C string.
        unsafe { CStr::from_ptr((self.print_system_info)()).to_string_lossy().into_owned() }
    }
}

pub struct LoadOptions {
    pub n_gpu_layers: i32,
    pub ctx: u32,
    pub threads: i32,
    /// No repacked copy of the weights (like llama-server's `--no-repack`).
    pub no_repack: bool,
    /// KV cache type (ggml type id): 1 = f16, 8 = q8_0, 2 = q4_0.
    pub kv_type: i32,
    /// -1 auto, 0 off, 1 on.
    pub flash_attn: i32,
    /// LoRA adapters loaded with the model; requests set their scales (default 0).
    pub adapters: Vec<std::path::PathBuf>,
    /// RAM for saved prompt states (the prompt cache), in bytes. 0 = keep only the live one.
    pub cache_bytes: usize,
}

/// A saved KV state of sequence 0: the tokens it holds and the LoRA scales it was made with.
struct Saved {
    tokens: Vec<llama_token>,
    lora: Vec<f32>,
    data: Vec<u8>,
    used: Instant,
}

/// The ggml type id of a KV cache type name (`-ctk` values of llama-server).
pub fn kv_type_id(name: &str) -> Option<i32> {
    Some(match name {
        "f32" => 0,
        "f16" => 1,
        "q4_0" => 2,
        "q4_1" => 3,
        "q5_0" => 6,
        "q5_1" => 7,
        "q8_0" => 8,
        "bf16" => 30,
        _ => return None,
    })
}

/// Sampling settings. `temperature <= 0` = greedy.
#[derive(Debug, Clone)]
pub struct SamplerOpts {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u32,
}

impl Default for SamplerOpts {
    fn default() -> Self {
        // llama-server's defaults.
        SamplerOpts { temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, seed: 0xFFFF_FFFF }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// End-of-generation token, or the caller asked to stop (a stop string).
    Stop,
    /// `max_tokens` or the context is full.
    Length,
    Cancelled,
}

/// What the caller wants after a piece of text.
pub enum Step {
    Continue,
    Stop,
}

#[derive(Debug, Clone)]
pub struct GenStats {
    pub prompt_tokens: usize,
    /// Prompt tokens reused from the previous request (shared prefix), not read again.
    pub cached_tokens: usize,
    pub prompt_ms: f64,
    pub tokens: usize,
    pub gen_ms: f64,
    /// Time spent in the pause hook between tokens.
    pub paused_ms: f64,
    pub finish: Finish,
}

/// One model with one context. Owned by one thread.
pub struct Model {
    lib: &'static Lib,
    model: Ptr,
    ctx: Ptr,
    vocab: Ptr,
    template: Option<CString>,
    n_ctx: usize,
    /// Tokens now in the KV cache (sequence 0), for prefix reuse.
    cached: Vec<llama_token>,
    /// Threads set on the context now.
    threads: i32,
    adapters: Vec<Ptr>,
    /// LoRA scales the live KV cache was computed with (one per adapter).
    lora: Vec<f32>,
    saved: Vec<Saved>,
    cache_bytes: usize,
}

/// What the pace hook asks for before the next token.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pace {
    pub pause: Duration,
    /// Change the thread count (takes effect from the next decode).
    pub threads: Option<i32>,
}

// The pointers are only used from the thread that owns the Model.
unsafe impl Send for Model {}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: pointers from llama.cpp, freed once.
        unsafe {
            (self.lib.free)(self.ctx);
            (self.lib.model_free)(self.model);
        }
    }
}

/// Cuts a byte stream into valid UTF-8 text, keeping an incomplete character for later.
#[derive(Default)]
struct Utf8Buf(Vec<u8>);

impl Utf8Buf {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.0.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.0) {
            Ok(_) => self.0.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(), // incomplete: wait for more
            Err(_) => self.0.len(),                              // invalid: pass it on lossily
        };
        let out: Vec<u8> = self.0.drain(..valid).collect();
        String::from_utf8_lossy(&out).into_owned()
    }
    fn flush(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.0).into_owned();
        self.0.clear();
        out
    }
}

impl Model {
    pub fn load(lib: &'static Lib, path: &Path, o: &LoadOptions) -> Result<Model> {
        // SAFETY: llama.cpp API calls with params from its own defaults.
        unsafe {
            let mut mp = (lib.model_default_params)();
            mp.n_gpu_layers = o.n_gpu_layers;
            mp.use_extra_bufts = !o.no_repack;
            let p = CString::new(path.to_string_lossy().as_bytes())?;
            let model = (lib.model_load_from_file)(p.as_ptr(), mp);
            if model.is_null() {
                bail!("cannot load model {}", path.display());
            }
            let mut cp = (lib.context_default_params)();
            cp.n_ctx = o.ctx;
            cp.n_batch = o.ctx.min(2048);
            cp.n_ubatch = cp.n_batch.min(512);
            // We only read the last token's logits. With a big vocabulary (Gemma: 262k) the default
            // (outputs for a whole batch) makes a ~540 MB compute buffer.
            cp.n_outputs_max = 1;
            cp.n_threads = o.threads;
            cp.n_threads_batch = o.threads;
            cp.type_k = o.kv_type;
            cp.type_v = o.kv_type;
            cp.flash_attn_type = o.flash_attn;
            cp.no_perf = true;
            let ctx = (lib.init_from_model)(model, cp);
            if ctx.is_null() {
                (lib.model_free)(model);
                bail!("cannot create a context (ctx {})", o.ctx);
            }
            let vocab = (lib.model_get_vocab)(model);
            let t = (lib.model_chat_template)(model, null());
            let template = (!t.is_null()).then(|| CStr::from_ptr(t).to_owned());
            let n_ctx = (lib.n_ctx)(ctx) as usize;
            let mut adapters = Vec::new();
            for a in &o.adapters {
                let p = CString::new(a.to_string_lossy().as_bytes())?;
                let ad = (lib.adapter_lora_init)(model, p.as_ptr());
                if ad.is_null() {
                    (lib.free)(ctx);
                    (lib.model_free)(model);
                    bail!("cannot load LoRA adapter {}", a.display());
                }
                adapters.push(ad);
            }
            let lora = vec![0.0; adapters.len()];
            let mut m = Model { lib, model, ctx, vocab, template, n_ctx, cached: Vec::new(), threads: o.threads, adapters, lora: Vec::new(), saved: Vec::new(), cache_bytes: o.cache_bytes };
            m.apply_lora(&lora)?;
            Ok(m)
        }
    }

    pub fn n_adapters(&self) -> usize {
        self.adapters.len()
    }

    /// Set the adapters' scales. The live KV cache is only valid for the scales it was made
    /// with, so a change drops it (saved states keep their own scales).
    fn apply_lora(&mut self, scales: &[f32]) -> Result<()> {
        if scales.len() != self.adapters.len() {
            bail!("{} LoRA scale(s) for {} adapter(s)", scales.len(), self.adapters.len());
        }
        if self.lora == scales {
            return Ok(());
        }
        if !self.adapters.is_empty() {
            let mut s = scales.to_vec();
            // SAFETY: adapters of this model, one scale each.
            let rc = unsafe { (self.lib.set_adapters_lora)(self.ctx, self.adapters.as_mut_ptr(), self.adapters.len(), s.as_mut_ptr()) };
            if rc != 0 {
                bail!("cannot set the LoRA adapters ({rc})");
            }
        }
        self.save_live();
        self.trim_cache(0);
        self.lora = scales.to_vec();
        Ok(())
    }

    /// Keep the live KV state in the prompt cache before it is overwritten (big enough ones only).
    fn save_live(&mut self) {
        if self.cache_bytes == 0 || self.cached.len() < 32 {
            return;
        }
        if self.saved.iter().any(|s| s.tokens == self.cached && s.lora == self.lora) {
            return;
        }
        // SAFETY: valid context; the buffer has the size llama.cpp asked for.
        let data = unsafe {
            let n = (self.lib.state_seq_get_size)(self.ctx, 0);
            if n == 0 || n > self.cache_bytes {
                return;
            }
            let mut buf = vec![0u8; n];
            let got = (self.lib.state_seq_get_data)(self.ctx, buf.as_mut_ptr(), n, 0);
            if got == 0 {
                return;
            }
            buf.truncate(got);
            buf
        };
        self.saved.push(Saved { tokens: self.cached.clone(), lora: self.lora.clone(), data, used: Instant::now() });
        // Least recently used first out, within the budget.
        while self.saved.iter().map(|s| s.data.len()).sum::<usize>() > self.cache_bytes {
            let (k, _) = self.saved.iter().enumerate().min_by_key(|(_, s)| s.used).expect("not empty");
            self.saved.remove(k);
        }
    }

    /// Restore the saved state that shares the longest prefix with `toks`, if it beats the
    /// live cache. Returns the shared prefix length now in the cache.
    fn best_prefix(&mut self, toks: &[llama_token]) -> usize {
        let common = |a: &[llama_token]| a.iter().zip(toks).take_while(|(x, y)| x == y).count();
        let live = common(&self.cached);
        let best = self
            .saved
            .iter()
            .enumerate()
            .filter(|(_, s)| s.lora == self.lora)
            .map(|(k, s)| (k, common(&s.tokens)))
            .max_by_key(|&(_, c)| c);
        let Some((k, c)) = best else { return live };
        if c <= live + 8 {
            return live;
        }
        self.save_live();
        // The save may have evicted entries: find the chosen one again by its tokens.
        let want = self.saved[k.min(self.saved.len().saturating_sub(1))].tokens.clone();
        let Some(k) = self.saved.iter().position(|s| s.tokens == want && s.lora == self.lora) else { return live };
        // SAFETY: valid context and memory; data from llama_state_seq_get_data on this model.
        let ok = unsafe {
            (self.lib.memory_clear)((self.lib.get_memory)(self.ctx), true);
            let s = &self.saved[k];
            (self.lib.state_seq_set_data)(self.ctx, s.data.as_ptr(), s.data.len(), 0) > 0
        };
        if !ok {
            self.saved.remove(k);
            self.cached.clear();
            return 0;
        }
        self.saved[k].used = Instant::now();
        self.cached = self.saved[k].tokens.clone();
        c
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    pub fn set_threads(&mut self, n: i32) {
        if n > 0 && n != self.threads {
            // SAFETY: valid context; between decodes.
            unsafe { (self.lib.set_n_threads)(self.ctx, n, n) }
            self.threads = n;
        }
    }

    pub fn threads(&self) -> i32 {
        self.threads
    }

    /// Format (role, content) messages with the model's built-in chat template.
    pub fn apply_template(&self, msgs: &[(String, String)]) -> Result<String> {
        let tmpl = self.template.as_ref().ok_or_else(|| anyhow!("the model has no chat template"))?;
        let owned: Vec<(CString, CString)> =
            msgs.iter().map(|(r, c)| Ok((CString::new(r.as_str())?, CString::new(c.as_str())?))).collect::<Result<_>>()?;
        let chat: Vec<ChatMessage> = owned.iter().map(|(r, c)| ChatMessage { role: r.as_ptr(), content: c.as_ptr() }).collect();
        let mut buf = vec![0u8; 2 * msgs.iter().map(|m| m.1.len() + m.0.len()).sum::<usize>() + 512];
        for _ in 0..2 {
            // SAFETY: buffer and messages are valid for the call.
            let n = unsafe { (self.lib.chat_apply_template)(tmpl.as_ptr(), chat.as_ptr(), chat.len(), true, buf.as_mut_ptr() as *mut c_char, buf.len() as i32) };
            if n < 0 {
                bail!("the model's chat template is not one llama_chat_apply_template knows");
            }
            if (n as usize) <= buf.len() {
                buf.truncate(n as usize);
                return Ok(String::from_utf8_lossy(&buf).into_owned());
            }
            buf.resize(n as usize, 0);
        }
        bail!("chat template output kept growing")
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<llama_token>> {
        let mut toks = vec![0; text.len() + 16];
        // SAFETY: buffers valid for the call.
        let n = unsafe { (self.lib.tokenize)(self.vocab, text.as_ptr() as *const c_char, text.len() as i32, toks.as_mut_ptr(), toks.len() as i32, true, true) };
        if n < 0 {
            bail!("tokenize failed ({n})");
        }
        toks.truncate(n as usize);
        Ok(toks)
    }

    fn piece(&self, t: llama_token) -> Vec<u8> {
        let mut buf = [0u8; 256];
        // SAFETY: buffer valid for the call.
        let n = unsafe { (self.lib.token_to_piece)(self.vocab, t, buf.as_mut_ptr() as *mut c_char, buf.len() as i32, 0, false) };
        buf[..n.max(0) as usize].to_vec()
    }

    /// Drop the KV cache after `keep` tokens. False if the model can't cut its cache there
    /// (some SWA or recurrent models); then the whole cache is cleared.
    fn trim_cache(&mut self, keep: usize) -> usize {
        // SAFETY: valid context and its memory.
        unsafe {
            let mem = (self.lib.get_memory)(self.ctx);
            if keep > 0 && (self.lib.memory_seq_rm)(mem, 0, keep as i32, -1) {
                self.cached.truncate(keep);
                return keep;
            }
            (self.lib.memory_clear)(mem, true);
        }
        self.cached.clear();
        0
    }

    fn sampler(&self, grammar: Option<&str>, s: &SamplerOpts) -> Result<Ptr> {
        // SAFETY: llama.cpp sampler API; the chain owns the samplers added to it.
        unsafe {
            let chain = (self.lib.sampler_chain_init)((self.lib.sampler_chain_default_params)());
            if let Some(g) = grammar {
                let gs = CString::new(g)?;
                let gr = (self.lib.sampler_init_grammar)(self.vocab, gs.as_ptr(), c"root".as_ptr());
                if gr.is_null() {
                    (self.lib.sampler_free)(chain);
                    bail!("the grammar does not parse");
                }
                (self.lib.sampler_chain_add)(chain, gr);
            }
            if s.temperature <= 0.0 {
                (self.lib.sampler_chain_add)(chain, (self.lib.sampler_init_greedy)());
            } else {
                if s.top_k > 0 {
                    (self.lib.sampler_chain_add)(chain, (self.lib.sampler_init_top_k)(s.top_k));
                }
                if s.top_p < 1.0 {
                    (self.lib.sampler_chain_add)(chain, (self.lib.sampler_init_top_p)(s.top_p, 1));
                }
                if s.min_p > 0.0 {
                    (self.lib.sampler_chain_add)(chain, (self.lib.sampler_init_min_p)(s.min_p, 1));
                }
                (self.lib.sampler_chain_add)(chain, (self.lib.sampler_init_temp)(s.temperature));
                (self.lib.sampler_chain_add)(chain, (self.lib.sampler_init_dist)(s.seed));
            }
            Ok(chain)
        }
    }

    /// Generate from a formatted prompt. The KV cache of the shared prefix with the previous
    /// prompt is reused. `on_text` gets valid UTF-8 pieces and may stop the generation; `pace`
    /// runs between tokens with the token count and the last step's compute time, and returns
    /// a pause and/or a new thread count (the control point a server can't give us). `cancel`
    /// is checked between tokens.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        grammar: Option<&str>,
        s: &SamplerOpts,
        lora: &[f32],
        cancel: &std::sync::atomic::AtomicBool,
        mut on_text: impl FnMut(&str) -> Step,
        mut pace: impl FnMut(usize, Duration) -> Pace,
    ) -> Result<GenStats> {
        use std::sync::atomic::Ordering;
        let toks = self.tokenize(prompt)?;
        if toks.len() >= self.n_ctx {
            bail!("the prompt has {} tokens; the context holds {}", toks.len(), self.n_ctx);
        }
        self.apply_lora(lora)?;
        // Reuse the longest shared prefix (live cache or a saved state), but always read at
        // least the last prompt token (for its logits).
        let common = self.best_prefix(&toks).min(toks.len() - 1);
        if common < self.cached.len() {
            self.save_live();
        }
        let keep = self.trim_cache(common);
        let chain = self.sampler(grammar, s)?;
        struct FreeOnDrop(&'static Lib, Ptr);
        impl Drop for FreeOnDrop {
            fn drop(&mut self) {
                // SAFETY: the chain is freed once.
                unsafe { (self.0.sampler_free)(self.1) }
            }
        }
        let _free = FreeOnDrop(self.lib, chain);

        let t0 = Instant::now();
        let mut rest = toks[keep..].to_vec();
        for chunk in rest.chunks_mut(2048) {
            // SAFETY: our own context and token buffer.
            let rc = unsafe { (self.lib.decode)(self.ctx, (self.lib.batch_get_one)(chunk.as_mut_ptr(), chunk.len() as i32)) };
            if rc != 0 {
                self.trim_cache(0);
                bail!("reading the prompt failed ({rc})");
            }
            self.cached.extend_from_slice(chunk);
        }
        // SAFETY: valid context. A GPU decode returns before it is done.
        unsafe { (self.lib.synchronize)(self.ctx) };
        let prompt_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let mut utf8 = Utf8Buf::default();
        let mut paused = Duration::ZERO;
        let mut n = 0;
        let mut work_start = Instant::now();
        let finish = loop {
            if cancel.load(Ordering::SeqCst) {
                break Finish::Cancelled;
            }
            if n >= max_tokens || self.cached.len() >= self.n_ctx {
                break Finish::Length;
            }
            // SAFETY: valid sampler chain and context.
            let mut t = unsafe { (self.lib.sampler_sample)(chain, self.ctx, -1) };
            // SAFETY: valid vocab.
            if unsafe { (self.lib.vocab_is_eog)(self.vocab, t) } {
                break Finish::Stop;
            }
            n += 1;
            let text = utf8.push(&self.piece(t));
            if !text.is_empty() {
                if let Step::Stop = on_text(&text) {
                    break Finish::Stop;
                }
            }
            let p = pace(n, work_start.elapsed());
            if let Some(t) = p.threads {
                self.set_threads(t);
            }
            if !p.pause.is_zero() {
                std::thread::sleep(p.pause);
                paused += p.pause;
            }
            work_start = Instant::now();
            // SAFETY: our own context; one token.
            let rc = unsafe { (self.lib.decode)(self.ctx, (self.lib.batch_get_one)(&mut t, 1)) };
            if rc != 0 {
                self.trim_cache(0);
                bail!("decode failed ({rc})");
            }
            self.cached.push(t);
        };
        let tail = utf8.flush();
        if !tail.is_empty() && finish != Finish::Cancelled {
            on_text(&tail);
        }
        Ok(GenStats {
            prompt_tokens: toks.len(),
            cached_tokens: keep,
            prompt_ms,
            tokens: n,
            gen_ms: t1.elapsed().as_secs_f64() * 1000.0,
            paused_ms: paused.as_secs_f64() * 1000.0,
            finish,
        })
    }
}

/// Load a llama.cpp build folder once for the whole process.
pub fn load_static(dir: &Path) -> Result<&'static Lib> {
    Ok(Box::leak(Box::new(Lib::load(dir)?)))
}

/// `jitmo-runtime --engine-test`: load a model in-process, generate, print timings as JSON.
pub fn run_test(args: &[String]) -> Result<()> {
    let mut lib_dir = r"V:\jitmo-sandbox\llama-b12606-bin-win-cpu-x64".to_string();
    let (mut model, mut prompt, mut system) = (String::new(), "Name three colors.".to_string(), String::new());
    let (mut ngl, mut threads, mut ctx, mut max, mut runs, mut pause_ms) = (0i32, 3i32, 2048u32, 64usize, 2usize, 0u64);
    let mut grammar = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut v = || it.next().cloned().ok_or_else(|| anyhow!("{a} needs a value"));
        match a.as_str() {
            "--lib-dir" => lib_dir = v()?,
            "--model" => model = v()?,
            "--prompt" => prompt = v()?,
            "--system" => system = v()?,
            "--ngl" => ngl = v()?.parse()?,
            "--threads" => threads = v()?.parse()?,
            "--ctx" => ctx = v()?.parse()?,
            "--max-tokens" => max = v()?.parse()?,
            "--runs" => runs = v()?.parse()?,
            "--pause-ms" => pause_ms = v()?.parse()?,
            "--grammar" => grammar = Some(v()?),
            other => bail!("unknown --engine-test option {other}"),
        }
    }
    if model.is_empty() {
        bail!("--engine-test needs --model <file.gguf>");
    }
    let t = Instant::now();
    let lib = load_static(Path::new(&lib_dir))?;
    let lib_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let o = LoadOptions { n_gpu_layers: ngl, ctx, threads, no_repack: true, kv_type: 1, flash_attn: -1, adapters: Vec::new(), cache_bytes: 0 };
    let mut m = Model::load(lib, Path::new(&model), &o)?;
    let load_ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut msgs = Vec::new();
    if !system.is_empty() {
        msgs.push(("system".to_string(), system.clone()));
    }
    msgs.push(("user".to_string(), prompt.clone()));
    let formatted = m.apply_template(&msgs)?;
    let greedy = SamplerOpts { temperature: 0.0, ..Default::default() };
    let never = std::sync::atomic::AtomicBool::new(false);
    for run in 1..=runs {
        if run > 1 {
            m.trim_cache(0); // time the whole prompt each run
        }
        let mut text = String::new();
        let g = m.generate(&formatted, max, grammar.as_deref(), &greedy, &[], &never, |p| {
            text.push_str(p);
            Step::Continue
        }, |_, _| Pace { pause: Duration::from_millis(pause_ms), threads: None })?;
        let tps = if g.gen_ms - g.paused_ms > 0.0 { g.tokens as f64 * 1000.0 / (g.gen_ms - g.paused_ms) } else { 0.0 };
        println!(
            "{}",
            serde_json::json!({
                "run": run, "lib_ms": lib_ms.round(), "load_ms": load_ms.round(), "ngl": ngl, "threads": threads,
                "prompt_tokens": g.prompt_tokens, "prompt_ms": (g.prompt_ms * 10.0).round() / 10.0,
                "prompt_tps": (g.prompt_tokens as f64 * 1000.0 / g.prompt_ms).round(),
                "tokens": g.tokens, "gen_tps": (tps * 10.0).round() / 10.0, "paused_ms": g.paused_ms.round(), "text": text,
            })
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_pieces_wait_for_whole_characters() {
        let mut b = Utf8Buf::default();
        let e = "é".as_bytes(); // 2 bytes
        assert_eq!(b.push(&[b'a', e[0]]), "a");
        assert_eq!(b.push(&[e[1], b'b']), "éb");
        assert_eq!(b.push(&[0xE2]), "");
        assert_eq!(b.flush(), "\u{FFFD}");
    }

    #[test]
    fn kv_type_names() {
        assert_eq!((kv_type_id("f16"), kv_type_id("q8_0"), kv_type_id("x")), (Some(1), Some(8), None));
    }
}
