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
use std::ptr::{null, null_mut};
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
    set_n_threads: unsafe extern "C" fn(Ptr, i32, i32),
    synchronize: unsafe extern "C" fn(Ptr),
    sampler_chain_init: unsafe extern "C" fn(SamplerChainParams) -> Ptr,
    sampler_chain_add: unsafe extern "C" fn(Ptr, Ptr),
    sampler_init_greedy: unsafe extern "C" fn() -> Ptr,
    sampler_init_grammar: unsafe extern "C" fn(Ptr, *const c_char, *const c_char) -> Ptr,
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
                set_n_threads: sym!(llama, "llama_set_n_threads"),
                synchronize: sym!(llama, "llama_synchronize"),
                sampler_chain_init: sym!(llama, "llama_sampler_chain_init"),
                sampler_chain_add: sym!(llama, "llama_sampler_chain_add"),
                sampler_init_greedy: sym!(llama, "llama_sampler_init_greedy"),
                sampler_init_grammar: sym!(llama, "llama_sampler_init_grammar"),
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
}

/// One model with one context (P0).
pub struct Model<'a> {
    lib: &'a Lib,
    model: Ptr,
    ctx: Ptr,
    vocab: Ptr,
    template: Option<CString>,
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        // SAFETY: pointers from llama.cpp, freed once.
        unsafe {
            (self.lib.free)(self.ctx);
            (self.lib.model_free)(self.model);
        }
    }
}

pub struct Generation {
    pub text: String,
    pub prompt_tokens: usize,
    pub prompt_ms: f64,
    pub tokens: usize,
    pub gen_ms: f64,
    /// Time spent in the pause hook between tokens.
    pub paused_ms: f64,
}

impl<'a> Model<'a> {
    pub fn load(lib: &'a Lib, path: &Path, o: &LoadOptions) -> Result<Model<'a>> {
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
            cp.n_threads = o.threads;
            cp.n_threads_batch = o.threads;
            cp.no_perf = true;
            let ctx = (lib.init_from_model)(model, cp);
            if ctx.is_null() {
                (lib.model_free)(model);
                bail!("cannot create a context");
            }
            let vocab = (lib.model_get_vocab)(model);
            let t = (lib.model_chat_template)(model, null());
            let template = (!t.is_null()).then(|| CStr::from_ptr(t).to_owned());
            Ok(Model { lib, model, ctx, vocab, template })
        }
    }

    pub fn set_threads(&self, n: i32) {
        // SAFETY: valid context.
        unsafe { (self.lib.set_n_threads)(self.ctx, n, n) }
    }

    /// Format (role, content) messages with the model's built-in chat template.
    pub fn apply_template(&self, msgs: &[(&str, &str)]) -> Result<String> {
        let tmpl = self.template.as_ref().ok_or_else(|| anyhow!("the model has no chat template"))?;
        let owned: Vec<(CString, CString)> =
            msgs.iter().map(|(r, c)| Ok((CString::new(*r)?, CString::new(*c)?))).collect::<Result<_>>()?;
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

    fn tokenize(&self, text: &str) -> Result<Vec<llama_token>> {
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

    /// Greedy generation from a formatted prompt. `grammar` = GBNF (root rule `root`).
    /// `pace` runs between tokens and may sleep: the pause point a server can't give us.
    pub fn generate(&self, prompt: &str, max_tokens: usize, grammar: Option<&str>, mut pace: impl FnMut(usize) -> Duration) -> Result<Generation> {
        let mut toks = self.tokenize(prompt)?;
        // SAFETY: llama.cpp calls on our own context, sampler and token buffers.
        unsafe {
            (self.lib.memory_clear)((self.lib.get_memory)(self.ctx), true);
            let chain = (self.lib.sampler_chain_init)((self.lib.sampler_chain_default_params)());
            if let Some(g) = grammar {
                let gs = CString::new(g)?;
                let s = (self.lib.sampler_init_grammar)(self.vocab, gs.as_ptr(), c"root".as_ptr());
                if s.is_null() {
                    (self.lib.sampler_free)(chain);
                    bail!("grammar does not parse");
                }
                (self.lib.sampler_chain_add)(chain, s);
            }
            (self.lib.sampler_chain_add)(chain, (self.lib.sampler_init_greedy)());

            let t0 = Instant::now();
            let n_prompt = toks.len();
            for chunk in toks.chunks_mut(2048) {
                let rc = (self.lib.decode)(self.ctx, (self.lib.batch_get_one)(chunk.as_mut_ptr(), chunk.len() as i32));
                if rc != 0 {
                    (self.lib.sampler_free)(chain);
                    bail!("decode of the prompt failed ({rc})");
                }
            }
            (self.lib.synchronize)(self.ctx); // a GPU decode returns before it is done
            let prompt_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let mut out = Vec::new();
            let mut paused = Duration::ZERO;
            let mut n = 0;
            while n < max_tokens {
                let mut t = (self.lib.sampler_sample)(chain, self.ctx, -1);
                if (self.lib.vocab_is_eog)(self.vocab, t) {
                    break;
                }
                out.extend(self.piece(t));
                n += 1;
                let wait = pace(n);
                if !wait.is_zero() {
                    std::thread::sleep(wait);
                    paused += wait;
                }
                let rc = (self.lib.decode)(self.ctx, (self.lib.batch_get_one)(&mut t, 1));
                if rc != 0 {
                    (self.lib.sampler_free)(chain);
                    bail!("decode failed ({rc})");
                }
            }
            (self.lib.sampler_free)(chain);
            Ok(Generation {
                text: String::from_utf8_lossy(&out).into_owned(),
                prompt_tokens: n_prompt,
                prompt_ms,
                tokens: n,
                gen_ms: t1.elapsed().as_secs_f64() * 1000.0,
                paused_ms: paused.as_secs_f64() * 1000.0,
            })
        }
    }
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
    let lib = Lib::load(Path::new(&lib_dir))?;
    let lib_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let m = Model::load(&lib, Path::new(&model), &LoadOptions { n_gpu_layers: ngl, ctx, threads, no_repack: true })?;
    let load_ms = t.elapsed().as_secs_f64() * 1000.0;
    let mut msgs = Vec::new();
    if !system.is_empty() {
        msgs.push(("system", system.as_str()));
    }
    msgs.push(("user", prompt.as_str()));
    let formatted = m.apply_template(&msgs)?;
    for run in 1..=runs {
        let g = m.generate(&formatted, max, grammar.as_deref(), |_| Duration::from_millis(pause_ms))?;
        let tps = if g.gen_ms - g.paused_ms > 0.0 { g.tokens as f64 * 1000.0 / (g.gen_ms - g.paused_ms) } else { 0.0 };
        println!(
            "{}",
            serde_json::json!({
                "run": run, "lib_ms": lib_ms.round(), "load_ms": load_ms.round(), "ngl": ngl, "threads": threads,
                "prompt_tokens": g.prompt_tokens, "prompt_ms": (g.prompt_ms * 10.0).round() / 10.0,
                "prompt_tps": (g.prompt_tokens as f64 * 1000.0 / g.prompt_ms).round(),
                "tokens": g.tokens, "gen_tps": (tps * 10.0).round() / 10.0, "paused_ms": g.paused_ms.round(), "text": g.text,
            })
        );
    }
    let _ = null_mut::<c_void>();
    Ok(())
}
