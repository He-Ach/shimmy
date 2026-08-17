//! Candle + CUDA backend.
//!
//! Exists because the WebGPU engine (Airframe) could not run the models this fork targets:
//! a 9B with a 151k vocab loses the GPU device during prefill
//! (`Device::poll Validation Error: Parent device is lost`), and that reproduced with
//! prefill chunks from 512 down to 16, so it is not the Windows TDR watchdog.
//!
//! Candle was measured on the same machine, same weights (Ollama's own `qwen3:8b` GGUF
//! blob), alone on the GPU:
//!
//! | engine        | decode        |
//! |---------------|---------------|
//! | Ollama        | 39.3 tok/s    |
//! | candle + CUDA | **42.8 tok/s**|
//!
//! It also supports the architectures we actually run — `quantized_qwen3` ships in
//! candle-transformers, where ruvllm's loader rejected qwen3 outright.
//!
//! What this backend deliberately does *not* own: the HTTP surface, model discovery, chat
//! templates, or tool calling. Those are shimmy's, and they are engine-agnostic — which is
//! the whole reason a second engine is a file rather than a rewrite.
//!
//! Cold start is the known weakness: ~69s to move an 8B into VRAM versus Ollama's ~2.5s
//! from page cache, plus a slow first forward pass while CUDA kernels warm. Steady-state
//! decode is where the win is, so a long-lived server amortises it and a one-shot CLI does
//! not.

use super::{GenOptions, InferenceEngine, LoadedModel, ModelSpec};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub struct CandleEngine;

impl CandleEngine {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CandleEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceEngine for CandleEngine {
    async fn load(&self, spec: &ModelSpec) -> Result<Box<dyn LoadedModel>> {
        let path = spec.base_path.clone();
        // Loading is seconds-to-a-minute of blocking file and PCIe work; doing it on the
        // async runtime would stall every other request on the server.
        let model = tokio::task::spawn_blocking(move || CandleModel::open(&path)).await??;
        Ok(Box::new(model))
    }
}

pub struct CandleModel {
    path: PathBuf,
    /// The weights are stateful across a turn (KV cache lives inside), so generation is
    /// serialised per model. One turn at a time is what the engine is for here.
    weights: Mutex<ModelWeights>,
    tokenizer: shimmytok::Tokenizer,
    device: Device,
}

impl CandleModel {
    fn open(path: &Path) -> Result<Self> {
        // CUDA or nothing: the CPU path exists in candle but is not what was measured, and
        // silently falling back to it would turn a 42.8 tok/s engine into an unusable one
        // while reporting success.
        let device = Device::new_cuda(0)
            .map_err(|e| anyhow!("candle backend requires a CUDA device: {e}"))?;

        let mut file = std::fs::File::open(path)
            .map_err(|e| anyhow!("cannot open {}: {e}", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow!("unreadable GGUF {}: {e}", path.display()))?;

        // Fail on the wrong architecture rather than producing noise: quantized_qwen3 reads
        // `qwen3.*` metadata keys, and a llama GGUF would fail deep inside with a missing
        // key rather than here with a sentence.
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        if !arch.starts_with("qwen3") && !arch.starts_with("qwen35") {
            return Err(anyhow!(
                "candle backend implements qwen3-family models; this GGUF is '{arch}'"
            ));
        }

        let weights = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| anyhow!("candle could not load {}: {e}", path.display()))?;
        let tokenizer = shimmytok::Tokenizer::from_gguf_file(path)
            .map_err(|e| anyhow!("no usable tokenizer in {}: {e}", path.display()))?;

        Ok(Self {
            path: path.to_path_buf(),
            weights: Mutex::new(weights),
            tokenizer,
            device,
        })
    }

    /// Greedy pick from the final position's logits.
    fn argmax(logits: &Tensor) -> Result<u32> {
        let last = logits.dims().len() - 2;
        let row = logits
            .narrow(last, logits.dim(last)? - 1, 1)?
            .squeeze(last)?
            .squeeze(0)?;
        let v: Vec<f32> = row.to_dtype(candle_core::DType::F32)?.to_vec1()?;
        let mut best = 0usize;
        for (i, x) in v.iter().enumerate() {
            if *x > v[best] {
                best = i;
            }
        }
        Ok(best as u32)
    }
}

#[async_trait]
impl LoadedModel for CandleModel {
    async fn generate(
        &self,
        prompt: &str,
        opts: GenOptions,
        mut on_token: Option<Box<dyn FnMut(String) + Send>>,
    ) -> Result<String> {
        let ids = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("tokenizing failed for {}: {e}", self.path.display()))?;
        if ids.is_empty() {
            return Ok(String::new());
        }

        let mut weights = self
            .weights
            .lock()
            .map_err(|_| anyhow!("model lock poisoned by an earlier panic"))?;

        // Prefill: the whole prompt in one forward pass, then decode one token at a time
        // against the KV cache — the regime the 42.8 tok/s figure describes.
        let input = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = weights.forward(&input, 0)?;
        let mut offset = ids.len();

        let mut out = String::new();
        for _ in 0..opts.max_tokens {
            let next = Self::argmax(&logits)?;
            if next == self.tokenizer.eos_token() {
                break;
            }
            if let Ok(piece) = self.tokenizer.decode(&[next], true) {
                if let Some(cb) = on_token.as_mut() {
                    cb(piece.clone());
                }
                out.push_str(&piece);
                // Stop strings are the template's business, but the engine has to honour
                // them or a chat turn runs to max_tokens every time.
                if opts.stop_tokens.iter().any(|s| !s.is_empty() && out.ends_with(s.as_str())) {
                    break;
                }
            }
            let step = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            logits = weights.forward(&step, offset)?;
            offset += 1;
        }
        Ok(out)
    }
}
