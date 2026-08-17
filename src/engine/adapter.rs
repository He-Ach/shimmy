use anyhow::Result;
use async_trait::async_trait;

#[cfg(feature = "huggingface")]
use super::GenOptions;
use super::{InferenceEngine, LoadedModel, ModelSpec};

#[cfg(feature = "huggingface")]
use super::{UniversalEngine, UniversalModel, UniversalModelSpec};

/// Universal adapter that bridges legacy and new engine interfaces
pub struct InferenceEngineAdapter {
    #[cfg(feature = "huggingface")]
    huggingface_engine: super::huggingface::HuggingFaceEngine,
    #[cfg(feature = "mlx")]
    mlx_engine: super::mlx::MLXEngine,
    #[cfg(feature = "candle")]
    candle_engine: super::candle_engine::CandleEngine,
    safetensors_engine: super::safetensors_native::SafeTensorsEngine,
    // Note: loaded_models removed as caching is not currently implemented
}

impl Default for InferenceEngineAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceEngineAdapter {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "huggingface")]
            huggingface_engine: super::huggingface::HuggingFaceEngine::new(),
            #[cfg(feature = "mlx")]
            mlx_engine: super::mlx::MLXEngine::new(),
            #[cfg(feature = "candle")]
            candle_engine: super::candle_engine::CandleEngine::new(),
            safetensors_engine: super::safetensors_native::SafeTensorsEngine::new(),
        }
    }

    /// Create adapter with specific GPU backend from CLI
    pub fn new_with_backend(_gpu_backend: Option<&str>) -> Self {
        Self {
            #[cfg(feature = "huggingface")]
            huggingface_engine: super::huggingface::HuggingFaceEngine::new(),
            #[cfg(feature = "mlx")]
            mlx_engine: super::mlx::MLXEngine::new(),
            #[cfg(feature = "candle")]
            candle_engine: super::candle_engine::CandleEngine::new(),
            safetensors_engine: super::safetensors_native::SafeTensorsEngine::new(),
        }
    }

    /// Auto-detect best backend for model
    fn select_backend(&self, spec: &ModelSpec) -> BackendChoice {
        // Check file extension and path patterns to determine optimal backend
        let path_str = spec.base_path.to_string_lossy();

        // FIRST: Check for explicit file extensions - these take priority over model IDs
        if let Some(ext) = spec.base_path.extension().and_then(|s| s.to_str()) {
            match ext {
                "safetensors" => {
                    // SafeTensors files ALWAYS use SafeTensors engine, regardless of source
                    return BackendChoice::SafeTensors;
                }
                "gguf" => {
                    // Candle wins over Airframe for GGUF when both are compiled in: it is the
                    // engine that actually completed a prefill on a 9B, and it was faster on
                    // the models measured (42.8 vs 39.3 tok/s against Ollama on the same card).
                    #[cfg(feature = "candle")]
                    {
                        return BackendChoice::Candle;
                    }
                    // GGUF models require the Airframe GPU engine (shimmy_server_gpu)
                    // or a build compiled with the GPU backend.
                    #[cfg(not(feature = "candle"))]
                    {
                        return BackendChoice::Unsupported(
                            "GGUF models require a GPU engine. \
                             Build with --features candle (CUDA) or --features airframe (WebGPU). \
                             See https://github.com/Michael-A-Kuykendall/shimmy for setup."
                                .to_string(),
                        );
                    }
                }
                #[cfg(feature = "mlx")]
                "npz" | "mlx" => {
                    // MLX native format
                    return BackendChoice::MLX;
                }
                _ => {} // Continue with other checks
            }
        }

        // SECOND: Check for HuggingFace model IDs (format: "org/model-name")
        // This check runs regardless of feature flags to prevent false positives
        // in the GGUF name-pattern check below.
        if path_str.contains('/') && !path_str.contains('\\') && !path_str.contains('.') {
            // Looks like a HuggingFace model ID (has slash, no backslash, no file extension)
            #[cfg(feature = "huggingface")]
            {
                return BackendChoice::HuggingFace;
            }
            #[cfg(not(feature = "huggingface"))]
            {
                // HuggingFace ID detected but HF feature not compiled in — SafeTensors is the
                // closest stateless fallback; Airframe handles the real inference path.
                return BackendChoice::SafeTensors;
            }
        }

        // THIRD: Check for MLX compatibility on Apple Silicon
        #[cfg(feature = "mlx")]
        {
            // Check if we're on Apple Silicon and model is MLX-compatible
            if super::mlx::MLXEngine::is_hardware_supported() {
                let model_name = spec.name.to_lowercase();
                if model_name.contains("llama")
                    || model_name.contains("mistral")
                    || model_name.contains("phi")
                    || model_name.contains("qwen")
                {
                    // Prefer MLX for known compatible models on Apple Silicon
                    return BackendChoice::MLX;
                }
            }
        }

        // FOURTH: Check for Ollama blob files and other patterns

        // Check for Ollama blob files (GGUF files without extension)
        if path_str.contains("ollama") && path_str.contains("blobs") && path_str.contains("sha256-")
        {
            // Ollama blobs are GGUF without the extension, so they belong to the GGUF engine
            // before the HuggingFace ID heuristics get a look at them.
            #[cfg(feature = "candle")]
            {
                return BackendChoice::Candle;
            }
            #[cfg(all(feature = "huggingface", not(feature = "candle")))]
            {
                return BackendChoice::HuggingFace;
            }
            #[cfg(not(any(feature = "huggingface", feature = "candle")))]
            {
                return BackendChoice::Unsupported(
                    "Ollama blob models require a GPU engine or the HuggingFace backend. \
                     Build with --features candle (CUDA) or --features airframe (WebGPU)."
                        .to_string(),
                );
            }
        }

        // Check for other patterns that indicate GGUF files
        if path_str.contains(".gguf")
            || spec.name.contains("llama")
            || spec.name.contains("phi")
            || spec.name.contains("qwen")
            || spec.name.contains("gemma")
            || spec.name.contains("mistral")
        {
            #[cfg(feature = "candle")]
            {
                return BackendChoice::Candle;
            }
            #[cfg(all(feature = "huggingface", not(feature = "candle")))]
            {
                return BackendChoice::HuggingFace;
            }
            #[cfg(not(any(feature = "huggingface", feature = "candle")))]
            {
                return BackendChoice::Unsupported(
                    "GGUF-named models require a GPU engine or the HuggingFace backend. \
                     Build with --features candle (CUDA) or --features airframe (WebGPU)."
                        .to_string(),
                );
            }
        }

        // Default to HuggingFace for other models
        #[cfg(feature = "huggingface")]
        {
            BackendChoice::HuggingFace
        }
        #[cfg(not(feature = "huggingface"))]
        {
            BackendChoice::Unsupported(
                "No inference backend enabled. Build with --features airframe or --features huggingface."
                    .to_string(),
            )
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum BackendChoice {
    #[cfg(feature = "huggingface")]
    HuggingFace,
    #[cfg(feature = "mlx")]
    #[allow(clippy::upper_case_acronyms)]
    MLX,
    #[cfg(feature = "candle")]
    Candle,
    SafeTensors,
    Unsupported(String),
}

#[async_trait]
impl InferenceEngine for InferenceEngineAdapter {
    async fn load(&self, spec: &ModelSpec) -> Result<Box<dyn LoadedModel>> {
        // Select backend and load model directly (no caching for now to avoid complexity)
        let backend = self.select_backend(spec);
        match backend {
            BackendChoice::Unsupported(msg) => {
                return Err(anyhow::anyhow!("{}", msg));
            }
            BackendChoice::SafeTensors => {
                // Use native SafeTensors engine - NO Python dependency!
                self.safetensors_engine.load(spec).await
            }
            #[cfg(feature = "mlx")]
            BackendChoice::MLX => {
                // Use MLX engine for Apple Silicon Metal GPU acceleration
                self.mlx_engine.load(spec).await
            }
            #[cfg(feature = "candle")]
            BackendChoice::Candle => self.candle_engine.load(spec).await,
            #[cfg(feature = "huggingface")]
            BackendChoice::HuggingFace => {
                // Convert to UniversalModelSpec for huggingface backend (for HF model IDs)
                let universal_spec = UniversalModelSpec {
                    name: spec.name.clone(),
                    backend: super::ModelBackend::HuggingFace {
                        base_model_id: spec.base_path.to_string_lossy().to_string(),
                        peft_path: spec.lora_path.as_ref().map(|p| p.to_path_buf()),
                        use_local: true,
                    },
                    template: spec.template.clone(),
                    ctx_len: spec.ctx_len,
                    device: "cpu".to_string(),
                    n_threads: spec.n_threads,
                };
                let universal_model = self.huggingface_engine.load(&universal_spec).await?;
                Ok(Box::new(UniversalModelWrapper {
                    model: universal_model,
                }))
            }
        }
    }
}

/// Wrapper to adapt UniversalModel to LoadedModel interface
#[cfg(feature = "huggingface")]
struct UniversalModelWrapper {
    model: Box<dyn UniversalModel>,
}

#[cfg(feature = "huggingface")]
#[async_trait]
impl LoadedModel for UniversalModelWrapper {
    async fn generate(
        &self,
        prompt: &str,
        opts: GenOptions,
        on_token: Option<Box<dyn FnMut(String) + Send>>,
    ) -> Result<String> {
        self.model.generate(prompt, opts, on_token).await
    }
}

// Note: Cached model references removed as they were unused placeholder code.
// Future implementation should use Arc<dyn LoadedModel> for proper model sharing.

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn create_test_spec(name: &str, path: &str) -> ModelSpec {
        ModelSpec {
            name: name.to_string(),
            base_path: PathBuf::from(path),
            lora_path: None,
            template: None,
            ctx_len: 2048,
            n_threads: None,
        }
    }

    /// A .gguf file must reach the candle engine when it is compiled in.
    ///
    /// This is the regression the server hit on a rented GPU: the candle backend existed,
    /// the model loaded from disk, and every request still failed with "GGUF models require
    /// the Airframe GPU engine" because nothing routed to it. An engine that is never
    /// selected is an engine that does not exist.
    #[cfg(feature = "candle")]
    #[test]
    fn gguf_routes_to_candle() {
        let adapter = InferenceEngineAdapter::new();

        let explicit = create_test_spec("qwen3-8b", "/workspace/models/qwen3-8b.gguf");
        assert_eq!(adapter.select_backend(&explicit), BackendChoice::Candle);

        // Ollama stores GGUF as extensionless blobs; same engine, different naming.
        let blob = create_test_spec(
            "qwen3",
            "/root/.ollama/models/blobs/sha256-af63361d2ac3ba2cb454f440842e345b926a50fe89",
        );
        assert_eq!(adapter.select_backend(&blob), BackendChoice::Candle);
    }

    #[test]
    fn test_huggingface_model_id_detection() {
        let adapter = InferenceEngineAdapter::new();

        // Test HuggingFace model IDs
        let hf_spec = create_test_spec("qwen", "Qwen/Qwen3-Next-80B-A3B-Instruct");
        let backend = adapter.select_backend(&hf_spec);
        #[cfg(feature = "huggingface")]
        assert_eq!(backend, BackendChoice::HuggingFace);

        let hf_spec2 = create_test_spec("llama", "meta-llama/Llama-2-7b-chat-hf");
        let backend2 = adapter.select_backend(&hf_spec2);
        #[cfg(feature = "huggingface")]
        assert_eq!(backend2, BackendChoice::HuggingFace);
    }

    #[test]
    fn test_local_file_detection() {
        let adapter = InferenceEngineAdapter::new();

        // Test local files still work
        let safetensors_spec = create_test_spec("local", "model.safetensors");
        let backend = adapter.select_backend(&safetensors_spec);
        assert_eq!(backend, BackendChoice::SafeTensors);

        // Test Windows paths (should not be treated as HF model IDs)
        let windows_spec = create_test_spec("local", "C:\\path\\to\\model.safetensors");
        let backend2 = adapter.select_backend(&windows_spec);
        assert_eq!(backend2, BackendChoice::SafeTensors);
    }

    #[test]
    fn test_safetensors_priority_over_huggingface() {
        let adapter = InferenceEngineAdapter::new();

        // SafeTensors files should ALWAYS use SafeTensors engine, even if from HuggingFace
        let safetensors_from_hf = create_test_spec("model", "/path/to/model.safetensors");
        let backend = adapter.select_backend(&safetensors_from_hf);
        assert_eq!(backend, BackendChoice::SafeTensors);

        // Even with complex paths containing slashes
        let safetensors_complex = create_test_spec(
            "model",
            "/models/huggingface/org/model/pytorch_model.safetensors",
        );
        let backend2 = adapter.select_backend(&safetensors_complex);
        assert_eq!(backend2, BackendChoice::SafeTensors);

        // Windows paths with safetensors
        let safetensors_windows =
            create_test_spec("model", "C:\\models\\org\\model\\model.safetensors");
        let backend3 = adapter.select_backend(&safetensors_windows);
        assert_eq!(backend3, BackendChoice::SafeTensors);
    }

    #[test]
    fn test_file_extension_priority() {
        let adapter = InferenceEngineAdapter::new();

        // File extensions should take priority over everything else
        let safetensors_spec = create_test_spec("llama-model", "path/to/llama.safetensors");
        let backend = adapter.select_backend(&safetensors_spec);
        assert_eq!(backend, BackendChoice::SafeTensors);

        #[cfg(feature = "mlx")]
        {
            let mlx_spec = create_test_spec("qwen-model", "path/to/qwen.mlx");
            let backend2 = adapter.select_backend(&mlx_spec);
            assert_eq!(backend2, BackendChoice::MLX);
        }
    }
}
