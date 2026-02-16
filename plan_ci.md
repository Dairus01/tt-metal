# CI Refactor Plans

The goal is to run and test the status of unit/demo tests and then add any missing tests to CI for the list of the HuggingFace models provided.

## Model List
### Tier 1

### Tier 2
- meta-llama/Llama-3.3-70B-Instruct
- Qwen/Qwen3-32B
- Qwen/Qwen2.5-Coder-32B-Instruct
- mistralai/Mistral-7B-Instruct-v0.3
- google/gemma-3-27b-it
- mistralai/Mixtral-8x7B-Instruct-v0.1
- meta-llama/Llama-3.1-8B-Instruct

### Tier 3

### HF Token
To access the huggingFace repo, use the HF_TOKEN environment variable (set externally).

## Test List
We require a shell script that runs both unit tests and demo tests for all the models in the list above.
The script must have the option to run either only the unit tests or the demo tests, or both.

When running the tests, keep in mind that sometimes a test might hang. To detect hangs, you should look at the output of the test every 3 minutes. If the output hasn't change, it's safe to assume it hanged and the job needs to be killed.
After killing the job, you should also reset the machine using the command: `tt-smi -r`. Afterwards, resume execution on the next model.

A summary of each test status needs to be stored into an output markdown file.

---

### meta-llama/Llama-3.3-70B-Instruct

**Unit tests** (CI function: `run_t3000_llama3.3-70b_tests`, timeout: 900s each)
- `test_attention.py` — PASS
- `test_attention_prefill.py` — PASS
- `test_mlp.py` — PASS
- `test_rms_norm.py` — PASS
- `test_decoder.py` — PASS
- `test_decoder_prefill.py` — PASS

Note: CI also runs `test_embedding.py` for this model.

**Demo tests** (CI function: `run_t3000_llama3.3-70b_demo_tests`, timeout: 1800s) — PASS
- `pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"`

**Demo tests — Data Parallel** (CI function: `run_t3000_llama3_70b_dp_tests`, timeout: 1800s)
- `pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci and DP"`

---

### meta-llama/Llama-3.1-8B-Instruct

**Unit tests** (CI function: `run_t3000_llama3.1-8b_tests`, timeout: 600s each)
- `test_attention.py` — PASS
- `test_attention_prefill.py` — PASS
- `test_mlp.py` — PASS
- `test_rms_norm.py` — PASS
- `test_decoder.py` — PASS
- `test_decoder_prefill.py` — PASS

**Demo tests** (CI function: `run_t3000_llama3.1-8b_demo_tests`, timeout: 600s) — PASS
- `pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"`

---

### Qwen/Qwen3-32B

**Unit tests** (CI function: `run_t3000_qwen3-32b_tests`, timeout: 900s each)
- `test_attention.py` — PASS
- `test_attention_prefill.py` — PASS
- `test_mlp.py` — PASS
- `test_rms_norm.py` — PASS
- `test_decoder.py` — PASS
- `test_decoder_prefill.py` — PASS

**Demo tests** (CI function: `run_t3000_qwen3-32b_demo_tests`, timeout: 1800s) — PASS
- `pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"`

---

### Qwen/Qwen2.5-Coder-32B-Instruct

**Unit tests** (CI function: `run_t3000_qwen25-coder-32b_tests`, timeout: 900s each)
- `test_attention.py` — PASS
- `test_attention_prefill.py` — PASS
- `test_mlp.py` — PASS
- `test_rms_norm.py` — PASS
- `test_decoder.py` — PASS
- `test_decoder_prefill.py` — PASS

**Demo tests** (CI function: `run_t3000_qwen25-coder-32b_demo_tests`, timeout: 1800s) — PASS
- `pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"`

---

### mistralai/Mistral-7B-Instruct-v0.3

**Unit tests** (CI function: `run_t3000_mistral_tests`, timeout: 600s each)
- `test_attention.py` — PASS
- `test_attention_prefill.py` — PASS
- `test_mlp.py` — PASS
- `test_rms_norm.py` — PASS
- `test_decoder.py` — PASS
- `test_decoder_prefill.py` — PASS

Note: CI also runs `test_embedding.py` for this model.

**Demo tests** (CI function: `run_t3000_mistral_demo_tests`, timeout: 1800s) — PASS
- `pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"`

---

### google/gemma-3-27b-it

**Exception**: This model does NOT use the generic `tt_transformers` unit tests. It has its own dedicated tests.

**Unit tests** (CI function: `run_t3000_gemma3-small_tests`, timeout: 600s)
- `pytest models/demos/multimodal/gemma3/tests/test_ci_dispatch.py -k "27b"`
  - Internally runs: test_mmp, test_attention, test_attention_prefill, test_patch_embedding, test_vision_attention, test_vision_cross_attention_transformer, test_vision_embedding, test_vision_layernorm, test_vision_mlp, test_vision_pipeline, test_vision_transformer_block, test_vision_transformer, test_embedding, test_rms_norm, test_mlp, test_decoder, test_decoder_prefill

Generic test results (for reference — these are NOT used in CI):
- `test_attention.py` — FAIL
- `test_attention_prefill.py` — FAIL
- `test_mlp.py` — PASS
- `test_rms_norm.py` — FAIL
- `test_decoder.py` — FAIL
- `test_decoder_prefill.py` — FAIL

**Demo tests** (CI function: `run_t3000_gemma3_tests`) — PASS
- `pytest models/demos/multimodal/gemma3/demo/text_demo.py -k "performance and ci-1"`
- `pytest models/demos/multimodal/gemma3/demo/vision_demo.py -k "performance and batch1-multi-image-trace"`

---

### mistralai/Mixtral-8x7B-Instruct-v0.1

**Exception**: This model does NOT use the generic `tt_transformers` unit tests due to its Mixture-of-Experts (MoE) architecture. It has its own dedicated tests under `models/tt_transformers/tests/mixtral/`.

**Unit tests** (CI function: `run_t3000_mixtral_tests`, timeout: 720s each, HF_MODEL=`mistralai/Mixtral-8x7B-v0.1`)
- `test_mixtral_rms_norm.py` — test_rms_norm_inference
- `test_mixtral_mlp.py` — test_mixtral_mlp_inference
- `test_mixtral_moe.py` — test_mixtral_moe_inference
- `test_mixtral_decoder.py` — test_mixtral_decoder_inference
- `test_mixtral_decoder_prefill.py` — test_mixtral_decoder_inference
- `test_mixtral_model.py::test_model_inference[...-paged_attention-quick]`
- `test_mixtral_model.py::test_model_inference[...-default_attention-quick]`
- `test_mixtral_model_prefill.py::test_model_inference[...-paged_attention-8]`
- `test_mixtral_model_prefill.py::test_model_inference[...-default_attention-8]`

**Demo tests** (CI function: `run_t3000_mixtral_demo_tests`, timeout: 3600s, HF_MODEL=`mistralai/Mixtral-8x7B-v0.1`) — HANG (hangs after some decode iterations)
- `pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"`

# Add tests to CI Pipelines

After we validate the tests we have to make sure we have them in our CI coverage.
For each model, we should add the relevant test matrix, following the structure present in the t3k_unit_tests.yaml file (using the unit tests as example). The relevant files are shown below.

In short, we want:
- Each missing test added to the yaml test matrix, with all the tests we covered above. If a test was failing (check tt-metal/test_results_unit.md for the latest summary), we want to have that test in the yaml, but commented out with a comment specifying that it failed and the reason.

- **Exception: google/gemma-3-27b-it** — This model does NOT use the generic `tt_transformers` unit tests. It has its own dedicated test at `models/demos/multimodal/gemma3/tests/test_ci_dispatch.py` which internally calls all relevant unit tests for the model. The existing CI entry (`run_t3000_gemma3-small_tests`) already covers gemma3 via `pytest --timeout 600 models/demos/multimodal/gemma3/tests/test_ci_dispatch.py -k "27b"`. No additional generic tests should be added for this model.
This model also does not use the generic `tt_transformers` demo tests. Instead it has its own dedicated demo test in `pytest --timeout 1000 models/demos/multimodal/gemma3/demo/text_demo.py -k "performance and ci-1"`. No additional demo tests should be added for this model.

- **Exception: mistralai/Mixtral-8x7B-Instruct-v0.1** — This model does NOT use the generic `tt_transformers` unit tests due to its Mixture-of-Experts (MoE) architecture being incompatible (e.g. `MixtralDecoderLayer` has no `mlp` attribute, `state_dict` key mismatches). Instead, it has its own dedicated tests under `models/tt_transformers/tests/mixtral/` which are already covered in CI by `run_t3000_mixtral_tests`. No additional generic tests should be added for this model.

- Do not forget to add the tests to the all the relevant files.

- Do the same for the demo tests.


## CI files relevant for Unit tests
- tt-metal/tests/pipeline_reorg/t3k_unit_tests.yaml
- tt-metal/tests/scripts/t3000/run_t3000_unit_tests.sh
tt-metal/.github/workflows/t3000-unit-tests.yaml

## CI files relevant for Demo tests
- tt-metal/tests/pipeline_reorg/t3k_demo_tests.yaml
- tt-metal/tests/scripts/t3000/run_t3000_demo_tests.sh
- tt-metal/.github/workflows/t3000-demo-tests.yaml
