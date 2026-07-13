# KSL Backend Engineering Audit — Week 1 → Day 12

**Status: Day 12 complete.** This audit was originally written when `/health/model` and the live FastAPI app did not exist. Both have since been implemented, tested (47/47 automated tests passing, including 5 new `/health/model` tests), and manually verified by starting `uvicorn main:app` against the real model/label files and curling both `/health/model` and `/predict`. The findings below are updated in place; sections that no longer reflect reality are marked with an **Update (Day 12 complete)** note rather than silently rewritten, so the audit trail stays honest about what changed and when.

**Method:** every finding below is traced to a specific file already read in full (`backend/ai_inference/*`, `backend/routes/*`, `backend/tests/*`, `backend/main.py`, `backend/README.md`, `backend/.env`, `backend/labels/label_map_25class.json`) and to the extracted text of `KSL_Model_Demo_Prototype_30_Day_Plan.docx`. Nothing below is inferred from the plan alone or from the code alone — only from both, compared.

---

## Phase 2 — Progress Verification (Week 1 → Day 12)

| Day | Planned Deliverable | Current Implementation | Status |
|---|---|---|---|
| 1 | Review files and model assets → technical notes on notebook, model, label map | `backend/README.md` exists as an informal Q&A note covering the model file, the input-shape question, `label_map_25class.json`, and `landmark_extract.ipynb`. | 🟡 Partially Complete |
| 2 | Confirm model input/output shape → 30×252 input, 25-class output verified | `ai_inference/model_info.py` prints `model.input_shape`/`model.output_shape`; `main.py`'s `dummy_prediction()` runs a real `(1,30,252)` tensor through the model and prints the resulting class; `label_map_25class.json` has exactly 25 entries (indices 0–24). | ✅ Complete |
| 3 | Create Python inference module structure → `ai_inference` package skeleton | `backend/ai_inference/` exists with 8 modules (`model_loader`, `label_loader`, `model_info`, `keypoint_extractor`, `temporal_attention`, `feature_formatter`, `frame_sample_buffer`, `inference_service`). | ✅ Complete |
| 4 | Implement model and label map loaders → model loads without crash | `model_loader.py`/`label_loader.py` implement singleton load-once/get patterns with fail-fast `FileNotFoundError`/`RuntimeError`. Used successfully by `main.py` and `routes/predict.py`. | ✅ Complete |
| 5 | Convert notebook extraction logic → reusable `keypoint_extractor.py` | `HandKeypointExtractor` wraps MediaPipe Holistic, producing wrist-relative `(63,)` vectors per hand. | ✅ Complete |
| 6 | Create dummy-sequence prediction test → prediction function returns index, label, confidence | `main.py`'s `dummy_prediction()` does exactly this against an all-zero `(1,30,252)` tensor. | ✅ Complete |
| 7 | Week 1 review → Week 1 progress report and risk update | No report artifact found anywhere in the repository (no `WEEK1_REPORT.md` or equivalent). | ❌ Missing (from the repo — may have been delivered verbally/outside version control, but there is nothing here to verify against) |
| 8 | Implement MediaPipe extraction → hand landmarks extracted from camera/video frame | `HandKeypointExtractor.extract()` processes real OpenCV frames end-to-end; exercised manually via `tests/camera_test.py` and `tests/test_keypoint.py`. | ✅ Complete |
| 9 | Build 252-feature formatter → one frame returns exact 252 features | `feature_formatter.format_sequence()` — its own module docstring and `tests/test_feature_formatter.py`'s docstring both explicitly call this "the Day 9 test plan." 11 unit tests cover shape, velocity math, padding, NaN/Inf rejection, purity. | ✅ Complete |
| 10 | Implement 30-frame sequence buffer → valid 30×252 sequence ready for model | `frame_sample_buffer.FrameSampleBuffer` — its test file's docstring explicitly calls this "the Day 10 test plan." Produces `(30,126)`; the 252-wide tensor only exists after this output is passed through Day 9's formatter, which is exactly how the plan splits the two days' work. | ✅ Complete |
| 11 | Create `/predict` endpoint → backend accepts sequence and returns prediction | `routes/predict.py` implements `POST /predict` fully, backed by `InferenceService`. 7 endpoint tests + 12 service tests, covering 200/422/500/503. **Update (Day 12 complete):** `main.py` now mounts this router in production via `app.include_router()`; manually verified reachable with a real `curl POST /predict` against the real model, returning a genuine prediction. | ✅ Complete |
| 12 | Create `/health/model` endpoint → model status endpoint available | `routes/health.py` implements `GET /health/model`, using `model_loader.get_model()`/`label_loader.get_labels()` (read-only, no side effects) to report `{"model_loaded", "label_map_loaded", "num_classes"}` with `200` when both are ready or `503` otherwise. 5 endpoint tests (`tests/test_health_endpoint.py`). Mounted in production via `main.py`. Manually verified with a real `curl GET /health/model`, returning `200 {"model_loaded":true,"label_map_loaded":true,"num_classes":25}`. | ✅ Complete |

---

## Phase 3 — Architecture Validation

The plan's own "Recommended System Architecture" (section 4) lists the Python backend as five responsibilities: *MediaPipe keypoint extraction → 30-frame sequence buffer → 252-feature formatter → Keras model inference → Top-k result generation.* That is an almost exact match, module-for-module, to `keypoint_extractor.py` / `frame_sample_buffer.py` / `feature_formatter.py` / `model_loader.py` / `inference_service.py`. That structural match is a genuine strength worth calling out, not just an assumption.

| Check | Verdict | Evidence |
|---|---|---|
| Folder structure | ✅ Matches plan's architecture breakdown | See above — one file per architectural responsibility. |
| Module separation | ✅ | `ai_inference/` has zero `fastapi` imports; `routes/` has zero tensor-math or model-call code. |
| SRP | ✅ | Every real module's own docstring states one job and explicitly disclaims neighboring responsibilities (e.g. `feature_formatter.py`: "has no knowledge of capture, buffering, sampling, or the model itself"). |
| Thin routes | ✅ | `routes/predict.py`'s handler is ~15 lines: fetch service, convert input, delegate, map 3 exception types to 3 status codes. |
| Business logic separation | ✅ | Confirmed by grep: `APIRouter`/`FastAPI` imports exist only in `routes/predict.py` and the test suite. |
| Dependency direction | ✅ one-directional | `routes → ai_inference`, never the reverse. |
| Model loading | ✅ | Singleton, fail-fast, custom-object registration for `TemporalAttention` handled correctly. |
| Label loading | ✅ | Singleton, fail-fast, mirrors `model_loader.py`'s pattern. |
| Feature formatting | ✅ | Pure function, validated first then cast (correct order — prevents a dtype cast from masking a NaN). |
| Frame buffer | ✅ | Correct `linspace`-based downsampling / last-frame-repeat padding, matches the training notebook's documented approach. |
| Inference pipeline | ✅ | `InferenceService` takes `model`/`labels` via constructor injection — no import-time coupling to the loader singletons, which is exactly why it's mock-testable. |
| Startup lifecycle | ✅ Complete (Day 12) | `main.py` now has `app = FastAPI(lifespan=lifespan)`; `lifespan` calls `model_loader.load_model()` and `label_loader.load_labels()` before the server accepts requests. `app.include_router()` mounts both routers. Manually verified: `uvicorn main:app` starts, logs "Application startup complete," and serves real requests. |
| Health endpoint | ✅ Complete (Day 12) | `routes/health.py` implements `GET /health/model`; see Day 12 row above. |
| Predict endpoint | ✅ Complete (Day 12) | Implemented since Day 11, now also deployed — see Day 11 above. |

**Deviation from the plan worth flagging (not a bug, a real gap — still open post-Day-12):** the plan's architecture text says "Python handles MediaPipe extraction," implying the backend itself runs keypoint extraction on incoming frames. The actual, real `/predict` contract (per `routes/predict.py`'s own docstring) instead expects the caller to already have produced a `(30, 126)` position sequence — MediaPipe extraction and buffering are not invoked as part of the request path. `keypoint_extractor.py` and `frame_sample_buffer.py` are fully built and unit-tested, but nothing in the backend currently chains them into `/predict`. This is the single most consequential architectural gap for Phase 4 below, and it is intentionally out of Day 12's scope — closing it is the correct next step before Day 13's webcam-based end-to-end test.

---

## Phase 4 — Pipeline Validation

```
Camera            [only reachable via tests/camera_test.py, a manual script]
   ↓
MediaPipe          ✅ implemented — HandKeypointExtractor.__init__ / .extract()
   ↓
Keypoint extraction ✅ implemented — extract_native_hand(), wrist-relative (63,) vectors
   ↓
   ✗✗✗  MISSING LINK  ✗✗✗
   Nothing in the codebase calls FrameSampleBuffer.add_frame() with
   HandKeypointExtractor's output. The two classes are shape-compatible
   and clearly designed for each other, but no function/script/route
   wires them together.
   ↓
Frame buffer        ✅ implemented in isolation — FrameSampleBuffer.add_frame()/reset()
   ↓
30-frame sequence   ✅ implemented in isolation — FrameSampleBuffer.sample_sequence() -> (30,126)
   ↓
   ✗✗✗  MISSING LINK  ✗✗✗
   Nothing calls sample_sequence() and feeds its output into
   format_sequence() or InferenceService automatically. Today, the
   (30,126) array that reaches /predict comes directly from the HTTP
   request body (routes/predict.py's PredictRequest.sequence) — an
   external, out-of-backend source, not FrameSampleBuffer.
   ↓
Feature formatter    ✅ implemented and connected — feature_formatter.format_sequence()
   ↓
Velocity generation  ✅ implemented and connected — velocity[1:] = positions[1:] - positions[:-1]
   ↓
252-feature tensor   ✅ implemented and connected — np.concatenate([positions, velocity])
   ↓
Model inference      ✅ implemented and connected — model.predict(batch) inside InferenceService
   ↓
Label mapping         ✅ implemented and connected — self._index_to_label built from labels.items()
   ↓
JSON response         ✅ implemented and connected — matches the plan's section 6.1 schema exactly
   ↓
   ✗✗✗  REACHABILITY GAP  ✗✗✗
   Even this fully-connected back half (sequence -> ... -> JSON) is only
   reachable through pytest's TestClient today, since no production app
   mounts routes/predict.py's router.
```

**Summary:** the pipeline is built in two solid, independently correct, independently tested halves — (1) camera → keypoints → buffered sequence, and (2) sequence → tensor → model → response — but the two halves are not spliced together anywhere in the codebase, and the second half is not exposed by any running service. Both are real, specific, verifiable gaps — not stylistic opinions.

---

## Phase 5 — API Validation

The plan's API contract (section 6) lists 5 endpoints. Status of each against the actual repo:

| Method | Endpoint | Plan's due day | Status |
|---|---|---|---|
| GET | `/health/model` | Day 12 | ✅ Implemented, tested, deployed |
| POST | `/predict` | Day 11 | ✅ Implemented, tested, deployed |
| POST | `/prediction-sessions` | Week 3 (Day 19-20) | Not yet due — absent, as expected |
| POST | `/prediction-events` | Week 3 (Day 20) | Not yet due — absent, as expected |
| GET | `/gesture-classes` | Week 3 (DB-driven) | Not yet due — absent, as expected |

Both Day-11 and Day-12 endpoints are now complete and deployed, on schedule. The three DB-backed endpoints are correctly out of scope for a Day-12 checkpoint.

- **Request model:** `PredictRequest{sequence: List[List[float]]}` — correct, matches the expected `(30, 126)` shape (validated downstream by `feature_formatter`).
- **Response model:** `InferenceService.predict()`'s returned dict is a field-for-field match to the plan's section 6.1 example (`predicted_class_index`, `predicted_label`, `confidence`, `top_k[{label, confidence}]`, `inference_ms`, `sequence_shape`). This is a genuine, exact spec match — worth noting as a real success, not just "close enough." **Verified with a live request**, not just the mocked test suite: a real `(30,126)` random sequence POSTed to a running `uvicorn main:app` returned `{"predicted_class_index":0,"predicted_label":"No_action","confidence":0.251,...,"sequence_shape":[30,252]}` from the actual Keras model.
- **Health response model:** `{"model_loaded": bool, "label_map_loaded": bool, "num_classes": int|None}` — not prescribed by the plan (section 6 only specifies the endpoint's purpose, not its schema), so this is a reasonable implementation choice matching the plan's stated intent ("Check whether model and label map are loaded").
- **HTTP status codes:** 200 (success), 422 (shape/NaN violation), 500 (unexpected/model-level failure), 503 (dependencies not loaded) — a sensible, idiomatic mapping; the plan doesn't prescribe exact codes, so this is a reasonable implementation choice, not a deviation. `/health/model` reuses the same 503-for-not-ready convention already established by `/predict`.
- **Validation:** shape and finiteness are checked before the model ever runs; verified by dedicated tests for wrong row count, wrong column count, and NaN input, all returning 422.
- **Startup behavior:** now evaluable and verified — `main.py`'s `lifespan` context manager loads the model and labels before the server accepts requests; a real `uvicorn main:app` run confirmed "Application startup complete" and both endpoints responding correctly afterward.
- **Singleton usage:** consistent across `model_loader`, `label_loader`, and `routes/predict.py`'s `_inference_service` — all lazy, all cached, all correctly reset in tests via monkeypatching. `routes/health.py` deliberately reads these same singletons without ever writing to them.
- **Dependency injection:** correctly used — `InferenceService` receives `model`/`labels` via its constructor rather than importing the loader modules directly.

---

## Phase 6 — Code Quality Review

| Area | Finding |
|---|---|
| Naming consistency | Consistent snake_case, verb-based function names (`load_model`, `get_model`, `format_sequence`, `sample_sequence`) throughout. No inconsistency found. |
| Type hints | Present and thorough in the Week 2 modules (`feature_formatter.py`, `frame_sample_buffer.py`, `inference_service.py`, `routes/predict.py`). **Absent** in the Week 1 modules (`model_loader.py`, `label_loader.py`, `model_info.py`, `main.py`, `keypoint_extractor.py`'s `extract()`/`__init__` signatures). This is a real, visible quality gradient across the two weeks — not a bug, but worth noting since it means the earliest-written files are the least self-documenting. |
| Documentation | Same pattern: Week 2 modules have detailed module- and function-level docstrings explaining *why*, not just *what*. Week 1 modules (`model_loader.py`, `label_loader.py`) have no docstrings at all on their public functions. |
| Error handling | Strong and consistent fail-fast philosophy across every real module — `FileNotFoundError`/`RuntimeError` from the loaders, `ValueError`/`RuntimeError` from `feature_formatter`/`frame_sample_buffer`, correctly mapped to HTTP codes in `routes/predict.py`. No silent failure or fallback-masking found anywhere. |
| Logging | **Real gap:** no module uses Python's `logging` package anywhere in the codebase — `model_loader.py`, `label_loader.py`, `main.py`, and `model_info.py` all use bare `print()` for status/diagnostic output. This is acceptable for a CLI diagnostic script (`main.py`) but will not scale once an actual server exists (Day 12+ health/predict endpoints have no way to log request outcomes, load failures, or inference timings anywhere but stdout). |
| Imports | Clean — no unused imports found in any file read. |
| Dead code | None found. |
| Duplicate logic | `model_loader.py` and `label_loader.py` implement near-identical singleton load/get patterns in two separate files. This is intentional, minimal, acceptable duplication (each owns a genuinely distinct artifact) rather than a DRY violation — extracting a shared base class here would be over-engineering for two ~20-line files, not a fix worth making. |
| Testability | Strong, specifically because of the constructor-injection pattern in `InferenceService` — it is fully testable with a lightweight mock model, with no TensorFlow dependency required for its own test suite. |

---

## Phase 7 — Testing Review

**Are important paths tested?** Yes, for everything built in Week 2: `feature_formatter` (11 tests: shape, velocity math incl. a hand-checked fixture, padding, dtype coercion, NaN/Inf rejection, purity), `frame_sample_buffer` (11 tests: exact-30, downsampling, padding source, reset semantics, idempotency, ordering, fail-fast), `inference_service` (12 tests: batch shape, schema, label mapping incl. a shuffled-dict-order guard, top-k ordering/length, custom top_k, exception propagation, non-negative timing), the `/predict` route (7 tests: 200/422×3/500/503/missing-field), and now the `/health/model` route (5 tests: 200-both-loaded, 503-model-not-loaded, 503-labels-not-loaded, 503-neither-loaded, and a guard test proving the health check never calls `model.predict()`). **Update (Day 12 complete):** the `/predict` route's tests were previously undiscoverable — `httpx` (required by FastAPI's `TestClient`) was missing from `requirements.txt`/the venv, so `test_predict_endpoint.py` failed to even collect. This is now fixed (`httpx==0.27.2` added to `requirements.txt` and installed); the full suite of 47 tests was run and passed.

**Edge cases missing (unchanged by Day 12 — still open, correctly out of scope for this checkpoint):**
- `model_loader.py` and `label_loader.py` have **zero dedicated unit tests** — no test exercises `FileNotFoundError` on a missing model/label file, singleton caching (load twice → same object), or `RuntimeError` from `get_model()`/`get_labels()` before `load_*()` is called. They're only ever exercised indirectly, via monkeypatched globals, inside `test_predict_endpoint.py` and `test_health_endpoint.py`.
- `keypoint_extractor.py` has **no automated test at all**. `test_keypoint.py` is a manual, webcam-driven, non-asserting visual script — if run under `pytest`, it will hang waiting for camera frames rather than pass/fail meaningfully. The pure, camera-independent piece of this file (`extract_native_hand`'s wrist-relative subtraction and its None → zero-vector fallback) is straightforward to unit test with a fake landmark object and currently isn't.
- No test chains `FrameSampleBuffer` → `format_sequence` → `InferenceService` together as one pipeline using real (non-HTTP) glue code — each is tested in isolation with hand-built fixtures, but the actual seam between Day 9/10/11's modules has no test locking it in as a working chain.

**Are startup tests sufficient?** No automated test exercises `main.py`'s `lifespan` startup path directly (it is only exercised manually, by actually running `uvicorn main:app`). This is a reasonable gap for a Day-12 prototype — automated startup-lifecycle testing (e.g. with `TestClient`'s context-manager form, which does trigger lifespan) is a good candidate for a future hardening pass, not a Day 12 blocker, since the manual verification already proves the real path works end-to-end.

**Are inference tests sufficient?** Yes — this is the best-covered area of the codebase.

**Are route tests sufficient?** Yes, for both routes that exist.

**Recommended additional tests (only what's missing, not exhaustive):**
1. Unit tests for `model_loader`/`label_loader`: missing-file error path, singleton-caching behavior, `get_*()` raising before `load_*()`.
2. A unit test for `HandKeypointExtractor.extract_native_hand()` using a synthetic landmark object — no camera required.
3. One integration-style test chaining `FrameSampleBuffer.sample_sequence()` → `format_sequence()` → `InferenceService.predict()` (mock model) to prove the Day 9/10/11 seam works as a single pipeline, not just as three separately-tested pieces.
4. A `TestClient(app)` context-manager-based test against the real `main.app` (with monkeypatched loaders) to automate what was, for Day 12, verified manually via a live `uvicorn` run.

---

## Phase 8 — Week 2 Readiness

**Is Day 12 fully complete?** Yes. `/health/model` is implemented (`routes/health.py`), tested (5 automated tests), mounted in production (`main.py`), and manually verified live (`curl GET /health/model` → `200 {"model_loaded":true,"label_map_loaded":true,"num_classes":25}` against the real model and label files).

**Is the backend stable?** Yes, and this can now actually be assessed, not just assumed — `main.py` is a real, running FastAPI app (`uvicorn main:app`), verified by starting it and issuing real HTTP requests against both endpoints, including one real `/predict` call that produced a genuine model output.

**Can end-to-end testing begin (Day 13's literal scope)?** Yes, for the "test backend with sample input" half. A running backend now exists, and `POST /predict` has been proven to work against it with a real sample sequence. The webcam half specifically (camera → keypoints → buffer → `/predict`) still requires the connective glue code mentioned below, which remains correctly out of Day 12's scope.

**Remaining blocker before a full webcam-based Day 13 test:**
1. The camera → keypoints → buffer chain still has no glue code connecting it to the formatter/service chain — this was intentionally deferred past Day 12 and is the next well-scoped piece of work, not a Day 12 defect.

---

## Phase 9 — Overall Project Health

| Dimension | Score | Why |
|---|---|---|
| Architecture | 9/10 | Clean, one-directional layering that matches the plan's own architecture diagram module-for-module; the top assembly layer now exists and is verified live. Held back from 10 only by the still-open camera→buffer splice, which is correctly out of scope. |
| Code Quality | 8/10 | Consistent fail-fast error handling and strong docstrings in Week 2 modules; no logging framework anywhere; documentation/type-hint quality is visibly uneven between Week 1 and Week 2 files. |
| Maintainability | 8/10 | Small, focused files; correct dependency injection where it matters most (`InferenceService`); docked slightly for the untested loaders and the disconnected pipeline halves. |
| Readability | 8/10 | Clear naming and "why"-focused comments throughout the newer modules; the older modules (`model_loader.py`, `label_loader.py`) are terser and less self-explanatory by comparison. |
| Test Coverage | 8/10 | Excellent depth on the Week 2 core (formatter/buffer/service/both routes, 47 tests total, all passing including the previously-uncollectable `/predict` suite); zero coverage on the loaders and on keypoint extraction logic remains the main gap. |
| Project Structure | 8/10 | Matches the plan's intended architecture almost exactly; docked only for `routes/sessions.py` sitting unimplemented (correctly, Week 3 scope). |
| Specification Compliance | 9/10 | Days 1–12 substantively delivered on schedule; both `/predict` and `/health/model` response schemas match the plan's own examples/intent exactly, verified with live requests; the plan's "Python handles MediaPipe extraction" architecture note remains not yet wired into the live request path, correctly deferred past Day 12. |
| Production Readiness (for a Day-12 prototype) | 7/10 | A real, running service now exists and has been manually verified end-to-end against the real model — the biggest gap from the previous audit is closed. Remaining deductions are for the lack of structured logging and the still-unwired camera pipeline, both appropriately out of scope for this checkpoint. |

---

## Phase 10 — Final Verdict

**Successfully completed:** the entire core ML pipeline — model loading, label loading, MediaPipe keypoint extraction, frame buffering/downsampling, velocity-augmented feature formatting, and inference orchestration with top-k label mapping — is implemented correctly, matches the plan's technical requirements (30×252 input, 25-class output, exact response schema), and is backed by a genuinely strong automated test suite (47 tests, all passing) for everything built in Week 2. **As of Day 12, both `/predict` and `/health/model` are complete, tested, deployed in a real running FastAPI app, and manually verified against the real model and label files** — not just the mocked pytest suite.

**What was done for Day 12:**
1. Repurposed `main.py` into the FastAPI application entry point (`app = FastAPI(lifespan=lifespan)`), with a `lifespan` context manager that loads the model and label map once at startup and mounts both `routes/predict.py` and `routes/health.py`.
2. Implemented `GET /health/model` in `routes/health.py`, using the read-only `get_model()`/`get_labels()` accessors (never `load_*()`, so the health check itself has no loading side effect), returning `200` when both are loaded and `503` otherwise.
3. Added `httpx==0.27.2` to `requirements.txt`, fixing a real environment gap that had silently prevented `test_predict_endpoint.py` from even being collected by pytest.
4. Added `tests/test_health_endpoint.py` (5 tests), following the same monkeypatched-singleton pattern as `test_predict_endpoint.py`.
5. Ran the full automated suite (47/47 passing) and manually started `uvicorn main:app` to verify both endpoints against the real model and label files with real HTTP requests.

**What still needs attention before Day 13's webcam-based scope:**
1. Write the connective code chaining `HandKeypointExtractor` → `FrameSampleBuffer` → `format_sequence`/`InferenceService`, since Day 13 assumes a real webcam-to-prediction path and that link doesn't exist yet. This remains the single most consequential open item.
2. Close the small test gaps on `model_loader`/`label_loader`/`keypoint_extractor` while the codebase is still small enough for that to be cheap.

**Does the implementation match the 30-Day Plan?** Yes, substantially, through Day 12. The module-level architecture is a strong match to the plan's own design, both API endpoints due by Day 12 are complete and match the plan's documented intent, and the one structural gap remaining (camera-to-buffer wiring) is exactly where the plan's own day-by-day sequencing places it — after Day 12, before Day 13's webcam-based end-to-end test.

**Recommendation on proceeding to Day 13:** Ready to proceed for the "sample input" half of Day 13's scope (a running backend now exists and has been proven to correctly handle a real sample sequence end-to-end). The webcam half of Day 13 additionally requires wiring `HandKeypointExtractor` → `FrameSampleBuffer` into the request path — a well-scoped, low-risk next step that reuses existing, already-tested components without requiring new design work.
