# Khmer Sign Language AI Backend — Engineering Documentation

**Status of the codebase this document describes:** Week 1 → Week 2 Day 12 (complete).
**Audience:** a junior developer joining the project who needs to understand the backend as it exists *right now*, not as it is eventually planned to be.

> ### ⚠️ Reality check — read this before anything else
> **Update (Day 12 complete):** `backend/main.py` is now a real, runnable FastAPI application. `uvicorn main:app` (run from `backend/`) starts a live server, loads the model and label map once at startup via a `lifespan` context manager, and serves both `/predict` and `/health/model`. This has been verified by actually starting the server and curling both endpoints (not just running the mocked pytest suite) — see Part 7 below.
>
> What is still true and still worth knowing:
> - `backend/routes/__init__.py` and `backend/routes/sessions.py` remain **empty files (0 bytes)**. There is still no session-management endpoint — that is correctly Week 3 (database-backed) scope, not a Day 12 gap.
> - `backend/routes/predict.py` and `backend/routes/health.py` both contain real, tested `APIRouter`s, and as of Day 12 both are **mounted in production** via `main.py`'s `app.include_router()` calls, not just inside the test suite.
> - `keypoint_extractor.py` (MediaPipe hand extraction) and `frame_sample_buffer.py` (frame accumulation/sampling) are fully implemented and unit-tested, but **nothing in the HTTP request path calls them yet** — this splice is intentionally still out of scope (see the 30-Day Plan's Week 2/Week 3 boundary). `routes/predict.py` still accepts an already-sampled `(30, 126)` sequence directly from the client. These two modules are currently exercised only by manual scripts (`camera_test.py`, `test_keypoint.py`) and their own unit tests.
>
> Every section below says explicitly which parts are live, wired, and tested versus which parts are implemented-but-not-yet-connected versus which parts are empty stubs. Part 9 (Engineering Review) revisits this distinction formally.

---

## PART 1 — Project Overview

### Purpose

This backend is the inference layer for a Khmer Sign Language (KSL) recognition prototype. Its job, end to end, is:

1. Accept a sequence of hand-position data representing a signed gesture.
2. Reshape that data into the exact tensor format the trained model expects.
3. Run the trained Keras model to classify the gesture into one of 25 classes (24 Khmer words/letters + `No_action`).
4. Return the predicted label, a confidence score, and the top-k alternative candidates as JSON.

It is deliberately **not** responsible for rendering UI (that's `frontend/`) or for long-term storage (that's `database/`, not yet touched by this backend).

### Folder structure (as it exists today)

```
Khmer-Sign-Language-AI-Prototype/
├── backend/
│   ├── .env                        # MODEL_PATH / LABEL_MAP_PATH overrides
│   ├── .gitignore
│   ├── README.md                   # informal project Q&A notes, not real docs
│   ├── requirements.txt
│   ├── main.py                     # FastAPI app entry point (app = FastAPI(), lifespan startup)
│   ├── labels/
│   │   └── label_map_25class.json  # label -> class-index map, 25 classes
│   ├── models/
│   │   └── best_model_25class_fix.h5  # trained Keras model (.h5)
│   ├── notebooks/                  # training/data notebooks (landmark_extract.ipynb etc.)
│   ├── ai_inference/                # the core ML/business-logic package
│   │   ├── keypoint_extractor.py   # MediaPipe hand-landmark extraction
│   │   ├── temporal_attention.py   # custom Keras layer used inside the .h5 model
│   │   ├── model_loader.py         # singleton loader for the Keras model
│   │   ├── label_loader.py         # singleton loader for the label map JSON
│   │   ├── model_info.py           # debug/printing helper
│   │   ├── frame_sample_buffer.py  # accumulate frames -> fixed (30,126) sequence
│   │   ├── feature_formatter.py    # (30,126) -> (30,252) velocity-augmented tensor
│   │   └── inference_service.py    # orchestrates format -> predict -> label -> top-k
│   ├── routes/                      # FastAPI routers package
│   │   ├── __init__.py             # EMPTY (package marker only)
│   │   ├── health.py               # GET /health/model — real, tested, mounted in prod
│   │   ├── sessions.py             # EMPTY — no session endpoints exist (Week 3 scope)
│   │   └── predict.py              # POST /predict — real, tested, mounted in prod
│   └── tests/                       # pytest suite + manual scripts
│       ├── test_feature_formatter.py
│       ├── test_frame_sample_buffer.py
│       ├── test_inference_service.py
│       ├── test_predict_endpoint.py
│       ├── test_health_endpoint.py
│       ├── test_keypoint.py         # manual visual script (opens a webcam), not assertions
│       └── camera_test.py           # manual webcam smoke test, not a pytest test
├── database/                        # not touched by backend code yet
└── frontend/                        # consumes the (future) HTTP API
```

### Major modules and responsibilities

| Package/module | Responsibility |
|---|---|
| `ai_inference` | All ML/business logic: loading the model and labels, converting raw hand positions into model-ready tensors, running inference, and formatting the result. Framework-agnostic — nothing here imports FastAPI. |
| `routes` | The HTTP boundary. Parses requests, delegates to `ai_inference`, translates exceptions into HTTP status codes. Should contain no ML logic itself (and, per `predict.py`'s own docstring, deliberately does not). |
| `main.py` | The FastAPI application entry point. Constructs `app = FastAPI(lifespan=lifespan)`, loads the model and label map once at startup, and mounts `routes/predict.py` and `routes/health.py`. Run with `uvicorn main:app` from within `backend/`. |
| `tests` | Automated unit/integration tests for `ai_inference` and `routes.predict`/`routes.health`, plus two manual, non-asserting webcam scripts left over from early MediaPipe experimentation. |

### High-level architecture

Logically, the codebase is layered like this. As of Day 12, every layer down to the buffering/extraction stage is assembled and live:

```
┌─────────────────────────────────────────────────────────┐
│  main.py — app = FastAPI(lifespan=...), router mounting  │
│  (REAL, verified: uvicorn main:app serves /predict        │
│   and /health/model)                                      │
├─────────────────────────────────────────────────────────┤
│  routes/predict.py, routes/health.py — HTTP boundary       │
│      (EXISTS, tested, mounted in prod)                     │
├─────────────────────────────────────────────────────────┤
│  ai_inference.inference_service.InferenceService          │
│      — orchestration layer (EXISTS, tested)                │
├─────────────────────────────────────────────────────────┤
│  ai_inference.feature_formatter                            │
│  ai_inference.model_loader / label_loader (EXISTS, tested)  │
├─────────────────────────────────────────────────────────┤
│  (Implemented but not called by the HTTP path yet)          │
│  ai_inference.keypoint_extractor                            │
│  ai_inference.frame_sample_buffer                            │
└─────────────────────────────────────────────────────────┘
```

---

## PART 2 — Complete Data Flow

The prompt's requested flow (`startup → FastAPI init → router registration → model loading → label loading → health → predict → inference → response`) is given below **annotated with what is real vs. planned**. As of Day 12, every stage in this flow is real and has been verified by actually starting `uvicorn main:app` and curling both endpoints (see Part 7).

```
[REAL — main.py]               Application startup: python process starts, uvicorn
                                imports main.py, evaluates app = FastAPI(lifespan=lifespan)
        ↓
[REAL — main.py]               FastAPI initialization: app object constructed
        ↓
[REAL — main.py]               Router registration: app.include_router(predict_route.router)
                                and app.include_router(health_route.router)
        ↓
[REAL — main.py lifespan]      Startup event fires: lifespan() context manager runs before
                                the server accepts requests
        ↓
[REAL — model_loader.py]       Model loading: load_model() reads MODEL_PATH from .env,
                                loads the .h5 Keras model with the custom
                                TemporalAttention layer registered, caches it in a
                                module-level singleton
        ↓
[REAL — label_loader.py]       Label loading: load_labels() reads LABEL_MAP_PATH from
                                .env, parses the JSON label→index map, caches it in a
                                module-level singleton
        ↓
[REAL — routes/health.py]      Health endpoint: GET /health/model reports current
                                model_loaded / label_map_loaded state via get_model()/
                                get_labels() (read-only, no side effects)
        ↓
[REAL, mounted in prod]        Prediction endpoint: routes/predict.py's POST /predict
        ↓
[REAL — inference_service.py]  Inference pipeline: InferenceService.predict()
        ↓
[REAL]                         Prediction response: JSON dict returned to the caller
```

### Function-by-function trace of the REAL, working path

This is the flow that actually executes today, driven by `tests/test_predict_endpoint.py`'s locally-assembled FastAPI app (or any future app that mounts `routes.predict.router` the same way).

**1. `routes.predict.predict(request: PredictRequest)`**
- **Who calls it:** FastAPI's routing layer, in response to an HTTP `POST /predict` whose JSON body matches `PredictRequest`.
- **Input:** a `PredictRequest` Pydantic model — effectively `{"sequence": List[List[float]]}`, expected to represent a `(30, 126)` array.
- **Returns:** a plain `dict` (FastAPI serializes it to JSON), or raises `HTTPException` (422/500/503).
- **Consumed by:** the HTTP client (frontend or test client).
- **Why here:** this is the HTTP boundary — the only file allowed to know about `fastapi`, `HTTPException`, and JSON request/response shapes.

**2. `routes.predict.get_inference_service()`**
- **Who calls it:** `predict()`, on every request.
- **Input:** none (reads module-level globals).
- **Returns:** the process-wide singleton `InferenceService`, lazily constructed on first call.
- **Consumed by:** `predict()`, which calls `.predict(positions)` on the returned service.
- **Why here:** keeps singleton construction/caching logic at the HTTP boundary (where the model/label singletons are known to be ready or not), rather than inside `InferenceService` itself, which stays a plain, injectable class with no knowledge of *how* its dependencies get built.

**3. `ai_inference.model_loader.get_model()`**
- **Who calls it:** `get_inference_service()`.
- **Input:** none.
- **Returns:** the cached Keras model object, or raises `RuntimeError` if `load_model()` was never called.
- **Consumed by:** `get_inference_service()`, to build `InferenceService`.
- **Why here:** `model_loader` is the single owner of "has the model been loaded, and what is it" — no other file is allowed to touch the `_model` global.

**4. `ai_inference.label_loader.get_labels()`**
- Symmetric to `get_model()`, but for the label→index dict. Same call site, same raise-if-not-ready contract.

**5. `ai_inference.inference_service.InferenceService.__init__(model, labels, top_k=3)`**
- **Who calls it:** `get_inference_service()`, exactly once per process (singleton).
- **Input:** the model object and labels dict obtained above.
- **Returns:** an `InferenceService` instance. Internally it also builds `self._index_to_label`, an explicit reverse map (`{index: label}`) built from `labels.items()` — not from `list(labels.keys())[index]`, which would silently break if dict insertion order ever diverged from class-index order.
- **Why here:** dependency injection — `InferenceService` never imports `model_loader` or `label_loader`. This makes it trivially unit-testable with any mock object exposing `.predict()`.

**6. `routes.predict.predict()` (continued) → converts `request.sequence` to `np.ndarray(dtype=np.float64)`**
- This is the only numpy conversion done at the HTTP boundary — everything past this point is `ai_inference`'s responsibility.

**7. `InferenceService.predict(positions)`**
- **Who calls it:** `routes.predict.predict()`.
- **Input:** `positions`, a `(30, 126)` numpy array (raw hand-position sequence, *not yet* velocity-augmented).
- **Returns:** a response `dict` (schema shown in Part 3), or propagates `ValueError` unmodified from `feature_formatter`.
- **Consumed by:** `routes.predict.predict()`, which either returns it directly (200) or catches the propagated `ValueError` and raises `HTTPException(422)`.
- **Why here:** this function is the single place the three remaining architecture stages ("compute velocity", "construct 252-feature tensor", "model inference") are stitched together.

**8. `ai_inference.feature_formatter.format_sequence(positions)`**
- **Who calls it:** `InferenceService.predict()`, exactly once per prediction.
- **Input:** `(30, 126)` numpy array.
- **Returns:** `(30, 252)` `float32` numpy array — original positions concatenated with frame-to-frame velocity (row 0's velocity forced to zero).
- **Raises:** `ValueError` if shape isn't exactly `(30, 126)` or if any value is non-finite (NaN/Inf).
- **Consumed by:** `InferenceService.predict()`, which wraps it in a batch dimension.
- **Why here:** this is a pure, stateless transformation with zero knowledge of HTTP, the model, or buffering — easy to unit test in isolation (see `test_feature_formatter.py`).

**9. `InferenceService.predict()` (continued) → `np.expand_dims(features, axis=0)`**
- Produces the `(1, 30, 252)` batch the Keras model expects (batch size 1).

**10. `model.predict(batch, verbose=0)`** (the injected Keras model's own method, not project code)
- **Input:** `(1, 30, 252)` float32 tensor.
- **Returns:** `(1, 25)` array of class probabilities.
- Timed with `time.perf_counter()` before/after to compute `inference_ms`.

**11. `InferenceService.predict()` (continued) → label mapping and top-k**
- `predicted_index = argmax(probabilities)`, mapped through `self._index_to_label`.
- `top_k_indices = argsort(probabilities)[::-1][:top_k]`, each mapped to `{"label": ..., "confidence": ...}`.
- Assembles and returns the final response dict.

**12. `routes.predict.predict()` (continued)**
- Returns the dict as-is on success (FastAPI serializes to a `200 OK` JSON response).
- Catches `ValueError` → `422 Unprocessable Entity`.
- Catches any other `Exception` (e.g. a TensorFlow-internal failure) → `500 Internal Server Error` with a generic message (no internals leaked).

### The NOT-yet-wired upstream path (client-side today)

`keypoint_extractor.HandKeypointExtractor` and `frame_sample_buffer.FrameSampleBuffer` implement the steps that would turn raw webcam frames into the `(30, 126)` sequence `routes/predict.py` expects — but nothing in the backend currently calls them as part of a request. As of Day 12, whatever produces the `(30, 126)` sequence sent to `/predict` (frontend-side MediaPipe + buffering, most likely) is outside this backend's code. These two modules exist, are fully unit-tested, and are architecturally ready to be reused server-side later, but today they are exercised only by:
- `tests/test_keypoint.py` and `tests/camera_test.py` — manual, visual, webcam-driven scripts (no assertions, must be run and eyeballed, not part of `pytest`'s meaningful pass/fail signal beyond "did it crash on import").
- `tests/test_frame_sample_buffer.py` — real automated unit tests of `FrameSampleBuffer` in isolation.

---

## PART 3 — Component Breakdown (every backend file)

### `backend/main.py`

| | |
|---|---|
| **Purpose** | The FastAPI application entry point. Constructs the `app` object, loads the model and label map exactly once at process startup, and mounts both real routers. |
| **Responsibility** | Application assembly only: build `app`, register a `lifespan` startup hook, include `routes/predict.py` and `routes/health.py`. No ML logic, no HTTP business logic — those stay in `ai_inference/` and `routes/` respectively. |
| **Public objects** | `app` (the `FastAPI` instance uvicorn serves); `lifespan(app)` — an `@asynccontextmanager` startup hook that calls `model_loader.load_model()` and `label_loader.load_labels()` before `yield`, then returns control to FastAPI to start accepting requests. |
| **Internal helpers** | None. |
| **Dependencies** | `fastapi.FastAPI`, `contextlib.asynccontextmanager`, `routes.predict`, `routes.health`, `ai_inference.model_loader`, `ai_inference.label_loader`. |
| **Objects created** | The single `app = FastAPI(lifespan=lifespan)` instance. |
| **Objects injected** | None — this is the composition root; it is the one file allowed to wire concrete routers and loaders together. |
| **Objects returned** | N/A — `app` is a module-level object uvicorn imports and serves (`uvicorn main:app`), not a function return value. |
| **Relationship with other files** | Imports `routes.predict`, `routes.health`, `ai_inference.model_loader`, `ai_inference.label_loader`. Nothing imports `main.py` — it is the top of the dependency graph, run directly by uvicorn. |
| **Why this file exists** | Every other piece of the request path (`InferenceService`, `feature_formatter`, the loaders, both routers) already existed and was independently tested before Day 12; this file is the composition root that assembles them into one runnable service, mirroring exactly how `tests/test_predict_endpoint.py` already proved the pieces fit together. |
| **SRP** | Satisfied — this file's only job is assembly (construct `app`, register startup, mount routers). It contains zero ML logic and zero HTTP business logic. |
| **Verification** | Manually verified by starting `uvicorn main:app` and issuing real HTTP requests: `GET /health/model` returned `200 {"model_loaded":true,"label_map_loaded":true,"num_classes":25}`, and `POST /predict` with a real `(30,126)` sequence returned a real prediction (`No_action`, confidence ≈0.25, `top_k` populated, `sequence_shape: [30,252]`) from the actual 20+ MB Keras model — not a mock. |
| **What changed from Week 1** | This file previously was a one-shot CLI diagnostic script (`main()` + `dummy_prediction()`) that loaded the model, printed diagnostics, ran one dummy `(1,30,252)` all-zero prediction, and exited — no server, no port bound. That CLI role is superseded by `GET /health/model` (loading can now be checked any time the server is running, not just once at process start) and is not preserved in this file. `ai_inference/model_info.py`'s `print_model_info()` is consequently no longer called from anywhere in production code; the file itself was left in place (not deleted) since removing it was out of Day 12's scope. |

### `backend/routes/health.py`

| | |
|---|---|
| **Purpose** | The HTTP boundary for `GET /health/model`: report whether the model and label map are currently loaded, without loading them. |
| **Responsibility** | Per its own module docstring, a read-only status check — it deliberately calls `get_model()`/`get_labels()`, never `load_model()`/`load_labels()`, so hitting this endpoint can never trigger a load as a side effect. |
| **Public objects** | `router` (an `APIRouter` instance); `health_model(response: Response) -> dict` (the route handler, registered via `@router.get("/health/model")`). |
| **Internal state** | None — stateless, reads the `model_loader`/`label_loader` singletons on every call. |
| **Dependencies** | `fastapi` (`APIRouter`, `Response`), `ai_inference.model_loader`, `ai_inference.label_loader`. |
| **Objects created** | The response `dict` per call. |
| **Objects injected** | FastAPI injects a `Response` object into the handler, which the handler mutates (`response.status_code = 503`) to report "not ready" without raising an exception — a deliberate style difference from `routes/predict.py`, chosen because a health check reporting `503` with a descriptive JSON body (`{"model_loaded": false, ...}`) is more useful to a caller than an `HTTPException`'s single `detail` string. |
| **Objects returned** | `{"model_loaded": bool, "label_map_loaded": bool, "num_classes": int | None}`, with HTTP `200` if both are loaded, `503` if either is not. |
| **Relationship with other files** | Depends on `ai_inference.model_loader` and `ai_inference.label_loader` only. Mounted in production by `main.py`'s `app.include_router(health_route.router)`. Tested by `tests/test_health_endpoint.py`, which follows the same monkeypatched-singleton pattern as `tests/test_predict_endpoint.py`. |
| **Why this file exists** | The Day 11 `/predict` endpoint's own 503 response tells a caller "not ready yet" but gives no way to proactively check readiness before sending a full `(30,126)` payload. `/health/model` closes that gap and matches the 30-Day Plan's section 6 API contract (`GET /health/model — Check whether model and label map are loaded`) exactly. |
| **SRP** | Satisfied — one job, "report current load state," with no ML logic and no ability to mutate loader state. |
| **Common mistakes to watch for** | Calling `load_model()`/`load_labels()` instead of `get_model()`/`get_labels()` here would turn a passive health check into an active loader — the first health check hit would then be responsible for a multi-second `.h5` load, which is exactly the side effect this endpoint is designed to avoid (loading happens once, at app startup, via `main.py`'s `lifespan`). |

### `backend/ai_inference/keypoint_extractor.py`

| | |
|---|---|
| **Purpose** | Wrap MediaPipe Holistic to extract per-frame left/right hand landmarks from a single OpenCV BGR frame. |
| **Responsibility** | Pure computer-vision extraction for exactly one frame at a time. No buffering, no sequencing, no model knowledge. |
| **Public functions** | `HandKeypointExtractor.__init__(model_complexity, min_detection_confidence, min_tracking_confidence)`; `extract(frame) -> dict`; `close()`. |
| **Internal/static helpers** | `extract_native_hand(hand_landmarks)` (static) — converts one MediaPipe hand into a wrist-relative `(63,)` vector (21 landmarks × 3 coords), or a zero vector if the hand wasn't detected. |
| **Dependencies** | `cv2`, `mediapipe`, `numpy`. |
| **Objects created** | `self.mp_holistic`, `self.holistic` (a `mp.solutions.holistic.Holistic` instance) in `__init__`. |
| **Objects injected** | None — MediaPipe configuration knobs are passed as constructor kwargs with sane defaults; no external service dependency to inject. |
| **Objects returned** | `extract()` returns `{"left_hand": (63,), "right_hand": (63,), "results": <MediaPipe Results>}`. |
| **Relationship with other files** | Consumed by `tests/test_keypoint.py` and `tests/camera_test.py` (manual scripts) and by `test_keypoint.py`'s import; **not** consumed by `routes/predict.py` or `inference_service.py` today. Its `(63,)` output shape is exactly what `FrameSampleBuffer.add_frame()` expects, so it is the intended upstream producer for that class, even though nothing currently wires them together automatically. |
| **Why this file exists** | Isolates the only MediaPipe-dependent code in the project into one class, so nothing else needs to know MediaPipe's API. |
| **SRP** | Well satisfied: this class does exactly one thing (per-frame extraction) and explicitly documents that it has no opinion about gesture start/stop timing. |
| **Common mistakes** | Forgetting to call `close()` to release MediaPipe resources (both manual scripts do call it, but only on the "normal" exit path, not if an exception occurs mid-loop). Also: `extract_native_hand` makes the wrist the origin (`pts -= pts[0]`) — a junior dev changing this would silently break compatibility with the trained model, which was trained on wrist-relative coordinates. |

### `backend/ai_inference/temporal_attention.py`

| | |
|---|---|
| **Purpose** | Custom Keras `Layer` (`TemporalAttention`) implementing an attention mechanism over the time dimension, learned during training and baked into `best_model_25class_fix.h5`. |
| **Responsibility** | Pure Keras layer math: given `(batch, time, features)`, compute per-timestep attention weights and return a `(batch, features)` weighted-sum context vector. |
| **Public functions** | `build(input_shape)`, `call(inputs)`, `get_config()` — the standard Keras `Layer` API. |
| **Internal helpers** | None beyond the standard layer lifecycle methods. |
| **Dependencies** | `tensorflow`, `tensorflow.keras.layers.Layer`. |
| **Objects created** | Three trainable weight tensors: `att_W` (feature×feature), `att_b` (feature,), `att_u` (feature,), created in `build()`. |
| **Objects injected** | None — weights are created internally via Keras's `add_weight`, standard for custom layers. |
| **Objects returned** | `call()` returns a `(batch, features)` tensor (the attended context vector). |
| **Relationship with other files** | Imported by `model_loader.py` and passed into `tf.keras.models.load_model(..., custom_objects={"TemporalAttention": TemporalAttention})` — **required** for the `.h5` file to deserialize at all, since Keras cannot reconstruct a custom layer class from a serialized model without this explicit mapping. |
| **Why this file exists** | Keras only saves a custom layer's *class name and config* in the `.h5` file, not its Python implementation. This file must exist, unmodified in its math, for the trained model to load anywhere. |
| **SRP** | Satisfied — one layer, one piece of math, no side responsibilities. |
| **Common mistakes** | Editing this file's math (even cosmetically) will silently produce a model that loads successfully but predicts incorrectly, since the *trained weights* assume this exact computation graph. This file should be treated as frozen/read-only unless retraining the model from scratch. |

### `backend/ai_inference/model_loader.py`

| | |
|---|---|
| **Purpose** | Load the trained Keras model exactly once per process and expose it as a singleton. |
| **Responsibility** | Own the `_model` global; know the model's file path (via `.env`/default); know which custom Keras objects are required to deserialize it. |
| **Public functions** | `load_model()` — loads (or returns cached) model; `get_model()` — returns the cached model or raises `RuntimeError` if not yet loaded. |
| **Internal helpers** | None. |
| **Dependencies** | `tensorflow`, `pathlib.Path`, `os`, `python-dotenv`, `ai_inference.temporal_attention.TemporalAttention`. |
| **Objects created** | `MODEL_PATH` (module-level `Path`, resolved once at import time from `BASE_DIR` + `.env`'s `MODEL_PATH` or the default `models/best_model_25class_fix.h5`). |
| **Objects injected** | None — this module is itself the injection *source* for `InferenceService` (via `routes/predict.py`'s `get_inference_service()`). |
| **Objects returned** | `load_model()`/`get_model()` both return the Keras `Model` object. |
| **Relationship with other files** | Depended on by `routes/predict.py` (via `get_model()`) and `backend/main.py` (via `load_model()`). Depends on `temporal_attention.py` for the custom layer registration. |
| **Why this file exists** | Loading a `.h5` TensorFlow model is expensive (disk I/O + graph construction); doing it once and caching avoids repeating that cost per request. Centralizing the path resolution and custom-object registration also means only one file needs to change if the model format or path convention changes. |
| **SRP** | Satisfied: this file only knows "where is the model file, how do I load it, and is it loaded yet." It has zero knowledge of what the model is used for downstream. |
| **Common mistakes** | Calling `get_model()` before anything has called `load_model()` (e.g., in a test that forgets to seed `model_loader._model` or call `load_model()` first) raises `RuntimeError` by design — this is intentional fail-fast behavior, not a bug to work around by adding a fallback default. |

### `backend/ai_inference/label_loader.py`

| | |
|---|---|
| **Purpose** | Load the label→index JSON map exactly once per process and expose it as a singleton. |
| **Responsibility** | Own the `_labels` global; know the label file's path. |
| **Public functions** | `load_labels()`, `get_labels()` — structurally identical contract to `model_loader.py`'s pair. |
| **Dependencies** | `json`, `pathlib.Path`, `os`, `python-dotenv`. |
| **Objects created** | `LABEL_PATH` (module-level `Path`). |
| **Objects returned** | Both functions return the parsed `dict` (`{"No_action": 0, "កាតាប": 1, ...}`). |
| **Relationship with other files** | Depended on by `routes/predict.py` and `backend/main.py`; conceptually paired 1:1 with `model_loader.py` (same pattern, same lifecycle, deliberately kept in a separate file rather than merged, since a model and its label map are independently swappable artifacts). |
| **Why this file exists** | Same reasoning as `model_loader.py` — cache an expensive-enough-to-avoid-repeating I/O operation and centralize its path resolution. |
| **SRP** | Satisfied — one responsibility, symmetric with `model_loader.py`. |
| **Common mistakes** | Assuming label *insertion order* in the JSON matches class index order. `InferenceService` deliberately does **not** make this assumption (it builds an explicit reverse map from the dict's values) — but if a junior dev writes new code against `label_loader.get_labels()` directly, they must remember to map by *value* (index), not by *position* in the dict. |

### `backend/ai_inference/model_info.py`

| | |
|---|---|
| **Purpose** | Debug/introspection helper — prints TensorFlow version, model input/output shape, and `model.summary()`. |
| **Responsibility** | Presentation of already-loaded model metadata; nothing else. |
| **Public functions** | `print_model_info(model)`. |
| **Dependencies** | `tensorflow` (only for `tf.__version__`). |
| **Objects returned** | None (prints only). |
| **Relationship with other files** | Called only by `backend/main.py`. |
| **Why this file exists** | Keeps diagnostic printing out of `main.py` itself, and out of `model_loader.py` (which shouldn't be responsible for *displaying* information about what it loads, only loading it). |
| **SRP** | Satisfied — a single, narrow "print info about this model" responsibility. |
| **Common mistakes** | None significant; this is a low-risk, side-effect-only utility. |

### `backend/ai_inference/frame_sample_buffer.py`

| | |
|---|---|
| **Purpose** | Accumulate raw per-frame `(63,)` + `(63,)` hand-position vectors and, on demand, reduce them to a fixed-length `(30, 126)` sequence, replicating the training notebook's sampling/padding logic exactly. |
| **Responsibility** | Buffering and fixed-length sampling only. Explicitly does **not** decide when a gesture starts/stops (that's the caller's job), and has no knowledge of velocity or the 252-feature tensor. |
| **Public functions (class `FrameSampleBuffer`)** | `__init__(target_length=30)`, `reset()`, `add_frame(left_hand, right_hand)`, `sample_sequence() -> np.ndarray`. |
| **Internal state** | `self._frames` — a plain Python list of `(126,)` combined vectors (append-only until `reset()`). |
| **Dependencies** | `numpy` only. |
| **Objects created** | The `_frames` list; on `sample_sequence()`, a stacked `(N, 126)` array and (if padding is needed) a repeated-last-frame padding block. |
| **Objects injected** | None — this class has no external dependencies to inject; it's a pure data structure with sampling logic. |
| **Objects returned** | `sample_sequence()` returns a `(30, 126)` numpy array. Raises `ValueError` from `add_frame()` on wrong input shape, and `RuntimeError` from `sample_sequence()` if called with zero accumulated frames. |
| **Relationship with other files** | Intended upstream producer is `keypoint_extractor.HandKeypointExtractor.extract()` (shape-compatible, `(63,)` + `(63,)`); intended downstream consumer is `feature_formatter.format_sequence()` (shape-compatible, `(30, 126)` in). **Neither connection is wired in production code today** — this class is currently only exercised by its own unit tests (`test_frame_sample_buffer.py`). |
| **Why this file exists** | The trained model requires a fixed-length 30-frame sequence, but real gestures vary in duration. This class is the single place that encodes "how do we turn a variable number of captured frames into exactly 30" — via `np.linspace` even-sampling when there are ≥30 frames, or last-frame-repeat padding when there are fewer. |
| **SRP** | Well satisfied — one job (accumulate + sample), clearly documented boundaries with neighboring stages. |
| **Common mistakes** | Forgetting to call `reset()` between captures — `sample_sequence()` is explicitly *not* self-clearing and is idempotent if called twice without an intervening `reset()`, which is correct behavior but easy to misuse if a caller assumes each `sample_sequence()` call implicitly starts a new capture window. |

### `backend/ai_inference/feature_formatter.py`

| | |
|---|---|
| **Purpose** | Convert a `(30, 126)` position sequence into the `(30, 252)` tensor the model was trained on, by concatenating the raw positions with frame-to-frame velocity. |
| **Responsibility** | Exactly two computations: "compute velocity" and "construct 252-feature tensor." Nothing about capture, buffering, or the model itself. |
| **Public functions** | `format_sequence(positions: np.ndarray) -> np.ndarray`. |
| **Dependencies** | `numpy` only. |
| **Objects created** | `velocity` (a `zeros_like(positions)` array, then filled via `positions[1:] - positions[:-1]`); the concatenated output array. |
| **Objects injected** | None — pure function, no state, no side effects, never mutates its input (validated by `test_input_not_modified`). |
| **Objects returned** | `(30, 252)` `float32` array. Raises `ValueError` on wrong shape or non-finite (NaN/Inf) input — deliberately fail-fast rather than coercing or silently ignoring bad data. |
| **Relationship with other files** | Called by exactly one place: `InferenceService.predict()`. Consumes the output shape produced by `FrameSampleBuffer.sample_sequence()` (though, as noted, that connection isn't wired in production yet — today the `(30, 126)` array comes directly from the HTTP request body). |
| **Why this file exists** | This exact velocity-augmentation formula must match the training notebook (`landmark_extract.ipynb`, Cell 5) bit-for-bit, or the model's predictions will be wrong in ways that are very hard to debug (the model will still run, just badly). Isolating it in one pure function makes it independently unit-testable against hand-checked values (see `test_velocity_computation`) without needing the real model loaded. |
| **SRP** | Excellently satisfied — this is close to a textbook example of a single-purpose, side-effect-free, well-documented pure function. |
| **Common mistakes** | Validating shape/finiteness *after* casting to `float32` instead of before — the actual code validates first, then casts, specifically so a dtype conversion never has a chance to mask a NaN/Inf that should already have failed. A junior dev "cleaning up" the order of these two steps would silently break this safety guarantee. |

### `backend/ai_inference/inference_service.py`

| | |
|---|---|
| **Purpose** | Orchestrate the full "raw positions → formatted tensor → model prediction → label-mapped, top-k response" pipeline. |
| **Responsibility** | Stitch together `feature_formatter` and an injected model object; own timing measurement, label-index mapping, and top-k selection. It does **not** load the model or labels itself. |
| **Public functions (class `InferenceService`)** | `__init__(model, labels: dict, top_k: int = 3)`; `predict(positions: np.ndarray) -> dict`. |
| **Internal helpers** | None beyond `__init__`'s construction of `self._index_to_label`. |
| **Dependencies** | `numpy`, `time`, `ai_inference.feature_formatter.format_sequence`. Notably **not** dependent on `model_loader` or `label_loader` at the import level — those are the caller's problem. |
| **Objects created** | `self._index_to_label` (dict, built once in `__init__`); per-call: `features`, `batch`, `probabilities`, `top_k_indices`, the response `dict`. |
| **Objects injected** | `model` (any object with `.predict(batch) -> array-like`) and `labels` (dict), both passed into the constructor by the caller (`routes/predict.py`'s `get_inference_service()`). This injection is exactly what makes `test_inference_service.py` able to test this class with a lightweight `_RecordingModel` mock instead of the real 20+ MB Keras model. |
| **Objects returned** | `predict()` returns a dict: `{"predicted_class_index": int, "predicted_label": str, "confidence": float, "top_k": [{"label": str, "confidence": float}, ...], "inference_ms": float, "sequence_shape": [30, 252]}`. |
| **Relationship with other files** | Depended on by `routes/predict.py` (constructed and called there). Depends on `feature_formatter.format_sequence`. Has an implicit, documented (not code-enforced) contract with `model_loader`/`label_loader` as the expected *source* of its injected constructor arguments. |
| **Why this file exists** | Without this class, `routes/predict.py` would need to know about tensor formatting, batching, timing, and label mapping directly — turning the HTTP route into a fat controller. This class is the seam that keeps the route thin and keeps ML orchestration logic independently testable without an HTTP layer at all. |
| **SRP** | Satisfied — this class's one job is "orchestrate one prediction from formatted-but-not-yet-tensorized input to a finished response dict." It correctly delegates tensor construction to `feature_formatter` rather than reimplementing it. |
| **Common mistakes** | Building the label reverse-map from `list(labels.keys())[index]` instead of from `labels.items()` — the former assumes dict insertion order matches class-index order, which is not guaranteed by the JSON label file's structure. The actual code explicitly avoids this trap (and `test_predict_label_mapping_survives_shuffled_dict_order` exists specifically to guard it). |

### `backend/routes/__init__.py`, `backend/routes/health.py`, `backend/routes/sessions.py`

| | |
|---|---|
| **Purpose** | All three files are **empty (0 bytes)**. |
| **Current status** | `__init__.py` makes `routes` a regular Python package (required for `import routes.predict` to work as a dotted import, though as an empty file it contributes no behavior). `health.py` and `sessions.py` are placeholder files with no code — they exist in the folder structure (likely scaffolded in the Week 1 commit) but implement nothing. |
| **What this means practically** | There is no `/health` endpoint and no session-management endpoint anywhere in this codebase today. Any documentation, diagram, or teammate that references "the health endpoint" is referring to planned-but-unbuilt functionality, not something you can currently curl. |
| **Why flagged separately** | The prompt that generated this document explicitly asked for a health-endpoint step in the data flow; per the "assume current implementation is source of truth" instruction, that step is documented here as absent rather than invented. |

### `backend/routes/predict.py`

| | |
|---|---|
| **Purpose** | The HTTP boundary for gesture prediction: parse the request, delegate to `InferenceService`, translate exceptions to HTTP status codes. |
| **Responsibility** | Per its own module docstring: "owns only the HTTP boundary... No feature formatting, label lookup, or model prediction logic lives here." |
| **Public objects** | `router` (an `APIRouter` instance); `PredictRequest` (Pydantic model, `{sequence: List[List[float]]}`); `predict(request: PredictRequest) -> dict` (the route handler, registered via `@router.post("/predict")`); `get_inference_service() -> InferenceService`. |
| **Internal state** | `_inference_service: InferenceService | None` — module-level singleton cache, mutated by `get_inference_service()` (and reset to `None` by tests via monkeypatching). |
| **Dependencies** | `fastapi` (`APIRouter`, `HTTPException`), `pydantic` (`BaseModel`), `numpy`, `ai_inference.model_loader`, `ai_inference.label_loader`, `ai_inference.inference_service.InferenceService`. |
| **Objects created** | The `InferenceService` singleton (lazily, on first request after the model/labels are loaded); a `numpy.ndarray` conversion of `request.sequence`. |
| **Objects injected** | Indirectly builds `InferenceService`'s dependencies from `model_loader.get_model()` / `label_loader.get_labels()` — this is the one place in the codebase where the loader singletons and the service class are wired together. |
| **Objects returned** | A `dict` (200 OK, JSON-serialized by FastAPI) or an `HTTPException` with status 422 (bad/NaN input), 500 (unexpected/model-level error), or 503 (model/labels not loaded yet). |
| **Relationship with other files** | Depends on `ai_inference.model_loader`, `ai_inference.label_loader`, `ai_inference.inference_service`. Is itself depended on only by `tests/test_predict_endpoint.py`, which mounts `router` into a local test-only `FastAPI()` app — **no production file currently mounts this router.** |
| **Why this file exists** | To be the single, thin translation layer between "the outside world speaks HTTP/JSON" and "the inside world speaks numpy arrays and Python exceptions." |
| **SRP** | Well satisfied. The 503-vs-422-vs-500 exception mapping is a good example of a controller doing controller-appropriate work (translating error *meaning* into HTTP semantics) without doing any of the underlying business logic itself. |
| **Common mistakes** | Two are explicitly guarded against by the existing tests: (1) letting a `ValueError` from bad input leak through as a 500 instead of 422 (`test_predict_wrong_row_count_returns_422` etc.); (2) failing to distinguish "not ready yet" (503) from "genuinely broken" (500) (`test_predict_not_loaded_returns_503`). A junior dev adding new exception types to `InferenceService` must remember to add a corresponding `except` clause here, or that new error will fall through to the generic 500 handler. |

### `backend/tests/*` (test and manual-script files)

| File | What it actually is |
|---|---|
| `test_feature_formatter.py` | Real, automated `pytest` unit tests for `format_sequence()`: output shape, position-block correctness, velocity math (including a hand-checked fixture), padded-frame zero-velocity, shape-contract violations, dtype coercion, NaN/Inf rejection, and input-purity (no in-place mutation). |
| `test_frame_sample_buffer.py` | Real, automated `pytest` unit tests for `FrameSampleBuffer`: exact-30 passthrough, downsampling via `linspace`, padding with the last real frame (not zero/average), `reset()` semantics, idempotent re-sampling, frame-order preservation, and fail-fast on empty buffer / malformed frame shapes. |
| `test_inference_service.py` | Real, automated `pytest` unit tests for `InferenceService`, using a `_RecordingModel` mock (records the batch it was called with) instead of the real Keras model: correct batch shape passed to the model, response schema, label-mapping correctness (including the shuffled-dict-order guard), top-k ordering/length, custom `top_k`, `ValueError` propagation without calling the model, and non-negative `inference_ms`. |
| `test_predict_endpoint.py` | Real, automated `pytest` integration tests for the `/predict` route, using FastAPI's `TestClient` against a locally-assembled `FastAPI()` app (not the production app, because none exists) with `model_loader`/`label_loader` singletons monkeypatched to a `_MockModel`. Covers 200/422/500/503 paths. |
| `test_keypoint.py` | **Not an automated test** despite the filename and being picked up by `pytest`'s default discovery (no `test_*` functions with asserts — it's a `print`-and-`imshow` loop that opens a real webcam). Running it under `pytest` will hang waiting for webcam frames/keypresses; it should be run manually with `python test_keypoint.py`, not via `pytest`. |
| `camera_test.py` | A manual webcam smoke test with no `test_` prefix, so `pytest` won't collect it at all. Opens the default camera, displays frames, exits on `q`. |

---

## PART 4 — Call Graph

### The real, working call graph (as exercised by `test_predict_endpoint.py`, or any future app that mounts the router the same way)

```
(test-only or future) FastAPI app
        │
        │  app.include_router(routes.predict.router)
        ▼
POST /predict
        │
        ▼
routes.predict.predict(request: PredictRequest)
        │
        ├──► routes.predict.get_inference_service()
        │            │
        │            ├──► ai_inference.model_loader.get_model()
        │            ├──► ai_inference.label_loader.get_labels()
        │            └──► ai_inference.inference_service.InferenceService(model, labels)
        │                         └──► builds self._index_to_label from labels.items()
        │
        ├──► np.array(request.sequence, dtype=np.float64)
        │
        ▼
InferenceService.predict(positions)
        │
        ├──► ai_inference.feature_formatter.format_sequence(positions)
        │            (validates shape + finiteness, computes velocity,
        │             concatenates -> (30, 252) float32)
        │
        ├──► np.expand_dims(features, axis=0)          -> (1, 30, 252)
        │
        ├──► model.predict(batch, verbose=0)            -> (1, 25) probabilities
        │            (timed via time.perf_counter())
        │
        ├──► np.argmax / np.argsort over probabilities
        ├──► self._index_to_label[...] lookups
        │
        ▼
        returns response dict
        │
        ▼
routes.predict.predict() returns dict  ──►  FastAPI serializes  ──►  200 OK JSON
        │
        (on ValueError from format_sequence)  ──► HTTPException(422)
        (on RuntimeError from get_inference_service) ──► HTTPException(503)
        (on any other Exception) ──► HTTPException(500)
```

### The independent, not-yet-connected upstream call graph

```
cv2.VideoCapture frame  (BGR, H×W×3)
        │
        ▼
ai_inference.keypoint_extractor.HandKeypointExtractor.extract(frame)
        │
        ├──► cv2.cvtColor(frame, BGR2RGB)
        ├──► self.holistic.process(rgb)              [MediaPipe]
        ├──► extract_native_hand(left_hand_landmarks)  -> (63,)
        └──► extract_native_hand(right_hand_landmarks) -> (63,)
        │
        ▼
        {"left_hand": (63,), "right_hand": (63,), "results": ...}
        │
        ▼  (repeated once per captured frame, caller-driven loop)
ai_inference.frame_sample_buffer.FrameSampleBuffer.add_frame(left_hand, right_hand)
        │
        ▼  (once the caller decides the gesture capture window is done)
FrameSampleBuffer.sample_sequence()  -> (30, 126)
        │
        ▼  (THIS is where, today, the connection to the HTTP path is manual/external —
        ▼   e.g. a client sends this (30,126) array as the /predict request body)
        │
        ▼
[joins the real call graph above at routes.predict.predict()'s request.sequence]
```

### Startup call graph — as actually usable today (`python backend/main.py`)

```
python backend/main.py
        │
        ▼
main()
        │
        ├──► ai_inference.model_loader.load_model()
        │            ├──► resolves MODEL_PATH from .env / default
        │            ├──► tf.keras.models.load_model(..., custom_objects={"TemporalAttention": TemporalAttention})
        │            └──► caches into module-level _model
        │
        ├──► ai_inference.label_loader.load_labels()
        │            ├──► resolves LABEL_PATH from .env / default
        │            ├──► json.load(...)
        │            └──► caches into module-level _labels
        │
        ├──► ai_inference.model_info.print_model_info(model)
        │            └──► prints TF version, input/output shape, model.summary()
        │
        └──► dummy_prediction(model, labels)
                     ├──► builds a (1, 30, 252) zero array
                     ├──► model.predict(dummy, verbose=0)
                     └──► prints predicted index/label/confidence
```

---

## PART 5 — Data Transformations

```
Camera frame (BGR)
(H, W, 3)
        │  cv2.cvtColor(BGR2RGB) + MediaPipe Holistic.process()
        │  — reduces a full image to 21 landmarks per hand, each with (x, y, z)
        ▼
Left hand landmarks              Right hand landmarks
(21, 3) -> flattened, wrist-relative
        │                                 │
        ▼                                 ▼
Left hand vector                  Right hand vector
(63,)                              (63,)
   — 21 × 3 = 63; wrist subtracted from every point so the hand's
     position becomes translation-invariant (only shape/pose matters,
     not where the hand is in the camera frame)
        │                                 │
        └───────────────┬─────────────────┘
                         │  FrameSampleBuffer.add_frame(): np.concatenate([left, right])
                         ▼
                Combined per-frame vector
                (126,)
                         │  accumulated across N frames as the gesture is captured
                         ▼
                Accumulated frames
                (N, 126)     N = however many frames were captured (variable)
                         │  FrameSampleBuffer.sample_sequence():
                         │    if N >= 30: np.linspace(0, N-1, 30) even-sampling
                         │    if N <  30: repeat the last frame to pad
                         │  — the model requires a FIXED sequence length (30),
                         │    but real signing gestures take a variable number
                         │    of frames, so this step normalizes duration
                         ▼
                Sampled sequence
                (30, 126)
                         │  feature_formatter.format_sequence():
                         │    velocity[0]  = 0
                         │    velocity[1:] = positions[1:] - positions[:-1]
                         │    features = concatenate([positions, velocity], axis=1)
                         │  — velocity captures MOTION between frames, which the
                         │    static position alone cannot represent, and matches
                         │    the exact feature engineering used at training time
                         ▼
                Formatted feature tensor
                (30, 252)     252 = 126 position dims + 126 velocity dims
                         │  InferenceService.predict(): np.expand_dims(axis=0)
                         │  — Keras models always expect a batch dimension, even
                         │    for a single sample
                         ▼
                Batched model input
                (1, 30, 252)
                         │  model.predict(batch)
                         │  — TemporalAttention-augmented sequence model
                         │    (LSTM/GRU-style + attention, per model_info.py's
                         │    output_shape / label count of 25)
                         ▼
                Raw model output
                (1, 25)     25 = 24 Khmer word/letter classes + "No_action"
                         │  probabilities = raw_predictions[0]  -> (25,)
                         │  argmax -> predicted_class_index
                         │  argsort()[::-1][:top_k] -> ranked alternatives
                         ▼
                Top-k response
                {
                  "predicted_class_index": int,
                  "predicted_label": str,
                  "confidence": float,
                  "top_k": [{"label": str, "confidence": float}, ...],
                  "inference_ms": float,
                  "sequence_shape": [30, 252]
                }
```

**Note on where the chain is actually joined today:** everything from "Camera frame" through "Sampled sequence `(30, 126)`" is implemented and unit-tested, but is not invoked by any HTTP code path. The live `/predict` endpoint starts its part of the chain at "Sampled sequence `(30, 126)`" — it receives that shape directly as JSON (`request.sequence`) and performs only the last three transformations (velocity augmentation → batching → model inference → top-k response).

---

## PART 6 — Request Lifecycle: `POST /predict`

Since no production app currently mounts the router, this section describes the lifecycle **exactly as `test_predict_endpoint.py` exercises it** — the same sequence would apply to any future production app that mounts `routes.predict.router` the same way.

1. A test-local `FastAPI()` instance is created and `routes.predict.router` is mounted via `app.include_router(predict_route.router)`.
2. An HTTP client (FastAPI's `TestClient`, or in production a real HTTP client) issues `POST /predict` with a JSON body `{"sequence": [[...126 floats...], ...30 rows...]}`.
3. FastAPI's routing layer matches the path/method to `routes.predict.predict`, and validates/parses the JSON body against the `PredictRequest` Pydantic model. **If the JSON doesn't have a `sequence` key of the right nested type, FastAPI itself returns a 422 before `predict()`'s body ever runs** (this is what `test_predict_missing_sequence_field_returns_422` actually verifies — Pydantic's own validation, not application code).
4. Inside `predict()`: `get_inference_service()` is called.
   - If `model_loader.get_model()` or `label_loader.get_labels()` raises `RuntimeError` (nothing has loaded them yet), `predict()` catches it and raises `HTTPException(503)` — response ends here.
   - Otherwise, an `InferenceService` is constructed (first call only; cached in `_inference_service` afterward) or reused.
5. `request.sequence` (a nested Python list) is converted to `np.array(..., dtype=np.float64)`.
6. `service.predict(positions)` runs:
   a. `format_sequence(positions)` validates shape `(30, 126)` and finiteness. If invalid, raises `ValueError`.
   b. If valid: velocity is computed, tensor is built `(30, 252)`, batched to `(1, 30, 252)`.
   c. `model.predict(batch, verbose=0)` runs the actual Keras forward pass, timed.
   d. Probabilities are turned into `predicted_index`, `predicted_label`, `confidence`, and a `top_k` list.
   e. A response dict is returned.
7. Back in `predict()`:
   - If step 6 raised `ValueError` → `HTTPException(422)` with the exception's message as `detail`.
   - If step 6 raised any other exception (e.g. a mocked `_FailingModel` raising `RuntimeError`, simulating a real TensorFlow-level failure) → `HTTPException(500, detail="Internal inference error.")` — deliberately generic, no internals leaked.
   - Otherwise → the dict is returned as-is.
8. FastAPI serializes the returned dict (or exception) to a JSON HTTP response and sends it back to the client.

---

## PART 7 — Startup Lifecycle

**Update (Day 12): this section previously described two paths, only one of which worked. As of Day 12, `main.py` is the FastAPI app, and starting it via uvicorn has been manually verified end-to-end (not just unit-tested with mocks).**

### 7a. What `uvicorn main:app` actually does (run from within `backend/`)

```
1. uvicorn imports main.py. Module-level code runs: fastapi.FastAPI is imported,
   routes.predict and routes.health are imported (which transitively import
   ai_inference.model_loader / label_loader / inference_service), and
   app = FastAPI(lifespan=lifespan) is constructed.
2. app.include_router(predict_route.router) and
   app.include_router(health_route.router) register both routers.
3. uvicorn starts the ASGI server and invokes the lifespan context manager
   before accepting any requests:
     - model_loader.load_model()
         - reads .env (MODEL_PATH=models/best_model_25class_fix.h5)
         - resolves BASE_DIR = parents[1] of model_loader.py, i.e. backend/
         - checks the .h5 file exists; raises FileNotFoundError if not
           (an unhandled exception here aborts startup -- fail-fast, the
           server never starts serving with a missing model)
         - calls tf.keras.models.load_model() with TemporalAttention registered
         - caches the result in the module-level `_model` singleton
     - label_loader.load_labels()
         - reads .env (LABEL_MAP_PATH=labels/label_map_25class.json)
         - checks the JSON file exists; raises FileNotFoundError if not
         - json.load()s it and caches into module-level `_labels`
     - lifespan yields control back to FastAPI
4. "Application startup complete" -- uvicorn now accepts HTTP requests on the
   bound port. GET /health/model and POST /predict are both live.
5. The _model/_labels singletons persist for the lifetime of this server
   process (a single worker reloads them independently if run with multiple
   workers -- see PART 9's Stateless Services note).
```

**Manually verified**, not just unit-tested: `uvicorn main:app --port 8123` was started, `GET /health/model` returned `200 {"model_loaded":true,"label_map_loaded":true,"num_classes":25}`, and `POST /predict` with a real `(30,126)` random sequence returned a genuine model prediction (not a mock) with `sequence_shape: [30, 252]`. The server was then stopped cleanly.

### 7b. What `python main.py` does now — **no longer a CLI diagnostic**

`main.py` no longer has an `if __name__ == "__main__":` block or a `main()` function; running it directly with `python main.py` just imports the module (constructing `app` but never starting a server, since only uvicorn's process invokes `lifespan`). The correct way to run this file is `uvicorn main:app`, not `python main.py`. This is a deliberate trade-off documented under "What changed from Week 1" in `routes/health.py` and `main.py`'s entries in Part 3: the one-shot dummy-prediction CLI check that Week 1 relied on is superseded by `GET /health/model`, which can be checked at any time the server is running, not just once at process start.

### 7c. What remains genuinely unbuilt

Nothing about *starting the backend* remains unbuilt as of Day 12. What is still unbuilt is unrelated to startup: the camera→keypoints→buffer splice into the request path (intentionally out of Day 12 scope) and the Week 3 database-backed endpoints (`/prediction-sessions`, `/prediction-events`, `/gesture-classes`, and `routes/sessions.py`).

---

## PART 8 — Dependency Graph

```
backend/main.py
    ↓
routes/predict.py
routes/health.py
ai_inference/model_loader.py        (direct, via lifespan startup)
ai_inference/label_loader.py        (direct, via lifespan startup)

routes/predict.py
    ↓
ai_inference/model_loader.py        (via get_model())
ai_inference/label_loader.py        (via get_labels())
ai_inference/inference_service.py
    ↓
ai_inference/feature_formatter.py

routes/health.py
    ↓
ai_inference/model_loader.py        (via get_model(), read-only)
ai_inference/label_loader.py        (via get_labels(), read-only)

ai_inference/frame_sample_buffer.py     (no internal deps beyond numpy;
                                          intended-but-unwired producer:
                                          ai_inference/keypoint_extractor.py)

ai_inference/keypoint_extractor.py      (no internal deps beyond cv2/mediapipe/numpy)

tests/test_feature_formatter.py     ──► ai_inference/feature_formatter.py
tests/test_frame_sample_buffer.py   ──► ai_inference/frame_sample_buffer.py
tests/test_inference_service.py     ──► ai_inference/inference_service.py
tests/test_predict_endpoint.py      ──► routes/predict.py
                                     ──► ai_inference/model_loader.py  (monkeypatched)
                                     ──► ai_inference/label_loader.py  (monkeypatched)
tests/test_keypoint.py              ──► ai_inference/keypoint_extractor.py
tests/camera_test.py                ──► ai_inference/keypoint_extractor.py
tests/test_health_endpoint.py       ──► routes/health.py
                                     ──► ai_inference/model_loader.py  (monkeypatched)
                                     ──► ai_inference/label_loader.py  (monkeypatched)

routes/sessions.py     (empty — no dependencies, no dependents; Week 3 scope)
routes/__init__.py     (empty — package marker only)
```

**Key structural observation:** `ai_inference` has zero dependencies on `routes` or `fastapi` in either direction except through `routes/predict.py` depending *on* `ai_inference` — the dependency arrow only ever points one way (routes → ai_inference), never the reverse. This is exactly the shape a layered architecture should have.

---

## PART 9 — Engineering Review

This section evaluates the **existing, real code** against each named principle. Per the task's instruction, no redesign is proposed — only genuine defects or absences, factually confirmed by direct inspection above.

| Principle | Verdict | Evidence |
|---|---|---|
| **Single Responsibility Principle** | **Followed well** in every real module. `feature_formatter.py` does one pure transform; `frame_sample_buffer.py` only buffers/samples; `model_loader.py`/`label_loader.py` only load-and-cache; `keypoint_extractor.py` only extracts one frame; `inference_service.py` only orchestrates; `routes/predict.py` only handles HTTP translation. Each file's own docstring explicitly states its boundary and what it does *not* do — a strong, intentional pattern, not an accident. |
| **Dependency Injection** | **Followed correctly where it matters most.** `InferenceService` receives `model` and `labels` via its constructor and has no import-time dependency on `model_loader`/`label_loader` — this is precisely why it can be unit-tested with `_RecordingModel`/`_MockModel` instead of a real 20+ MB Keras model. The wiring of concrete singletons happens in exactly one place (`routes/predict.py`'s `get_inference_service()`), which is the correct place for composition-root-style wiring to live. |
| **Separation of Concerns** | **Followed.** ML/business logic (`ai_inference/`) has no `fastapi` import anywhere. HTTP concerns (`routes/`) contain no tensor math, no model calls, no label lookups. |
| **Stateless Services** | **Followed for `InferenceService` and `feature_formatter`** — both are stateless given their constructor/arguments (no request leaves mutated shared state, confirmed by `test_input_not_modified`). **Not applicable in the usual sense to `FrameSampleBuffer`**, which is intentionally *stateful* (it's a buffer/accumulator by design) — this is correct given its job, not a violation, since the caller owns one instance per capture session rather than sharing it across concurrent requests. The module-level singletons in `model_loader.py`/`label_loader.py`/`routes/predict.py` (`_model`, `_labels`, `_inference_service`) are shared mutable global state — acceptable for read-only, load-once artifacts in a single-process app, but worth knowing about if the app is ever run with multiple worker processes (each process gets its own independent singleton and will reload the model separately — not a bug, just a fact to plan around operationally). |
| **Thin Controllers** | **Followed.** `routes/predict.py`'s `predict()` function is ~15 lines of actual logic: fetch service, convert input, call service, map three exception types to three status codes. All computation is delegated. |
| **Layered Architecture** | **Followed, and now complete top-to-bottom as of Day 12.** The dependency graph in Part 8 shows a clean one-directional layering (`main.py` → routes → services → pure functions/loaders), and as of Day 12 there is a real layer *above* `routes/predict.py`: `main.py` performs app assembly, startup-time loading, and router mounting, verified by an actual `uvicorn main:app` run. |

### Remaining gaps (post-Day-12; not architectural defects — absences of not-yet-built, out-of-scope code)

1. **The keypoint-extraction → buffering → predict pipeline is not connected end-to-end anywhere in the backend.** Each piece is solid and independently tested, but a new developer expecting to `curl` a webcam frame all the way to a prediction cannot do so with backend code alone today — the frontend (or an external step) must currently produce the `(30, 126)` sequence itself. This splice was explicitly deferred past Day 12 (see the 30-Day Plan's Week 2/Week 3 boundary) and remains the correct next piece of work.
2. **`routes/sessions.py` is empty.** No session-management endpoint exists — correctly out of scope until Week 3's database-backed work begins.
3. **No `logging` module usage anywhere** (`model_loader.py`, `label_loader.py` still use bare `print()`). Acceptable for a Day-12 prototype server with a single worker; worth revisiting once the service needs to run unattended or with multiple workers.

Resolved as of Day 12 (previously listed here as gaps): a live application entry point now exists (`main.py`, verified via a real `uvicorn` run against the real model and label files); `routes/health.py` is implemented, tested, and mounted in production.

None of the remaining gaps are bugs in the sense of "existing code behaves incorrectly" — every real, wired code path (feature formatting, buffering, inference orchestration, both routes, and now app startup) is correctly implemented and covered by passing tests, confirmed both by the automated suite (47 tests) and by manually exercising the live server. They are, instead, honest, correctly-sequenced gaps against the 30-Day Plan's later milestones.
