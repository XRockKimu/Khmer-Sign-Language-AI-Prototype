# Khmer Sign Language AI Prototype — UML Documentation

**Scope of truth:** these diagrams describe the backend **as it actually exists at Week 2 / Day 12**, cross-checked against `BACKEND_DOCUMENTATION.md`, `DAY12_AUDIT.md`, the repository source, and the 30-Day Plan. Where the 30-Day Plan assumes something that is not built yet (a live FastAPI app, `/health/model`, the camera→buffer wiring), the diagrams show it as *planned* or *missing* rather than pretending it exists. That honesty is deliberate — a UML that lies about what is wired would be worse than none.

Every diagram below is given as **PlantUML source** (paste into any PlantUML renderer, or the VS Code PlantUML extension) followed by a short explanation.

## Colour / status convention (used across all diagrams)

| Colour | Meaning |
|---|---|
| 🟢 `#CCFFCC` light green | Real — implemented, wired into the live path, and tested |
| 🟡 `#FFF2CC` light yellow | Implemented and unit-tested, but **not connected** into the HTTP request path |
| 🔴 `#F8CECC` light red | Empty stub (0 bytes) or not built at all |
| 🔵 `#DAE8FC` light blue | Planned / external component (frontend, database, future app) |

---

## 1. Component Diagram

```plantuml
@startuml Component_Diagram
title KSL Prototype — Component Diagram (Day 12 state)
skinparam componentStyle rectangle
skinparam wrapWidth 180
left to right direction

legend right
  |= Colour |= Meaning |
  |<#CCFFCC>| Real, wired & tested |
  |<#FFF2CC>| Implemented, NOT wired |
  |<#F8CECC>| Empty stub / not built |
  |<#DAE8FC>| Planned / external |
endlegend

package "frontend/ (Next.js)" #DAE8FC {
  [Camera Demo Page] as Cam
  [Client-side MediaPipe\n+ buffering] as ClientMP
}

package "backend/" {
  component "FastAPI app\n(app = FastAPI())\nNOT BUILT" as App #F8CECC
  component "main.py\nCLI diagnostic" as Main #CCFFCC

  package "routes/" {
    [predict.py\nPOST /predict] as Predict #CCFFCC
    [health.py] as Health #F8CECC
    [sessions.py] as Sessions #F8CECC
  }

  package "ai_inference/" {
    [inference_service.py] as Infer #CCFFCC
    [feature_formatter.py] as Fmt #CCFFCC
    [model_loader.py] as ML #CCFFCC
    [label_loader.py] as LL #CCFFCC
    [temporal_attention.py] as TA #CCFFCC
    [model_info.py] as MI #CCFFCC
    [keypoint_extractor.py] as KP #FFF2CC
    [frame_sample_buffer.py] as Buf #FFF2CC
  }
}

database "database/\n(untouched)" as DB #DAE8FC

Cam --> ClientMP
ClientMP --> Predict : POST (30,126) JSON

Predict --> Infer : predict(positions)
Predict --> ML : get_model()
Predict --> LL : get_labels()
Infer --> Fmt : format_sequence()
ML --> TA : register custom layer
Infer ..> TA : (baked into .h5)

Main --> ML
Main --> LL
Main --> MI

KP ..> Buf : intended producer (UNWIRED)
Buf ..> Fmt : intended consumer (UNWIRED)

App ..> Predict : would include_router() (MISSING)
Predict ..> DB : logging — Week 3
@enduml
```

**Explanation.** The backend splits cleanly into an HTTP boundary (`routes/`) and framework-agnostic ML logic (`ai_inference/`). Only `predict.py` is a real, tested router, and even it is mounted **only inside the test suite** — no production `FastAPI` app exists to include it, so the green boxes are reachable today only via pytest. `keypoint_extractor.py` and `frame_sample_buffer.py` (yellow) are complete and tested but nothing in the request path calls them; today the `(30,126)` sequence arrives pre-computed from the client. `health.py`/`sessions.py` are 0-byte stubs, and the database is not touched by any backend code yet.

---

## 2. Class Diagram

```plantuml
@startuml Class_Diagram
title KSL Backend — Class / Module Diagram (Day 12)
skinparam classAttributeIconSize 0

class HandKeypointExtractor #FFF2CC {
  - holistic : mp.Holistic
  + __init__(model_complexity, min_detection_confidence, min_tracking_confidence)
  + extract(frame) : dict
  + close()
  + {static} extract_native_hand(hand_landmarks) : ndarray(63,)
}

class FrameSampleBuffer #FFF2CC {
  - _frames : list<ndarray(126,)>
  + __init__(target_length = 30)
  + reset()
  + add_frame(left_hand, right_hand)
  + sample_sequence() : ndarray(30,126)
}

class TemporalAttention #CCFFCC {
  - att_W : Weight
  - att_b : Weight
  - att_u : Weight
  + build(input_shape)
  + call(inputs) : tensor(batch, features)
  + get_config() : dict
}

class InferenceService #CCFFCC {
  - _model
  - _labels : dict
  - _index_to_label : dict
  - top_k : int
  + __init__(model, labels, top_k = 3)
  + predict(positions : ndarray(30,126)) : dict
}

class PredictRequest #CCFFCC {
  + sequence : List[List[float]]
}

class "model_loader\n«module»" as ModelLoader #CCFFCC {
  - _model
  + load_model()
  + get_model()
}

class "label_loader\n«module»" as LabelLoader #CCFFCC {
  - _labels : dict
  + load_labels()
  + get_labels() : dict
}

class "feature_formatter\n«module»" as FeatureFormatter #CCFFCC {
  + format_sequence(positions : ndarray(30,126)) : ndarray(30,252)
}

class "model_info\n«module»" as ModelInfo #CCFFCC {
  + print_model_info(model)
}

class "routes.predict\n«module»" as PredictRoute #CCFFCC {
  - _inference_service : InferenceService | None
  + predict(request : PredictRequest) : dict
  + get_inference_service() : InferenceService
}

PredictRoute --> PredictRequest : parses
PredictRoute ..> InferenceService : creates & caches
PredictRoute --> ModelLoader : get_model()
PredictRoute --> LabelLoader : get_labels()
InferenceService --> FeatureFormatter : format_sequence()
ModelLoader --> TemporalAttention : custom_objects
ModelLoader ..> ModelInfo : (via main.py)

HandKeypointExtractor ..> FrameSampleBuffer : (63,)+(63,) — intended, UNWIRED
FrameSampleBuffer ..> FeatureFormatter : (30,126) — intended, UNWIRED
@enduml
```

**Explanation.** The only true OOP classes are `HandKeypointExtractor`, `FrameSampleBuffer`, `TemporalAttention`, `InferenceService`, and the Pydantic `PredictRequest`; the rest of the backend is module-level functions with singleton globals, shown here with the `«module»` stereotype. The key design win is that `InferenceService` receives `model` and `labels` through its constructor (dependency injection) and never imports the loaders — which is exactly why it is testable with a mock model. `TemporalAttention` is a custom Keras layer that must be registered at load time or the `.h5` file cannot deserialize. The two dashed, yellow associations mark the shape-compatible-but-unconnected seam between the capture half and the inference half.

---

## 3. Sequence Diagram — `POST /predict`

```plantuml
@startuml Sequence_Predict
title Sequence — POST /predict (as exercised by test_predict_endpoint.py)
skinparam sequenceMessageAlign center
actor Client
participant "FastAPI\nrouting" as FastAPI
participant "routes.predict\n.predict()" as Route
participant "get_inference_\nservice()" as Getter
participant "model_loader" as ML
participant "label_loader" as LL
participant "InferenceService" as Svc
participant "feature_formatter" as Fmt
participant "Keras model" as Model

Client -> FastAPI : POST /predict\n{sequence:[[...126]x30]}
activate FastAPI
FastAPI -> FastAPI : validate body\nvs PredictRequest

alt body malformed (missing/typed wrong)
  FastAPI --> Client : 422 (Pydantic)
else body valid
  FastAPI -> Route : predict(request)
  activate Route
  Route -> Getter : get_inference_service()
  activate Getter
  Getter -> ML : get_model()
  Getter -> LL : get_labels()
  alt loaders not ready
    ML --> Getter : raise RuntimeError
    Getter --> Route : RuntimeError
    Route --> Client : 503 Service Unavailable
  else ready
    Getter -> Svc : new / cached InferenceService(model, labels)
    Getter --> Route : service
    deactivate Getter
    Route -> Route : np.array(sequence, float64)  (30,126)
    Route -> Svc : predict(positions)
    activate Svc
    Svc -> Fmt : format_sequence(positions)
    alt bad shape / NaN / Inf
      Fmt --> Svc : raise ValueError
      Svc --> Route : ValueError
      Route --> Client : 422 Unprocessable Entity
    else valid
      Fmt --> Svc : (30,252) float32
      Svc -> Svc : expand_dims -> (1,30,252)
      Svc -> Model : predict(batch)  [timed]
      Model --> Svc : (1,25) probabilities
      Svc -> Svc : argmax, argsort -> label + top-k
      Svc --> Route : response dict
      deactivate Svc
      Route --> Client : 200 OK JSON
    end
  end
  deactivate Route
end
alt any other exception (e.g. TF failure)
  Route --> Client : 500 Internal (generic)
end
deactivate FastAPI
@enduml
```

**Explanation.** This is the real, tested request path. Pydantic rejects a malformed body with `422` before the handler runs. The handler stays thin: it fetches the singleton service, converts JSON to a numpy array, delegates, and maps exactly three exception classes onto `503` (dependencies not loaded), `422` (bad/NaN input from `feature_formatter`), and `500` (anything else, with internals hidden). All tensor math, timing, and label mapping happen inside `InferenceService`. Note the diagram starts from an already-sampled `(30,126)` body — MediaPipe and buffering are *not* part of this sequence today.

---

## 4. Activity Diagram — AI Pipeline

```plantuml
@startuml Activity_Pipeline
title Activity — AI pipeline (frame to prediction)
start
:Capture camera frame (BGR, HxWx3);
:cv2.cvtColor BGR->RGB;
:MediaPipe Holistic.process();
fork
  :extract_native_hand(left) -> (63,);
fork again
  :extract_native_hand(right) -> (63,);
end fork
:concatenate -> per-frame (126,)
(wrist-relative, translation-invariant);
:FrameSampleBuffer.add_frame();
repeat
  :accumulate frames (N,126);
repeat while (more frames in gesture?) is (yes)
->no;
:sample_sequence();
if (N >= 30?) then (yes)
  :np.linspace even-sample 30 frames;
else (no)
  :repeat last frame to pad to 30;
endif
:sampled sequence (30,126);

note right #F8CECC
  MISSING LINK (Day 12):
  nothing automatically feeds
  sample_sequence() output into
  format_sequence(). Everything
  ABOVE this line is implemented
  but not wired into /predict.
end note

:format_sequence() — velocity
velocity[0]=0; velocity[1:]=pos[1:]-pos[:-1];
:concatenate -> (30,252);
:expand_dims -> (1,30,252);
:model.predict() [TemporalAttention model];
:probabilities (1,25);
:argmax -> predicted index/label;
:argsort[::-1][:k] -> top-k;
:assemble response dict
{label, confidence, top_k, inference_ms, shape};
stop
@enduml
```

**Explanation.** The pipeline is two independently-correct halves. The upper half (frame → MediaPipe → wrist-relative `(63,)` hands → `(126,)` per frame → variable `(N,126)` → normalised `(30,126)`) is fully implemented and tested but disconnected. The `MISSING LINK` note marks exactly where the two halves fail to meet: today the `(30,126)` array arrives from the HTTP body, not from `FrameSampleBuffer`. The lower half (velocity augmentation → `(30,252)` → batch → model → argmax/top-k → response) is fully wired inside `InferenceService`.

---

## 5. Deployment Diagram

```plantuml
@startuml Deployment_Diagram
title Deployment — Day 12 (what can actually run today)
skinparam node {
  BackgroundColor<<planned>> #DAE8FC
  BackgroundColor<<real>> #CCFFCC
  BackgroundColor<<missing>> #F8CECC
}

node "Client device" <<planned>> {
  artifact "Next.js frontend\n(camera + client MediaPipe)" as FE
}

node "Backend host" {
  artifact "uvicorn + FastAPI app\nNOT BUILT — uvicorn backend.main:app FAILS" <<missing>> as Server
  artifact "python backend/main.py\n(CLI diagnostic — RUNS)" <<real>> as CLI
  folder "backend/models" {
    artifact "best_model_25class_fix.h5" as H5
  }
  folder "backend/labels" {
    artifact "label_map_25class.json" as JSON
  }
  artifact ".env\n(MODEL_PATH / LABEL_MAP_PATH)" as ENV
}

database "PostgreSQL\n(Week 3 — not provisioned)" <<planned>> as PG

FE ..> Server : HTTPS POST /predict\n(once the app exists)
CLI --> H5 : load_model()
CLI --> JSON : load_labels()
CLI --> ENV : read paths
Server ..> H5 : (would load at startup)
Server ..> PG : (Week 3 logging)
@enduml
```

**Explanation.** The only thing deployable today is the `python backend/main.py` CLI diagnostic (green), which loads the model and label map, prints diagnostics, and runs one dummy prediction. The intended runtime — `uvicorn backend.main:app` — **fails today** because `main.py` never constructs a `FastAPI()` instance (red). The frontend and PostgreSQL are planned. The model `.h5`, label JSON, and `.env` are real on-disk artifacts consumed via path resolution.

---

## 6. Package Diagram

```plantuml
@startuml Package_Diagram
title Package Diagram — dependency direction (one-way: routes -> ai_inference)
skinparam packageStyle rectangle

package "backend" {
  package "ai_inference" #CCFFCC {
    [model_loader]
    [label_loader]
    [model_info]
    [temporal_attention]
    [feature_formatter]
    [inference_service]
    [keypoint_extractor] #FFF2CC
    [frame_sample_buffer] #FFF2CC
  }
  package "routes" {
    [predict] #CCFFCC
    [health] #F8CECC
    [sessions] #F8CECC
  }
  package "tests" {
    [test_feature_formatter]
    [test_frame_sample_buffer]
    [test_inference_service]
    [test_predict_endpoint]
  }
  [main]
  package "models / labels" #DAE8FC {
    [best_model_25class_fix.h5]
    [label_map_25class.json]
  }
}

routes ..> ai_inference : uses (one-directional)
main ..> ai_inference : uses
tests ..> ai_inference : verifies
tests ..> routes : verifies (mounts predict.router locally)
ai_inference ..> "models / labels" : loads at runtime

note bottom of ai_inference
  ai_inference has ZERO imports of
  fastapi or of routes. The dependency
  arrow only ever points routes -> ai_inference.
end note
@enduml
```

**Explanation.** The package layering is clean and one-directional: `routes → ai_inference`, never the reverse, and `ai_inference` contains no FastAPI imports at all. `main.py` and the test suite both depend on `ai_inference`; the tests additionally mount `routes.predict.router` into a throwaway app to verify it. `temporal_attention` sits inside `ai_inference` because it must be importable when `model_loader` deserialises the `.h5`.

---

## 7. Startup Lifecycle Diagram

```plantuml
@startuml Startup_Lifecycle
title Startup Lifecycle — two entry points, two outcomes
start
if (entry point?) then (python backend/main.py)
  :import numpy, model_loader,\nlabel_loader, model_info;
  :__main__ guard -> main();
  :model_loader.load_model()
  - resolve MODEL_PATH from .env
  - check .h5 exists (else FileNotFoundError)
  - tf.keras.models.load_model(
      custom_objects=TemporalAttention)
  - cache into _model singleton;
  :label_loader.load_labels()
  - resolve LABEL_MAP_PATH
  - json.load(), cache into _labels;
  :model_info.print_model_info();
  :dummy_prediction()
  - (1,30,252) zeros -> model.predict()
  - print index/label/confidence;
  :process exits — no server, no port;
  stop
else (uvicorn backend.main:app)
  #F8CECC:AttributeError:
  module 'main' has no attribute 'app';
  #F8CECC:No FastAPI() instance,
  no include_router(),
  no lifespan/startup handler
  anywhere in production code;
  end
endif
@enduml
```

**Explanation.** "Starting the backend" means two different things, and only one works. `python backend/main.py` runs a one-shot diagnostic that loads the model and labels into module-level singletons, prints info, runs a dummy forward pass, and exits. `uvicorn backend.main:app` fails immediately because there is no `app` object and no startup/lifespan handler anywhere in production code — this is the single most important fact before trying to serve predictions over HTTP.

---

## 8. Week 1 Architecture

```plantuml
@startuml Week1_Architecture
title Week 1 Architecture — Foundation & Validation (Days 1-7)
skinparam componentStyle rectangle

package "Delivered in Week 1" {
  [README.md\ninformal notes] as RM #CCFFCC
  [model_loader.py\nsingleton load/get] as ML #CCFFCC
  [label_loader.py\nsingleton load/get] as LL #CCFFCC
  [model_info.py\ndiagnostics] as MI #CCFFCC
  [keypoint_extractor.py\nMediaPipe -> (63,) hands] as KP #CCFFCC
  [main.py\nCLI: load + dummy predict] as MAIN #CCFFCC
}

artifact "best_model_25class_fix.h5" as H5
artifact "label_map_25class.json" as JSON

MAIN --> ML
MAIN --> LL
MAIN --> MI
ML --> H5
LL --> JSON

note bottom
  Focus: prove the model files, label map,
  and backend foundation load and produce a
  plausible dummy prediction (30x252 -> 25 classes).
  Day 7 "Week 1 report" artifact: NOT found in repo.
end note
@enduml
```

**Explanation.** Week 1 established the foundation: singleton loaders for the model and label map (fail-fast on missing files), a diagnostics printer, the MediaPipe keypoint extractor, and a CLI `main.py` that ties loading together and runs a dummy `(1,30,252)` prediction to confirm the `.h5` and label JSON are compatible. Everything here loads and runs; the only tracked miss is that no Day-7 written report artifact exists in version control.

---

## 9. Week 2 Architecture

```plantuml
@startuml Week2_Architecture
title Week 2 Architecture — Inference Pipeline (Days 8-12)
skinparam componentStyle rectangle

package "HTTP boundary" {
  [routes/predict.py\nPOST /predict] as PR #CCFFCC
  [routes/health.py] as H #F8CECC
  [routes/sessions.py] as S #F8CECC
}

package "ai_inference (orchestration + pure fns)" {
  [inference_service.py\norchestrate + top-k] as IS #CCFFCC
  [feature_formatter.py\n(30,126)->(30,252)] as FF #CCFFCC
  [frame_sample_buffer.py\n(N,126)->(30,126)] as FB #FFF2CC
  [model_loader / label_loader] as LD #CCFFCC
}

package "tests (pytest)" {
  [test_predict_endpoint\ntest_inference_service\ntest_feature_formatter\ntest_frame_sample_buffer] as T #CCFFCC
}

PR --> LD : get_model/get_labels
PR --> IS : predict(positions)
IS --> FF : format_sequence()
FB ..> FF : intended input (UNWIRED)
T ..> PR : mounts router locally
T ..> IS
T ..> FF
T ..> FB

note bottom #F8CECC
  Gaps at Day 12:
  - /health/model not implemented (health.py empty)
  - no production FastAPI app mounts predict.router
  - frame_sample_buffer not wired into the request path
end note
@enduml
```

**Explanation.** Week 2 built the inference pipeline behind a thin HTTP boundary. `feature_formatter` (velocity augmentation), `frame_sample_buffer` (fixed-length sampling), and `inference_service` (orchestration + top-k) are all implemented and heavily unit-tested, and `routes/predict.py` exposes them through `POST /predict`. The remaining Day-12 gaps are concrete and specific: `/health/model` is unimplemented, no production app mounts the predict router, and the buffer is not spliced into the request path.

---

## 10. End-to-End Camera → Prediction Workflow

```plantuml
@startuml EndToEnd_Workflow
title End-to-End — Camera to Prediction (real vs missing)
skinparam sequenceMessageAlign center
actor User
participant "Next.js\nCamera Page" as FE #DAE8FC
participant "HandKeypoint\nExtractor" as KP #FFF2CC
participant "FrameSample\nBuffer" as BUF #FFF2CC
participant "POST /predict\n(routes.predict)" as API #CCFFCC
participant "Inference\nService" as SVC #CCFFCC
participant "feature_\nformatter" as FMT #CCFFCC
participant "Keras\nmodel" as MDL #CCFFCC

User -> FE : perform sign to camera
activate FE

group Capture half — IMPLEMENTED but NOT wired server-side
  FE -> KP : extract(frame)  [per frame]
  KP --> FE : {left(63,), right(63,)}
  FE -> BUF : add_frame() x N
  FE -> BUF : sample_sequence()
  BUF --> FE : (30,126)
end

note over KP, BUF #F8CECC
  In the backend these two classes exist and are
  tested, but nothing calls them in the request path.
  Today this work happens client-side (or externally);
  the backend receives an already-sampled (30,126).
end note

FE -> API : POST /predict {sequence (30,126)}
activate API

note over API #F8CECC
  REACHABILITY GAP: no production FastAPI app
  mounts this router yet — reachable only via
  pytest TestClient today.
end note

API -> SVC : predict(positions)
activate SVC
SVC -> FMT : format_sequence() -> (30,252)
FMT --> SVC : (30,252)
SVC -> MDL : predict((1,30,252))
MDL --> SVC : (1,25) probabilities
SVC -> SVC : argmax + top-k + timing
SVC --> API : response dict
deactivate SVC
API --> FE : 200 OK\n{label, confidence, top_k, ...}
deactivate API
FE --> User : show Khmer label + confidence + top-3
deactivate FE
@enduml
```

**Explanation.** This ties everything together and marks the two real gaps between the plan's vision and Day-12 reality. First, the capture half (`HandKeypointExtractor` → `FrameSampleBuffer`) exists and is tested in the backend but isn't invoked in the request path — today the `(30,126)` sequence is produced client-side/externally. Second, even the fully-wired inference half is reachable only through pytest, because no production `FastAPI` app mounts the router. Close those two gaps (assemble the app + wire the buffer chain, both reusing code that already exists) and this workflow becomes real end-to-end.

---

## Summary of what these diagrams encode

| Area | Real & wired | Implemented, unwired | Missing / stub |
|---|---|---|---|
| Model & label loading | ✅ `model_loader`, `label_loader` | — | — |
| Feature formatting | ✅ `feature_formatter` (30,126→30,252) | — | — |
| Inference orchestration | ✅ `InferenceService` + top-k | — | — |
| `/predict` route logic | ✅ tested (200/422/500/503) | — | not mounted in a live app |
| Keypoint extraction | — | 🟡 `HandKeypointExtractor` | — |
| Frame buffering | — | 🟡 `FrameSampleBuffer` | — |
| Camera→buffer splice | — | — | 🔴 no glue code |
| FastAPI app / startup | — | — | 🔴 no `app`, no lifespan |
| `/health/model` | — | — | 🔴 `health.py` empty |
| Session endpoints | — | — | 🔴 `sessions.py` empty |
| Database logging | — | — | 🔵 Week 3 |

The consistent story across all ten diagrams: the ML core is genuinely production-quality and well-tested in isolation; the two remaining pieces of work are **assembly** (wire the existing halves into one runnable app) and the **one missing health route** — neither of which requires new business logic.
