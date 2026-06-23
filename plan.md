# Plan — Hoàn thành toàn bộ rubric.md (Day 19 Lab)

> Nhật ký kế hoạch + thực hiện cho nhiệm vụ "thực hiện toàn bộ nhiệm vụ trong `rubric.md`".
> Ngày: 2026-06-23. Path: **lite** (Qdrant in-memory + SQLite Feast). Python 3.12 (`.venv`).

---

## 0. Bối cảnh & phát hiện ban đầu

- 4 notebook `.py` (jupytext) **đã có sẵn code giải** (phần TODO đã điền) → nhiệm vụ là
  **chạy** để sinh output, không phải viết logic.
- `.venv/Scripts/` (Windows) đã cài đủ package: fastembed, qdrant_client, rank_bm25,
  feast, fastapi, uvicorn, httpx, polars, pandas, pyarrow, jupytext, jupyter, nbconvert,
  nbclient, ipykernel.
- `data/` chưa tồn tại → phải seed. Chưa có `.ipynb`.
- Makefile dùng path Unix (`.venv/bin`) → trên Windows chạy trực tiếp `.venv/Scripts/python.exe`.
- Lỗi encoding console Windows (cp1252) khi in ký tự Unicode → fix bằng
  `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` cho mọi lệnh.
- Khi notebook spawn subprocess (`uvicorn` ở NB3, `feast` ở NB4) → cần prepend
  `.venv/Scripts` vào `PATH`.

---

## 1. Kế hoạch (6 bước)

| # | Bước | Tiêu chí rubric |
|---|---|---|
| 1 | Seed data (`seed_corpus.py`) → 1000 docs + 50 golden queries | nền tảng |
| 2 | Convert `.py`→`.ipynb` (jupytext) + execute (nbconvert, giữ output) | NB1–NB4 |
| 3 | `benchmark.py` (hybrid thắng) + `verify_lite.py` | reproducible (5pts) |
| 4 | Screenshots → `submission/screenshots/` (≥1/NB) | submission |
| 5 | Điền `submission/REFLECTION.md` (số liệu thật, ≤200 chữ) | submission |
| 6 | Bonus: `bonus/ARCHITECTURE.md` + `agent.py` + `demo.py` | 20 bonus pts |

---

## 2. Thực hiện & kết quả

### Bước 1 — Seed
- `PYTHONIOENCODING=utf-8 python scripts/seed_corpus.py` → `data/corpus_vn.jsonl` (1000),
  `data/golden_set.jsonl` (50), deterministic seed=42.

### Bước 2 — Execute notebooks
- `jupytext --to notebook notebooks/*.py` → 4 `.ipynb`.
- `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900`
  cho từng notebook (PATH có `.venv/Scripts`, env UTF-8).

| NB | Kết quả chính |
|---|---|
| 1 | `Indexed: 1000 vectors`, vector dim 384; paraphrase query → cluster `cloud` |
| 2 | P@10: **hybrid 78.6% > keyword 77.8% > semantic 73.2%**; slice: exact→BM25 96.7%, mixed→hybrid 100%, paraphrase yếu (model EN) |
| 3 | `/search` latency_ms 13.9; bảng P99 kw 3.5 / sem 13.8 / **hybrid 18.8ms < 50ms PASS** |
| 4 | feast apply 3 views + materialize OK; online lookup u_001 OK; 100-call **P99 0.81ms < 10ms**; PIT join **3 dòng** |

**Sửa NB4 (`04_feast_feature_store.py`):** `entity_df` gốc query `u_001` ở `NOW-2h`
nhưng feature của u_001 ghi ở `NOW-1h` (tương lai) → PIT loại đúng → chỉ 2 dòng.
Đã đổi query timestamp để mỗi user query *sau* feature của họ → đủ **3 dòng**, vẫn
đúng ngữ nghĩa point-in-time (kèm comment giải thích). Re-convert + re-execute.

### Bước 3 — Benchmark + verify
- `python scripts/benchmark.py` → **PASS** (hybrid beats keyword +0.8pp, semantic +5.4pp);
  latency 5000 calls/mode (hybrid P99 19.6ms).
- `python scripts/verify_lite.py` → **All checks passed**.

### Bước 4 — Screenshots
- nbconvert `--to html` cho 4 notebook → serve qua `python -m http.server 8123`
  (file:// bị Playwright chặn) → Playwright MCP full-page screenshot.
- Output: `submission/screenshots/nb1..nb4_*.png` (4 ảnh). Dọn HTML tạm + dừng server.

### Bước 5 — REFLECTION
- `submission/REFLECTION.md` điền số liệu thật: mode nào thắng loại query nào, khi nào
  không dùng hybrid (corpus thuần exact → BM25; latency gắt / query thuần ngữ nghĩa → vector).

### Bước 6 — Bonus
- `bonus/agent.py` — `HybridMemoryAgent.remember()`/`.recall()`: episodic per-user
  (Qdrant in-memory, filter `user_id`) + hybrid RRF (k=60, reuse NB2) + profile từ
  Feast online (reuse NB4). Fallback khi Feast unavailable.
- `bonus/demo.py` — 5 query (vector / profile / fresh / paraphrase / mixed) → **exit 0**.
- `bonus/ARCHITECTURE.md` — **1042 từ**: sơ đồ data-flow, 3 quyết định có tradeoff
  (chunking ~280 ký tự; tabular vs embedding features; freshness 3 tốc độ theo TTL),
  rejected alternative (episodic trong feature store), VN-context (tokenizer pyvi/
  underthesea, code-switching, Nghị định 13), honest limitations.

---

## 3. Lưu ý reproducible

- `data/*.jsonl`, feast `registry.db`/`online_store.db`/`data/*.parquet` bị `.gitignore`
  **có chủ ý** → tái tạo bằng `python scripts/seed_corpus.py` + chạy NB4.
- Notebook `.ipynb` (giữ output) **được commit** làm deliverable.
- Lệnh tái tạo từ sạch (Windows):
  ```
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 .venv/Scripts/python.exe scripts/seed_corpus.py
  .venv/Scripts/python.exe -m jupytext --to notebook notebooks/*.py
  # execute từng nb với nbconvert --execute (PATH có .venv/Scripts, timeout 900)
  .venv/Scripts/python.exe scripts/benchmark.py
  .venv/Scripts/python.exe bonus/demo.py
  ```

---

## 4. Git / submission

- Commit `2a993d1` — submission (4 ipynb + REFLECTION + 4 screenshots + bonus/).
- Commit `e860911` — `docs/` (slides PDF + HTML).
- Push: `origin` = `github.com/Kaistory/Day19-Track2-VectorFeatureStore-Lab` (fork public),
  branch `main`. `upstream` = VinUni (không push).
- `.claude/` để **untracked** (config local).
- **Việc thủ công còn lại:** đảm bảo repo public → paste URL vào ô Day 19 trên VinUni LMS
  → giữ public đến khi có điểm. **Không cần PR.**

---

## 5. Trạng thái cuối

- Docker: `docker compose down` (3 container removed) + Docker Desktop engine đã tắt.
- Không còn server local nào chạy.
- Tổng: **Core 100/100 đạt mọi ngưỡng + Bonus 20/20 đủ deliverable.**
