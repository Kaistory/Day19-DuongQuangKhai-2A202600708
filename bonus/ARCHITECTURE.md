# Hybrid AI Memory — Architecture

**Contributors:** Dương Quang Khải (A20)
**POC:** `agent.py` (`HybridMemoryAgent.remember()` / `.recall()`) + `demo.py` (5 queries).

Một trợ lý AI cá nhân cho người dùng Việt Nam cần *nhớ* ba thứ với vòng đời rất
khác nhau: **episodic memory** (hội thoại, tài liệu đã đọc — đổi liên tục),
**stable profile** (ngôn ngữ, tốc độ đọc, lĩnh vực quan tâm — đổi theo tuần) và
**recent activity** (query 1 giờ qua — đổi theo giây). POC này tách đúng ba vòng
đời đó vào hai hệ lưu trữ: **Vector Store** cho episodic và **Feature Store**
(Feast) cho profile + activity.

## Sơ đồ kiến trúc

```
        ┌──────────────────────────── WRITE PATH ─────────────────────────────┐
 user text ──► chunk_text() ──► embed (bge-small-en, 384d) ──► Qdrant upsert
                                                              (payload: user_id, text)

 activity/profile events ──► Parquet (offline) ──► feast materialize ──► online store
                                                                         (SQLite/Redis)

        └──────────────────────────── READ PATH ─────────────────────────────┘
 query ─┬─► BM25 (per-user chunks) ─┐
        │                            ├─► RRF (k=60) ─► top-3 memories ─┐
        └─► Qdrant vector search ───┘   (filter user_id)               │
                                                                       ├─► assembled
 query ───► Feast get_online_features(user_id) ─► profile + activity ──┘   context ─► LLM
            (user_profile_features, query_velocity_features)              (string → prompt)
```

Hai store giao nhau **chỉ ở bước assemble context** (cuối `recall()`): episodic
trả "user đã đọc gì", feature store trả "user là ai + đang làm gì", rồi gộp thành
một context string cho LLM. Tách read path giúp mỗi store scale và refresh độc lập.

## 3 quyết định kiến trúc (tradeoff explicit)

### 1. Chunking strategy — sentence-pack ~280 ký tự (vs per-message vs per-conversation)

`chunk_text()` cắt theo ranh giới câu rồi đóng gói tới ~280 ký tự/chunk.
- **Per-message** cho retrieval quality cao nhất (mỗi vector rất focused) nhưng
  *storage cost* và số vector phình to, và một câu ngắn ("ok, cảm ơn") tạo ra
  embedding nhiễu kéo precision xuống.
- **Per-conversation** rẻ storage nhưng embedding bị *dilute*: một vector trung
  bình hoá nhiều chủ đề → recall kém, lại dễ tràn context window khi trả về.
- **Chọn sentence-pack ~280 ký tự** vì cân giữa ba trục *retrieval quality vs
  storage cost vs context window*: đủ ngữ cảnh để vector có nghĩa, đủ nhỏ để top-3
  chunk vừa prompt budget. 280 ký tự ≈ 1–3 câu tiếng Việt, hợp với độ dài ghi chú.

### 2. Feature schema — tabular features (vs embedding features)

Profile dùng **feature dạng bảng** (`reading_speed_wpm` Int64,
`preferred_language` String, `topic_affinity` String, `queries_last_hour` Int64),
mỗi feature gắn entity `user`, source Parquet, TTL riêng (xem §3).
- **Embedding feature** (vector "latent preference" học từ lịch sử) biểu diễn sở
  thích tinh vi hơn, nhưng *không giải thích được* (vì sao recommend?), tốn pipeline
  huấn luyện riêng, và khó debug khi sai.
- **Chọn tabular** vì với một POC, profile cần *minh bạch + cheap to compute +
  trực tiếp dùng để filter/boost* (vd boost doc theo `topic_affinity`). Khi scale
  có thể thêm một embedding feature view song song — không phải thay thế.

### 3. Freshness strategy — 3 tốc độ refresh theo use case

Vòng đời dữ liệu quyết định cơ chế cập nhật, gắn vào **TTL của từng feature view**:
- `query_velocity_features` **TTL=1h, streaming/Push API (sub-second)** — fraud &
  "tôi đang quan tâm gì gần đây" cần phản ánh tức thì; TTL dài sẽ trả tín hiệu chết.
- `item_popularity_features` **TTL=24h, batch hàng giờ** — độ phổ biến giảm dần,
  cập nhật mỗi giờ là đủ, streaming là over-engineering.
- `user_profile_features` **TTL=30 ngày, batch hàng ngày** — thuộc tính ổn định;
  refresh quá thường vừa tốn vừa vô ích.

Tradeoff cốt lõi: **freshness vs cost**. Sub-second không miễn phí (cần stream
infra + Push API); ta chỉ trả giá đó cho dữ liệu thật sự cần tươi. Đây chính là lý
do PIT join (NB4) + TTL đúng quan trọng — sai TTL → training-serving skew.

## Loại bỏ một lựa chọn — episodic trong Feature Store

Tôi đã cân nhắc lưu episodic memory **ngay trong feature store** dưới dạng một
embedding feature view (mỗi user một vector "memory"), để chỉ phải vận hành **một**
hệ. Nhưng tôi **tách riêng vào Vector Store** vì: (1) episodic cần *similarity
search top-K* — feature store chỉ làm key-value lookup theo entity, không có ANN;
(2) **re-index cycle khác hẳn** — memory mới sinh mỗi giờ (append liên tục) còn
profile đổi theo tuần, nhét chung làm materialize đắt vô lý; (3) episodic là
*unbounded growth*, cần TTL/decay và pruning riêng. Một store sai mục đích sẽ phải
bẻ cong cả hai workload.

## Vietnamese-context considerations

- **Tokenizer**: BM25 hiện `text.lower().split()` theo khoảng trắng — tiếng Việt là
  ngôn ngữ *âm tiết tách rời* nên "cơ sở dữ liệu" thành 3 token rời, làm hỏng tần
  suất. Production nên dùng `pyvi`/`underthesea` để gộp từ ghép; tradeoff là +độ trễ
  và phụ thuộc model. Đây là quyết định "think-hard", không để AI tự chọn.
- **Code-switching (vi/en)**: user VN trộn "deploy lên prod", "fix bug". Embedding
  model nên multilingual (`bge-m3`) thay vì `bge-small-en` (English-only, yếu trên
  paraphrase tiếng Việt — đã đo 24% recall ở NB2). `preferred_language` lưu cả
  giá trị `mix` để chọn nhánh xử lý.
- **Privacy / Nghị định 13/2023 (PDPD)**: episodic chứa dữ liệu cá nhân → cần
  per-user isolation (đã làm: Qdrant filter theo `user_id` payload) và đường xoá
  dữ liệu theo yêu cầu.

## What this POC doesn't handle yet

- **Privacy isolation cứng**: hiện chỉ filter payload `user_id` trong một
  collection dùng chung; chưa per-user collection / encryption at rest / xoá GDPR-PDPD.
- **Memory decay / forgetting**: episodic chưa có TTL hay pruning "30 ngày không
  truy cập → archive"; memory tăng vô hạn.
- **CRUD trên memory**: chỉ có `remember` (append); chưa update/delete một ký ức.
- **Personalization re-ranking**: chưa boost kết quả theo `topic_affinity` (có thể
  thành RRF 3-retriever: BM25 + vector + profile-prior).
- **Multi-device sync & memory consolidation** (gộp 5 ký ức tương tự thành 1 summary).

---

### Vibe-coding log (optional)

- **Prompt hiệu quả nhất**: đưa spec rõ "RRF k=60, rank 1-based, filter theo user_id
  payload, fallback khi Feast unavailable" → agent sinh `_hybrid_search` đúng một lần.
- **Prompt fail**: nhờ "chọn chunking strategy tốt nhất" mà không cho ràng buộc
  (storage/latency/context window) → câu trả lời chung chung; phải tự ra quyết định
  rồi mới nhờ implement.
