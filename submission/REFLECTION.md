# Reflection — Lab 19

**Tên:** _Dương Quang Khải_
**Cohort:** _A20_
**Path đã chạy:** _lite_

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Kết quả Precision@10 (benchmark.py): keyword 77.8%, semantic 73.2%, **hybrid 78.6%** — hybrid thắng trung bình.

Theo loại query:
- **`exact`** (15q): BM25 thắng — kw 96.7% = hyb 96.7% > sem 88.7%. Query chứa thuật ngữ verbatim ("Kubernetes", "OAuth JWT") nên keyword match là tín hiệu mạnh nhất; vector loãng hơn vì embed cả ngữ cảnh.
- **`paraphrase`** (15q): điểm thấp toàn bộ — kw 33.3%, hyb 32.0%, sem 24.0%. Model `bge-small-en` train cho tiếng Anh nên semantic recall trên paraphrase tiếng Việt yếu; đây là teaching moment — đổi sang `bge-m3` (docker path) sẽ giúp vector thắng nhóm này.
- **`mixed`** (20q): **hybrid thắng rõ** — hyb 100% > sem 98.5% > kw 97.0%. Query có cả thuật ngữ exact lẫn ý paraphrase, RRF gộp hai tín hiệu nên robust nhất.

**Khi không dùng hybrid:** (1) corpus thuần technical/code, query luôn là exact keyword → pure BM25 đủ tốt mà rẻ hơn (P99 3.6ms vs hybrid 19.6ms, không cần embed). (2) Latency/cost budget cực gắt hoặc query thuần ngữ nghĩa, không từ khoá → pure vector. Hybrid chỉ đáng khi query đa dạng (mix exact + paraphrase) — đúng pattern user thật.

---

## Điều ngạc nhiên nhất khi làm lab này

Embedding model English (`bge-small-en`) khiến nhóm `paraphrase` tiếng Việt tụt mạnh (24%) — chứng minh "chọn embedding model" là quyết định kiến trúc, không phải boilerplate. Online lookup Feast/SQLite đạt P99 0.81ms, nhanh hơn nhiều so với kỳ vọng.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
