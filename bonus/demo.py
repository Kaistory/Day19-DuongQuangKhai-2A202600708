"""Demo for HybridMemoryAgent — 5 queries spanning vector / profile / fresh / paraphrase / mixed.

Prereqs (already done by the main lab):
  - `python scripts/seed_corpus.py`
  - NB4 ran `feast apply` + `materialize-incremental` (so the online store has u_001).

Run: `python bonus/demo.py`   (exits 0 on success)
"""
from __future__ import annotations

import sys

from agent import HybridMemoryAgent


def main() -> int:
    agent = HybridMemoryAgent()
    user = "u_001"

    # Seed some episodic memories for the user (things they read / noted).
    agent.remember(
        "Hôm nay tôi đọc một bài về Kubernetes auto-scaling và cách Pod tự mở rộng "
        "theo lưu lượng. Bài viết nhắc tới HPA và cluster autoscaler.",
        user,
    )
    agent.remember(
        "Ghi chú: serverless với AWS Lambda giúp tối ưu chi phí khi traffic biến động, "
        "trả tiền theo từng lần gọi hàm.",
        user,
    )
    agent.remember(
        "Đọc về bảo mật: mã hoá dữ liệu nhạy cảm khi lưu trữ và dùng OAuth/JWT cho xác thực.",
        user,
    )
    agent.remember(
        "Tài liệu về tự động mở rộng hạ tầng theo số người dùng đang online, "
        "kết hợp cân bằng tải đa vùng.",
        user,
    )

    queries = [
        "Tôi đã đọc gì về Kubernetes?",            # 1. vector hit thuần
        "Recommend đọc gì tiếp",                     # 2. cần topic_affinity (profile)
        "Tôi đang quan tâm gì gần đây?",             # 3. cần fresh activity (queries_last_hour)
        "Tài liệu về tự động mở rộng hạ tầng?",      # 4. paraphrase (vector wins)
        "Cho tôi summary cloud security",            # 5. mixed (episodic + profile)
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n{'=' * 70}\n[Query {i}/5]")
        print(agent.recall(q, user))

    print(f"\n{'=' * 70}\nDemo complete — 5 queries answered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
