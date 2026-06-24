"""
索引耗时诊断脚本

用法：
    # 模式 A：所有文件一次性入库（推荐的真实用法）
    python scripts/profile_index.py --mode batch

    # 模式 B：逐文件入库（用来暴露 BM25 全量重建的 O(N^2) 问题）
    python scripts/profile_index.py --mode per-file

跑完后看日志里 [INDEX-TIMING] 行，对比两种模式下 bm25.index 的 rebuild 耗时差异。
"""

import argparse
import sys
from pathlib import Path

# 让脚本无论从哪运行都能 import app
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag.pipeline import RAGPipeline


def collect_files(data_dir: Path) -> list:
    exts = {".pdf", ".docx", ".doc", ".md", ".txt"}
    files = [str(p) for p in sorted(data_dir.iterdir()) if p.suffix.lower() in exts]
    if not files:
        print(f"[WARN] No documents under {data_dir}")
    return files


def main():
    parser = argparse.ArgumentParser(description="索引耗时诊断")
    parser.add_argument("--mode", choices=["batch", "per-file"], default="batch",
                        help="batch=一次全部入库; per-file=逐文件入库(暴露 BM25 重建)")
    parser.add_argument("--data", default=str(ROOT / "data"), help="文档目录")
    args = parser.parse_args()

    files = collect_files(Path(args.data))
    print(f"[RUN] mode={args.mode}, files={len(files)}")
    print("=" * 70)

    pipeline = RAGPipeline()

    if args.mode == "batch":
        pipeline.index_files(files)
    else:
        total = 0
        for i, f in enumerate(files, 1):
            print(f"\n--- [{i}/{len(files)}] {Path(f).name} ---")
            total += pipeline.index_files([f])
        print(f"\n[per-file TOTAL chunks] = {total}")

    print("=" * 70)
    print("[DONE] 检查上方 [INDEX-TIMING] 各行，定位最慢阶段。")
    print("关注点：")
    print("  1) bm25.index 的 rebuild 时间 vs tokenize 时间（rebuild 占比大 = BM25 全量重建拖累）")
    print("  2) vector.index 的 embed 时间 vs upsert 时间（embed 占比大 = 向量化是瓶颈）")
    print("  3) 对比 batch vs per-file：per-file 模式下 bm25.rebuild 会随文件数增长")


if __name__ == "__main__":
    main()
