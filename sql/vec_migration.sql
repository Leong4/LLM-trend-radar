-- 向量表：记录每个 chunk 的嵌入元数据
CREATE TABLE IF NOT EXISTS vec_chunk (
  chunk_id   INTEGER PRIMARY KEY,
  model      TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  norm       REAL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 全文检索（用于 BM25 混合召回）
-- 约定：rowid = chunk_id，文本由后续脚本同步写入
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunk
USING fts5(text, content='');

-- 索引（可选）
CREATE INDEX IF NOT EXISTS idx_vec_model ON vec_chunk(model);