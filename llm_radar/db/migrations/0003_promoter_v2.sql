PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS items (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  created_at DATETIME NOT NULL,
  state TEXT NOT NULL,
  last_score REAL DEFAULT 0,
  last_ucb REAL DEFAULT 0,
  obs_days INTEGER DEFAULT 0,
  cluster_id TEXT,
  meta_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_items_state ON items(state);
CREATE INDEX IF NOT EXISTS idx_items_cluster ON items(cluster_id);
CREATE INDEX IF NOT EXISTS idx_items_created ON items(created_at);

CREATE TABLE IF NOT EXISTS signals (
  id TEXT NOT NULL,
  day DATE NOT NULL,
  prior REAL DEFAULT 0,
  velocity REAL DEFAULT 0,
  semantic REAL DEFAULT 0,
  reputation REAL DEFAULT 0,
  score REAL DEFAULT 0,
  ucb REAL DEFAULT 0,
  PRIMARY KEY (id, day)
);
CREATE INDEX IF NOT EXISTS idx_signals_day ON signals(day);

CREATE TABLE IF NOT EXISTS transitions (
  id TEXT NOT NULL,
  ts DATETIME NOT NULL,
  from_state TEXT,
  to_state TEXT,
  reason TEXT,
  details TEXT
);
CREATE INDEX IF NOT EXISTS idx_transitions_id ON transitions(id);
CREATE INDEX IF NOT EXISTS idx_transitions_ts ON transitions(ts);

CREATE TABLE IF NOT EXISTS labels (
  id TEXT NOT NULL,
  ts DATETIME NOT NULL,
  label TEXT,
  note TEXT
);
CREATE INDEX IF NOT EXISTS idx_labels_id ON labels(id);
CREATE INDEX IF NOT EXISTS idx_labels_ts ON labels(ts);
