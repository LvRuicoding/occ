#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PREFIX="${PG_PREFIX:-$ROOT_DIR/pgsql}"
DATA_DIR="${PGDATA:-$PREFIX/data}"
LOG_FILE="${PG_LOGFILE:-$ROOT_DIR/postgres.log}"
PGPORT="${PGPORT:-55432}"
PGHOST="${PGHOST:-/tmp/lab6-pgsocket}"
PSQL="$PREFIX/bin/psql"
PG_CTL="$PREFIX/bin/pg_ctl"
INITDB="$PREFIX/bin/initdb"

if [ ! -x "$PSQL" ] || [ ! -x "$PG_CTL" ] || [ ! -x "$INITDB" ]; then
  echo "PostgreSQL binaries were not found under $PREFIX/bin. Run ./build_postgres.sh first." >&2
  exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
  "$INITDB" -D "$DATA_DIR" --locale=C
fi

export PGPORT PGHOST
mkdir -p "$PGHOST"

"$PG_CTL" -D "$DATA_DIR" -l "$LOG_FILE" -o "-c listen_addresses='' -k '$PGHOST' -p $PGPORT" start >/dev/null 2>&1 || true

"$PSQL" -h "$PGHOST" -p "$PGPORT" postgres -v ON_ERROR_STOP=1 <<'SQL'
DROP TABLE IF EXISTS lab6_outer;
DROP TABLE IF EXISTS lab6_inner;
CREATE TABLE lab6_outer (id int, name text);
CREATE TABLE lab6_inner (id int, name text);
INSERT INTO lab6_outer VALUES
  (1, 'a'), (2, 'b'), (3, 'b'), (4, 'c'), (5, 'd');
INSERT INTO lab6_inner VALUES
  (10, 'a'), (20, 'b'), (30, 'b'), (40, 'x');
SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET block_nested_loop_size = 1;
SELECT 'block=1', count(*) FROM lab6_outer o, lab6_inner i WHERE o.name = i.name;
SET block_nested_loop_size = 2;
SELECT 'block=2', count(*) FROM lab6_outer o, lab6_inner i WHERE o.name = i.name;
SET block_nested_loop_size = 8;
SELECT 'block=8', count(*) FROM lab6_outer o, lab6_inner i WHERE o.name = i.name;
EXPLAIN SELECT count(*) FROM lab6_outer o, lab6_inner i WHERE o.name = i.name;
SQL

