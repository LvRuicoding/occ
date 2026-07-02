#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PREFIX="${PG_PREFIX:-$ROOT_DIR/pgsql}"
DATA_DIR="${PGDATA:-$PREFIX/data}"
LOG_FILE="${PG_LOGFILE:-$ROOT_DIR/postgres.log}"
RESULT_FILE="$ROOT_DIR/block_performance.txt"
PGPORT="${PGPORT:-55432}"
PGHOST="${PGHOST:-/tmp/lab6-pgsocket}"
PSQL="$PREFIX/bin/psql"
PG_CTL="$PREFIX/bin/pg_ctl"
INITDB="$PREFIX/bin/initdb"

QUERY="SELECT count(*) FROM restaurantaddress ra, restaurantphone rp WHERE ra.name = rp.name;"

if [ ! -x "$PSQL" ] || [ ! -x "$PG_CTL" ] || [ ! -x "$INITDB" ]; then
  echo "PostgreSQL binaries were not found under $PREFIX/bin. Run ./build_postgres.sh first." >&2
  exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
  "$INITDB" -D "$DATA_DIR" --locale=C
fi

export PGPORT
export PGHOST
mkdir -p "$PGHOST"

"$PG_CTL" -D "$DATA_DIR" -l "$LOG_FILE" -o "-c listen_addresses='' -k '$PGHOST' -p $PGPORT" start >/dev/null 2>&1 || \
  "$PG_CTL" -D "$DATA_DIR" -l "$LOG_FILE" -o "-c listen_addresses='' -k '$PGHOST' -p $PGPORT" restart >/dev/null

"$PSQL" -h "$PGHOST" -p "$PGPORT" postgres -v ON_ERROR_STOP=1 -tc "SELECT 1 FROM pg_database WHERE datname = 'similarity';" | grep -q 1 || \
  "$PSQL" -h "$PGHOST" -p "$PGPORT" postgres -v ON_ERROR_STOP=1 -c "CREATE DATABASE similarity;"

"$PSQL" -h "$PGHOST" -p "$PGPORT" similarity -f "$ROOT_DIR/similarity_data.sql" >/dev/null

{
  for size in 1 2 8 64 128 1024; do
    "$PSQL" -h "$PGHOST" -p "$PGPORT" similarity -v ON_ERROR_STOP=1 -At <<SQL
SET enable_hashjoin = off;
SET enable_mergejoin = off;
SET block_nested_loop_size = $size;
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) $QUERY
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) $QUERY
SQL
  done
} | python3 "$ROOT_DIR/parse_experiment.py" > "$RESULT_FILE"

echo "Wrote $RESULT_FILE"
