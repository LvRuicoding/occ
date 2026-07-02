#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SRC_DIR="$ROOT_DIR/postgresql-12.0"
PREFIX="${PG_PREFIX:-$ROOT_DIR/pgsql}"

cd "$SRC_DIR"

if [ ! -f config.status ] || ! ./config.status --config | grep -q -- "--prefix=$PREFIX"; then
  ./configure --enable-depend --enable-cassert --enable-debug CFLAGS="-O0" --prefix="$PREFIX"
fi

make
make install
