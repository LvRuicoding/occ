#!/usr/bin/env python3
import json
import sys

SIZES = [1, 2, 8, 64, 128, 1024]


def read_json_arrays(text):
    decoder = json.JSONDecoder()
    pos = 0
    arrays = []

    while True:
        start = text.find("[", pos)
        if start < 0:
            return arrays

        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            pos = start + 1
            continue

        if isinstance(value, list):
            arrays.append(value)
        pos = start + end


def main():
    text = sys.stdin.read()
    blocks = read_json_arrays(text)
    if len(blocks) % 2 != 0:
        raise SystemExit("Expected two EXPLAIN JSON results per block size")

    print("# block_size first_ms second_ms shared_hit shared_read")
    for size, first_json, second_json in zip(SIZES, blocks[0::2], blocks[1::2]):
        first = first_json[0]
        second = second_json[0]
        plan = second["Plan"]
        print(
            size,
            f"{first['Execution Time']:.3f}",
            f"{second['Execution Time']:.3f}",
            int(plan.get("Shared Hit Blocks", 0)),
            int(plan.get("Shared Read Blocks", 0)),
        )


if __name__ == "__main__":
    main()
