"""
Run this script to see the exact SQL queries for a natural-language input.
Usage:
    python -m agent2.utils.show_sql "Give me population for Hessen from 2022 to 2025"
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from agent2.pipeline.query_parser import parse_query


def build_sql(query: str) -> None:
    params = parse_query(query)

    states = params.spatial
    years  = params.temporal
    attrs  = params.attributes

    attr_cols  = ",\n       ".join(f"sd.{a}" for a in attrs)
    states_arr = ", ".join(f"'{s}'" for s in states)
    years_arr  = ", ".join(str(y) for y in years)

    state_filter = (
        f"WHERE s.state_name = ANY(ARRAY[{states_arr}])\n  AND sd.stat_year = ANY(ARRAY[{years_arr}])"
        if "all" not in states
        else f"WHERE sd.stat_year = ANY(ARRAY[{years_arr}])"
    )

    sql_main = f"""-- Query 1: Main data lookup
SELECT s.state_name,
       sd.stat_year,
       {attr_cols}
FROM state_demographics sd
JOIN states s ON s.state_id = sd.state_id
{state_filter}
ORDER BY s.state_name, sd.stat_year;"""

    sql_partition = f"""-- Query 2: Partition check (which states exist in Agent-2)
SELECT DISTINCT s.state_name
FROM state_demographics sd
JOIN states s ON s.state_id = sd.state_id
WHERE s.state_name = ANY(ARRAY[{states_arr if 'all' not in states else "-- all states"}]);"""

    print("\n" + "=" * 60)
    print(f"  NL Query  : {query}")
    print(f"  Type      : {params.query_type}")
    print(f"  States    : {states}")
    print(f"  Years     : {years}")
    print(f"  Attributes: {attrs}")
    print("=" * 60)
    print()
    print(sql_main)
    print()
    print(sql_partition)
    print()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Give me population for Hessen from 2022 to 2025"
    build_sql(q)
