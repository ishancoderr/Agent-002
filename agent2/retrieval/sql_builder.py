"""
Build the SQL for a parsed query.

Kept apart from the code that runs it so the statement can be inspected,
logged and tested without touching a database. Every builder returns the
statement twice over: a bound form for execution (parameters kept separate,
never interpolated) and a readable form with the values written in, which is
what the evaluation log shows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class BuiltQuery:
    """A statement ready to run, plus the readable form for the log."""
    sql: str                    # bound form, :named parameters
    params: Dict[str, Any]      # values for those parameters
    readable: str               # same statement with values written in, for logging
    label: str = ""             # what this statement is for


class SqlBuilder:
    """Turns parsed query fields into the statements the local store runs.

    Values always travel as bound parameters. The readable form exists only to
    be logged; it is never executed, so a state name with a quote in it cannot
    change what runs."""

    @staticmethod
    def _as_sql_list(values) -> str:
        return ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in values)

    def partition_check(self, states: List[str]) -> BuiltQuery:
        """Which of the requested states this agent holds any row for at all.

        Asked first because a state with no rows is a spatial gap, which is a
        different thing from a state whose rows exist but are empty."""
        sql = """
            SELECT DISTINCT s.state_name
            FROM state_demographics sd
            JOIN states s ON s.state_id = sd.state_id
            WHERE s.state_name = ANY(:states)
        """
        readable = (
            "SELECT DISTINCT s.state_name\n"
            "FROM state_demographics sd\n"
            "JOIN states s ON s.state_id = sd.state_id\n"
            f"WHERE s.state_name = ANY(ARRAY[{self._as_sql_list(states)}]);"
        )
        return BuiltQuery(sql, {"states": states}, readable,
                          "partition check (which states exist here)")

    def demographics(self, states: List[str], years: List[int],
                     attributes: List[str]) -> BuiltQuery:
        """The values themselves, for the states this agent actually holds.

        Attribute names are whitelisted by the caller before they reach here —
        they name columns, so they cannot be bound as parameters."""
        columns = ", ".join(f"sd.{attribute}" for attribute in attributes)
        sql = f"""
            SELECT s.state_name, sd.stat_year, {columns}
            FROM state_demographics sd
            JOIN states s ON s.state_id = sd.state_id
            WHERE s.state_name = ANY(:states)
              AND sd.stat_year  = ANY(:years)
            ORDER BY s.state_name, sd.stat_year
        """
        readable = (
            f"SELECT s.state_name, sd.stat_year, {columns}\n"
            "FROM state_demographics sd\n"
            "JOIN states s ON s.state_id = sd.state_id\n"
            f"WHERE s.state_name = ANY(ARRAY[{self._as_sql_list(states)}])\n"
            f"  AND sd.stat_year  = ANY(ARRAY[{self._as_sql_list(years)}])\n"
            "ORDER BY s.state_name, sd.stat_year;"
        )
        return BuiltQuery(sql, {"states": states, "years": years}, readable,
                          "main data lookup (local states only)")

    def state_geometry(self, name: str) -> BuiltQuery:
        sql = """
            SELECT ST_AsText(geo_shape), state_name, ST_SRID(geo_shape)
            FROM states WHERE state_name = :name LIMIT 1
        """
        readable = ("SELECT ST_AsText(geo_shape), state_name, ST_SRID(geo_shape) "
                    f"FROM states WHERE state_name = '{name}' LIMIT 1;")
        return BuiltQuery(sql, {"name": name}, readable, "state polygon lookup")

    def city_geometry(self, name: str) -> BuiltQuery:
        sql = """
            SELECT ST_AsText(centroid), city_name, ST_SRID(centroid)
            FROM cities WHERE city_name = :name LIMIT 1
        """
        readable = ("SELECT ST_AsText(centroid), city_name, ST_SRID(centroid) "
                    f"FROM cities WHERE city_name = '{name}' LIMIT 1;")
        return BuiltQuery(sql, {"name": name}, readable, "city point lookup")
