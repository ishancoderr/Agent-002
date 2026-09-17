"""
Generates and validates the SQL this agent runs against its own database —
the one file for that. A plain-English request ("population for these
states/years") goes in; a checked-safe (sql, params) statement, ready for
bound execution, comes out.

Three things live together here because they're one pipeline, not three
separate concerns: what schema exists (read from config/schema/entities.yaml
+ attributes.yaml) -> what the model is told it's allowed to write (the
prompt built from that same schema, config/prompts/sql_generation.yaml) ->
what's actually allowed to run (validate_sql(), a real SQL parser checking
table/column/function references against that same schema — not regex).

Adding an attribute table is a YAML edit to attributes.yaml; nothing here
needs to change for it to become queryable. partition_check()/demographics()
take an explicit `entity_type` (e.g. "state", "city") and get their wording
("federal state names", "the city attributes table", ...) from that entity's
own entities.yaml label — not hardcoded to states, so a table for a
different entity type is fully usable through this file already. (What
still only ever asks for entity_type="state" today is local_store.py's
LocalStore.lookup(), since QueryParams itself has no entity_type field yet
— a pipeline/-side change, out of scope here.)

Safety model: fails closed. Anything validate_sql() cannot positively
confirm safe is rejected, not allowed through — see its own docstring for
the five checks. This is one layer of defense-in-depth, not the only one
that should exist: running this against a database role with SELECT-only
privileges on just these tables remains the strongest guarantee, since it
holds even if a flaw is ever found here.

Values always travel as bound parameters, never interpolated — the model
writes a query template with named placeholders (:name), never a literal
value baked into the SQL text. Result columns are requested under fixed,
explicit aliases (ENTITY_ALIAS, YEAR_ALIAS, then each attribute by its own
name) regardless of what the underlying table actually calls them, and
local_store.py reads results by those same names, not by position — so
column order coming back from the model can never silently swap one
attribute's value for another's.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import openai
import sqlglot
import yaml
from sqlglot import exp

log = logging.getLogger("agent2.retrieval.sql_generator")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCHEMA_DIR = _REPO_ROOT / "config" / "schema"
_PROMPT_PATH = _REPO_ROOT / "config" / "prompts" / "sql_generation.yaml"

_ENTITIES: dict = yaml.safe_load((_SCHEMA_DIR / "entities.yaml").read_text(encoding="utf-8"))["entities"]
_ATTRIBUTE_TABLES: list = yaml.safe_load(
    (_SCHEMA_DIR / "attributes.yaml").read_text(encoding="utf-8")
)["attribute_tables"]

# The exact result-column aliases every request below demands the model use
# — the one contract local_store.py's row._mapping[...] reads depend on.
# Deliberately NOT the real column names (attributes.yaml's period column is
# "stat_year", not "year", and the key column isn't always "state_name" —
# see ENTITY_ALIAS just below) — fixed, table-agnostic output names, so
# local_store.py never needs to know what the underlying schema calls them.
ENTITY_ALIAS = "entity_name"
YEAR_ALIAS = "year"

# Same contract, for the geometry/spatial shapes below — geometry_resolver.py
# and spatial_validator.py's row._mapping[...] reads depend on these exactly
# as local_store.py's depend on ENTITY_ALIAS/YEAR_ALIAS.
WKT_ALIAS = "wkt"
SRID_ALIAS = "srid"
FOUND_ALIAS = "found"
LAT_ALIAS = "lat"
LNG_ALIAS = "lng"
MEETS_ZONE_ALIAS = "meets_zone"


# ═══════════════════════════════════════════════════════════════════════════
# SCHEMA — what tables/columns exist, for the prompt and for the validator
# ═══════════════════════════════════════════════════════════════════════════

def _entity_columns(entity: dict) -> Set[str]:
    """Every column one entities.yaml entry (e.g. "state" or "city") declares:
    its key/id/geometry columns, plus lat/lng if it's a point entity."""
    cols = {entity["key_column"], entity["id_column"], entity["geometry_column"]}
    if "lat_column" in entity:
        cols.add(entity["lat_column"])
    if "lng_column" in entity:
        cols.add(entity["lng_column"])
    return cols


def _attribute_table_columns(table: dict) -> Set[str]:
    """Every column one attributes.yaml table declares: its join column
    (the FK back to the entity it's about), its period column if it has
    one, and each of its actual data columns."""
    cols = {table["join_column"]}
    if table.get("period_column"):
        cols.add(table["period_column"])
    cols.update(table["columns"].keys())
    return cols


def allowed_tables_and_columns() -> Dict[str, Set[str]]:
    """{table_name: {every column name a query may reference in that table}}
    — the whitelist validate_sql() checks generated SQL against. A disabled
    entity (entities.yaml's `enabled: false`) is left out, same as it's
    excluded everywhere else in this codebase."""
    allowed: Dict[str, Set[str]] = {}
    for entity in _ENTITIES.values():
        if not entity.get("enabled", True):
            continue
        allowed.setdefault(entity["table"], set()).update(_entity_columns(entity))
    for table in _ATTRIBUTE_TABLES:
        allowed.setdefault(table["table"], set()).update(_attribute_table_columns(table))
    return allowed


def schema_description() -> str:
    """Human-readable table/column list for the LLM prompt — describes only
    what's actually queryable, so the model has an accurate map instead of
    guessing column names that don't exist."""
    lines = []
    for entity in _ENTITIES.values():
        if not entity.get("enabled", True):
            continue
        cols = sorted(_entity_columns(entity))
        lines.append(f"- {entity['table']} ({entity['label_plural']}): {', '.join(cols)}")
    for table in _ATTRIBUTE_TABLES:
        cols = sorted(_attribute_table_columns(table))
        lines.append(
            f"- {table['table']} ({table['entity']} attributes, "
            f"join to the {table['entity']} table on {table['join_column']}): "
            f"{', '.join(cols)}"
        )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# SAFETY — is a statement the model wrote actually safe to run?
# ═══════════════════════════════════════════════════════════════════════════

# Functions with no legitimate role in a plain data SELECT for this system,
# and a real one in an attack: reading arbitrary files, sleeping (blind
# timing exfiltration / DoS), reaching another database, running a shell
# command, or altering server state. Checked by name, case-insensitively,
# regardless of which of the tables/columns checks above would also have
# caught the surrounding query — a second, independent net.
_FORBIDDEN_FUNCTIONS = {
    "pg_sleep", "pg_sleep_for", "pg_sleep_until",
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "lo_read", "lo_write",
    "dblink", "dblink_connect", "dblink_exec",
    "copy_from_program", "copy_to_program",
    "pg_terminate_backend", "pg_cancel_backend",
    "pg_reload_conf", "set_config", "current_setting",
    "query_to_xml", "xpath",
}


class UnsafeSQLError(ValueError):
    """Raised when generated SQL fails validation. Always raised, never
    swallowed by a caller — there is no partial-trust way to run SQL that
    failed this check."""


def validate_sql(sql: str) -> None:
    """Raise UnsafeSQLError if `sql` is not a single, read-only SELECT
    referencing only allowlisted tables, columns, and functions. Returns
    normally (does nothing) if every check passes.

    Uses a real SQL parser (sqlglot) rather than regex/string matching —
    regex-based SQL filtering is well known to be bypassable via comments,
    string literals, whitespace, and encoding tricks that still parse as
    valid SQL a person didn't anticipate a pattern for."""
    # Pre-check, ahead of check 1: reject comments on the raw text itself,
    # before any parsing happens — a mismatch between how this validator's
    # parser and Postgres's own parser handle an edge case is its own
    # vulnerability class (a "parser differential"), and a comment has no
    # legitimate purpose in a generated data query anyway.
    if "--" in sql or "/*" in sql:
        raise UnsafeSQLError("Comments are not allowed in generated SQL.")

    # Check 1: the text has to parse as SQL at all.
    try:
        parsed = sqlglot.parse(sql, read="postgres")
    except Exception as exc:
        raise UnsafeSQLError(f"SQL failed to parse: {exc}") from exc

    # Check 2: exactly one statement — sqlglot.parse() splits on ';', so
    # "SELECT ...; DROP TABLE ...;" comes back as two statements here.
    statements = [s for s in parsed if s is not None]
    if len(statements) != 1:
        raise UnsafeSQLError(
            f"Expected exactly one SQL statement, got {len(statements)} — "
            f"statement stacking is rejected outright."
        )
    stmt = statements[0]

    # Check 3: SELECT only, and not SELECT ... INTO (which creates a table
    # as a side effect of what looks like a plain read).
    if not isinstance(stmt, exp.Select):
        raise UnsafeSQLError(
            f"Only SELECT statements are allowed; got {type(stmt).__name__}."
        )
    if stmt.args.get("into") is not None:
        raise UnsafeSQLError("SELECT ... INTO is rejected — it creates a table.")

    allowed = allowed_tables_and_columns()

    # A CTE's own name (`WITH geoms AS (...)`) is a local label the query
    # defines and reads back within itself, never a real database
    # identifier — so it doesn't belong in, or need to pass, the table
    # allowlist below. Excluding it by name doesn't relax anything: every
    # real table actually read to produce that local data is still walked
    # and checked exactly like any other reference, since find_all()
    # traverses the whole tree including inside a CTE's own definition —
    # and a real table sharing a CTE's name could never be *reached* by
    # that name once the CTE shadows it for the rest of the statement, so
    # excluding it opens no path to an unauthorized table.
    cte_names = {c.alias for c in stmt.find_all(exp.CTE)}

    # Check 4: every table referenced anywhere in the statement (including
    # inside a JOIN, a subquery, or a CTE — find_all() walks the whole tree)
    # must be one this system itself declares, or a CTE this same statement
    # defines.
    referenced_tables = {t.name for t in stmt.find_all(exp.Table)} - cte_names
    unknown_tables = referenced_tables - set(allowed)
    if unknown_tables:
        raise UnsafeSQLError(
            f"Query references table(s) not in the allowed schema: {sorted(unknown_tables)}."
        )

    # Check 5a: same for every column — but a CTE's or a derived table's own
    # declared output-column names (`... AS t(col1, col2)`, a CTE's own
    # `name(col1, col2)` form included) are local labels too, not real
    # database columns, so a reference to one shouldn't need to match the
    # schema allowlist either.
    #
    # That exemption has to be scoped precisely, not just "this name was
    # declared as a local alias *somewhere* in the statement" — a flat,
    # query-wide exemption would let an unrelated, genuinely-forbidden
    # column slip through anywhere it happens to share a name with some
    # CTE's own declared output column, e.g.:
    #   WITH x(internal_notes) AS (SELECT state_name FROM states)
    #   SELECT r.internal_notes FROM state_demographics r, x
    # `r.internal_notes` has nothing to do with `x` — it's a real,
    # unqualified-would-be-forbidden column read off an unrelated table —
    # and must still be rejected even though "internal_notes" also happens
    # to be x's own column name. So a column is only exempt when the
    # *specific reference* actually resolves to that local construct:
    # qualified by its alias (`g1.shape` where g1 is a usage of the "geoms"
    # CTE), or unqualified and inside the one SELECT scope where that
    # source is directly in FROM/JOIN (the CTE's own body reading its
    # UNNEST'd source unqualified, for instance) — never by bare name
    # anywhere in the statement.
    all_allowed_columns = {col for cols in allowed.values() for col in cols}

    # {CTE name: the output columns it declares} — needed to resolve a
    # second local alias on a re-reference (`FROM geoms g1 JOIN geoms g2`)
    # back to the same declared column set.
    cte_columns: Dict[str, Set[str]] = {}
    for cte in stmt.find_all(exp.CTE):
        alias_node = cte.args.get("alias")
        cols = {ident.name for ident in (alias_node.args.get("columns") or [])} if alias_node else set()
        if cols:
            cte_columns[cte.alias] = cols

    # {local alias name: columns it's allowed to expose under that alias} —
    # covers a construct's own declared alias (a CTE's `geoms(...)`, an
    # UNNEST's `t(...)`) and any further alias a CTE picks up on reuse
    # (`g1`/`g2` above), which carry no columns list of their own but
    # expose exactly the CTE's.
    alias_exposed: Dict[str, Set[str]] = {}
    for alias_node in stmt.find_all(exp.TableAlias):
        cols = {ident.name for ident in (alias_node.args.get("columns") or [])}
        if cols:
            alias_exposed[alias_node.this.name] = cols
    for t in stmt.find_all(exp.Table):
        if t.name in cte_columns:
            alias_exposed[t.alias or t.name] = cte_columns[t.name]

    def _owning_select(node: exp.Expression):
        """Nearest enclosing SELECT — the only scope an unqualified column
        can legally resolve a same-scope FROM source's column against."""
        parent = node.parent
        while parent is not None and not isinstance(parent, exp.Select):
            parent = parent.parent
        return parent

    def _local_names_in_scope(select_node: exp.Select) -> Set[str]:
        """Column names visible unqualified inside `select_node`, from a
        local (CTE or derived-table) source directly in its own FROM/JOIN —
        not inherited from any outer or unrelated SELECT."""
        names: Set[str] = set()
        from_clause = select_node.args.get("from_")
        sources = [from_clause.this] if from_clause else []
        sources += [j.this for j in select_node.args.get("joins") or []]
        for src in sources:
            if isinstance(src, exp.Table) and src.name in cte_columns:
                names |= cte_columns[src.name]
                continue
            src_alias = src.args.get("alias") if hasattr(src, "args") else None
            if src_alias is not None:
                names |= {ident.name for ident in (src_alias.args.get("columns") or [])}
        return names

    def _is_locally_exempt(column: exp.Column) -> bool:
        if column.table:
            return column.name in alias_exposed.get(column.table, set())
        enclosing = _owning_select(column)
        return enclosing is not None and column.name in _local_names_in_scope(enclosing)

    unknown_columns = {
        c.name
        for c in stmt.find_all(exp.Column)
        if c.name not in all_allowed_columns and not _is_locally_exempt(c)
    }
    if unknown_columns:
        raise UnsafeSQLError(
            f"Query references column(s) not in the allowed schema: {sorted(unknown_columns)}."
        )

    # Check 5b: every function call, against the denylist above — a second,
    # independent net even for a function that happens to pass checks 4/5a
    # (e.g. dblink() takes no table/column argument at all).
    called_functions = {
        f.name.lower()
        for f in stmt.find_all((exp.Func, exp.Anonymous))
        if getattr(f, "name", None)
    }
    forbidden_used = called_functions & _FORBIDDEN_FUNCTIONS
    if forbidden_used:
        raise UnsafeSQLError(f"Query calls forbidden function(s): {sorted(forbidden_used)}.")


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT — one system prompt per query category, assembled from the RULES
# shared by every category (config/prompts/sql_generation.yaml) plus the
# EXAMPLES specific to that one category (that category's own extraction
# prompt file, under its own `sql_generation.examples` key) — see
# config/prompts/sql_generation.yaml's own top comment for why the split is
# there and not, say, one shared examples list or one file per category
# duplicating the rules.
# ═══════════════════════════════════════════════════════════════════════════

SQL_MODEL = os.getenv("SQL_MODEL", "gpt-4o-mini")

# A literal marker substituted with schema_description()'s live output at
# call time (not baked in at import time — see SqlGenerator._generate()),
# replaced with plain string .replace(), not .format(): the assembled
# examples below already contain real '{'/'}' characters (JSON), which
# .format() would misparse as its own placeholders.
_SCHEMA_MARKER = "{schema}"

_SHARED_PROMPT_DOC: dict = yaml.safe_load(_PROMPT_PATH.read_text(encoding="utf-8"))
_SHARED_RULES = "\n".join(f"  - {rule}" for rule in _SHARED_PROMPT_DOC["rules"])


def _load_system_prompt_template(category: str) -> str:
    """Assemble the system prompt template for one query category —
    `category` names a file in config/prompts/ (e.g. "direct_lookup"),
    whose own `sql_generation.examples` supplies the worked examples; the
    intro/schema framing/rules come from the shared doc above. Still
    contains _SCHEMA_MARKER for SqlGenerator._generate() to fill in."""
    category_path = _REPO_ROOT / "config" / "prompts" / f"{category}.yaml"
    doc = yaml.safe_load(category_path.read_text(encoding="utf-8"))
    try:
        examples = doc["sql_generation"]["examples"]
    except KeyError as exc:
        raise KeyError(
            f"{category_path} has no sql_generation.examples — every category "
            f"SqlGenerator is asked to write SQL for needs its own worked "
            f"examples there."
        ) from exc

    example_lines = ["EXAMPLES:"]
    for example in examples:
        example_lines.append("")
        example_lines.append(f"Request: {example['request']}")
        example_lines.append(
            "→ " + json.dumps(example["output"], ensure_ascii=False, separators=(",", ":"))
        )

    parts = [
        _SHARED_PROMPT_DOC["intro"],
        _SHARED_PROMPT_DOC["schema_intro"] + "\n\n" + _SCHEMA_MARKER,
        _SHARED_PROMPT_DOC["rules_intro"] + "\n" + _SHARED_RULES,
        "\n".join(example_lines),
    ]
    return "\n\n".join(parts) + "\n"


# Built lazily, one entry per category on first use, not all five at import
# time — a category this process never actually asks SqlGenerator about
# (e.g. a deployment that never sees a SPATIAL_OPERATION query) never pays
# to have its prompt assembled.
_SYSTEM_PROMPT_TEMPLATES: Dict[str, str] = {}


def _system_prompt_template(category: str) -> str:
    if category not in _SYSTEM_PROMPT_TEMPLATES:
        _SYSTEM_PROMPT_TEMPLATES[category] = _load_system_prompt_template(category)
    return _SYSTEM_PROMPT_TEMPLATES[category]


# ═══════════════════════════════════════════════════════════════════════════
# GENERATION — ask the model for one statement, validate it, hand it back
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BuiltQuery:
    """A statement ready to run, plus the readable form for the log."""
    sql: str                    # bound form, :named parameters
    params: Dict[str, Any]      # values for those parameters
    readable: str               # same statement with values written in, for logging
    label: str = ""             # what this statement is for


class SqlGenerator:
    """Turns a plain-English description of needed data into one validated,
    ready-to-run SELECT statement — not one hand-written method per table."""

    def __init__(self, client: openai.OpenAI | None = None, model: str = SQL_MODEL):
        """Reuse a passed-in OpenAI client, or make one from the env key.
        `model` defaults to SQL_MODEL but can be overridden per instance."""
        self._client = client or openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        # Fixed-shape statements (every method below partition_check()/
        # demographics()) never vary in SQL text by the runtime values a
        # call supplies — only by which entity_type/operation the shape is
        # for, which `shape_key` encodes — so the LLM only has to write
        # each one once per process, ever; see _cached_query(). Keyed by
        # instance, not module-level, so a test using its own SqlGenerator
        # never sees another instance's (possibly different-model) cache.
        self._template_cache: Dict[str, str] = {}

    def _generate(self, request: str, category: str = "direct_lookup") -> Tuple[str, Dict[str, Any]]:
        """`request` is a plain-English description of the data needed, e.g.
        "population and marriages for states ['Bayern'] in years [2021]".
        `category` picks which config/prompts/*.yaml's worked examples the
        model sees — see _system_prompt_template().

        Returns (sql, params) — sql has already passed validate_sql(), so
        it is safe to execute with params bound. Raises UnsafeSQLError if
        the model's statement fails validation, or ValueError if its
        response wasn't usable JSON at all — either way, nothing is ever
        returned that hasn't been validated; there is no partial-trust path."""
        system_prompt = _system_prompt_template(category).replace(_SCHEMA_MARKER, schema_description())
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ],
        )
        raw = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        log.info("       | LLM SQL (%s): %s  (tokens=%d)", self.model, raw, tokens)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM SQL response was not valid JSON: {raw!r}") from exc

        sql = data.get("sql", "")
        params = data.get("params", {})
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError(f"LLM SQL response had no usable 'sql' field: {data!r}")
        if not isinstance(params, dict):
            raise ValueError(f"LLM SQL response's 'params' was not an object: {data!r}")

        validate_sql(sql)  # raises UnsafeSQLError if unsafe — never swallowed here
        return sql, params

    @staticmethod
    def _as_readable(sql: str, params: Dict[str, Any]) -> str:
        """Same statement with bound values written in, for the log only —
        never executed, so a value with a quote in it cannot change what
        actually runs (that's `sql` + bound `params`, always)."""
        readable = sql
        for key, value in params.items():
            if isinstance(value, list):
                literal = "ARRAY[" + ", ".join(
                    f"'{v}'" if isinstance(v, str) else str(v) for v in value
                ) + "]"
            elif isinstance(value, str):
                literal = f"'{value}'"
            else:
                literal = str(value)
            readable = readable.replace(f":{key}", literal)
        return readable + ";"

    def _build(self, request: str, label: str, category: str = "direct_lookup") -> BuiltQuery:
        """Generate + validate `request`, then wrap the result as a
        BuiltQuery — adding the human-readable log form and the `label`
        callers use to say what this statement was for. Always calls the
        model: used where the SQL text itself depends on the runtime
        values (partition_check()/demographics() — a different attribute
        list is a different SELECT list, not just different bound values)."""
        sql, params = self._generate(request, category)
        return BuiltQuery(sql=sql, params=params, readable=self._as_readable(sql, params), label=label)

    def _cached_query(self, shape_key: str, category: str, request: str,
                       label: str, params: Dict[str, Any]) -> BuiltQuery:
        """Like _build(), for a statement whose SQL text is fully determined
        by `shape_key` alone (e.g. "entity_exact_geometry:city") — every
        call for the same shape_key runs the identical statement, only the
        bound values differ, so the model is asked to write it once per
        process and the validated text is then reused.

        `request` must prescribe the exact placeholder name for every value
        `params` supplies (sql_generation.yaml's own rule for this) — the
        model is never asked again after the first call, so there is no
        second chance for it to tell the caller what name it chose; `params`
        has to already match. `request` should describe the shape in
        general terms (a representative example value, not this call's
        real one) since it is only ever sent to the model once, before any
        specific caller's real values are known."""
        sql = self._template_cache.get(shape_key)
        if sql is None:
            sql, _unused_example_params = self._generate(request, category)
            self._template_cache[shape_key] = sql
            log.info("       | SQL template cached for %r", shape_key)
        return BuiltQuery(sql=sql, params=params, readable=self._as_readable(sql, params), label=label)

    def partition_check(self, keys: List[str], entity_type: str) -> BuiltQuery:
        """Which of the requested `keys` this agent holds any row for at all,
        for entities.yaml's `entity_type` (e.g. "state", "city") — the
        wording below ("federal state names", "the city attributes table")
        comes from that entity's own label/label_plural, not hardcoded.

        Asked first because a key with no rows is a spatial gap, which is a
        different thing from a key whose rows exist but are empty."""
        entity = _ENTITIES[entity_type]
        label, label_plural = entity["label"], entity["label_plural"]
        request = (
            f"Which of these {label} names does this agent hold at least one row "
            f"for in the {label} attributes table, regardless of year or which "
            f"attribute values are actually filled in: {keys}. Return the "
            f"distinct {label} names found, in a single column aliased exactly "
            f"as {ENTITY_ALIAS}."
        )
        return self._build(request, f"partition check (which {label_plural} exist here)")

    def demographics(self, keys: List[str], years: List[int],
                     attributes: List[str], entity_type: str) -> BuiltQuery:
        """The values themselves, for the `keys` this agent actually holds,
        for entities.yaml's `entity_type`.

        Attribute names are whitelisted by the caller before they reach here —
        they name real columns, checked again by validate_sql() against the
        same config/schema/attributes.yaml whitelist independently of that
        caller."""
        entity = _ENTITIES[entity_type]
        label, label_plural = entity["label"], entity["label_plural"]
        attrs_csv = ", ".join(attributes)
        request = (
            f"For {label_plural} {keys} and years {years}, return {attrs_csv} "
            f"from the {label} attributes table, one row per {label} per year. "
            f"Alias the {label} name column exactly as {ENTITY_ALIAS}, the year "
            f"column exactly as {YEAR_ALIAS}, and each attribute column exactly "
            f"by its own name ({attrs_csv}) — no other aliases — in that exact "
            f"column order: {ENTITY_ALIAS}, {YEAR_ALIAS}, then each attribute in "
            f"the order listed. Order the rows by {label} name then year."
        )
        return self._build(request, f"main data lookup (local {label_plural} only)")

    # ═══════════════════════════════════════════════════════════════════
    # GEOMETRY_LOOKUP shapes — resolving one named entity to its own
    # geometry. Used by agent2/retrieval/geometry_resolver.py, whichever
    # higher-level function (a GEOMETRY_LOOKUP request, or a SPATIAL_
    # OPERATION's own entity resolution) needs a name resolved.
    # ═══════════════════════════════════════════════════════════════════

    def entity_exact_geometry(self, name: str, entity_type: str) -> BuiltQuery:
        """WKT + matched name for `name`, exact match, only if the row's
        geometry isn't NULL. None back (no row/fetchone()) covers both "no
        such row" and "row exists, geometry NULL" — callers distinguish
        those with entity_exists() when that distinction matters."""
        label = _ENTITIES[entity_type]["label"]
        request = (
            f"The WKT text and matched name of the {label} whose name "
            f"exactly equals a given value, only if it has a non-null "
            f"geometry. Use exactly :name for the name to match. Alias the "
            f"WKT column exactly as {WKT_ALIAS} and the matched name column "
            f"exactly as {ENTITY_ALIAS}."
        )
        return self._cached_query(
            shape_key=f"entity_exact_geometry:{entity_type}",
            category="geometry_lookup",
            request=request,
            label=f"exact {label} geometry lookup",
            params={"name": name},
        )

    def entity_exists(self, name: str, entity_type: str) -> BuiltQuery:
        """Whether any row at all exists for `name`, geometry NULL or not —
        the "row exists but its shape is missing" case entity_exact_geometry()
        alone can't distinguish from "not held here at all"."""
        label = _ENTITIES[entity_type]["label"]
        request = (
            f"Whether this agent holds any row at all for the {label} whose "
            f"name exactly equals a given value, regardless of whether its "
            f"geometry is filled in. Use exactly :name for the name to "
            f"match. Return one row aliased exactly as {FOUND_ALIAS} if a "
            f"row exists, no rows otherwise."
        )
        return self._cached_query(
            shape_key=f"entity_exists:{entity_type}",
            category="geometry_lookup",
            request=request,
            label=f"{label} existence check",
            params={"name": name},
        )

    def entity_ilike_geometry(self, name: str, entity_type: str) -> BuiltQuery:
        """Case-insensitive fallback for entity_exact_geometry() — tried
        only once the exact form and the existence check have both missed."""
        label = _ENTITIES[entity_type]["label"]
        request = (
            f"The WKT text and matched name of the {label} whose name "
            f"case-insensitively matches a given value, only if it has a "
            f"non-null geometry. Use exactly :name for the value to match "
            f"against. Alias the WKT column exactly as {WKT_ALIAS} and the "
            f"matched name column exactly as {ENTITY_ALIAS}."
        )
        return self._cached_query(
            shape_key=f"entity_ilike_geometry:{entity_type}",
            category="geometry_lookup",
            request=request,
            label=f"case-insensitive {label} geometry lookup",
            params={"name": name},
        )

    # ═══════════════════════════════════════════════════════════════════
    # SPATIAL_RELATIONSHIP_BUFFER shapes — Scenario 21's open-ended
    # "which cities lie within X km of Y" test. Buffer/within are always
    # about cities (a fact about what this category means, not something
    # to parameterize — see geometry_resolver.py's own note on this).
    # ═══════════════════════════════════════════════════════════════════

    def city_buffer(self, name: str, distance_km: float) -> BuiltQuery:
        """A metres-radius buffer around a locally-held city's own centroid."""
        request = (
            "The matched name, the WKT text of a buffer polygon (in "
            "metres) around the centroid of the city whose name exactly "
            "equals a given value, and that geometry's SRID — only if the "
            "city has a non-null centroid. Use exactly :name for the name "
            f"to match and exactly :dist for the buffer distance in "
            f"metres. Alias the matched name column exactly as "
            f"{ENTITY_ALIAS}, the buffer WKT column exactly as {WKT_ALIAS}, "
            f"and the SRID column exactly as {SRID_ALIAS}."
        )
        return self._cached_query(
            shape_key="city_buffer",
            category="spatial_relationship_buffer",
            request=request,
            label="buffer around a named city",
            params={"name": name, "dist": distance_km * 1000},
        )

    def point_buffer(self, wkt_point: str, srid: int, distance_km: float) -> BuiltQuery:
        """The same buffer, around a point already resolved some other way
        (e.g. from the peer) — no local row required."""
        request = (
            "The WKT text of a buffer polygon (in metres) around an "
            "already-known point geometry, given as WKT text and its SRID. "
            "Use exactly :wkt for the point's WKT text, exactly :srid for "
            "its SRID, and exactly :dist for the buffer distance in "
            f"metres. Alias the result column exactly as {WKT_ALIAS}."
        )
        return self._cached_query(
            shape_key="point_buffer",
            category="spatial_relationship_buffer",
            request=request,
            label="buffer around a known point",
            params={"wkt": wkt_point, "srid": srid, "dist": distance_km * 1000},
        )

    def within_buffer(self, wkt: str, srid: int, exclude: List[str]) -> BuiltQuery:
        """Every local city whose centroid falls inside an already-built
        buffer polygon — the local-catalogue side of Scenario 21's
        achieve/spatial-query pattern."""
        request = (
            "The name, WKT text and SRID of every city in the local "
            "cities catalogue whose centroid lies within an already-known "
            "buffer polygon (given as WKT text and its SRID), excluding a "
            "given list of names. Use exactly :wkt for the polygon's WKT "
            "text, exactly :srid for its SRID, and exactly :exclude for "
            f"the list of names to exclude. Alias the name column exactly "
            f"as {ENTITY_ALIAS}, the WKT column exactly as {WKT_ALIAS}, and "
            f"the SRID column exactly as {SRID_ALIAS}."
        )
        return self._cached_query(
            shape_key="within_buffer:city",
            category="spatial_relationship_buffer",
            request=request,
            label="cities within a buffer polygon",
            params={"wkt": wkt, "srid": srid, "exclude": exclude},
        )

    # ═══════════════════════════════════════════════════════════════════
    # SPATIAL_OPERATION shapes — combining or testing already-resolved
    # geometries (Section 3: Scenarios 13-16, 20). An operation is never
    # itself missing (both agents run the same code); only an input shape
    # can be, and that's resolved separately, through the geometry-lookup
    # shapes above, before either of these ever runs.
    # ═══════════════════════════════════════════════════════════════════

    _OP_PHRASING = {
        "Union": "the Union of two already-known geometries",
        "Intersection": "the Intersection of two already-known geometries",
        "Difference": "the Difference of two already-known geometries (the first minus the second)",
        "SymDifference": "the SymDifference of two already-known geometries",
    }

    def binary_operation(self, operation: str, wkt_a: str, srid_a: int,
                         wkt_b: str, srid_b: int) -> BuiltQuery:
        """Union/Intersection/Difference/SymDifference on two already-resolved
        geometries. `operation` must be one of _OP_PHRASING's keys — the
        same four names classify.yaml's SPATIAL_OPERATION extraction
        already constrains "operation" to."""
        phrasing = self._OP_PHRASING[operation]
        request = (
            f"The WKT text of {phrasing}, each given as WKT text and its "
            f"SRID. Use exactly :wkt_a and :srid_a for the first, exactly "
            f":wkt_b and :srid_b for the second. Alias the result column "
            f"exactly as {WKT_ALIAS}."
        )
        return self._cached_query(
            shape_key=f"binary_operation:{operation}",
            category="spatial_operation",
            request=request,
            label=f"{operation} of two geometries",
            params={"wkt_a": wkt_a, "srid_a": srid_a, "wkt_b": wkt_b, "srid_b": srid_b},
        )

    def intersects_buffer(self, wkt_target: str, srid_target: int, wkt_ref: str,
                          srid_ref: int, distance_km: float) -> BuiltQuery:
        """Named-target buffer test (Scenario 20's BufferWithin): does an
        already-resolved target geometry meet a buffer built around an
        already-resolved reference geometry. Purely local scratch — every
        input here was already named and fetched the ordinary way, unlike
        within_buffer()'s open-ended candidate scan."""
        request = (
            "Whether an already-known target geometry intersects a buffer "
            "polygon (in metres) built around an already-known reference "
            "geometry, both given as WKT text and their own SRID. Use "
            "exactly :wkt_t and :srid_t for the target, exactly :wkt_ref "
            "and :srid_ref for the reference, and exactly :dist for the "
            f"buffer distance in metres. Alias the result column exactly "
            f"as {MEETS_ZONE_ALIAS}."
        )
        return self._cached_query(
            shape_key="intersects_buffer",
            category="spatial_operation",
            request=request,
            label="target-within-buffer test",
            params={"wkt_t": wkt_target, "srid_t": srid_target,
                    "wkt_ref": wkt_ref, "srid_ref": srid_ref, "dist": distance_km * 1000},
        )

    # ═══════════════════════════════════════════════════════════════════
    # SPATIAL_ADJACENCY / SPATIAL_DIRECTION / SPATIAL_DISTANCE shapes —
    # agent2/pipeline/spatial_validator.py's verdict-style relationship
    # queries. Every candidate state's geometry has to be tested, not just
    # the ones already known to have a local shape (see spatial_validator's
    # own note on why a shapeless state must not just vanish from a
    # result) — so `names`/`wkts` here are the FULL set of (name, WKT)
    # pairs already resolved in Python for every state, turned into rows
    # with unnest() inside a CTE rather than one placeholder pair per
    # state. That keeps the SQL text the same size no matter how many
    # states are actually being tested, which is exactly what makes these
    # shapes cacheable at all.
    # ═══════════════════════════════════════════════════════════════════

    def all_geometries(self, entity_type: str) -> BuiltQuery:
        """Every entity's own (name, WKT) pair, straight from its table —
        the starting set _adjacency()/_direction()/_distance() build their
        `names`/`wkts` arrays from before any peer fetch fills gaps."""
        entity = _ENTITIES[entity_type]
        label, label_plural = entity["label"], entity["label_plural"]
        request = (
            f"The name and WKT text of every {label}'s own geometry, "
            f"straight from the {label} table, no filtering. Alias the "
            f"name column exactly as {ENTITY_ALIAS} and the WKT column "
            f"exactly as {WKT_ALIAS}."
        )
        return self._cached_query(
            shape_key=f"all_geometries:{entity_type}",
            category="spatial_relationship",
            request=request,
            label=f"every {label_plural}'s geometry",
            params={},
        )

    def adjacency(self, names: List[str], wkts: List[str], ref: str) -> BuiltQuery:
        """Every entity_name in `names` that touches `ref` (ST_Intersects
        minus the touching-itself case)."""
        request = (
            "Given the full set of federal states' own (name, WKT) pairs "
            "as two parallel array parameters, which of them touch (share "
            "a border with, but are not identical to) a given reference "
            "state name. Use exactly :names for the array of names, "
            "exactly :wkts for the array of matching WKT text, and exactly "
            f":ref for the reference state's name. Build a CTE named "
            f"exactly geoms with columns {ENTITY_ALIAS} and shape from "
            f"unnest(:names, :wkts), self-join it to compare every pair, "
            f"and require ST_Intersects true while ST_Equals is false so a "
            f"state is never reported as touching itself. Alias the result "
            f"column exactly as {ENTITY_ALIAS}."
        )
        return self._cached_query(
            shape_key="adjacency",
            category="spatial_relationship",
            request=request,
            label="adjacency test",
            params={"names": names, "wkts": wkts, "ref": ref},
        )

    _DIRECTION_PHRASING = {
        "north_of": "lie north of a given reference state's centroid (compass bearing at most 45 degrees or at least 315 degrees from the reference's centroid)",
        "south_of": "lie south of a given reference state's centroid (compass bearing between 135 and 225 degrees from the reference's centroid)",
        "east_of":  "lie east of a given reference state's centroid (compass bearing between 45 and 135 degrees from the reference's centroid)",
        "west_of":  "lie west of a given reference state's centroid (compass bearing between 225 and 315 degrees from the reference's centroid)",
    }

    def direction(self, direction_key: str, names: List[str], wkts: List[str], ref: str) -> BuiltQuery:
        """Every entity_name in `names` that lies in `direction_key`'s
        compass sector from `ref`'s centroid, nearest bearing first.
        `direction_key` must be one of _DIRECTION_PHRASING's keys."""
        phrasing = self._DIRECTION_PHRASING[direction_key]
        request = (
            f"Given the full set of federal states' own (name, WKT) pairs "
            f"as two parallel array parameters, which of them {phrasing}, "
            f"nearest bearing first. Use exactly :names for the array of "
            f"names, exactly :wkts for the array of matching WKT text, and "
            f"exactly :ref for the reference state's name. Build a CTE "
            f"named exactly geoms with columns {ENTITY_ALIAS} and shape "
            f"from unnest(:names, :wkts), then a second CTE named exactly "
            f"azimuths with its own explicit columns {ENTITY_ALIAS} and az "
            f"computing each other state's compass bearing in degrees "
            f"from the reference's centroid to its own centroid, then "
            f"filter and order by that bearing. Alias the result column "
            f"exactly as {ENTITY_ALIAS}."
        )
        return self._cached_query(
            shape_key=f"direction:{direction_key}",
            category="spatial_relationship",
            request=request,
            label=f"{direction_key} test",
            params={"names": names, "wkts": wkts, "ref": ref},
        )

    def distance(self, names: List[str], wkts: List[str], lat: float,
                lng: float, dist_m: float) -> BuiltQuery:
        """Every entity_name in `names` within `dist_m` metres of
        (lat, lng), nearest first."""
        request = (
            "Given the full set of federal states' own (name, WKT) pairs "
            "as two parallel array parameters, which of them lie within a "
            "given distance (in metres) of a given reference point "
            "(longitude/latitude), nearest first. Use exactly :names for "
            "the array of names, exactly :wkts for the array of matching "
            "WKT text, exactly :lng and :lat for the reference point, and "
            "exactly :dist for the distance in metres. Build a CTE named "
            f"exactly geoms with columns {ENTITY_ALIAS} and shape from "
            f"unnest(:names, :wkts), filter with ST_DWithin against the "
            f"point (cast both sides to geography) and order by "
            f"ST_Distance the same way. Alias the result column exactly as "
            f"{ENTITY_ALIAS}."
        )
        return self._cached_query(
            shape_key="distance",
            category="spatial_relationship",
            request=request,
            label="distance test",
            params={"names": names, "wkts": wkts, "lat": lat, "lng": lng, "dist": dist_m},
        )

    def city_coords_exact(self, name: str) -> BuiltQuery:
        """(lat, lng) for a city, exact name match."""
        request = (
            "The (latitude, longitude) of the city whose name exactly "
            "equals a given value. Use exactly :name for the name to "
            f"match. Alias the latitude column exactly as {LAT_ALIAS} and "
            f"the longitude column exactly as {LNG_ALIAS}."
        )
        return self._cached_query(
            shape_key="city_coords_exact",
            category="spatial_relationship",
            request=request,
            label="exact city coordinate lookup",
            params={"name": name},
        )

    def city_coords_ilike(self, name: str) -> BuiltQuery:
        """(lat, lng) for a city, case-insensitive full-name match — tried
        once the exact match has missed."""
        request = (
            "The matched name, latitude and longitude of the city whose "
            "name case-insensitively equals a given value. Use exactly "
            ":name for the value to match against. Alias the matched name "
            f"column exactly as {ENTITY_ALIAS}, the latitude column "
            f"exactly as {LAT_ALIAS}, and the longitude column exactly as "
            f"{LNG_ALIAS}."
        )
        return self._cached_query(
            shape_key="city_coords_ilike",
            category="spatial_relationship",
            request=request,
            label="case-insensitive city coordinate lookup",
            params={"name": name},
        )

    def city_coords_partial(self, name_pattern: str) -> BuiltQuery:
        """(lat, lng) for a city whose name contains `name_pattern`
        (already %-wrapped by the caller), shortest match first — the last
        resort once both the exact and full ILIKE forms have missed."""
        request = (
            "The matched name, latitude and longitude of the city whose "
            "name contains a given value (partial, case-insensitive "
            "match), shortest matching name first. Use exactly :name for "
            "the value to search for (already wrapped in %...% wildcards). "
            f"Alias the matched name column exactly as {ENTITY_ALIAS}, the "
            f"latitude column exactly as {LAT_ALIAS}, and the longitude "
            f"column exactly as {LNG_ALIAS}."
        )
        return self._cached_query(
            shape_key="city_coords_partial",
            category="spatial_relationship",
            request=request,
            label="partial city coordinate lookup",
            params={"name": name_pattern},
        )
