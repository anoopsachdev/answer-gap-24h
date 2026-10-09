"""Build the question-level analysis table.

The StackSample CSVs contain multiline HTML and mixed timestamp formats.
DuckDB's parallel CSV reader dropped most answers on this file, so the
extract step uses Python's csv module and DuckDB only joins the slim tables.

The label is "no answer from someone else within 24 hours." Score, answer
counts, and closed dates are not features. The dump stores the latest
revision of each post, which the report treats as a leakage risk.
"""

from __future__ import annotations

import csv
import re

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    ANSWERS,
    EDIT_PATTERN,
    OUTPUTS,
    PARQUET,
    PROCESSED,
    QUESTIONS,
    TAGS,
    TMP,
)
from src.io_utils import write_json

EDIT_RE = re.compile(EDIT_PATTERN, re.IGNORECASE)
CHUNK = 250_000


def _blank(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if text == "" or text.upper() == "NA":
        return None
    return text


def _write_rows(path, schema: pa.Schema, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    if not hasattr(_write_rows, "writers"):
        _write_rows.writers = {}
    writer = _write_rows.writers.get(path)
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(path, schema, compression="zstd")
        _write_rows.writers[path] = writer
    writer.write_table(table)


def _close_writers() -> None:
    for writer in getattr(_write_rows, "writers", {}).values():
        writer.close()
    _write_rows.writers = {}


def _extract() -> dict:
    """Stream the three CSVs into slim parquet files. One row in, one row out."""
    answers_path = PROCESSED / "answers_slim.parquet"
    questions_path = PROCESSED / "questions_slim.parquet"
    tags_path = PROCESSED / "tags_slim.parquet"
    for path in (answers_path, questions_path, tags_path):
        if path.exists():
            path.unlink()

    answer_schema = pa.schema(
        [
            ("question_id", pa.int64()),
            ("owner_user_id", pa.int64()),
            ("created", pa.string()),
        ]
    )
    question_schema = pa.schema(
        [
            ("question_id", pa.int64()),
            ("owner_user_id", pa.int64()),
            ("created", pa.string()),
            ("closed", pa.string()),
            ("title", pa.string()),
            ("title_len", pa.int32()),
            ("body_len", pa.int32()),
            ("code_block_count", pa.int32()),
            ("link_count", pa.int32()),
            ("has_edit_marker", pa.bool_()),
        ]
    )
    tag_schema = pa.schema(
        [
            ("question_id", pa.int64()),
            ("tag", pa.string()),
        ]
    )

    print("Extracting answers...")
    n_answers = 0
    batch: list[dict] = []
    with ANSWERS.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parent = _blank(row["ParentId"])
            created = _blank(row["CreationDate"])
            if parent is None or created is None:
                continue
            owner = _blank(row["OwnerUserId"])
            batch.append(
                {
                    "question_id": int(parent),
                    "owner_user_id": int(owner) if owner is not None else None,
                    "created": created,
                }
            )
            n_answers += 1
            if len(batch) >= CHUNK:
                _write_rows(answers_path, answer_schema, batch)
                batch = []
                print(f"  answers {n_answers:,}")
    if batch:
        _write_rows(answers_path, answer_schema, batch)

    print("Extracting questions...")
    n_questions = 0
    n_bad_questions = 0
    batch = []
    with QUESTIONS.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qid = _blank(row["Id"])
            created = _blank(row["CreationDate"])
            if qid is None or created is None:
                n_bad_questions += 1
                continue
            body = row.get("Body") or ""
            title = row.get("Title") or ""
            owner = _blank(row["OwnerUserId"])
            batch.append(
                {
                    "question_id": int(qid),
                    "owner_user_id": int(owner) if owner is not None else None,
                    "created": created,
                    "closed": _blank(row.get("ClosedDate")),
                    "title": title,
                    "title_len": len(title),
                    "body_len": len(body),
                    "code_block_count": body.lower().count("<pre"),
                    "link_count": body.lower().count("href="),
                    "has_edit_marker": bool(EDIT_RE.search(body)),
                }
            )
            n_questions += 1
            if len(batch) >= CHUNK:
                _write_rows(questions_path, question_schema, batch)
                batch = []
                print(f"  questions {n_questions:,}")
    if batch:
        _write_rows(questions_path, question_schema, batch)

    print("Extracting tags...")
    n_tags = 0
    batch = []
    with TAGS.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            qid = _blank(row["Id"])
            tag = _blank(row["Tag"])
            if qid is None or tag is None:
                continue
            batch.append({"question_id": int(qid), "tag": tag})
            n_tags += 1
            if len(batch) >= CHUNK:
                _write_rows(tags_path, tag_schema, batch)
                batch = []
    if batch:
        _write_rows(tags_path, tag_schema, batch)
    _close_writers()
    print(f"Extracted questions={n_questions:,} answers={n_answers:,} tags={n_tags:,}")
    return {
        "n_questions": n_questions,
        "n_answers": n_answers,
        "n_tags": n_tags,
        "n_bad_questions": n_bad_questions,
        "answers_path": answers_path,
        "questions_path": questions_path,
        "tags_path": tags_path,
    }


def _ts(column: str) -> str:
    """Parse the two timestamp spellings that show up in the dump, both as UTC."""
    return (
        "coalesce("
        f"try_strptime({column}, '%Y-%m-%dT%H:%M:%SZ'), "
        f"try_strptime({column}, '%Y-%m-%dT%H:%M:%S.%fZ'), "
        f"try_strptime({column}, '%Y-%m-%d %H:%M:%S')"
        ")"
    )


def build_dataset(force: bool = False) -> dict:
    if PARQUET.exists() and not force:
        print(f"Using existing {PARQUET}")
        summary_path = OUTPUTS / "build_summary.json"
        if summary_path.exists():
            from src.io_utils import read_json

            return read_json(summary_path)

    for path in (QUESTIONS, ANSWERS, TAGS):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python -m src.download` or copy the StackSample CSVs into data/raw/."
            )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    extracted = _extract()
    if extracted["n_answers"] < 1_900_000 or extracted["n_questions"] < 1_200_000:
        raise RuntimeError(
            f"CSV extract looks truncated: {extracted['n_questions']} questions, {extracted['n_answers']} answers."
        )

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{TMP.as_posix()}'")
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=4")

    print("Parsing timestamps and loading slim tables...")
    con.execute(
        f"""
        CREATE TABLE questions AS
        SELECT
            question_id,
            owner_user_id,
            {_ts('created')} AS created,
            {_ts('closed')} AS closed,
            title,
            title_len,
            body_len,
            code_block_count,
            link_count,
            has_edit_marker
        FROM read_parquet('{extracted['questions_path'].as_posix()}')
        WHERE {_ts('created')} IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE TABLE answers AS
        SELECT
            question_id,
            owner_user_id,
            {_ts('created')} AS created
        FROM read_parquet('{extracted['answers_path'].as_posix()}')
        WHERE {_ts('created')} IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE TABLE tags AS
        SELECT question_id, tag
        FROM read_parquet('{extracted['tags_path'].as_posix()}')
        """
    )

    summary = con.execute(
        """
        SELECT
            (SELECT count(*) FROM questions) AS n_questions,
            (SELECT count(*) FROM answers) AS n_answers,
            (SELECT count(*) FROM tags) AS n_tag_rows,
            (SELECT count(*) FROM questions WHERE question_id % 10 = 0) AS n_id_multiple_of_10,
            (SELECT count(*) FROM questions WHERE owner_user_id IS NULL) AS n_null_owner,
            (SELECT min(created) FROM questions) AS created_min,
            (SELECT max(created) FROM questions) AS created_max
        """
    ).fetchdf()
    print(summary.to_string(index=False))

    print("Attaching the first non-self answer...")
    con.execute(
        """
        CREATE TABLE first_other AS
        SELECT
            q.question_id,
            min(a.created) FILTER (
                WHERE a.created >= q.created - INTERVAL 5 MINUTE
                  AND NOT (
                      a.owner_user_id IS NOT NULL
                      AND q.owner_user_id IS NOT NULL
                      AND a.owner_user_id = q.owner_user_id
                  )
            ) AS first_other_answer,
            count(a.question_id) FILTER (
                WHERE a.owner_user_id IS NOT NULL
                  AND q.owner_user_id IS NOT NULL
                  AND a.owner_user_id = q.owner_user_id
            ) AS n_self_answers
        FROM questions q
        LEFT JOIN answers a ON a.question_id = q.question_id
        GROUP BY q.question_id
        """
    )

    print("Choosing each question's primary tag...")
    con.execute(
        """
        CREATE TABLE tag_freq AS
        SELECT tag, count(*) AS n
        FROM tags
        GROUP BY tag
        """
    )
    con.execute(
        """
        CREATE TABLE question_tags AS
        SELECT
            q.question_id,
            count(t.tag) AS tag_count,
            arg_max(t.tag, f.n) AS primary_tag
        FROM questions q
        LEFT JOIN tags t ON t.question_id = q.question_id
        LEFT JOIN tag_freq f ON f.tag = t.tag
        GROUP BY q.question_id
        """
    )

    print("Writing the analysis table...")
    con.execute(
        f"""
        COPY (
            WITH bounds AS (
                SELECT max(created) AS dump_end FROM questions
            ),
            labeled AS (
                SELECT
                    q.question_id,
                    q.owner_user_id,
                    q.created,
                    q.title,
                    q.title_len,
                    q.body_len,
                    q.code_block_count,
                    q.link_count,
                    coalesce(t.tag_count, 0) AS tag_count,
                    coalesce(t.primary_tag, 'untagged') AS primary_tag,
                    q.has_edit_marker,
                    hour(q.created) AS hour_utc,
                    isodow(q.created) IN (6, 7) AS weekend,
                    year(q.created) AS year,
                    q.closed,
                    f.first_other_answer,
                    coalesce(f.n_self_answers, 0) AS n_self_answers,
                    CASE
                        WHEN f.first_other_answer IS NULL THEN NULL
                        ELSE date_diff('minute', q.created, f.first_other_answer) / 60.0
                    END AS hours_to_other_answer,
                    (q.created > b.dump_end - INTERVAL 24 HOUR) AS right_censored,
                    (
                        q.closed IS NOT NULL
                        AND q.closed <= q.created + INTERVAL 24 HOUR
                        AND (
                            f.first_other_answer IS NULL
                            OR f.first_other_answer > q.created + INTERVAL 24 HOUR
                        )
                    ) AS closed_fast_unanswered,
                    (
                        f.first_other_answer IS NOT NULL
                        AND f.first_other_answer <= q.created + INTERVAL 24 HOUR
                    ) AS answered_24h
                FROM questions q
                CROSS JOIN bounds b
                LEFT JOIN first_other f ON f.question_id = q.question_id
                LEFT JOIN question_tags t ON t.question_id = q.question_id
            )
            SELECT
                question_id,
                owner_user_id,
                created,
                title,
                title_len,
                body_len,
                code_block_count,
                link_count,
                tag_count,
                primary_tag,
                has_edit_marker,
                hour_utc,
                weekend,
                year,
                n_self_answers,
                hours_to_other_answer,
                right_censored,
                closed_fast_unanswered,
                answered_24h,
                (first_other_answer IS NOT NULL) AS eventually_answered,
                (NOT right_censored AND NOT closed_fast_unanswered) AS in_primary,
                (NOT answered_24h) AS unanswered_24h
            FROM labeled
            WHERE NOT right_censored
        ) TO '{PARQUET.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    counts = con.execute(
        f"""
        SELECT
            count(*) AS n_saved,
            count(*) FILTER (WHERE in_primary) AS n_primary,
            count(*) FILTER (WHERE closed_fast_unanswered) AS n_closed_fast,
            count(*) FILTER (WHERE in_primary AND answered_24h) AS n_answered_24h,
            count(*) FILTER (WHERE in_primary AND eventually_answered) AS n_eventual,
            count(*) FILTER (WHERE in_primary AND has_edit_marker) AS n_edit_marker,
            count(*) FILTER (WHERE in_primary AND code_block_count > 0) AS n_with_code,
            count(*) FILTER (WHERE in_primary AND owner_user_id IS NULL) AS n_null_owner_primary,
            avg(CASE WHEN in_primary THEN cast(answered_24h AS DOUBLE) END) AS answer_rate_primary
        FROM read_parquet('{PARQUET.as_posix()}')
        """
    ).fetchdf()
    print(counts.to_string(index=False))

    n_questions = int(summary.loc[0, "n_questions"])
    n_id_mod = int(summary.loc[0, "n_id_multiple_of_10"])
    answer_rate = float(counts.loc[0, "answer_rate_primary"])
    if answer_rate < 0.25:
        raise RuntimeError(
            f"24-hour answer rate is {answer_rate:.3f}. The answer join is probably incomplete."
        )
    payload = {
        "n_questions_extracted": extracted["n_questions"],
        "n_answers_extracted": extracted["n_answers"],
        "n_tags_extracted": extracted["n_tags"],
        "n_questions_parsed": n_questions,
        "n_answers_parsed": int(summary.loc[0, "n_answers"]),
        "n_tag_rows": int(summary.loc[0, "n_tag_rows"]),
        "share_ids_multiple_of_10": n_id_mod / n_questions if n_questions else None,
        "n_null_owner": int(summary.loc[0, "n_null_owner"]),
        "created_min": summary.loc[0, "created_min"],
        "created_max": summary.loc[0, "created_max"],
        "n_saved_excluding_right_censored": int(counts.loc[0, "n_saved"]),
        "n_primary": int(counts.loc[0, "n_primary"]),
        "n_closed_fast_unanswered": int(counts.loc[0, "n_closed_fast"]),
        "answer_rate_24h_primary": float(counts.loc[0, "answer_rate_primary"]),
        "n_with_code_primary": int(counts.loc[0, "n_with_code"]),
        "n_edit_marker_primary": int(counts.loc[0, "n_edit_marker"]),
        "n_null_owner_primary": int(counts.loc[0, "n_null_owner_primary"]),
        "notes": [
            "CSVs were read with Python's csv module because DuckDB's parallel reader dropped most multiline answers.",
            "Timestamps that matched none of the known formats were dropped.",
            "Right-censored questions (posted in the last 24 hours of the dump) are excluded.",
            "Self-answers are ignored when both owner ids are present and equal.",
            "Closed within 24 hours with no other answer is excluded from the primary label.",
            "Banned as features: score, answer count, answer score, closed date.",
        ],
    }
    write_json(OUTPUTS / "build_summary.json", payload)
    con.close()
    return payload


if __name__ == "__main__":
    build_dataset(force=True)
