"""
build_dataset.py
----------------
Executes the DuckDB ETL and Feature Engineering SQL scripts on raw BTS Form 234
ASCII files and exports the leak-free analytical dataset to Parquet format.
"""

from pathlib import Path
import time
import duckdb


def run_pipeline() -> None:
    start_time = time.time()

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    schema_sql_path = project_root / "sql" / "01_schema_setup.sql"
    feature_sql_path = project_root / "sql" / "02_feature_query.sql"
    output_parquet_path = processed_dir / "flights_clean.parquet"

    asc_files = list(raw_dir.glob("*.asc"))
    if not asc_files:
        raise FileNotFoundError(
            f"No .asc files found in {raw_dir}. Please place BTS Form 234 files there."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Found {len(asc_files)} raw .asc file(s) in {raw_dir}")

    con = duckdb.connect(database=":memory:")

    try:
        print(f"[INFO] Executing schema setup: {schema_sql_path.name}...")
        schema_sql = schema_sql_path.read_text(encoding="utf-8")
        con.execute(schema_sql)

        raw_count = con.execute("SELECT COUNT(*) FROM raw_flights").fetchone()[0]
        print(f"[INFO] Raw records ingested: {raw_count:,}")

        print(f"[INFO] Executing feature engineering: {feature_sql_path.name}...")
        feature_sql = feature_sql_path.read_text(encoding="utf-8")
        con.execute(feature_sql)

        stats = con.execute("""
            SELECT
                COUNT(*) AS total_rows,
                ROUND(AVG(is_delayed) * 100, 2) AS delay_rate_pct,
                COUNT(DISTINCT origin) AS unique_origins
            FROM model_features
        """).fetchone()

        print(f"[INFO] Clean dataset rows: {stats[0]:,}")
        print(f"[INFO] Target positive class (is_delayed=1): {stats[1]}%")
        print(f"[INFO] Distinct origin airports: {stats[2]}")

        print(f"[INFO] Exporting dataset to {output_parquet_path}...")
        con.execute(f"""
            COPY model_features
            TO '{output_parquet_path.as_posix()}'
            (FORMAT PARQUET, COMPRESSION 'SNAPPY');
        """)

        elapsed = time.time() - start_time
        print(f"[SUCCESS] Pipeline completed in {elapsed:.2f} seconds.")

    finally:
        con.close()


if __name__ == "__main__":
    run_pipeline()