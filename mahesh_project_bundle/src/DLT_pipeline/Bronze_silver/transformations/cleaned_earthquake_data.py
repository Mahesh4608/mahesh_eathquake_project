from pyspark import pipelines as dp
from pyspark.sql.functions import (
    col,
    count,
    count_if,
    from_json,
    explode,
    from_unixtime,
    current_timestamp,
    when,
    year,
    month,
    dayofmonth,
    regexp_extract,
)
import dlt
from pyspark.sql.types import *

# This file defines a sample transformation.
# Edit the sample below or add new transformations
# using "+ Add" in the file browser.
catalog_name = spark.conf.get("catalog_name")
volume_path = f"/Volumes/{catalog_name}/bronze/earthqauke_data"
primary_key = "id"
properties_schema = StructType(
    [
        StructField("mag", StringType()),
        StructField("place", StringType()),
        StructField("time", StringType()),
        StructField("status", StringType()),
        StructField("tsunami", StringType()),
        StructField("type", StringType()),
        StructField("url", StringType()),
        StructField("detail", StringType()),
        StructField("felt", StringType()),
        StructField("cdi", StringType()),
        StructField("mmi", StringType()),
        StructField("alert", StringType()),
        StructField("sig", StringType()),
        StructField("net", StringType()),
        StructField("code", StringType()),
        StructField("ids", StringType()),
        StructField("sources", StringType()),
        StructField("types", StringType()),
        StructField("nst", StringType()),
        StructField("dmin", StringType()),
        StructField("rms", StringType()),
        StructField("gap", StringType()),
        StructField("magType", StringType()),
        StructField("title", StringType()),
    ]
)

geometry_schema = StructType([StructField("coordinates", ArrayType(DoubleType()))])

feature_schema = StructType(
    [
        StructField("id", StringType()),
        StructField("properties", properties_schema),
        StructField("geometry", geometry_schema),
    ]
)

schema = ArrayType(feature_schema)


@dlt.view(name="earthqauake_data_vw")
def earthquake_data():
    df = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .load(volume_path)
        .withColumn("_load_timestamp", current_timestamp())
    )
    df = df.withColumn("parsed_data", from_json(col("features"), schema))
    df = df.select(explode(col("parsed_data")).alias("features"), "_load_timestamp")
    df = df.select(
        "features.properties.*",
        "features.id",
        col("features.geometry.coordinates")[0].alias("longitude"),
        col("features.geometry.coordinates")[1].alias("latitude"),
        col("features.geometry.coordinates")[2].alias("depth"),
        "_load_timestamp",
    )
    df = (
        df.withColumn("time", from_unixtime(col("time") / 1000).cast("timestamp"))
        .withColumn("mag", col("mag").cast("double"))
        .withColumn("nst", col("nst").cast("double"))
        .withColumn("sig", col("sig").cast("double"))
        .withColumn("tsunami", col("tsunami").cast("double"))
        .withColumn("felt", col("felt").cast("double"))
        .withColumn("gap", col("gap").cast("double"))
        .withColumn("dmin", col("dmin").cast("double"))
        .withColumn("rms", col("rms").cast("double"))
        .withColumn("magType", col("magType").cast("double"))
        .withColumn("cdi", col("cdi").cast("double"))
        .withColumn("mmi", col("mmi").cast("double"))
    )

    df = (
        df
        # Magnitude Category
        .withColumn(
            "magnitude_category",
            when(col("mag") < 2, "Micro")
            .when(col("mag") < 4, "Minor")
            .when(col("mag") < 6, "Moderate")
            .when(col("mag") < 7, "Strong")
            .otherwise("Major"),
        )
        # Risk Score
        .withColumn("risk_score", col("mag") * col("sig"))
        # Is Tsunami
        .withColumn("is_tsunami", when(col("tsunami") == 1, True).otherwise(False))
        # Year / Month / Day for partitioning
        .withColumn("event_year", year(col("time")))
        .withColumn("event_month", month(col("time")))
        .withColumn("event_day", dayofmonth(col("time")))
    )
    df = df.withColumn("region", regexp_extract(col("place"), r",\s*(.*)$", 1))

    return df


dlt.create_streaming_table(name="earthquake_data_final")
dlt.apply_changes(
    target="earthquake_data_final",
    source="earthqauake_data_vw",
    keys=[primary_key],
    sequence_by=col("_load_timestamp"),
    stored_as_scd_type="1",
)
