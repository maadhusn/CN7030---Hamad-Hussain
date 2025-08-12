from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float64, Int64, String
from .entities import eurusd_entity
from .data_sources import eurusd_source

eurusd_features = FeatureView(
    name="eurusd_features",
    entities=[eurusd_entity],
    ttl=timedelta(days=1),
    schema=[
        Field(name="close", dtype=Float64),
        Field(name="ret_1", dtype=Float64),
        Field(name="ret_3", dtype=Float64),
        Field(name="ret_6", dtype=Float64),
        Field(name="rv_6", dtype=Float64),
        Field(name="rv_24", dtype=Float64),
        Field(name="ema_6", dtype=Float64),
        Field(name="ema_24", dtype=Float64),
        Field(name="g_tone_mean", dtype=Float64),
        Field(name="g_event_total_count_1h", dtype=Int64),
        Field(name="wiki_usd_views", dtype=Int64),
        Field(name="cpi_yoy", dtype=Float64),
        Field(name="pce_yoy", dtype=Float64),
        Field(name="sp_cpi_yoy_dev", dtype=Float64),
        Field(name="sp_pce_yoy_dev", dtype=Float64),
        Field(name="label", dtype=Int64),
    ],
    source=eurusd_source,
)
