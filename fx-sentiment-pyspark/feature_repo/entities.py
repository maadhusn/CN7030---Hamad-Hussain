from feast import Entity, ValueType

eurusd_entity = Entity(
    name="eurusd",
    value_type=ValueType.STRING,
    description="EURUSD currency pair entity"
)
