from feast import FeatureService
from .feature_views import eurusd_features

eurusd_service = FeatureService(
    name="eurusd_prediction_service",
    features=[eurusd_features],
)
