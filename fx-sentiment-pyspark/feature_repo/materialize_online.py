#!/usr/bin/env python3
"""
Materialize features to online store for real-time serving
"""
import os
from datetime import datetime, timedelta
from feast import FeatureStore

def materialize_features():
    """Materialize features from offline to online store"""
    store = FeatureStore(repo_path=".")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    store.materialize(start_date, end_date)
    print(f"Materialized features from {start_date} to {end_date}")

if __name__ == "__main__":
    materialize_features()
