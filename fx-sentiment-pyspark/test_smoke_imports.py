#!/usr/bin/env python3
"""
Smoke test for import verification
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all key modules can be imported"""
    try:
        import code_spark.train_big
        print("✅ train_big imports OK")
    except Exception as e:
        print(f"❌ train_big import failed: {e}")
        return False
    
    try:
        import code_spark.ingest_fx
        print("✅ ingest_fx imports OK")
    except Exception as e:
        print(f"❌ ingest_fx import failed: {e}")
        return False
    
    try:
        import code_spark.ingest_fred
        print("✅ ingest_fred imports OK")
    except Exception as e:
        print(f"❌ ingest_fred import failed: {e}")
        return False
    
    try:
        import apps.signal_ui.streamlit_app
        print("✅ streamlit_app imports OK")
    except Exception as e:
        print(f"❌ streamlit_app import failed: {e}")
        return False
    
    try:
        import code_spark.viz.report_plots
        print("✅ viz.report_plots imports OK")
    except Exception as e:
        print(f"❌ viz.report_plots import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
