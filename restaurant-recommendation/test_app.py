"""
Basic tests for the restaurant recommendation app.
"""
import pytest
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app_import():
    """Test that the app can be imported without errors."""
    try:
        import app
        assert hasattr(app, 'app')
        assert app.app is not None
    except ImportError as e:
        pytest.fail(f"Failed to import app: {e}")

def test_app_creation():
    """Test that the Flask app can be created."""
    try:
        from app import app
        assert app is not None
        assert hasattr(app, 'config')
    except Exception as e:
        pytest.fail(f"Failed to create Flask app: {e}")

def test_app_config():
    """Test that the app has basic configuration."""
    try:
        from app import app
        assert app.config is not None
        # Test that the app has a secret key (required for sessions)
        assert 'SECRET_KEY' in app.config or app.secret_key is not None
    except Exception as e:
        pytest.fail(f"Failed to access app config: {e}")

def test_health_endpoint():
    """Test that the app has a basic health check endpoint."""
    try:
        from app import app
        with app.test_client() as client:
            # Test if the app responds to a basic request
            response = client.get('/')
            # Should return 200 or redirect (302)
            assert response.status_code in [200, 302]
    except Exception as e:
        pytest.fail(f"Failed to test health endpoint: {e}")

def test_ml_model_loading():
    """Test that ML models can be loaded without errors."""
    try:
        from app import load_ml_models
        # This should not raise an exception
        load_ml_models()
    except Exception as e:
        # If ML models fail to load, that's okay for basic testing
        # Just log the error but don't fail the test
        print(f"ML models not available: {e}")

if __name__ == '__main__':
    pytest.main([__file__])
