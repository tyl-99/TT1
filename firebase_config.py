import os
import firebase_admin
from firebase_admin import credentials

def initialize_firebase():
    """Initialize Firebase using environment variables"""
    
    # Check if Firebase is already initialized
    if firebase_admin._apps:
        return firebase_admin.get_app()
    
    # Load credentials from environment variables
    firebase_config = {
        "type": os.getenv("FIREBASE_TYPE", "service_account"),
        "project_id": os.getenv("FIREBASE_PROJECT_ID"),
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
        "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL"),
        "universe_domain": os.getenv("FIREBASE_UNIVERSE_DOMAIN", "googleapis.com")
    }
    
    # Validate required fields
    required_fields = ["project_id", "private_key", "client_email"]
    for field in required_fields:
        if not firebase_config.get(field):
            raise ValueError(f"Missing required Firebase configuration: {field}")
    
    # Initialize Firebase with storage bucket
    cred = credentials.Certificate(firebase_config)
    
    # Add storage bucket configuration
    storage_bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
    if not storage_bucket:
        # Use default bucket name: project-id.appspot.com
        storage_bucket = f"{firebase_config['project_id']}.appspot.com"
    
    app = firebase_admin.initialize_app(cred, {
        'storageBucket': storage_bucket
    })
    
    return app

def get_firestore_client():
    """Get Firestore client (for database operations)"""
    from firebase_admin import firestore
    initialize_firebase()
    return firestore.client()

def get_storage_bucket(bucket_name=None):
    """Get Firebase Storage bucket (for file storage)"""
    from firebase_admin import storage
    initialize_firebase()
    
    if bucket_name:
        return storage.bucket(bucket_name)
    else:
        # Use default bucket that was configured during initialization
        return storage.bucket()

def get_auth_client():
    """Get Firebase Auth client (for user authentication)"""
    from firebase_admin import auth
    initialize_firebase()
    return auth 