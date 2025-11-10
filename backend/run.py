#!/usr/bin/env python3
"""
SCC Classification API Runner
"""

import uvicorn
import os

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models/vit", exist_ok=True)
    os.makedirs("models/convnext", exist_ok=True)
    os.makedirs("models/coatnet", exist_ok=True)
    
    print("🚀 Starting SCC Classification API...")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )