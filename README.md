# T-800 Server Setup

## Prerequisites
- Python 3.11 or higher
- (Recommended) Virtual environment
- Git
- Ngrok (for public API access)

## Step-by-Step Setup Guide

### 1. Clone the Repository
If you haven't already, clone your GitHub repository:
```bash
git clone <your-repo-url>
cd t-800-server
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
python server.py
```

### 5. Expose Your API with Ngrok
Start ngrok to forward your local server to a public domain:
```bash
ngrok http 5000
```
- Copy the HTTPS forwarding URL from ngrok and use it in your Android app or anywhere you need public API access.

## Notes
- Edit `config.py` for custom settings if needed.
- Make sure your Android app uses the ngrok HTTPS URL for API calls.
- For troubleshooting, check logs and error messages in your terminal.

