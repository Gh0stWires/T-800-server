# T-800 Server Setup

## Prerequisites
- Python 3.11 or higher
- (Optional) Virtual environment
- Git

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd t-800-server
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(If you don't have a requirements.txt, install needed packages manually)*

4. **Run the server**
   ```bash
   python server.py
   ```

## Pushing Changes to GitHub

1. **Add files**
   ```bash
   git add .
   ```
2. **Commit changes**
   ```bash
   git commit -m "Add server setup instructions to README"
   ```
3. **Push to GitHub**
   ```bash
   git push
   ```

## Notes
- For SSL, place your certificates (`cert.pem`, `key.pem`) in the project directory.
- Edit `config.py` for custom settings.

---
Feel free to expand this guide for advanced configuration or troubleshooting.
