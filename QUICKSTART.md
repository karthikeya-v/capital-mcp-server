# Quick Start Guide

Get your Capital.com MCP server running in 5 minutes!

## Step 1: Get API Credentials

1. Sign up at [capital.com](https://capital.com)
2. Create a **demo account** (recommended for testing)
3. Go to Settings → API
4. Generate new API credentials (save the key and password)

## Step 2: Run Setup

### macOS/Linux
```bash
./setup.sh
```

### Windows
```cmd
setup.bat
```

## Step 3: Configure Credentials

Edit the `.env` file:

```bash
nano .env  # or use any text editor
```

Update with your credentials:
```env
CAPITAL_API_KEY=abc123xyz...
CAPITAL_API_PASSWORD=YourPassword123
CAPITAL_USE_DEMO=true
```

## Step 4: Test the Server

```bash
# Activate virtual environment first
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate  # Windows

# Run the server
python3 server.py
```

You should see: `Starting Capital.com MCP Server in DEMO mode...`

Press `Ctrl+C` to stop.

## Step 5: Connect to Claude Desktop

### Find Your Config File

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

### Get Your Full Path

```bash
# macOS/Linux
pwd
# Shows: /Users/yourname/capital-mcp-server

# Windows (in PowerShell)
Get-Location
# Shows: C:\Users\yourname\capital-mcp-server
```

### Edit Config File

**macOS/Linux Example**:
```json
{
  "mcpServers": {
    "capital-com": {
      "command": "/Users/yourname/capital-mcp-server/venv/bin/python3",
      "args": ["/Users/yourname/capital-mcp-server/server.py"],
      "env": {
        "CAPITAL_API_KEY": "your_api_key_here",
        "CAPITAL_API_PASSWORD": "your_password_here",
        "CAPITAL_USE_DEMO": "true"
      }
    }
  }
}
```

**Windows Example**:
```json
{
  "mcpServers": {
    "capital-com": {
      "command": "C:\\Users\\yourname\\capital-mcp-server\\venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\yourname\\capital-mcp-server\\server.py"],
      "env": {
        "CAPITAL_API_KEY": "your_api_key_here",
        "CAPITAL_API_PASSWORD": "your_password_here",
        "CAPITAL_USE_DEMO": "true"
      }
    }
  }
}
```

⚠️ **Important**: Use **absolute paths**, not relative paths!

## Step 6: Restart Claude Desktop

1. Completely close Claude Desktop (not just the window)
2. Reopen Claude Desktop
3. Look for Capital.com tools in the MCP section

## Step 7: Try It Out!

Ask Claude:

```
"Search for EUR/USD and show me the current price"
```

```
"Get the last 50 daily candles for Bitcoin and analyze the trend"
```

```
"What's my account balance?"
```

## Troubleshooting

### Server not showing in Claude?

1. **Check paths are absolute**: Not `./server.py` but `/full/path/server.py`
2. **Check Python path**: Should be inside your `venv` folder
3. **Check logs**: Look in `~/Library/Logs/Claude/` (macOS) or `%APPDATA%\Claude\logs\` (Windows)

### Authentication errors?

1. Double-check API credentials in `.env`
2. Verify demo account is active
3. Try generating new API credentials

### Command not found?

Make sure virtual environment is activated:
```bash
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

## What's Next?

- Read the full [README.md](README.md) for detailed documentation
- Try different market analysis queries
- Explore technical analysis features
- **Test thoroughly in demo mode before considering live trading**

## Support

Having issues?

1. Check the [README.md](README.md) troubleshooting section
2. Verify all paths are absolute
3. Check Claude Desktop logs for errors
4. Ensure Python 3.10+ is installed

---

**Remember**: Always start with **demo account** (`CAPITAL_USE_DEMO=true`)!
