#!/bin/bash

echo "🚀 Capital.com MCP Server Setup"
echo "================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION detected"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created - PLEASE EDIT IT WITH YOUR API CREDENTIALS"
    echo ""
fi

# Make server executable
chmod +x server.py

echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your Capital.com API credentials"
echo "2. Test the server: python3 server.py"
echo "3. Add to Claude Desktop config (see README.md)"
echo "4. Restart Claude Desktop"
echo ""
echo "⚠️  IMPORTANT: Start with demo account (CAPITAL_USE_DEMO=true)"
echo ""
