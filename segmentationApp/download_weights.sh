#!/bin/bash

# ============================================================================
#  ChronoRoot Model Downloader
# ============================================================================

# Visual Styling
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# 1. Dynamic Path Setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
MODELS_DIR="$SCRIPT_DIR/models"

# 2. Dependency Check & Auto-Install
# We check if 'hf' is in the path. If not, we install it.
if ! command -v hf &> /dev/null; then
    echo -e "${YELLOW}[!]${NC} 'hf' command not found. Installing huggingface_hub..."
    
    # Attempt install
    if pip install huggingface_hub==1.3.4; then
        echo -e "${GREEN}[OK]${NC} huggingface_hub installed successfully."
    else
        echo -e "${RED}[ERROR]${NC} Failed to install huggingface_hub."
        exit 1
    fi
fi

echo -e "${BLUE}[STATUS]${NC} Scanning Hugging Face for models by author 'ngaggion'..."

# 3. Fetch Repos
# We capture stderr (2>&1) to print the error message if the connection fails
if ! RAW_OUTPUT=$(hf models ls --author "ngaggion" 2>&1); then
    echo -e "${RED}[ERROR]${NC} Failed to connect to Hugging Face."
    echo "Details: $RAW_OUTPUT"
    exit 1
fi

# Match repo IDs only (hf models ls prints a table; avoid trailing columns)
REPOS=($(echo "$RAW_OUTPUT" | grep -oE 'ngaggion/ChronoRoot2-[A-Za-z0-9_-]+' | sort -u))

if [ ${#REPOS[@]} -eq 0 ]; then
    echo -e "${YELLOW}[!]${NC} No models found matching 'ngaggion/ChronoRoot2-'."
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Found ${#REPOS[@]} matching models."

# 4. Download Loop
for REPO in "${REPOS[@]}"; do
    # Extract species name (ngaggion/ChronoRoot2-Tomato -> Tomato)
    NAME=$(echo "$REPO" | sed "s|.*/ChronoRoot2-||")
    TARGET_PATH="$MODELS_DIR/$NAME"

    echo -e "\n${BLUE}[SYNC]${NC} Processing: ${BOLD}$NAME${NC}"
    
    # hf download automatically checks for updates/hashes
    hf download "$REPO" --local-dir "$TARGET_PATH"
done

echo -e "\n${GREEN}[SUCCESS]${NC} All models are synchronized."