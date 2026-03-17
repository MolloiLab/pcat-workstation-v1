#!/usr/bin/env bash
# Build PCAT Workstation as a macOS .dmg installer.
#
# Prerequisites:
#   pip install pyinstaller
#
# Usage:
#   ./scripts/build_dmg.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/dist"
APP_NAME="PCAT Workstation"
DMG_NAME="PCAT_Workstation"
VERSION="1.0.0"

echo "=== Building $APP_NAME v$VERSION ==="

# Step 1: Clean previous build
echo "[1/4] Cleaning previous build..."
rm -rf "$BUILD_DIR/$APP_NAME" "$BUILD_DIR/$APP_NAME.app" "$BUILD_DIR/$DMG_NAME.dmg"
rm -rf "$PROJECT_DIR/build"

# Step 2: Run PyInstaller
echo "[2/4] Running PyInstaller..."
cd "$PROJECT_DIR"
pyinstaller pcat_workstation.spec --noconfirm

# Step 3: Verify .app was created
APP_PATH="$BUILD_DIR/$APP_NAME.app"
if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: $APP_PATH not found. Build failed."
    exit 1
fi
echo "    Built: $APP_PATH"

# Step 4: Create DMG
echo "[3/4] Creating DMG..."
DMG_PATH="$BUILD_DIR/${DMG_NAME}_${VERSION}.dmg"

# Use hdiutil to create a simple DMG
hdiutil create -volname "$APP_NAME" \
    -srcfolder "$BUILD_DIR/$APP_NAME.app" \
    -ov -format UDBZ \
    "$DMG_PATH"

echo "[4/4] Done!"
echo "    DMG: $DMG_PATH"
echo ""
echo "To install: Open the DMG and drag '$APP_NAME' to Applications."
