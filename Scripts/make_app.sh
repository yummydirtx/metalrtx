#!/bin/bash
# Builds MetalRTX in release mode and packages it into a standalone .app bundle.
#
# Usage: Scripts/make_app.sh [output_dir]
#   output_dir defaults to ./dist

set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${1:-dist}"
APP_NAME="MetalRTX"
BUNDLE_ID="com.metalrtx.app"

echo "Building release binary..."
swift build -c release

BIN_PATH=".build/release/${APP_NAME}"
RESOURCE_BUNDLE=".build/release/${APP_NAME}_${APP_NAME}.bundle"

if [[ ! -f "$BIN_PATH" ]]; then
    echo "error: could not find built binary at $BIN_PATH" >&2
    exit 1
fi

APP_DIR="${OUT_DIR}/${APP_NAME}.app"
echo "Assembling ${APP_DIR}..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

cp "$BIN_PATH" "$APP_DIR/Contents/MacOS/${APP_NAME}"

# Place shader resources under Contents/Resources — the standard macOS bundle
# location. codesign only seals content inside Contents/, so anything left at the
# .app root would break the signature ("unsealed contents present in bundle root").
# ShaderLibrary looks these up via Bundle.main.url(forResource: "Shaders", ...).
SHADERS_SRC="${RESOURCE_BUNDLE}/Shaders"
if [[ ! -d "$SHADERS_SRC" ]]; then
    # Fall back to the source tree if the SPM bundle layout changes.
    SHADERS_SRC="Sources/${APP_NAME}/Shaders"
fi
if [[ -d "$SHADERS_SRC" ]]; then
    cp -R "$SHADERS_SRC" "$APP_DIR/Contents/Resources/Shaders"
else
    echo "error: could not find Shaders directory to bundle" >&2
    exit 1
fi

cat > "$APP_DIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>Metal RTX</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>15.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.graphics-design</string>
</dict>
</plist>
PLIST

SIGN_IDENTITY="${CODESIGN_IDENTITY:-}"

if [[ -n "$SIGN_IDENTITY" ]]; then
    echo "Signing with identity: $SIGN_IDENTITY"
    # --options runtime enables the Hardened Runtime, required for notarization.
    # --timestamp adds a secure timestamp, also required for notarization.
    codesign --force --deep --options runtime --timestamp \
        --sign "$SIGN_IDENTITY" "$APP_DIR"
    codesign --verify --deep --strict --verbose=2 "$APP_DIR"
else
    echo "CODESIGN_IDENTITY not set; ad-hoc signing (local use only, not distributable)."
    codesign --force --deep --sign - "$APP_DIR"
fi

echo "Done: $APP_DIR"
