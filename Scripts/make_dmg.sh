#!/bin/bash
# Packages the (signed, notarized) MetalRTX.app into a distributable DMG with an
# Applications symlink for drag-to-install.
#
# Usage: Scripts/make_dmg.sh [app_path] [output_dmg]
#   app_path   defaults to dist/MetalRTX.app
#   output_dmg defaults to dist/MetalRTX.dmg

set -euo pipefail

cd "$(dirname "$0")/.."

APP_PATH="${1:-dist/MetalRTX.app}"
DMG_PATH="${2:-dist/MetalRTX.dmg}"
VOL_NAME="MetalRTX"

if [[ ! -d "$APP_PATH" ]]; then
    echo "error: $APP_PATH not found. Build it first with Scripts/make_app.sh." >&2
    exit 1
fi

STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

echo "Staging contents..."
cp -R "$APP_PATH" "$STAGING/"
ln -s /Applications "$STAGING/Applications"

echo "Creating ${DMG_PATH}..."
rm -f "$DMG_PATH"
hdiutil create \
    -volname "$VOL_NAME" \
    -srcfolder "$STAGING" \
    -fs HFS+ \
    -format UDZO \
    -ov \
    "$DMG_PATH"

# Sign the DMG so it, too, passes Gatekeeper cleanly when downloaded.
if [[ -n "${CODESIGN_IDENTITY:-}" ]]; then
    echo "Signing DMG..."
    codesign --force --sign "$CODESIGN_IDENTITY" --timestamp "$DMG_PATH"
fi

echo "Done: $DMG_PATH"
