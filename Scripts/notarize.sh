#!/bin/bash
# Notarizes and staples a signed MetalRTX.app produced by make_app.sh.
#
# One-time setup (run once, stores credentials securely in your login keychain,
# never share these with anyone else):
#   xcrun notarytool store-credentials "notarytool-profile" \
#       --apple-id "you@example.com" \
#       --team-id "YOURTEAMID" \
#       --password "app-specific-password"
#
# Usage: Scripts/notarize.sh [app_path] [keychain_profile]
#   app_path defaults to dist/MetalRTX.app
#   keychain_profile defaults to "notarytool-profile"

set -euo pipefail

cd "$(dirname "$0")/.."

APP_PATH="${1:-dist/MetalRTX.app}"
PROFILE="${2:-notarytool-profile}"
ZIP_PATH="${APP_PATH%.app}.zip"

if [[ ! -d "$APP_PATH" ]]; then
    echo "error: $APP_PATH not found. Run Scripts/make_app.sh with CODESIGN_IDENTITY set first." >&2
    exit 1
fi

echo "Zipping ${APP_PATH}..."
ditto -c -k --keepParent "$APP_PATH" "$ZIP_PATH"

echo "Submitting to Apple notary service (profile: ${PROFILE})..."
xcrun notarytool submit "$ZIP_PATH" --keychain-profile "$PROFILE" --wait

echo "Stapling ticket to ${APP_PATH}..."
xcrun stapler staple "$APP_PATH"

rm -f "$ZIP_PATH"
echo "Done. ${APP_PATH} is signed, notarized, and stapled."
