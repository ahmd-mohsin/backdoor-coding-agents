#!/bin/bash
# One-time setup on your Mac.
#
#   scripts/deltaai/local/setup_ssh.sh [ncsa_username]
#
# Adds DeltaAI host aliases to ~/.ssh/config (after backing it up), and adds the
# login nodes' host keys to ~/.ssh/known_hosts only if they match the fingerprints
# published in the DeltaAI login docs.
#
# Aliases: deltaai = round-robin login node; deltaai1..deltaai4 = gh-login01..04.
# The username defaults to NCSA_USER in scripts/deltaai/config.env.
set -euo pipefail
. "$(dirname "$0")/common.sh"

case "${1:-}" in -h|--help) usage; exit 0 ;; esac
user="${1:-${NCSA_USER:-}}"
if [ -z "$user" ]; then
    echo "usage: $0 <ncsa_username>   (or set NCSA_USER in $DTAI_DIR/config.env)" >&2
    exit 1
fi

cfg="$HOME/.ssh/config"
begin="# >>> deltaai (managed by scripts/deltaai/local/setup_ssh.sh) >>>"
end="# <<< deltaai <<<"

mkdir -p "$HOME/.ssh/cm"
chmod 700 "$HOME/.ssh" "$HOME/.ssh/cm"
touch "$cfg"
chmod 600 "$cfg"
backup="$cfg.bak.$(date +%Y%m%d%H%M%S)"
cp "$cfg" "$backup"

# Remove any earlier copy of our block (and trailing blank lines), then append the new one.
tmp="$(mktemp)"
awk -v b="$begin" -v e="$end" '
    $0 == b { skip = 1 }
    !skip {
        if ($0 ~ /^[[:space:]]*$/) { blank++ }
        else { while (blank > 0) { print ""; blank-- } print }
    }
    $0 == e { skip = 0 }
' "$cfg" > "$tmp"
cat >> "$tmp" <<EOF

$begin
# deltaai is the round-robin alias; deltaai1-4 pin one login node (use those with tmux).
Host deltaai
    HostName dtai-login.delta.ncsa.illinois.edu
Host deltaai1
    HostName gh-login01.delta.ncsa.illinois.edu
Host deltaai2
    HostName gh-login02.delta.ncsa.illinois.edu
Host deltaai3
    HostName gh-login03.delta.ncsa.illinois.edu
Host deltaai4
    HostName gh-login04.delta.ncsa.illinois.edu
Host deltaai deltaai1 deltaai2 deltaai3 deltaai4
    User $user
    # SSH keys are disabled on DeltaAI; log in with your NCSA password + Duo.
    PreferredAuthentications keyboard-interactive,password
    PubkeyAuthentication no
    # Share one authenticated connection with later ssh/rsync/scp to the same alias.
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 12h
    ServerAliveInterval 60
    ServerAliveCountMax 10
    TCPKeepAlive yes
$end
EOF
mv "$tmp" "$cfg"
chmod 600 "$cfg"
echo "Updated $cfg for NCSA user '$user' (backup: $backup)."

# Host key fingerprints published at
# https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/user-guide/login.html
known_fps="SHA256:NtMT6SiYO/SS6HbZw1a7nct6/UnKplU46BoQTwXnWJA SHA256:uu7sIdxzzcLVCDnKFrBr9PHb1Y6Ox4PNd8xS+m+DFjM SHA256:1RAR7PSHTxrqOn978/us/4loY0uwnpNd6aWs30AX4c8"
is_known_fp() {
    case " $known_fps " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

kh="$HOME/.ssh/known_hosts"
touch "$kh"
chmod 600 "$kh"
echo "Checking login node host keys:"
for h in dtai-login.delta.ncsa.illinois.edu gh-login01.delta.ncsa.illinois.edu \
         gh-login02.delta.ncsa.illinois.edu gh-login03.delta.ncsa.illinois.edu \
         gh-login04.delta.ncsa.illinois.edu; do
    existing="$(ssh-keygen -l -F "$h" -f "$kh" 2>/dev/null | grep -o 'SHA256:[^ ]*' || true)"
    if [ -n "$existing" ]; then
        bad=""
        for fp in $existing; do is_known_fp "$fp" || bad="$bad $fp"; done
        if [ -z "$bad" ]; then
            echo "  $h: already in known_hosts, matches the published fingerprint"
        else
            echo "  WARNING: known_hosts has a key for $h that does not match the published fingerprints:$bad" >&2
        fi
        continue
    fi
    line="$(ssh-keyscan -T 10 -t ed25519 "$h" 2>/dev/null | grep -v '^#' | head -n 1 || true)"
    if [ -z "$line" ]; then
        echo "  $h: could not fetch its host key (offline?); ssh will ask you to confirm it on first login"
        continue
    fi
    fp="$(printf '%s\n' "$line" | ssh-keygen -l -f - | awk '{print $2}')"
    if is_known_fp "$fp"; then
        printf '%s\n' "$line" >> "$kh"
        echo "  $h: host key matches the published fingerprint, added"
    else
        echo "  WARNING: $h presented $fp, which is not a published DeltaAI fingerprint. Not added." >&2
    fi
done

cat <<EOF

Done. Next:
  scripts/deltaai/local/login.sh -s main    # log in, inside tmux session "main" on gh-login01
EOF
