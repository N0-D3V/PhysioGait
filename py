#!/usr/bin/env bash
# NixOS: pip wheels need libstdc++/libz on the loader path
export LD_LIBRARY_PATH=/nix/store/0vqb1mcas5j8dv6bhbrshinlgsg6bvgi-gcc-15.3.0-lib/lib:/nix/store/g74kkxb1gz0r6gn1bgv6akyh4qgyvx9p-libz-1.2.8.2025.03.07/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec "$(dirname "$(readlink -f "$0")")/.venv/bin/python" "$@"
