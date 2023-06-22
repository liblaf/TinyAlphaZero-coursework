#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

while true; do
  tail --lines=64 "${@}"
  sleep 2
done
