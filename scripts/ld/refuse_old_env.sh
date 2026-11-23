# Sourced by the LD shell runners: refuse a variable under an old name rather than ignore it.
# opengwasdb.ld.env does the same for Python (ADR 0094); a variable set under the old name would
# be read by nothing, and the runner's default would apply in its place.
for name in $(compgen -e); do
  case "$name" in
    OGLD_*) new="OPENGWASDB_LD_${name#OGLD_}" ;;
    *) continue ;;
  esac
  echo "opengwasdb ld: an old environment variable is set: $name is now $new" >&2
  exit 2
done
