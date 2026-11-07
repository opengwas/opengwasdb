- **Every writer refuses a missing marker other than the canonical NaN (issue
  #350, spec §15, ADR 0077).** A completion read a source cell whose SE marker
  was a float16 NaN with a payload (`0x7e01`) as missing, and where nothing
  filled the cell it wrote the payload into the completed Store, which then
  failed `validate` (`associations.markers_canonical`). `storable` now refuses
  such a z or SE in every writer, and every completion refuses the source cell
  (`NotAdmitted`) rather than carry it.
