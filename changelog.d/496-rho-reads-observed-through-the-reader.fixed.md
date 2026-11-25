- **`build-dense-rho` reads observed Z through the Association Reader (issue #496).**
  It tested `isfinite(z)` and an `imputed` array by itself, so a Reference-Completed
  grid without its imputed mask counted every imputed Z as observed and fed it to the
  estimate. It now reads observed associations as every query does: such a grid is
  refused, naming the mask, and a §15-invalid cell is refused rather than read.
