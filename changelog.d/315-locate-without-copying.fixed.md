- **A Dense band finds its Top-Hit tier cells without copying the tier (issue
  #315).** `Block.locate` searched a tier's `uint32` rows with Python-int bounds, so
  numpy copied every row of the tier to `int64` on each call: 1.12 GB for OGS-00011's
  140,478,063-cell tier, several times a band, in the serial pass as well. It was
  what made OGS-00011's workers overrun their memory budget.
