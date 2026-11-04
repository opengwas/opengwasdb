- **`build-ragged-ssf` names a variant by the GWAS-SSF reader's rsid rule (issue
  #56, O1).** A variant's rsid is the first of its row's `rsid` and `variant_id`
  cells that is an rs identifier. The build used to take the first non-empty
  cell and store no rsid when that cell was something else, such as a provider
  token in `rsid` beside the rs id in `variant_id`. Rebuilt, OGS-00006 gains
  these 21 rsids; it loses and replaces none, and nothing else in it changes.
  OGS-00007 is unchanged.

  | ALID | rsid |
  |---|---|
  | `3:38603929:C:T` | `rs1805124` |
  | `3:38633221:C:T` | `rs6599230` |
  | `4:94657499:C:T` | `rs148939806` |
  | `5:38370449:C:T` | `rs140280982` |
  | `5:170722733:A:G` | `rs147147696` |
  | `6:26505040:A:G` | `rs149886741` |
  | `7:100893176:G:T` | `rs1799805` |
  | `15:42746201:C:T` | `rs146279300` |
  | `16:11281137:A:C` | `rs35576928` |
  | `17:3092700:C:T` | `rs769424` |
  | `17:3092923:C:T` | `rs769423` |
  | `17:3216103:G:T` | `rs769425` |
  | `17:3216379:A:G` | `rs769426` |
  | `17:3216473:C:T` | `rs769427` |
  | `17:3278090:C:T` | `rs769434` |
  | `17:3292209:C:T` | `rs703903` |
  | `17:3310887:G:T` | `rs231678` |
  | `17:3397640:A:T` | `rs769422` |
  | `17:3398255:A:G` | `rs150988` |
  | `17:53824436:A:G` | `rs116955880` |
  | `19:43175269:A:G` | `rs150587441` |
