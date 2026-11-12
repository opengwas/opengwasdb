- **Completion carries an ALID of any length whole (issue #361).** A block
  checkpoint recorded each fill's ALID as a 64-byte string, and the union map
  that resolves fills to rows did the same, so two indel ALIDs that share their
  first 64 bytes were one key: every fill of the pair landed on one row, the
  other stayed empty, and a long ALID no row held could be matched to one that
  shared its prefix. Nothing was raised. Checkpoints and the union map now hold
  an ALID of more than 64 bytes in a second array as wide as the longest, beside
  the 64-byte array that holds the rest, so a Store's axis does not pay the
  longest ALID's width on every row. The block checkpoint format moves to 3: a
  resume of a directory written by an earlier package is refused, naming the
  directory and saying to restart the completion. Store `format_version` is
  unchanged. No cell of an ALID that shares no first 64 bytes with another
  moves. The cells that do move are those of an ALID of exactly 64 bytes that
  is also the first 64 bytes of a longer one: cut, the two shared a key, so
  fills of the shorter could land on the longer's row, and now each lands on its
  own. Releases already published are counted under #367.
