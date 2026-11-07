- **`AssociationCodec.encode` refuses a `carried` mask of the wrong length,
  and the exact-carry rule reads a frequency the way a reader does (issue
  #301).** `carried` marks the cells decoded from another Store, whose SE must
  decode to exactly its value; it was the one array of a batch the parallel
  check skipped, so a mask of another length was broadcast over the cells — a
  single `True` held every cell as an exact exception — or reached numpy's
  error, which names neither the array nor the batch. It is now refused by
  name, as every other array of a batch is. The exact rule also predicted each
  carried SE from the EAF as `encode_se` received it, while `decode_se`
  narrows that to `float32` first: handed a `float64` frequency, it held cells
  a reader decodes exactly as exceptions the Store does not need. It now
  predicts from the narrowed value. The codes `encode_se` writes still come
  from the prediction over the EAF as given, so a Store this package already
  built is unchanged.
