"""Dataset adapters: turn external data formats into domain objects.

Everything specific to a dataset's file layout (column names, label syntax, date format,
quirks such as UIT-ViSFD's OTHERS token) stays inside its adapter module. The rest of the
system only ever sees domain enums and `GoldReview` objects.
"""
