from dataclasses import dataclass


@dataclass
class Evidence:
    # Optional anchors: paths, commits, urls, ids — anything to re-check later
    kind: str = ""  # "file" | "commit" | "url" | "ticket" | "log" | ...
    ref: str = ""  # path/hash/url/identifier
    note: str = ""
