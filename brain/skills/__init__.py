"""Built-in skills.

Module imports register their @skill-decorated functions in the global
registry. Each skill is a thin shim around the existing Vault / Memory APIs.
"""

# Importing these modules has the side-effect of registering their skills.
from brain.skills import notes, search, links, graph, brain_meta  # noqa: F401
from brain.skills import effectors  # noqa: F401  — registers file/terminal/CC skills
