"""Argparse extension for the fl-offload plugin (commit 2 placeholder).

Will host:

* ``add_fl_offload_args(parser)`` — registers the eight ``--fl-offload-*``
  CLI flags plus the two observability flags introduced in commit 7.
* ``chain_extra_args_provider(extra_args_provider)`` — returns a new provider
  that first defers to the caller's provider and then calls
  ``add_fl_offload_args``.

Empty in commit 1.
"""
