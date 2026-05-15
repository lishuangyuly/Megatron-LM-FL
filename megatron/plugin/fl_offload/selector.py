"""``get_forward_backward_func`` wrapper / selector (commit 5 placeholder).

Will host ``get_forward_backward_func_wrapper(original)`` that, depending on
``--pipeline-schedule-backend`` and ``FlOffloadConfig.enable``, either
returns the FL original or hands back the plugin's wrapped schedule.

Empty in commit 1.
"""
