from . import emacs, notemacs, vi

__all__ = ["emacs", "notemacs", "vi"]

editingmodes = [emacs.EmacsMode, notemacs.NotEmacsMode, vi.ViMode]

# add check to ensure all modes have unique mode names
