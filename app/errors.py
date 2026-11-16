"""Shared user-facing exceptions for pipeline failures."""


class PipelineError(RuntimeError):
    """A recoverable job error that should be displayed in the UI and job log."""
