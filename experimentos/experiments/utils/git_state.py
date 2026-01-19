"""Helpers for tracking git state between command runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

from experiments.config.settings import PathSettings


def _get_default_project_root() -> Path:
    """Get the default project root from settings."""
    return PathSettings().project_root


@dataclass
class GitStateTracker:
    """Tracks last processed commit to detect new code changes."""

    key: str
    project_root: Path = field(default_factory=_get_default_project_root)

    def __post_init__(self) -> None:
        self._state_dir = self.project_root / ".ldp"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        safe_key = self.key.replace("/", "_").replace("\\", "_")
        self._state_file = self._state_dir / f"{safe_key}.commit"
        self._repo_root = self.project_root

    def _current_commit(self) -> Optional[str]:
        try:
            repo = Repo(self._repo_root)
            if repo.head.is_detached:
                return repo.head.commit.hexsha
            return repo.head.commit.hexsha
        except (InvalidGitRepositoryError, NoSuchPathError, GitCommandError):
            return None

    def _read_last_commit(self) -> Optional[str]:
        if not self._state_file.exists():
            return None
        content = self._state_file.read_text(encoding="utf-8").strip()
        return content or None

    def has_new_commit(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Returns whether a new commit exists since the last recorded run."""
        last_commit = self._read_last_commit()
        current_commit = self._current_commit()
        changed = bool(current_commit and current_commit != last_commit)
        return changed, last_commit, current_commit

    def record_current_commit(self) -> None:
        commit = self._current_commit()
        if commit is None:
            # Remove stale records if repository information is unavailable.
            if self._state_file.exists():
                self._state_file.unlink()
            return
        self._state_file.write_text(commit, encoding="utf-8")
