"""Internationalization protocols for analysis artifacts.

This module defines the Translator protocol and Locale enum for
internationalizing analysis outputs such as plots, tables, and reports.
"""

from enum import StrEnum
from typing import Protocol


class Locale(StrEnum):
    """Supported locales for analysis artifacts."""

    EN_US = "en_US"
    PT_BR = "pt_BR"


class Translator(Protocol):
    """Protocol for translation services.

    Provides a standard interface for translating message IDs to
    localized strings with optional parameter interpolation.
    """

    @property
    def locale(self) -> Locale:
        """Get the current locale.

        Returns:
            The locale used for translations.
        """
        ...

    def translate(self, msgid: str, **kwargs: str) -> str:
        """Translate a message ID to a localized string.

        Args:
            msgid: The message identifier to translate.
            **kwargs: Optional parameters to interpolate into the translated string.

        Returns:
            The translated string with parameters interpolated.

        Example:
            >>> translator.translate("Stability Analysis: {metric_name}", metric_name="G-Mean")
            "Análise de Estabilidade: G-Mean"
        """
        ...


# Alias for the translate method signature for convenience
_ = Translator.translate
"""Common alias for translation function."""


__all__ = [
    "Locale",
    "Translator",
]
