"""Gettext-based translator service implementation.

This module provides a Translator implementation using Python's gettext
module for internationalization of analysis artifacts.
"""

import gettext
from gettext import GNUTranslations, NullTranslations
from pathlib import Path

from loguru import logger

from experiments.core.analysis.translator import Locale


def _get_locales_dir() -> Path:
    """Get the path to the locales directory.

    Returns:
        Path to the locales directory in the project root.
    """
    return Path(__file__).resolve().parent.parent.parent / "locales"


class GettextTranslator:
    """Translator implementation using Python's gettext module.

    Loads translations from .mo files in the locales directory and provides
    message translation with parameter interpolation.

    Attributes:
        _locale: The locale used for translations.
        _translations: The gettext translations object.
    """

    def __init__(
        self,
        locale: Locale,
        domain: str = "base",
        locales_dir: Path | None = None,
    ) -> None:
        """Initialize the translator with a specific locale.

        Args:
            locale: The locale to use for translations.
            domain: The gettext domain (name of .mo file without extension).
            locales_dir: Optional custom path to locales directory.
                If not provided, uses the default locales directory.
        """
        self._locale = locale
        self._domain = domain
        self._locales_dir = locales_dir or _get_locales_dir()
        self._translations: GNUTranslations | NullTranslations

        try:
            self._translations = gettext.translation(
                domain=domain,
                localedir=str(self._locales_dir),
                languages=[locale.value],
            )
            logger.debug(f"Loaded translations for locale '{locale.value}'")
        except FileNotFoundError:
            logger.warning(
                f"No translations found for locale '{locale.value}', "
                f"using null translations (passthrough)"
            )
            self._translations = NullTranslations()

    @property
    def locale(self) -> Locale:
        """Get the current locale.

        Returns:
            The locale used for translations.
        """
        return self._locale

    def translate(self, msgid: str, **kwargs: str) -> str:
        """Translate a message ID to a localized string.

        If the translation is not found, logs a warning and returns the
        original msgid with parameters interpolated (graceful fallback).

        Args:
            msgid: The message identifier to translate.
            **kwargs: Optional parameters to interpolate into the translated string.

        Returns:
            The translated string with parameters interpolated.
        """
        translated = self._translations.gettext(msgid)

        # If translation returns the same msgid and we're not in en_US,
        # it means the translation is missing
        if translated == msgid and self._locale != Locale.EN_US:
            logger.warning(f"Missing translation for locale '{self._locale.value}': {msgid!r}")

        # Interpolate parameters if provided
        if kwargs:
            try:
                return translated.format(**kwargs)
            except KeyError as e:
                logger.error(
                    f"Missing format parameter in translation: {e}. "
                    f"msgid={msgid!r}, kwargs={kwargs}"
                )
                return translated

        return translated


def create_translator(
    locale: Locale | str,
    domain: str = "base",
    locales_dir: Path | None = None,
) -> GettextTranslator:
    """Factory function to create a translator instance.

    Args:
        locale: The locale to use for translations.
            Can be a Locale enum or a string (e.g., "pt_BR").
        domain: The gettext domain (name of .mo file without extension).
        locales_dir: Optional custom path to locales directory.

    Returns:
        A configured GettextTranslator instance.
    """
    if isinstance(locale, str):
        locale = Locale(locale)

    return GettextTranslator(
        locale=locale,
        domain=domain,
        locales_dir=locales_dir,
    )


__all__ = [
    "GettextTranslator",
    "create_translator",
]
