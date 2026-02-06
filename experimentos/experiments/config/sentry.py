import sentry_sdk

from experiments.config.settings import LdpSettings


def init_sentry(settings: LdpSettings) -> None:
    """Initialize Sentry for error tracking."""
    sentry_sdk.init(
        dsn=settings.sentry_dns,
        send_default_pii=True,
    )
