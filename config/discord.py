from dataclasses import dataclass

@dataclass(frozen=True)
class DiscordConfig:
    """Discord configuration constants."""

    # Bot token
    BOT_TOKEN: str = 'your_bot_token_here'

    # Channel IDs
    TRADE_CHANNEL_ID: int = 123456789012345678
    LOG_CHANNEL_ID: int = 987654321098765432

# Create singleton instance
DISCORD = DiscordConfig()

# Convenience exports
__all__ = [
    'DISCORD',
]
