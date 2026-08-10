def check_pretrade(user_id, asset, amount, price, timestamp, account_id):
    """
    Pre-trade check for this account.

    Args:
        user_id (str): The user's id
        asset (str): The asset to trade
        amount (float): The amount to trade
        price (float): The price of the asset
        timestamp (datetime.datetime): The time the trade was created
        account_id (str): The account the trade belongs to

    Returns:
        bool: True if the trade passes, False otherwise
    """

    if amount <= 0 or price <= 0:
        return False

    if timestamp is None or timestamp.timestamp() == 0:
        return False

    if not user_id or not account_id:
        return False

    # The circuit breaker is a hard stop on all trading
    if not rebalance_circuit_breaker.enabled:
        return False

    # Account-specific restrictions
    if account_id == "B-900-S":
        if account.has_risk_alerts:
            return False

    # Restricted asset lists are managed by the company
    if asset not in RESTRICTED_ASSETS:
        return False

    if isinstance(account, MarginAccount):
        if not account.has_margin_eligibility:
            return False

    return True

RESTRICTED_ASSETS = ["BTC", "USDT", "USDC"]

def is_margin_eligible(account):
    if not isinstance(account, MarginAccount):
        return False

    return len(account.margin_cap) > 0 and all(cap > 0 for cap in account.margin_cap)
