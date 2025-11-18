#!/usr/bin/env python3
"""
Data Preprocessing Utilities for Cappuccino

Provides sklearn-compatible transformers for financial data preprocessing.
Ported from FinRL library.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler


class GroupByScaler(BaseEstimator, TransformerMixin):
    """
    Sklearn-like scaler that scales data independently per group.

    Essential for multi-asset portfolios where different tickers have
    vastly different price scales (e.g., BTC at $40,000 vs SOL at $100).

    Example:
        >>> from preprocessor import GroupByScaler
        >>> import pandas as pd
        >>> from sklearn.preprocessing import StandardScaler
        >>>
        >>> # Create sample data
        >>> df = pd.DataFrame({
        ...     'tic': ['BTC', 'BTC', 'ETH', 'ETH'],
        ...     'close': [40000, 41000, 2500, 2600],
        ...     'volume': [1000, 1100, 5000, 5200]
        ... })
        >>>
        >>> # Scale each ticker independently
        >>> scaler = GroupByScaler(by='tic', scaler=StandardScaler)
        >>> scaled_df = scaler.fit_transform(df)
        >>>
        >>> # BTC and ETH are normalized separately
        >>> print(scaled_df)

    Attributes:
        by (str): Column name to group by (usually 'tic' for ticker)
        scaler (class): Sklearn scaler class (StandardScaler, MinMaxScaler, etc.)
        columns (list): Columns to scale (None = all numeric)
        scaler_kwargs (dict): Arguments to pass to scaler
        scalers (dict): Fitted scalers for each group
    """

    def __init__(
        self,
        by: str,
        scaler=MaxAbsScaler,
        columns=None,
        scaler_kwargs=None
    ):
        """
        Initialize GroupByScaler.

        Args:
            by: Name of column to group by (e.g., 'tic')
            scaler: Scikit-learn scaler class (default: MaxAbsScaler)
            columns: List of columns to scale (default: all numeric)
            scaler_kwargs: Keyword arguments for scaler (default: None)
        """
        self.scalers = {}  # Dictionary of fitted scalers per group
        self.by = by
        self.scaler = scaler
        self.columns = columns
        self.scaler_kwargs = {} if scaler_kwargs is None else scaler_kwargs

    def fit(self, X, y=None):
        """
        Fit scaler independently for each group.

        Args:
            X: DataFrame to fit
            y: Not used (for sklearn compatibility)

        Returns:
            self: Fitted scaler
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("GroupByScaler requires a pandas DataFrame")

        # If columns not specified, use all numeric columns (excluding groupby column)
        if self.columns is None:
            self.columns = X.select_dtypes(exclude=["object"]).columns.tolist()
            if self.by in self.columns:
                self.columns.remove(self.by)

        # Fit one scaler for each group
        for group_value in X[self.by].unique():
            X_group = X.loc[X[self.by] == group_value, self.columns]

            # Create and fit scaler for this group
            self.scalers[group_value] = self.scaler(**self.scaler_kwargs).fit(X_group)

        return self

    def transform(self, X, y=None):
        """
        Transform data using fitted scalers.

        Args:
            X: DataFrame to transform
            y: Not used (for sklearn compatibility)

        Returns:
            Transformed DataFrame (copy)
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("GroupByScaler requires a pandas DataFrame")

        # Create copy to avoid modifying original
        X_transformed = X.copy()

        # Apply scaler for each group
        for group_value in X[self.by].unique():
            if group_value not in self.scalers:
                raise ValueError(
                    f"Group '{group_value}' was not seen during fit(). "
                    f"Available groups: {list(self.scalers.keys())}"
                )

            select_mask = X[self.by] == group_value
            X_transformed.loc[select_mask, self.columns] = self.scalers[group_value].transform(
                X.loc[select_mask, self.columns]
            )

        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.

        Args:
            X: DataFrame to fit and transform
            y: Not used (for sklearn compatibility)

        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X):
        """
        Reverse the scaling transformation.

        Args:
            X: Scaled DataFrame

        Returns:
            Original-scale DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("GroupByScaler requires a pandas DataFrame")

        X_inv = X.copy()

        for group_value in X[self.by].unique():
            if group_value not in self.scalers:
                raise ValueError(
                    f"Group '{group_value}' was not seen during fit(). "
                    f"Available groups: {list(self.scalers.keys())}"
                )

            select_mask = X[self.by] == group_value
            X_inv.loc[select_mask, self.columns] = self.scalers[group_value].inverse_transform(
                X.loc[select_mask, self.columns]
            )

        return X_inv


def data_split(df, start, end, target_date_col="date"):
    """
    Split dataset by date range.

    Args:
        df: DataFrame with time series data
        start: Start date (str or datetime)
        end: End date (str or datetime)
        target_date_col: Name of date column (default: "date")

    Returns:
        Filtered and sorted DataFrame
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)].copy()
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


# Example usage and tests
if __name__ == "__main__":
    print("GroupByScaler - Example Usage")
    print("=" * 60)

    # Create sample multi-asset data
    df = pd.DataFrame({
        'tic': ['BTC'] * 5 + ['ETH'] * 5 + ['SOL'] * 5,
        'date': pd.date_range('2024-01-01', periods=5).tolist() * 3,
        'close': [40000, 41000, 42000, 41500, 43000,  # BTC
                  2500, 2600, 2700, 2650, 2750,      # ETH
                  100, 105, 110, 108, 112],          # SOL
        'volume': [1000, 1100, 1050, 1200, 1150,
                   5000, 5200, 5100, 5300, 5250,
                   10000, 10500, 10200, 10800, 10600]
    })

    print("\nOriginal Data (3 tickers):")
    print(df.head(10))

    print("\n" + "=" * 60)
    print("Scaling with GroupByScaler (StandardScaler per ticker)")
    print("=" * 60)

    # Create and fit scaler
    scaler = GroupByScaler(
        by='tic',
        scaler=StandardScaler,
        columns=['close', 'volume']
    )

    # Fit and transform
    df_scaled = scaler.fit_transform(df)

    print("\nScaled Data:")
    print(df_scaled.head(10))

    print("\n" + "=" * 60)
    print("Inverse Transform (back to original scale)")
    print("=" * 60)

    df_inv = scaler.inverse_transform(df_scaled)
    print(df_inv.head(10))

    # Verify inverse is correct
    close_diff = (df['close'] - df_inv['close']).abs().max()
    vol_diff = (df['volume'] - df_inv['volume']).abs().max()

    print(f"\nMax difference after inverse transform:")
    print(f"  Close: {close_diff:.2e} (should be ~0)")
    print(f"  Volume: {vol_diff:.2e} (should be ~0)")

    if close_diff < 1e-10 and vol_diff < 1e-10:
        print("\n✓ Inverse transform verified!")
    else:
        print("\n✗ Inverse transform has errors")

    print("\n" + "=" * 60)
    print("Why GroupByScaler is Important:")
    print("=" * 60)
    print("""
    Without GroupByScaler:
    - BTC ($40,000) dominates ETH ($2,500) and SOL ($100)
    - Model focuses on BTC, ignores smaller coins
    - Poor diversification, missed opportunities

    With GroupByScaler:
    - Each ticker normalized independently
    - BTC, ETH, SOL treated equally in feature space
    - Model learns patterns from all assets
    - Better portfolio allocation
    """)
