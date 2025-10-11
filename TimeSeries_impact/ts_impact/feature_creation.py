
from pandas.tseries.holiday import USFederalHolidayCalendar


def add_day_of_week(df):
    """
    Add day of week as a feature to the dataframe.
    """
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek
    return df

def add_month_of_year(df):
    """
    Add month of year as a feature to the dataframe.
    """
    df = df.copy()
    df['month_of_year'] = df.index.month
    return df

def add_holiday_flags(df, country='US'):
    """
    Add holiday flags as features to the dataframe.
    """
    df = df.copy()
    if country != 'US':
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    else:
        raise NotImplementedError("Currently only US holidays are implemented")
    df['is_holiday'] = df.index.isin(holidays).astype(int)
    return df
