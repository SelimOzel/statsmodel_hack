import numpy as np
from typing import Any, Optional

def dummy_function():
	print("Stat tools hack entered.")


# this needs to be converted to a class like HetGoldfeldQuandt,
# 3 different returns are a mess
# See:
# Ng and Perron(2001), Lag length selection and the construction of unit root
# tests with good size and power, Econometrica, Vol 69 (6) pp 1519-1554
# TODO: include drift keyword, only valid with regression == "c"
# just changes the distribution of the test statistic to a t distribution
# TODO: autolag is untested
def adfuller(
    x,
    maxlag=None,
    regression="c",
    autolag="AIC",
    store=False,
    regresults=False,
):
    """
    Augmented Dickey-Fuller unit root test.

    The Augmented Dickey-Fuller test can be used to test for a unit root in a
    univariate process in the presence of serial correlation.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    maxlag : int
        Maximum lag which is included in test, default 12*(nobs/100)^{1/4}.
    regression : {"c","ct","ctt","nc"}
        Constant and trend order to include in regression.

        * "c" : constant only (default).
        * "ct" : constant and trend.
        * "ctt" : constant, and linear and quadratic trend.
        * "nc" : no constant, no trend.

    autolag : {"AIC", "BIC", "t-stat", None}
        Method to use when automatically determining the lag length among the
        values 0, 1, ..., maxlag.

        * If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
        * If None, then the number of included lags is set to maxlag.
    store : bool
        If True, then a result instance is returned additionally to
        the adf statistic. Default is False.
    regresults : bool, optional
        If True, the full regression results are returned. Default is False.

    Returns
    -------
    adf : float
        The test statistic.
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010).
    usedlag : int
        The number of lags used.
    nobs : int
        The number of observations used for the ADF regression and calculation
        of the critical values.
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010).
    icbest : float
        The maximized information criterion if autolag is not None.
    resstore : ResultStore, optional
        A dummy class with results attached as attributes.

    Notes
    -----
    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then we cannot reject that there is a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon 1994, but using the updated 2010 tables. If the p-value is close
    to significant, then the critical values should be used to judge whether
    to reject the null.

    The autolag option and maxlag for it are described in Greene.

    References
    ----------
    .. [1] W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.

    .. [2] Hamilton, J.D.  "Time Series Analysis".  Princeton, 1994.

    .. [3] MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
        unit-root and cointegration tests.  `Journal of Business and Economic
        Statistics` 12, 167-76.

    .. [4] MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."  Queen"s
        University, Dept of Economics, Working Papers.  Available at
        http://ideas.repec.org/p/qed/wpaper/1227.html

    Examples
    --------
    See example notebook
    """
    x = array_like(x, "x")
    maxlag = int_like(maxlag, "maxlag", optional=True)
    regression = string_like(
        regression, "regression", options=("c", "ct", "ctt", "nc")
    )
    autolag = string_like(
        autolag, "autolag", optional=True, options=("aic", "bic", "t-stat")
    )
    store = bool_like(store, "store")
    regresults = bool_like(regresults, "regresults")

    if regresults:
        store = True

    trenddict = {None: "nc", 0: "c", 1: "ct", 2: "ctt"}
    if regression is None or isinstance(regression, int):
        regression = trenddict[regression]
    regression = regression.lower()
    nobs = x.shape[0]

    ntrend = len(regression) if regression != "nc" else 0
    if maxlag is None:
        # from Greene referencing Schwert 1989
        maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        # -1 for the diff
        maxlag = min(nobs // 2 - ntrend - 1, maxlag)
        if maxlag < 0:
            raise ValueError(
                "sample size is too short to use selected "
                "regression component"
            )
    elif maxlag > nobs // 2 - ntrend - 1:
        raise ValueError(
            "maxlag must be less than (nobs/2 - 1 - ntrend) "
            "where n trend is the number of included "
            "deterministic regressors"
        )
    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:, None], maxlag, trim="both", original="in")
    nobs = xdall.shape[0]

    xdall[:, 0] = x[-nobs - 1 : -1]  # replace 0 xdiff with level of x
    xdshort = xdiff[-nobs:]

    if store:
        from statsmodels.stats.diagnostic import ResultsStore

        resstore = ResultsStore()
    if autolag:
        if regression != "nc":
            fullRHS = add_trend(xdall, regression, prepend=True)
        else:
            fullRHS = xdall
        startlag = fullRHS.shape[1] - xdall.shape[1] + 1
        # 1 for level
        # search for lag length with smallest information criteria
        # Note: use the same number of observations to have comparable IC
        # aic and bic: smaller is better

        if not regresults:
            icbest, bestlag = _autolag(
                OLS, xdshort, fullRHS, startlag, maxlag, autolag
            )
        else:
            icbest, bestlag, alres = _autolag(
                OLS,
                xdshort,
                fullRHS,
                startlag,
                maxlag,
                autolag,
                regresults=regresults,
            )
            resstore.autolag_results = alres

        bestlag -= startlag  # convert to lag not column index

        # rerun ols with best autolag
        xdall = lagmat(xdiff[:, None], bestlag, trim="both", original="in")
        nobs = xdall.shape[0]
        xdall[:, 0] = x[-nobs - 1 : -1]  # replace 0 xdiff with level of x
        xdshort = xdiff[-nobs:]
        usedlag = bestlag
    else:
        usedlag = maxlag
        icbest = None
    if regression != "nc":
        resols = OLS(
            xdshort, add_trend(xdall[:, : usedlag + 1], regression)
        ).fit()
    else:
        resols = OLS(xdshort, xdall[:, : usedlag + 1]).fit()

    adfstat = resols.tvalues[0]
    #    adfstat = (resols.params[0]-1.0)/resols.bse[0]
    # the "asymptotically correct" z statistic is obtained as
    # nobs/(1-np.sum(resols.params[1:-(trendorder+1)])) (resols.params[0] - 1)
    # I think this is the statistic that is used for series that are integrated
    # for orders higher than I(1), ie., not ADF but cointegration tests.

    # Get approx p-value and critical values
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    critvalues = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    critvalues = {
        "1%": critvalues[0],
        "5%": critvalues[1],
        "10%": critvalues[2],
    }
    if store:
        resstore.resols = resols
        resstore.maxlag = maxlag
        resstore.usedlag = usedlag
        resstore.adfstat = adfstat
        resstore.critvalues = critvalues
        resstore.nobs = nobs
        resstore.H0 = (
            "The coefficient on the lagged level equals 1 - " "unit root"
        )
        resstore.HA = "The coefficient on the lagged level < 1 - stationary"
        resstore.icbest = icbest
        resstore._str = "Augmented Dickey-Fuller Test Results"
        return adfstat, pvalue, critvalues, resstore
    else:
        if not autolag:
            return adfstat, pvalue, usedlag, nobs, critvalues
        else:
            return adfstat, pvalue, usedlag, nobs, critvalues, icbest

def lagmat(x, maxlag, trim='forward', original='ex', use_pandas=False):
    """
    Create 2d array of lags.

    Parameters
    ----------
    x : array_like
        Data; if 2d, observation in rows and variables in columns.
    maxlag : int
        All lags from zero to maxlag are included.
    trim : {'forward', 'backward', 'both', 'none', None}
        The trimming method to use.

        * 'forward' : trim invalid observations in front.
        * 'backward' : trim invalid initial observations.
        * 'both' : trim invalid observations on both sides.
        * 'none', None : no trimming of observations.
    original : {'ex','sep','in'}
        How the original is treated.

        * 'ex' : drops the original array returning only the lagged values.
        * 'in' : returns the original array and the lagged values as a single
          array.
        * 'sep' : returns a tuple (original array, lagged values). The original
                  array is truncated to have the same number of rows as
                  the returned lagmat.
    use_pandas : bool
        If true, returns a DataFrame when the input is a pandas
        Series or DataFrame.  If false, return numpy ndarrays.

    Returns
    -------
    lagmat : ndarray
        The array with lagged observations.
    y : ndarray, optional
        Only returned if original == 'sep'.

    Notes
    -----
    When using a pandas DataFrame or Series with use_pandas=True, trim can only
    be 'forward' or 'both' since it is not possible to consistently extend
    index values.

    Examples
    --------
    >>> from statsmodels.tsa.tsatools import lagmat
    >>> import numpy as np
    >>> X = np.arange(1,7).reshape(-1,2)
    >>> lagmat(X, maxlag=2, trim="forward", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="backward", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])

    >>> lagmat(X, maxlag=2, trim="both", original='in')
    array([[ 5.,  6.,  3.,  4.,  1.,  2.]])

    >>> lagmat(X, maxlag=2, trim="none", original='in')
    array([[ 1.,  2.,  0.,  0.,  0.,  0.],
       [ 3.,  4.,  1.,  2.,  0.,  0.],
       [ 5.,  6.,  3.,  4.,  1.,  2.],
       [ 0.,  0.,  5.,  6.,  3.,  4.],
       [ 0.,  0.,  0.,  0.,  5.,  6.]])
    """
    maxlag = int_like(maxlag, 'maxlag')
    use_pandas = bool_like(use_pandas, 'use_pandas')
    trim = string_like(trim, 'trim', optional=True,
                       options=('forward', 'backward', 'both', 'none'))
    original = string_like(original, 'original', options=('ex', 'sep', 'in'))

    # TODO:  allow list of lags additional to maxlag
    orig = x
    x = array_like(x, 'x', ndim=2, dtype=None)
    #is_pandas = _is_using_pandas(orig, None) and use_pandas
    trim = 'none' if trim is None else trim
    trim = trim.lower()
    #if is_pandas and trim in ('none', 'backward'):
    #    raise ValueError("trim cannot be 'none' or 'forward' when used on "
    #                     "Series or DataFrames")

    dropidx = 0
    nobs, nvar = x.shape
    if original in ['ex', 'sep']:
        dropidx = nvar
    if maxlag >= nobs:
        raise ValueError("maxlag should be < nobs")
    lm = np.zeros((nobs + maxlag, nvar * (maxlag + 1)))
    for k in range(0, int(maxlag + 1)):
        lm[maxlag - k:nobs + maxlag - k,
        nvar * (maxlag - k):nvar * (maxlag - k + 1)] = x

    if trim in ('none', 'forward'):
        startobs = 0
    elif trim in ('backward', 'both'):
        startobs = maxlag
    else:
        raise ValueError('trim option not valid')

    if trim in ('none', 'backward'):
        stopobs = len(lm)
    else:
        stopobs = nobs
    '''
    if is_pandas:
        x = orig
        x_columns = x.columns if isinstance(x, DataFrame) else [x.name]
        columns = [str(col) for col in x_columns]
        for lag in range(maxlag):
            lag_str = str(lag + 1)
            columns.extend([str(col) + '.L.' + lag_str for col in x_columns])
        lm = DataFrame(lm[:stopobs], index=x.index, columns=columns)
        lags = lm.iloc[startobs:]
        if original in ('sep', 'ex'):
            leads = lags[x_columns]
            lags = lags.drop(x_columns, 1)
    '''
    #else:
    lags = lm[startobs:stopobs, dropidx:]
    if original == 'sep':
        leads = lm[startobs:stopobs, :dropidx]

    if original == 'sep':
        return lags, leads
    else:
        return lags

def int_like(
    value: Any, name: str, optional: bool = False, strict: bool = False
) -> Optional[int]:
    """
    Convert to int or raise if not int_like

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow int or np.integer that are not bool. If False,
        allow types that support integer division by 1 and conversion to int.

    Returns
    -------
    converted : int
        value converted to a int
    """
    if optional and value is None:
        return None
    is_bool_timedelta = isinstance(value, (bool, np.timedelta64))

    if hasattr(value, "squeeze") and callable(value.squeeze):
        value = value.squeeze()

    if isinstance(value, (int, np.integer)) and not is_bool_timedelta:
        return int(value)
    elif not strict and not is_bool_timedelta:
        try:
            if value == (value // 1):
                return int(value)
        except Exception:
            pass
    extra_text = " or None" if optional else ""
    raise TypeError(
        "{0} must be integer_like (int or np.integer, but not bool"
        " or timedelta64){1}".format(name, extra_text)
    )

# Obtained from statsmodels.tools.validation
def array_like(
    obj,
    name,
    dtype=np.double,
    ndim=1,
    maxdim=None,
    shape=None,
    order=None,
    contiguous=False,
    optional=False,
):
    """
    Convert array-like to a ndarray and check conditions

    Parameters
    ----------
    obj : array_like
         An array, any object exposing the array interface, an object whose
        __array__ method returns an array, or any (nested) sequence.
    name : str
        Name of the variable to use in exceptions
    dtype : {None, numpy.dtype, str}
        Required dtype. Default is double. If None, does not change the dtype
        of obj (if present) or uses NumPy to automatically detect the dtype
    ndim : {int, None}
        Required number of dimensions of obj. If None, no check is performed.
        If the numebr of dimensions of obj is less than ndim, additional axes
        are inserted on the right. See examples.
    maxdim : {int, None}
        Maximum allowed dimension.  Use ``maxdim`` instead of ``ndim`` when
        inputs are allowed to have ndim 1, 2, ..., or maxdim.
    shape : {tuple[int], None}
        Required shape obj.  If None, no check is performed. Partially
        restricted shapes can be checked using None. See examples.
    order : {'C', 'F', None}
        Order of the array
    contiguous : bool
        Ensure that the array's data is contiguous with order ``order``
    optional : bool
        Flag indicating whether None is allowed

    Returns
    -------
    ndarray
        The converted input.

    Examples
    --------
    Convert a list or pandas series to an array
    >>> import pandas as pd
    >>> x = [0, 1, 2, 3]
    >>> a = array_like(x, 'x', ndim=1)
    >>> a.shape
    (4,)

    >>> a = array_like(pd.Series(x), 'x', ndim=1)
    >>> a.shape
    (4,)

    >>> type(a.orig)
    pandas.core.series.Series

    Squeezes singleton dimensions when required
    >>> x = np.array(x).reshape((4, 1))
    >>> a = array_like(x, 'x', ndim=1)
    >>> a.shape
    (4,)

    Right-appends when required size is larger than actual
    >>> x = [0, 1, 2, 3]
    >>> a = array_like(x, 'x', ndim=2)
    >>> a.shape
    (4, 1)

    Check only the first and last dimension of the input
    >>> x = np.arange(4*10*4).reshape((4, 10, 4))
    >>> y = array_like(x, 'x', ndim=3, shape=(4, None, 4))

    Check only the first two dimensions
    >>> z = array_like(x, 'x', ndim=3, shape=(4, 10))

    Raises ValueError if constraints are not satisfied
    >>> z = array_like(x, 'x', ndim=2)
    Traceback (most recent call last):
     ...
    ValueError: x is required to have ndim 2 but has ndim 3

    >>> z = array_like(x, 'x', shape=(10, 4, 4))
    Traceback (most recent call last):
     ...
    ValueError: x is required to have shape (10, 4, 4) but has shape (4, 10, 4)

    >>> z = array_like(x, 'x', shape=(None, 4, 4))
    Traceback (most recent call last):
     ...
    ValueError: x is required to have shape (*, 4, 4) but has shape (4, 10, 4)
    """
    if optional and obj is None:
        return None
    arr = np.asarray(obj, dtype=dtype, order=order)
    if maxdim is not None:
        if arr.ndim > maxdim:
            msg = "{0} must have ndim <= {1}".format(name, maxdim)
            raise ValueError(msg)
    elif ndim is not None:
        if arr.ndim > ndim:
            arr = _right_squeeze(arr, stop_dim=ndim)
        elif arr.ndim < ndim:
            arr = np.reshape(arr, arr.shape + (1,) * (ndim - arr.ndim))
        if arr.ndim != ndim:
            msg = "{0} is required to have ndim {1} but has ndim {2}"
            raise ValueError(msg.format(name, ndim, arr.ndim))
    if shape is not None:
        for actual, req in zip(arr.shape, shape):
            if req is not None and actual != req:
                req_shape = str(shape).replace("None, ", "*, ")
                msg = "{0} is required to have shape {1} but has shape {2}"
                raise ValueError(msg.format(name, req_shape, arr.shape))
    if contiguous:
        arr = np.ascontiguousarray(arr, dtype=dtype)
    return arr

# Obtained from statsmodels.tools.validation
def string_like(value, name, optional=False, options=None, lower=True):
    """
    Check if object is string-like and raise if not

    Parameters
    ----------
    value : object
        Value to verify.
    name : str
        Variable name for exceptions.
    optional : bool
        Flag indicating whether None is allowed.
    options : tuple[str]
        Allowed values for input parameter `value`.
    lower : bool
        Convert all case-based characters in `value` into lowercase.

    Returns
    -------
    str
        The validated input

    Raises
    ------
    TypeError
        If the value is not a string or None when optional is True.
    ValueError
        If the input is not in ``options`` when ``options`` is set.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        extra_text = " or None" if optional else ""
        raise TypeError("{0} must be a string{1}".format(name, extra_text))
    if lower:
        value = value.lower()
    if options is not None and value not in options:
        extra_text = "If not None, " if optional else ""
        options_text = "'" + "', '".join(options) + "'"
        msg = "{0}{1} must be one of: {2}".format(
            extra_text, name, options_text
        )
        raise ValueError(msg)
    return value

# Obtained from statsmodels.tools.validation
def bool_like(value, name, optional=False, strict=False):
    """
    Convert to bool or raise if not bool_like

    Parameters
    ----------
    value : object
        Value to verify
    name : str
        Variable name for exceptions
    optional : bool
        Flag indicating whether None is allowed
    strict : bool
        If True, then only allow bool. If False, allow types that support
        casting to bool.

    Returns
    -------
    converted : bool
        value converted to a bool
    """
    if optional and value is None:
        return value
    extra_text = " or None" if optional else ""
    if strict:
        if isinstance(value, bool):
            return value
        else:
            raise TypeError("{0} must be a bool{1}".format(name, extra_text))

    if hasattr(value, "squeeze") and callable(value.squeeze):
        value = value.squeeze()
    try:
        return bool(value)
    except Exception:
        raise TypeError(
            "{0} must be a bool (or bool-compatible)"
            "{1}".format(name, extra_text)
        )

# Obtained from statsmodels.tools.data
#def _is_using_pandas(endog, exog):
#    from statsmodels.compat.pandas import data_klasses as klasses
#    return (isinstance(endog, klasses) or isinstance(exog, klasses))