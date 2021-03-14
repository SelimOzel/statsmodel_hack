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