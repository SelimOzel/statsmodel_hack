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

def add_trend(x, trend="c", prepend=False, has_constant='skip'):
	"""
	Add a trend and/or constant to an array.

	Parameters
	----------
	x : array_like
		Original array of data.
	trend : str {'n', 'c', 't', 'ct', 'ctt'}
		The trend to add.

		* 'n' add no trend.
		* 'c' add constant only.
		* 't' add trend only.
		* 'ct' add constant and linear trend.
		* 'ctt' add constant and linear and quadratic trend.
	prepend : bool
		If True, prepends the new data to the columns of X.
	has_constant : str {'raise', 'add', 'skip'}
		Controls what happens when trend is 'c' and a constant column already
		exists in x. 'raise' will raise an error. 'add' will add a column of
		1s. 'skip' will return the data without change. 'skip' is the default.

	Returns
	-------
	array_like
		The original data with the additional trend columns.  If x is a
		recarray or pandas Series or DataFrame, then the trend column names
		are 'const', 'trend' and 'trend_squared'.

	See Also
	--------
	statsmodels.tools.tools.add_constant
		Add a constant column to an array.

	Notes
	-----
	Returns columns as ['ctt','ct','c'] whenever applicable. There is currently
	no checking for an existing trend.
	"""
	prepend = bool_like(prepend, 'prepend')
	trend = string_like(trend, 'trend', options=('n', 'c', 't', 'ct', 'ctt'))
	has_constant = string_like(has_constant, 'has_constant',
							   options=('raise', 'add', 'skip'))

	# TODO: could be generalized for trend of aribitrary order
	columns = ['const', 'trend', 'trend_squared']
	if trend == 'n':
		return x.copy()
	elif trend == "c":  # handles structured arrays
		columns = columns[:1]
		trendorder = 0
	elif trend == "ct" or trend == "t":
		columns = columns[:2]
		if trend == "t":
			columns = columns[1:2]
		trendorder = 1
	elif trend == "ctt":
		trendorder = 2

	#is_recarray = _is_recarray(x)
	#is_pandas = _is_using_pandas(x, None) or is_recarray
	'''
	if is_pandas or is_recarray:
		if is_recarray:
			# deprecated: remove recarray support after 0.12
			import warnings
			from statsmodels.tools.sm_exceptions import recarray_warning
			warnings.warn(recarray_warning, FutureWarning)

			descr = x.dtype.descr
			x = pd.DataFrame.from_records(x)
		elif isinstance(x, pd.Series):
			x = pd.DataFrame(x)
		else:
			x = x.copy()
	'''
	#else:
	x = np.asanyarray(x)

	nobs = len(x)
	trendarr = np.vander(np.arange(1, nobs + 1, dtype=np.float64), trendorder + 1)
	# put in order ctt
	trendarr = np.fliplr(trendarr)
	if trend == "t":
		trendarr = trendarr[:, 1]

	if "c" in trend:
		'''
		if is_pandas or is_recarray:
			# Mixed type protection
			def safe_is_const(s):
				try:
					return np.ptp(s) == 0.0 and np.any(s != 0.0)
				except:
					return False
			col_const = x.apply(safe_is_const, 0)
		'''
		#else:
		ptp0 = np.ptp(np.asanyarray(x), axis=0)
		col_is_const = ptp0 == 0
		nz_const = col_is_const & (x[0] != 0)
		col_const = nz_const

		if np.any(col_const):
			if has_constant == 'raise':
				if x.ndim == 1:
					base_err = "x is constant."
				else:
					columns = np.arange(x.shape[1])[col_const]
					if isinstance(x, pd.DataFrame):
						columns = x.columns
					const_cols = ", ".join([str(c) for c in columns])
					base_err = (
						"x contains one or more constant columns. Column(s) "
						f"{const_cols} are constant."
					)
				msg = (
					f"{base_err} Adding a constant with trend='{trend}' is not allowed."
				)
				raise ValueError(msg)
			elif has_constant == 'skip':
				columns = columns[1:]
				trendarr = trendarr[:, 1:]

	order = 1 if prepend else -1
	'''
	if is_recarray or is_pandas:
		trendarr = pd.DataFrame(trendarr, index=x.index, columns=columns)
		x = [trendarr, x]
		x = pd.concat(x[::order], 1)
	'''
	#else:
	x = [trendarr, x]
	x = np.column_stack(x[::order])

	'''
	if is_recarray:
		x = x.to_records(index=False)
		new_descr = x.dtype.descr
		extra_col = len(new_descr) - len(descr)
		if prepend:
			descr = new_descr[:extra_col] + descr
		else:
			descr = descr + new_descr[-extra_col:]

		x = x.astype(np.dtype(descr))
	'''

	return x

def _autolag(
    mod,
    endog,
    exog,
    startlag,
    maxlag,
    method,
    modargs=(),
    fitargs=(),
    regresults=False,
):
    """
    Returns the results for the lag length that maximizes the info criterion.

    Parameters
    ----------
    mod : Model class
        Model estimator class
    endog : array_like
        nobs array containing endogenous variable
    exog : array_like
        nobs by (startlag + maxlag) array containing lags and possibly other
        variables
    startlag : int
        The first zero-indexed column to hold a lag.  See Notes.
    maxlag : int
        The highest lag order for lag length selection.
    method : {"aic", "bic", "t-stat"}
        aic - Akaike Information Criterion
        bic - Bayes Information Criterion
        t-stat - Based on last lag
    modargs : tuple, optional
        args to pass to model.  See notes.
    fitargs : tuple, optional
        args to pass to fit.  See notes.
    regresults : bool, optional
        Flag indicating to return optional return results

    Returns
    -------
    icbest : float
        Best information criteria.
    bestlag : int
        The lag length that maximizes the information criterion.
    results : dict, optional
        Dictionary containing all estimation results

    Notes
    -----
    Does estimation like mod(endog, exog[:,:i], *modargs).fit(*fitargs)
    where i goes from lagstart to lagstart+maxlag+1.  Therefore, lags are
    assumed to be in contiguous columns from low to high lag length with
    the highest lag in the last column.
    """
    # TODO: can tcol be replaced by maxlag + 2?
    # TODO: This could be changed to laggedRHS and exog keyword arguments if
    #    this will be more general.

    results = {}
    method = method.lower()
    for lag in range(startlag, startlag + maxlag + 1):
        mod_instance = mod(endog, exog[:, :lag], *modargs)
        results[lag] = mod_instance.fit()

    if method == "aic":
        icbest, bestlag = min((v.aic, k) for k, v in results.items())
    elif method == "bic":
        icbest, bestlag = min((v.bic, k) for k, v in results.items())
    elif method == "t-stat":
        # stop = stats.norm.ppf(.95)
        stop = 1.6448536269514722
        # Default values to ensure that always set
        bestlag = startlag + maxlag
        icbest = 0.0
        for lag in range(startlag + maxlag, startlag - 1, -1):
            icbest = np.abs(results[lag].tvalues[-1])
            bestlag = lag
            if np.abs(icbest) >= stop:
                # Break for first lag with a significant t-stat
                break
    else:
        raise ValueError(f"Information Criterion {method} not understood.")

    if not regresults:
        return icbest, bestlag
    else:
        return icbest, bestlag, results


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

_model_params_doc = """Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    exog : array_like
        A nobs x k array where `nobs` is the number of observations and `k`
        is the number of regressors. An intercept is not included by default
        and should be added by the user. See
        :func:`statsmodels.tools.add_constant`."""

_missing_param_doc = """\
missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'."""
_extra_param_doc = """
    hasconst : None or bool
        Indicates whether the RHS includes a user-supplied constant. If True,
        a constant is not checked for and k_constant is set to 1 and all
        result statistics are calculated as if a constant is present. If
        False, a constant is not checked for and k_constant is set to 0.
    **kwargs
        Extra arguments that are used to set model properties when using the
        formula interface."""

class Model(object):
    __doc__ = """
    A (predictive) statistical model. Intended to be subclassed not used.

    %(params_doc)s
    %(extra_params_doc)s

    Attributes
    ----------
    exog_names
    endog_names

    Notes
    -----
    `endog` and `exog` are references to any data provided.  So if the data is
    already stored in numpy arrays and it is changed then `endog` and `exog`
    will change as well.
    """ % {'params_doc': _model_params_doc,
           'extra_params_doc': _missing_param_doc + _extra_param_doc}

    # Maximum number of endogenous variables when using a formula
    # Default is 1, which is more common. Override in models when needed
    # Set to None to skip check
    _formula_max_endog = 1

    def __init__(self, endog, exog=None, **kwargs):
        missing = kwargs.pop('missing', 'none')
        hasconst = kwargs.pop('hasconst', None)
        self.data = self._handle_data(endog, exog, missing, hasconst,
                                      **kwargs)
        self.k_constant = self.data.k_constant
        self.exog = self.data.exog
        self.endog = self.data.endog
        self._data_attr = []
        self._data_attr.extend(['exog', 'endog', 'data.exog', 'data.endog'])
        if 'formula' not in kwargs:  # will not be able to unpickle without these
            self._data_attr.extend(['data.orig_endog', 'data.orig_exog'])
        # store keys for extras if we need to recreate model instance
        # we do not need 'missing', maybe we need 'hasconst'
        self._init_keys = list(kwargs.keys())
        if hasconst is not None:
            self._init_keys.append('hasconst')

    def _get_init_kwds(self):
        """return dictionary with extra keys used in model.__init__
        """
        kwds = dict(((key, getattr(self, key, None))
                     for key in self._init_keys))

        return kwds

    def _handle_data(self, endog, exog, missing, hasconst, **kwargs):
        data = handle_data(endog, exog, missing, hasconst, **kwargs)
        # kwargs arrays could have changed, easier to just attach here
        for key in kwargs:
            if key in ['design_info', 'formula']:  # leave attached to data
                continue
            # pop so we do not start keeping all these twice or references
            try:
                setattr(self, key, data.__dict__.pop(key))
            except KeyError:  # panel already pops keys in data handling
                pass
        return data

    @classmethod
    def from_formula(cls, formula, data, subset=None, drop_cols=None,
                     *args, **kwargs):
        """
        Create a Model from a formula and dataframe.

        Parameters
        ----------
        formula : str or generic Formula object
            The formula specifying the model.
        data : array_like
            The data for the model. See Notes.
        subset : array_like
            An array-like object of booleans, integers, or index values that
            indicate the subset of df to use in the model. Assumes df is a
            `pandas.DataFrame`.
        drop_cols : array_like
            Columns to drop from the design matrix.  Cannot be used to
            drop terms involving categoricals.
        *args
            Additional positional argument that are passed to the model.
        **kwargs
            These are passed to the model with one exception. The
            ``eval_env`` keyword is passed to patsy. It can be either a
            :class:`patsy:patsy.EvalEnvironment` object or an integer
            indicating the depth of the namespace to use. For example, the
            default ``eval_env=0`` uses the calling namespace. If you wish
            to use a "clean" environment set ``eval_env=-1``.

        Returns
        -------
        model
            The model instance.

        Notes
        -----
        data must define __getitem__ with the keys in the formula terms
        args and kwargs are passed on to the model instantiation. E.g.,
        a numpy structured or rec array, a dictionary, or a pandas DataFrame.
        """
        # TODO: provide a docs template for args/kwargs from child models
        # TODO: subset could use syntax. issue #469.
        if subset is not None:
            data = data.loc[subset]
        eval_env = kwargs.pop('eval_env', None)
        if eval_env is None:
            eval_env = 2
        elif eval_env == -1:
            from patsy import EvalEnvironment
            eval_env = EvalEnvironment({})
        elif isinstance(eval_env, int):
            eval_env += 1  # we're going down the stack again
        missing = kwargs.get('missing', 'drop')
        if missing == 'none':  # with patsy it's drop or raise. let's raise.
            missing = 'raise'

        tmp = handle_formula_data(data, None, formula, depth=eval_env,
                                  missing=missing)
        ((endog, exog), missing_idx, design_info) = tmp
        max_endog = cls._formula_max_endog
        if (max_endog is not None and
                endog.ndim > 1 and endog.shape[1] > max_endog):
            raise ValueError('endog has evaluated to an array with multiple '
                             'columns that has shape {0}. This occurs when '
                             'the variable converted to endog is non-numeric'
                             ' (e.g., bool or str).'.format(endog.shape))
        if drop_cols is not None and len(drop_cols) > 0:
            cols = [x for x in exog.columns if x not in drop_cols]
            if len(cols) < len(exog.columns):
                exog = exog[cols]
                cols = list(design_info.term_names)
                for col in drop_cols:
                    try:
                        cols.remove(col)
                    except ValueError:
                        pass  # OK if not present
                design_info = design_info.subset(cols)

        kwargs.update({'missing_idx': missing_idx,
                       'missing': missing,
                       'formula': formula,  # attach formula for unpckling
                       'design_info': design_info})
        mod = cls(endog, exog, *args, **kwargs)
        mod.formula = formula

        # since we got a dataframe, attach the original
        mod.data.frame = data
        return mod

    @property
    def endog_names(self):
        """
        Names of endogenous variables.
        """
        return self.data.ynames

    @property
    def exog_names(self):
        """
        Names of exogenous variables.
        """
        return self.data.xnames

    def fit(self):
        """
        Fit a model to data.
        """
        raise NotImplementedError

    def predict(self, params, exog=None, *args, **kwargs):
        """
        After a model has been fit predict returns the fitted values.

        This is a placeholder intended to be overwritten by individual models.
        """
        raise NotImplementedError

class LikelihoodModel(Model):
    """
    Likelihood model is a subclass of Model.
    """

    def __init__(self, endog, exog=None, **kwargs):
        super(LikelihoodModel, self).__init__(endog, exog, **kwargs)
        self.initialize()

    def initialize(self):
        """
        Initialize (possibly re-initialize) a Model instance.

        For example, if the the design matrix of a linear model changes then
        initialized can be used to recompute values using the modified design
        matrix.
        """
        pass

    # TODO: if the intent is to re-initialize the model with new data then this
    # method needs to take inputs...

    def loglike(self, params):
        """
        Log-likelihood of model.

        Parameters
        ----------
        params : ndarray
            The model parameters used to compute the log-likelihood.

        Notes
        -----
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def score(self, params):
        """
        Score vector of model.

        The gradient of logL with respect to each parameter.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The score vector evaluated at the parameters.
        """
        raise NotImplementedError

    def information(self, params):
        """
        Fisher information matrix of model.

        Returns -1 * Hessian of the log-likelihood evaluated at params.

        Parameters
        ----------
        params : ndarray
            The model parameters.
        """
        raise NotImplementedError

    def hessian(self, params):
        """
        The Hessian matrix of the model.

        Parameters
        ----------
        params : ndarray
            The parameters to use when evaluating the Hessian.

        Returns
        -------
        ndarray
            The hessian evaluated at the parameters.
        """
        raise NotImplementedError

    def fit(self, start_params=None, method='newton', maxiter=100,
            full_output=True, disp=True, fargs=(), callback=None, retall=False,
            skip_hessian=False, **kwargs):
        """
        Fit method for likelihood based models

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson, 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver
            - 'minimize' for generic wrapper of scipy minimize (BFGS by default)

            The explicit arguments in `fit` are passed to the solver,
            with the exception of the basin-hopping solver. Each
            solver has several optional arguments that are not the same across
            solvers. See the notes section below (or scipy.optimize) for the
            available arguments and for the list of explicit arguments that the
            basin-hopping solver supports.
        maxiter : int, optional
            The maximum number of iterations to perform.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
        disp : bool, optional
            Set to True to print convergence messages.
        fargs : tuple, optional
            Extra arguments passed to the likelihood function, i.e.,
            loglike(x,*args)
        callback : callable callback(xk), optional
            Called after each iteration, as callback(xk), where xk is the
            current parameter vector.
        retall : bool, optional
            Set to True to return list of solutions at each iteration.
            Available in Results object's mle_retvals attribute.
        skip_hessian : bool, optional
            If False (default), then the negative inverse hessian is calculated
            after the optimization. If True, then the hessian will not be
            calculated. However, it will be available in methods that use the
            hessian in the optimization (currently only with `"newton"`).
        kwargs : keywords
            All kwargs are passed to the chosen solver with one exception. The
            following keyword controls what happens after the fit::

                warn_convergence : bool, optional
                    If True, checks the model for the converged flag. If the
                    converged flag is False, a ConvergenceWarning is issued.

        Notes
        -----
        The 'basinhopping' solver ignores `maxiter`, `retall`, `full_output`
        explicit arguments.

        Optional arguments for solvers (see returned Results.mle_settings)::

            'newton'
                tol : float
                    Relative error in params acceptable for convergence.
            'nm' -- Nelder Mead
                xtol : float
                    Relative error in params acceptable for convergence
                ftol : float
                    Relative error in loglike(params) acceptable for
                    convergence
                maxfun : int
                    Maximum number of function evaluations to make.
            'bfgs'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.Inf is max, -np.Inf is min)
                epsilon
                    If fprime is approximated, use this value for the step
                    size. Only relevant if LikelihoodModel.score is None.
            'lbfgs'
                m : int
                    This many terms are used for the Hessian approximation.
                factr : float
                    A stop condition that is a variant of relative error.
                pgtol : float
                    A stop condition that uses the projected gradient.
                epsilon
                    If fprime is approximated, use this value for the step
                    size. Only relevant if LikelihoodModel.score is None.
                maxfun : int
                    Maximum number of function evaluations to make.
                bounds : sequence
                    (min, max) pairs for each element in x,
                    defining the bounds on that parameter.
                    Use None for one of min or max when there is no bound
                    in that direction.
            'cg'
                gtol : float
                    Stop when norm of gradient is less than gtol.
                norm : float
                    Order of norm (np.Inf is max, -np.Inf is min)
                epsilon : float
                    If fprime is approximated, use this value for the step
                    size. Can be scalar or vector.  Only relevant if
                    Likelihoodmodel.score is None.
            'ncg'
                fhess_p : callable f'(x,*args)
                    Function which computes the Hessian of f times an arbitrary
                    vector, p.  Should only be supplied if
                    LikelihoodModel.hessian is None.
                avextol : float
                    Stop when the average relative error in the minimizer
                    falls below this amount.
                epsilon : float or ndarray
                    If fhess is approximated, use this value for the step size.
                    Only relevant if Likelihoodmodel.hessian is None.
            'powell'
                xtol : float
                    Line-search error tolerance
                ftol : float
                    Relative error in loglike(params) for acceptable for
                    convergence.
                maxfun : int
                    Maximum number of function evaluations to make.
                start_direc : ndarray
                    Initial direction set.
            'basinhopping'
                niter : int
                    The number of basin hopping iterations.
                niter_success : int
                    Stop the run if the global minimum candidate remains the
                    same for this number of iterations.
                T : float
                    The "temperature" parameter for the accept or reject
                    criterion. Higher "temperatures" mean that larger jumps
                    in function value will be accepted. For best results
                    `T` should be comparable to the separation (in function
                    value) between local minima.
                stepsize : float
                    Initial step size for use in the random displacement.
                interval : int
                    The interval for how often to update the `stepsize`.
                minimizer : dict
                    Extra keyword arguments to be passed to the minimizer
                    `scipy.optimize.minimize()`, for example 'method' - the
                    minimization method (e.g. 'L-BFGS-B'), or 'tol' - the
                    tolerance for termination. Other arguments are mapped from
                    explicit argument of `fit`:
                      - `args` <- `fargs`
                      - `jac` <- `score`
                      - `hess` <- `hess`
            'minimize'
                min_method : str, optional
                    Name of minimization method to use.
                    Any method specific arguments can be passed directly.
                    For a list of methods and their arguments, see
                    documentation of `scipy.optimize.minimize`.
                    If no method is specified, then BFGS is used.
        """
        Hinv = None  # JP error if full_output=0, Hinv not defined

        if start_params is None:
            if hasattr(self, 'start_params'):
                start_params = self.start_params
            elif self.exog is not None:
                # fails for shape (K,)?
                start_params = [0] * self.exog.shape[1]
            else:
                raise ValueError("If exog is None, then start_params should "
                                 "be specified")

        # TODO: separate args from nonarg taking score and hessian, ie.,
        # user-supplied and numerically evaluated estimate frprime does not take
        # args in most (any?) of the optimize function

        nobs = self.endog.shape[0]
        # f = lambda params, *args: -self.loglike(params, *args) / nobs

        def f(params, *args):
            return -self.loglike(params, *args) / nobs

        if method == 'newton':
            # TODO: why are score and hess positive?
            def score(params, *args):
                return self.score(params, *args) / nobs

            def hess(params, *args):
                return self.hessian(params, *args) / nobs
        else:
            def score(params, *args):
                return -self.score(params, *args) / nobs

            def hess(params, *args):
                return -self.hessian(params, *args) / nobs

        warn_convergence = kwargs.pop('warn_convergence', True)
        optimizer = Optimizer()
        xopt, retvals, optim_settings = optimizer._fit(f, score, start_params,
                                                       fargs, kwargs,
                                                       hessian=hess,
                                                       method=method,
                                                       disp=disp,
                                                       maxiter=maxiter,
                                                       callback=callback,
                                                       retall=retall,
                                                       full_output=full_output)

        # NOTE: this is for fit_regularized and should be generalized
        cov_params_func = kwargs.setdefault('cov_params_func', None)
        if cov_params_func:
            Hinv = cov_params_func(self, xopt, retvals)
        elif method == 'newton' and full_output:
            Hinv = np.linalg.inv(-retvals['Hessian']) / nobs
        elif not skip_hessian:
            H = -1 * self.hessian(xopt)
            invertible = False
            if np.all(np.isfinite(H)):
                eigvals, eigvecs = np.linalg.eigh(H)
                if np.min(eigvals) > 0:
                    invertible = True

            if invertible:
                Hinv = eigvecs.dot(np.diag(1.0 / eigvals)).dot(eigvecs.T)
                Hinv = np.asfortranarray((Hinv + Hinv.T) / 2.0)
            else:
                warnings.warn('Inverting hessian failed, no bse or cov_params '
                              'available', HessianInversionWarning)
                Hinv = None

        if 'cov_type' in kwargs:
            cov_kwds = kwargs.get('cov_kwds', {})
            kwds = {'cov_type': kwargs['cov_type'], 'cov_kwds': cov_kwds}
        else:
            kwds = {}
        if 'use_t' in kwargs:
            kwds['use_t'] = kwargs['use_t']
        # TODO: add Hessian approximation and change the above if needed
        mlefit = LikelihoodModelResults(self, xopt, Hinv, scale=1., **kwds)

        # TODO: hardcode scale?
        mlefit.mle_retvals = retvals
        if isinstance(retvals, dict):
            if warn_convergence and not retvals['converged']:
                from statsmodels.tools.sm_exceptions import ConvergenceWarning
                warnings.warn("Maximum Likelihood optimization failed to "
                              "converge. Check mle_retvals",
                              ConvergenceWarning)

        mlefit.mle_settings = optim_settings
        return mlefit

class RegressionModel(LikelihoodModel):
    """
    Base class for linear regression models. Should not be directly called.

    Intended for subclassing.
    """
    def __init__(self, endog, exog, **kwargs):
        super(RegressionModel, self).__init__(endog, exog, **kwargs)
        self._data_attr.extend(['pinv_wexog', 'weights'])

    def initialize(self):
        """Initialize model components."""
        self.wexog = self.whiten(self.exog)
        self.wendog = self.whiten(self.endog)
        # overwrite nobs from class Model:
        self.nobs = float(self.wexog.shape[0])

        self._df_model = None
        self._df_resid = None
        self.rank = None

    @property
    def df_model(self):
        """
        The model degree of freedom.

        The dof is defined as the rank of the regressor matrix minus 1 if a
        constant is included.
        """
        if self._df_model is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_model = float(self.rank - self.k_constant)
        return self._df_model

    @df_model.setter
    def df_model(self, value):
        self._df_model = value

    @property
    def df_resid(self):
        """
        The residual degree of freedom.

        The dof is defined as the number of observations minus the rank of
        the regressor matrix.
        """

        if self._df_resid is None:
            if self.rank is None:
                self.rank = np.linalg.matrix_rank(self.exog)
            self._df_resid = self.nobs - self.rank
        return self._df_resid

    @df_resid.setter
    def df_resid(self, value):
        self._df_resid = value

    def whiten(self, x):
        """
        Whiten method that must be overwritten by individual models.

        Parameters
        ----------
        x : array_like
            Data to be whitened.
        """
        raise NotImplementedError("Subclasses must implement.")

    def fit(self, method="pinv", cov_type='nonrobust', cov_kwds=None,
            use_t=None, **kwargs):
        """
        Full fit of the model.

        The results include an estimate of covariance matrix, (whitened)
        residuals and an estimate of scale.

        Parameters
        ----------
        method : str, optional
            Can be "pinv", "qr".  "pinv" uses the Moore-Penrose pseudoinverse
            to solve the least squares problem. "qr" uses the QR
            factorization.
        cov_type : str, optional
            See `regression.linear_model.RegressionResults` for a description
            of the available covariance estimators.
        cov_kwds : list or None, optional
            See `linear_model.RegressionResults.get_robustcov_results` for a
            description required keywords for alternative covariance
            estimators.
        use_t : bool, optional
            Flag indicating to use the Student's t distribution when computing
            p-values.  Default behavior depends on cov_type. See
            `linear_model.RegressionResults.get_robustcov_results` for
            implementation details.
        **kwargs
            Additional keyword arguments that contain information used when
            constructing a model using the formula interface.

        Returns
        -------
        RegressionResults
            The model estimation results.

        See Also
        --------
        RegressionResults
            The results container.
        RegressionResults.get_robustcov_results
            A method to change the covariance estimator used when fitting the
            model.

        Notes
        -----
        The fit method uses the pseudoinverse of the design/exogenous variables
        to solve the least squares minimization.
        """
        if method == "pinv":
            if not (hasattr(self, 'pinv_wexog') and
                    hasattr(self, 'normalized_cov_params') and
                    hasattr(self, 'rank')):

                self.pinv_wexog, singular_values = pinv_extended(self.wexog)
                self.normalized_cov_params = np.dot(
                    self.pinv_wexog, np.transpose(self.pinv_wexog))

                # Cache these singular values for use later.
                self.wexog_singular_values = singular_values
                self.rank = np.linalg.matrix_rank(np.diag(singular_values))

            beta = np.dot(self.pinv_wexog, self.wendog)

        elif method == "qr":
            if not (hasattr(self, 'exog_Q') and
                    hasattr(self, 'exog_R') and
                    hasattr(self, 'normalized_cov_params') and
                    hasattr(self, 'rank')):
                Q, R = np.linalg.qr(self.wexog)
                self.exog_Q, self.exog_R = Q, R
                self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))

                # Cache singular values from R.
                self.wexog_singular_values = np.linalg.svd(R, 0, 0)
                self.rank = np.linalg.matrix_rank(R)
            else:
                Q, R = self.exog_Q, self.exog_R

            # used in ANOVA
            self.effects = effects = np.dot(Q.T, self.wendog)
            beta = np.linalg.solve(R, effects)
        else:
            raise ValueError('method has to be "pinv" or "qr"')

        if self._df_model is None:
            self._df_model = float(self.rank - self.k_constant)
        if self._df_resid is None:
            self.df_resid = self.nobs - self.rank

        if isinstance(self, OLS):
            lfit = OLSResults(
                self, beta,
                normalized_cov_params=self.normalized_cov_params,
                cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
        else:
            lfit = RegressionResults(
                self, beta,
                normalized_cov_params=self.normalized_cov_params,
                cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
                **kwargs)
        return RegressionResultsWrapper(lfit)

class WLS(RegressionModel):
    __doc__ = """
    Weighted Least Squares

    The weights are presumed to be (proportional to) the inverse of
    the variance of the observations.  That is, if the variables are
    to be transformed by 1/sqrt(W) you must supply weights = 1/W.

    %(params)s
    weights : array_like, optional
        A 1d array of weights.  If you supply 1/W then the variables are
        pre- multiplied by 1/sqrt(W).  If no weights are supplied the
        default value is 1 and WLS results are the same as OLS.
    %(extra_params)s

    Attributes
    ----------
    weights : ndarray
        The stored weights supplied as an argument.

    See Also
    --------
    GLS : Fit a linear model using Generalized Least Squares.
    OLS : Fit a linear model using Ordinary Least Squares.

    Notes
    -----
    If the weights are a function of the data, then the post estimation
    statistics such as fvalue and mse_model might not be correct, as the
    package does not yet support no-constant regression.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> Y = [1,3,4,5,2,3,4]
    >>> X = range(1,8)
    >>> X = sm.add_constant(X)
    >>> wls_model = sm.WLS(Y,X, weights=list(range(1,8)))
    >>> results = wls_model.fit()
    >>> results.params
    array([ 2.91666667,  0.0952381 ])
    >>> results.tvalues
    array([ 2.0652652 ,  0.35684428])
    >>> print(results.t_test([1, 0]))
    <T test: effect=array([ 2.91666667]), sd=array([[ 1.41224801]]), t=array([[ 2.0652652]]), p=array([[ 0.04690139]]), df_denom=5>
    >>> print(results.f_test([0, 1]))
    <F test: F=array([[ 0.12733784]]), p=[[ 0.73577409]], df_denom=5, df_num=1>
    """ 

    def __init__(self, endog, exog, weights=1., missing='none', hasconst=None,
                 **kwargs):
        weights = np.array(weights)
        if weights.shape == ():
            if (missing == 'drop' and 'missing_idx' in kwargs and
                    kwargs['missing_idx'] is not None):
                # patsy may have truncated endog
                weights = np.repeat(weights, len(kwargs['missing_idx']))
            else:
                weights = np.repeat(weights, len(endog))
        # handle case that endog might be of len == 1
        if len(weights) == 1:
            weights = np.array([weights.squeeze()])
        else:
            weights = weights.squeeze()
        super(WLS, self).__init__(endog, exog, missing=missing,
                                  weights=weights, hasconst=hasconst, **kwargs)
        nobs = self.exog.shape[0]
        weights = self.weights
        # Experimental normalization of weights
        weights = weights / np.sum(weights) * nobs
        if weights.size != nobs and weights.shape[0] != nobs:
            raise ValueError('Weights must be scalar or same length as design')

    def whiten(self, x):
        """
        Whitener for WLS model, multiplies each column by sqrt(self.weights).

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        array_like
            The whitened values sqrt(weights)*X.
        """

        x = np.asarray(x)
        if x.ndim == 1:
            return x * np.sqrt(self.weights)
        elif x.ndim == 2:
            return np.sqrt(self.weights)[:, None] * x

    def loglike(self, params):
        r"""
        Compute the value of the gaussian log-likelihood function at params.

        Given the whitened design matrix, the log-likelihood is evaluated
        at the parameter vector `params` for the dependent variable `Y`.

        Parameters
        ----------
        params : array_like
            The parameter estimates.

        Returns
        -------
        float
            The value of the log-likelihood function for a WLS Model.

        Notes
        --------
        .. math:: -\frac{n}{2}\log SSR
                  -\frac{n}{2}\left(1+\log\left(\frac{2\pi}{n}\right)\right)
                  -\frac{1}{2}\log\left(\left|W\right|\right)

        where :math:`W` is a diagonal weight matrix matrix and
        :math:`SSR=\left(Y-\hat{Y}\right)^\prime W \left(Y-\hat{Y}\right)` is
        the sum of the squared weighted residuals.
        """
        nobs2 = self.nobs / 2.0
        SSR = np.sum((self.wendog - np.dot(self.wexog, params))**2, axis=0)
        llf = -np.log(SSR) * nobs2      # concentrated likelihood
        llf -= (1+np.log(np.pi/nobs2))*nobs2  # with constant
        llf += 0.5 * np.sum(np.log(self.weights))
        return llf

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Compute the weights for calculating the Hessian.

        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """

        return self.weights

    def fit_regularized(self, method="elastic_net", alpha=0.,
                        L1_wt=1., start_params=None, profile_scale=False,
                        refit=False, **kwargs):
        # Docstring attached below
        if not np.isscalar(alpha):
            alpha = np.asarray(alpha)
        # Need to adjust since RSS/n in elastic net uses nominal n in
        # denominator
        alpha = alpha * np.sum(self.weights) / len(self.weights)

        rslt = OLS(self.wendog, self.wexog).fit_regularized(
            method=method, alpha=alpha,
            L1_wt=L1_wt,
            start_params=start_params,
            profile_scale=profile_scale,
            refit=refit, **kwargs)

        from statsmodels.base.elastic_net import (
            RegularizedResults, RegularizedResultsWrapper)
        rrslt = RegularizedResults(self, rslt.params)
        return RegularizedResultsWrapper(rrslt)


# Obtained from statsmodels.regression.linear_model 
class OLS(WLS):
    __doc__ = """
    Ordinary Least Squares

    %(params)s
    %(extra_params)s

    Attributes
    ----------
    weights : scalar
        Has an attribute weights = array(1.0) due to inheritance from WLS.

    See Also
    --------
    WLS : Fit a linear model using Weighted Least Squares.
    GLS : Fit a linear model using Generalized Least Squares.

    Notes
    -----
    No constant is added by the model unless you are using formulas.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> import numpy as np
    >>> duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
    >>> Y = duncan_prestige.data['income']
    >>> X = duncan_prestige.data['education']
    >>> X = sm.add_constant(X)
    >>> model = sm.OLS(Y,X)
    >>> results = model.fit()
    >>> results.params
    const        10.603498
    education     0.594859
    dtype: float64

    >>> results.tvalues
    const        2.039813
    education    6.892802
    dtype: float64

    >>> print(results.t_test([1, 0]))
                                 Test for Constraints
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    c0            10.6035      5.198      2.040      0.048       0.120      21.087
    ==============================================================================

    >>> print(results.f_test(np.identity(2)))
    <F test: F=array([[159.63031026]]), p=1.2607168903696672e-20, df_denom=43, df_num=2>
    """

    def __init__(self, endog, exog=None, missing='none', hasconst=None,
                 **kwargs):
        super(OLS, self).__init__(endog, exog, missing=missing,
                                  hasconst=hasconst, **kwargs)
        if "weights" in self._init_keys:
            self._init_keys.remove("weights")

    def loglike(self, params, scale=None):
        """
        The likelihood function for the OLS model.

        Parameters
        ----------
        params : array_like
            The coefficients with which to estimate the log-likelihood.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        float
            The likelihood function evaluated at params.
        """
        nobs2 = self.nobs / 2.0
        nobs = float(self.nobs)
        resid = self.endog - np.dot(self.exog, params)
        if hasattr(self, 'offset'):
            resid -= self.offset
        ssr = np.sum(resid**2)
        if scale is None:
            # profile log likelihood
            llf = -nobs2*np.log(2*np.pi) - nobs2*np.log(ssr / nobs) - nobs2
        else:
            # log-likelihood
            llf = -nobs2 * np.log(2 * np.pi * scale) - ssr / (2*scale)
        return llf

    def whiten(self, x):
        """
        OLS model whitener does nothing.

        Parameters
        ----------
        x : array_like
            Data to be whitened.

        Returns
        -------
        array_like
            The input array unmodified.

        See Also
        --------
        OLS : Fit a linear model using Ordinary Least Squares.
        """
        return x

    def score(self, params, scale=None):
        """
        Evaluate the score function at a given point.

        The score corresponds to the profile (concentrated)
        log-likelihood in which the scale parameter has been profiled
        out.

        Parameters
        ----------
        params : array_like
            The parameter vector at which the score function is
            computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        ndarray
            The score vector.
        """

        if not hasattr(self, "_wexog_xprod"):
            self._setup_score_hess()

        xtxb = np.dot(self._wexog_xprod, params)
        sdr = -self._wexog_x_wendog + xtxb

        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T,
                                                  params)
            ssr += np.dot(params, xtxb)
            return -self.nobs * sdr / ssr
        else:
            return -sdr / scale

    def _setup_score_hess(self):
        y = self.wendog
        if hasattr(self, 'offset'):
            y = y - self.offset
        self._wendog_xprod = np.sum(y * y)
        self._wexog_xprod = np.dot(self.wexog.T, self.wexog)
        self._wexog_x_wendog = np.dot(self.wexog.T, y)

    def hessian(self, params, scale=None):
        """
        Evaluate the Hessian function at a given point.

        Parameters
        ----------
        params : array_like
            The parameter vector at which the Hessian is computed.
        scale : float or None
            If None, return the profile (concentrated) log likelihood
            (profiled over the scale parameter), else return the
            log-likelihood using the given scale value.

        Returns
        -------
        ndarray
            The Hessian matrix.
        """

        if not hasattr(self, "_wexog_xprod"):
            self._setup_score_hess()

        xtxb = np.dot(self._wexog_xprod, params)

        if scale is None:
            ssr = self._wendog_xprod - 2 * np.dot(self._wexog_x_wendog.T,
                                                  params)
            ssr += np.dot(params, xtxb)
            ssrp = -2*self._wexog_x_wendog + 2*xtxb
            hm = self._wexog_xprod / ssr - np.outer(ssrp, ssrp) / ssr**2
            return -self.nobs * hm / 2
        else:
            return -self._wexog_xprod / scale

    def hessian_factor(self, params, scale=None, observed=True):
        """
        Calculate the weights for the Hessian.

        Parameters
        ----------
        params : ndarray
            The parameter at which Hessian is evaluated.
        scale : None or float
            If scale is None, then the default scale will be calculated.
            Default scale is defined by `self.scaletype` and set in fit.
            If scale is not None, then it is used as a fixed scale.
        observed : bool
            If True, then the observed Hessian is returned. If false then the
            expected information matrix is returned.

        Returns
        -------
        ndarray
            A 1d weight vector used in the calculation of the Hessian.
            The hessian is obtained by `(exog.T * hessian_factor).dot(exog)`.
        """

        return np.ones(self.exog.shape[0])

    def fit_regularized(self, method="elastic_net", alpha=0.,
                        L1_wt=1., start_params=None, profile_scale=False,
                        refit=False, **kwargs):

        # In the future we could add support for other penalties, e.g. SCAD.
        if method not in ("elastic_net", "sqrt_lasso"):
            msg = "Unknown method '%s' for fit_regularized" % method
            raise ValueError(msg)

        # Set default parameters.
        defaults = {"maxiter":  50, "cnvrg_tol": 1e-10,
                    "zero_tol": 1e-8}
        defaults.update(kwargs)

        if method == "sqrt_lasso":
            from statsmodels.base.elastic_net import (
                RegularizedResults, RegularizedResultsWrapper
            )
            params = self._sqrt_lasso(alpha, refit, defaults["zero_tol"])
            results = RegularizedResults(self, params)
            return RegularizedResultsWrapper(results)

        from statsmodels.base.elastic_net import fit_elasticnet

        if L1_wt == 0:
            return self._fit_ridge(alpha)

        # If a scale parameter is passed in, the non-profile
        # likelihood (residual sum of squares divided by -2) is used,
        # otherwise the profile likelihood is used.
        if profile_scale:
            loglike_kwds = {}
            score_kwds = {}
            hess_kwds = {}
        else:
            loglike_kwds = {"scale": 1}
            score_kwds = {"scale": 1}
            hess_kwds = {"scale": 1}

        return fit_elasticnet(self, method=method,
                              alpha=alpha,
                              L1_wt=L1_wt,
                              start_params=start_params,
                              loglike_kwds=loglike_kwds,
                              score_kwds=score_kwds,
                              hess_kwds=hess_kwds,
                              refit=refit,
                              check_step=False,
                              **defaults)

    def _sqrt_lasso(self, alpha, refit, zero_tol):

        try:
            import cvxopt
        except ImportError:
            msg = 'sqrt_lasso fitting requires the cvxopt module'
            raise ValueError(msg)

        n = len(self.endog)
        p = self.exog.shape[1]

        h0 = cvxopt.matrix(0., (2*p+1, 1))
        h1 = cvxopt.matrix(0., (n+1, 1))
        h1[1:, 0] = cvxopt.matrix(self.endog, (n, 1))

        G0 = cvxopt.spmatrix([], [], [], (2*p+1, 2*p+1))
        for i in range(1, 2*p+1):
            G0[i, i] = -1
        G1 = cvxopt.matrix(0., (n+1, 2*p+1))
        G1[0, 0] = -1
        G1[1:, 1:p+1] = self.exog
        G1[1:, p+1:] = -self.exog

        c = cvxopt.matrix(alpha / n, (2*p + 1, 1))
        c[0] = 1 / np.sqrt(n)

        from cvxopt import solvers
        solvers.options["show_progress"] = False

        rslt = solvers.socp(c, Gl=G0, hl=h0, Gq=[G1], hq=[h1])
        x = np.asarray(rslt['x']).flat
        bp = x[1:p+1]
        bn = x[p+1:]
        params = bp - bn

        if not refit:
            return params

        ii = np.flatnonzero(np.abs(params) > zero_tol)
        rfr = OLS(self.endog, self.exog[:, ii]).fit()
        params *= 0
        params[ii] = rfr.params

        return params

    def _fit_ridge(self, alpha):
        """
        Fit a linear model using ridge regression.

        Parameters
        ----------
        alpha : scalar or array_like
            The penalty weight.  If a scalar, the same penalty weight
            applies to all variables in the model.  If a vector, it
            must have the same length as `params`, and contains a
            penalty weight for each coefficient.

        Notes
        -----
        Equivalent to fit_regularized with L1_wt = 0 (but implemented
        more efficiently).
        """

        u, s, vt = np.linalg.svd(self.exog, 0)
        v = vt.T
        q = np.dot(u.T, self.endog) * s
        s2 = s * s
        if np.isscalar(alpha):
            sd = s2 + alpha * self.nobs
            params = q / sd
            params = np.dot(v, params)
        else:
            alpha = np.asarray(alpha)
            vtav = self.nobs * np.dot(vt, alpha[:, None] * v)
            d = np.diag(vtav) + s2
            np.fill_diagonal(vtav, d)
            r = np.linalg.solve(vtav, q)
            params = np.dot(v, r)

        from statsmodels.base.elastic_net import RegularizedResults
        return RegularizedResults(self, params)