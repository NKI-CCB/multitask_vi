import toolz
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressor_

from .util import calculate_proximity, calculate_perm_vi


class _RandomForestRegressorWrapper(BaseEstimator, RegressorMixin):

    _params = ()

    def __init__(self, **kwargs):
        """Base wrapper for a RandomForestRegressor class."""

        super().__init__()
        self._forest = RandomForestRegressor_(**kwargs)

    def fit(self, X, y, **kwargs):
        self._forest.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self._forest.predict(X)

    def __getattr__(self, attr):
        # Check if own attribute.
        if attr in self.__dict__:
            return getattr(self, attr)

        # Proxy to forest.
        return getattr(self._forest, attr)

    def get_params(self, deep=True):
        # Merge own parameters with the forests.
        forest_params = self._forest.get_params(deep=deep)
        own_params = {p: getattr(self, p) for p in self._params}
        return toolz.merge(forest_params, own_params)

    def set_params(self, **parameters):
        # Copy dict.
        parameters = dict(parameters)

        # Extract own parameters.
        for own_param in self._params:
            if own_param in parameters:
                self.setattr(own_param, parameters.pop(own_param))

        # Pass others to the forest.
        self._forest.set_params(parameters)

        return self


class CaseSpecificRandomForestRegressor(_RandomForestRegressorWrapper):

    _params = ('w', 'feature_importances')

    def __init__(self, w=10, feature_importances=False, **kwargs):
        """Case-specific RandomForestRegressor.

        Args:
            w (int): Min number of features in leaf node when calculating
                weights using the proximity measure.
            feature_importances (bool): Whether to calculate the feature
                importance metric.
            **kwargs: Passed to internal scikit-learn RandomForestRegressor.

        Returns:
            RandomForestRegressor: Basic model.

        """

        # Check unsupported arguments.
        if 'oob_score' in kwargs and kwargs['oob_score']:
            raise NotImplementedError('OOB score not yet implemented')

        super().__init__(**kwargs)

        self.w = w
        self.feature_importances = feature_importances

        # Predefine fit attribtutes.
        self.x0_ = None
        self.sampling_weight_ = None
        self.feature_importances_ = None

    def fit(self, X, y, x0=None, **kwargs):
        if x0 is None:
            raise ValueError('No x0 provided')

        # Set x0, wrapping in list if needed.
        self.x0_ = [x0] if isinstance(x0, int) else x0

        # Determine weights and fit forest.
        self._set_weights(X, y, self.x0_)
        super().fit(X, y, sampling_weight=self.sampling_weight_)

        # Set feature importances.
        if self.feature_importances:
            self._set_importances(X, y)

        return self

    def _set_weights(self, X, y, x0):
        # Fit model with same parameters.
        prox_params = toolz.merge(self._forest.get_params(),
                                  dict(min_samples_leaf=self.w))

        prox_model = RandomForestRegressor_(**prox_params)
        prox_model.fit(X=X, y=y)

        # Calculate proximity.
        proximity = calculate_proximity(prox_model, X=X)

        # Calculate weights.
        weights = np.mean(proximity[x0, :], axis=0)
        weights /= np.sum(weights)

        self.proximity_ = proximity
        self.proximity_forest_ = prox_model

        self.sampling_weight_ = weights

    def _set_importances(self, X, y):
        self.feature_importances_ = calculate_perm_vi(
            self._forest, X, y, sampling_weight=self.sampling_weight_)


class RandomForestRegressor(_RandomForestRegressorWrapper):
    _params = ('feature_importances',)

    def __init__(self, feature_importances=False, **kwargs):
        """RandomForestRegressor that uses a permuted feature importance.

        Args:
            feature_importances (bool): Whether to calculate the feature
                importance metric.
            **kwargs: Passed to internal scikit-learn RandomForestRegressor.

        Returns:
            RandomForestRegressor: Basic model.

        """

        super().__init__(**kwargs)
        self.feature_importances = feature_importances
        self.feature_importances_ = None

    def fit(self, X, y, **kwargs):
        self._forest.fit(X, y, **kwargs)

        if self.feature_importances:
            self._set_importances(X, y)

        return self

    def _set_importances(self, X, y):
        self.feature_importances_ = calculate_perm_vi(self._forest, X, y)
