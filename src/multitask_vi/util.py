"""Utility functions."""

from concurrent import futures
from tqdm import tqdm


def parallel_map(fn, iterable, n_jobs=1, **kwargs):
    """Runs jobs in parallel using ProcessPoolExecutor.

    Displays a progressbar using tqdm to indicate progress. If n_jobs == 1,
    jobs executed serially without using the executor to avoid overhead
    (and possibly simplify debugging).

    Parameters
    ----------
    fn : function
        Function to map.
    iterable : iterable
        Iterable to map over.
    n_jobs : int
        Number of workers to use.
    **kwargs
        Any kwargs are passed to fn.

    """

    if n_jobs > 1:
        with futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures_list = [executor.submit(fn, i, **kwargs) for i in iterable]

            for future in tqdm(
                    futures.as_completed(futures_list),
                    total=len(futures_list),
                    ncols=80):  # yapf: disable
                yield future.result()
    else:
        for item in tqdm(list(iterable), ncols=80):
            yield fn(item, **kwargs)
