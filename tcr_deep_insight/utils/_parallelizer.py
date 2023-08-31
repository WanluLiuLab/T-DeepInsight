import os
from multiprocessing import Manager
from threading import Thread
from typing import (
    Any, Callable, Optional, Sequence, Union, Iterable
)
from joblib import delayed, Parallel
import numpy as np
from scipy.sparse import issparse


class Parallelizer:
    def __init__(self, n_jobs:int):
        self.n_jobs = self.get_n_jobs(n_jobs=n_jobs)
        self._msg_shown = False 
        
    def get_n_jobs(self, n_jobs):
        if n_jobs is None or (n_jobs < 0 and os.cpu_count() + 1 + n_jobs <= 0):
            return 1
        elif n_jobs > os.cpu_count():
            return os.cpu_count()
        elif n_jobs < 0:
            return os.cpu_count() + 1 + n_jobs
        else:
            return n_jobs

    @staticmethod
    def __update__(pbar, queue, n_total):
        n_finished = 0
        while n_finished < n_total:
            try:
                res = queue.get()
            except EOFError as e:
                if not n_finished != n_total:
                    raise RuntimeError(
                        f"Finished only `{n_finished} out of `{n_total}` tasks.`"
                    ) from e
                break
            assert res in (None, (1, None), 1)  # (None, 1) means only 1 job
            if res == (1, None):
                n_finished += 1
                if pbar is not None:
                    pbar.update()
            elif res is None:
                n_finished += 1
            elif pbar is not None:
                pbar.update()

        if pbar is not None:
            pbar.close()
        
    def parallelize(
        self,
        map_func: Callable[[Any], Any],
        map_data: Union[Sequence[Any], Iterable[Any]],
        n_split: Optional[int] = None,
        progress: bool = True,
        progress_unit: str = "",
        use_ixs: bool = False,
        backend: str = "loky",
        reduce_func: Optional[Callable[[Any], Any]] = None,
        reduce_as_array: bool = True,
    ):
        if progress:
            try:
                try:
                    from tqdm.notebook import tqdm
                except ImportError:
                    from tqdm import tqdm_notebook as tqdm
                import ipywidgets  # noqa
            except ImportError:
                global _msg_shown
                tqdm = None

                self._msg_shown = True
            else:
                tqdm = None

        col_len = map_data.shape[0] if issparse(map_data) else len(map_data)

        if n_split is None:
            n_split = self.n_jobs

        if issparse(map_data):
            if n_split == map_data.shape[0]:
                map_datas = [map_data[[ix], :] for ix in range(map_data.shape[0])]
            else:
                step = map_data.shape[0] // n_split

                ixs = [
                    np.arange(i * step, min((i + 1) * step, map_data.shape[0]))
                    for i in range(n_split)
                ]
                ixs[-1] = np.append(
                    ixs[-1], np.arange(ixs[-1][-1] + 1, map_data.shape[0])
                )

                map_datas = [map_data[ix, :] for ix in filter(len, ixs)]
        else:
            map_datas = list(filter(len, np.array_split(map_data, n_split)))

        pass_queue = not hasattr(map_func, "py_func")  # we'd be inside a numba function

        def wrapper(*args, **kwargs):
            if pass_queue and progress:
                pbar = None if tqdm is None else tqdm(total=col_len, unit=progress_unit)
                queue = Manager().Queue()
                thread = Thread(target=Parallelizer.__update__, args=(pbar, queue, len(map_datas)))
                thread.start()
            else:
                pbar, queue, thread = None, None, None

            res = Parallel(n_jobs=self.n_jobs, backend=backend)(
                delayed(map_func)(
                    *((i, cs) if use_ixs else (cs,)),
                    *args,
                    **kwargs,
                    queue=queue,
                )
                for i, cs in enumerate(map_datas)
            )

            res = np.array(res) if reduce_as_array else res
            if thread is not None:
                thread.join()

            return res if reduce_func is None else reduce_func(res)
        return wrapper