import polars as pl
from abc import ABC, abstractmethod
from collections import deque
from itertools import chain
from typing import Any, Dict, Generator, Iterable
import random
import bisect
from tqdm import tqdm


class ActionType:
    VIEW = 'AT_View'
    CLICK = 'AT_Click'
    CART_UPDATE = 'AT_CartUpdate'
    PURCHASE = 'AT_Purchase'


class Preprocessor:
    mapping_action_types = {
        ActionType.VIEW: 0,
        ActionType.CART_UPDATE: 1,
        ActionType.CLICK: 2,
        ActionType.PURCHASE: 3
    }
    def __init__(
        self,
        train_df: pl.DataFrame,
    ):
        self.train_df = train_df

    def _map_col(self, column: str, cast: pl.DataType = None) -> dict:
        uniques = sorted(self.train_df.select(pl.col(column)).unique().to_series().to_list())
        mapping = {val: idx for idx, val in enumerate(uniques)}

        for attr in ("train_df",):
            df = getattr(self, attr)
            df = df.with_columns(
                pl.col(column)
                .replace(mapping)
                .alias(column)
            )
            if cast is not None:
                df = df.with_columns(pl.col(column).cast(cast))
            setattr(self, attr, df)

        return mapping

    def run(self):
        self.train_df = self.train_df.with_columns(
            pl.col("source_type").fill_null("").alias("source_type")
        )

        self.mapping_product_ids = self._map_col("product_id")
        self.mapping_user_ids = self._map_col("user_id")
        self.mapping_source_types = self._map_col("source_type", cast=pl.Int8)

        self.train_df = self.train_df.with_columns(
            pl.col("action_type")
            .replace(self.mapping_action_types)
            .cast(pl.Int8)
            .alias("action_type")
        )

        self.targets = (
            self.train_df
            .filter(
                pl.col("request_id").is_not_null() &
                pl.col("action_type").is_in([0, 1]) &
                (pl.col("source_type") != self.mapping_source_types["ST_Catalog"])
            )
            .group_by([
                "user_id",
                "request_id",
                "product_id",
            ])
            .agg([
                pl.col("action_type").max(),
                pl.col("timestamp").min(),
                pl.col("source_type").mode().first()
            ])
        )

        requests_with_cartupdate_and_view = (
            self.targets
            .select(["request_id", "action_type", "timestamp"])
            .group_by("request_id")
            .agg([
                pl.col("action_type").max().alias("max_t"),
                pl.col("action_type").min().alias("min_t"),
                pl.len(),
                pl.col("timestamp").min().alias("req_ts")
            ])
            .with_columns(sum_targets=pl.col('max_t').add(pl.col('min_t')))
            .filter(pl.col('sum_targets') == 1)
            .filter(pl.col('len') >= 10)
            .select(["request_id", "req_ts"])
        )
        self.targets = (
            self.targets
            .drop("timestamp")
            .join(requests_with_cartupdate_and_view, on="request_id", how="inner")
            .with_columns(pl.col("req_ts").alias("timestamp"))
            .drop("req_ts")
        )
        self.targets = (
            self.targets
            .group_by(['user_id', 'request_id', 'timestamp', 'source_type'])
            .agg([
                pl.col('product_id'),
                pl.col('action_type'),
            ])
        )

        self.timesplit_valid_end = self.train_df["timestamp"].max()
        self.timesplit_valid_start = self.timesplit_valid_end - 30 * 24 * 60 * 60
        self.timesplit_train_end = self.timesplit_valid_start - 2 * 24 * 60 * 60
        self.timesplit_train_start = self.train_df["timestamp"].min()

        self.train_df = (
            self.train_df
            .filter(pl.col("action_type") != 0)
            .drop("request_id")
        )

        self.train_targets = self.targets.filter(
            pl.col("timestamp") <= self.timesplit_train_end
        )
        self.valid_targets = self.targets.filter(
            (pl.col("timestamp") > self.timesplit_valid_start) &
            (pl.col("timestamp") <= self.timesplit_valid_end)
        )
        self.train_history = self.train_df.filter(pl.col('timestamp') <= self.timesplit_train_end)
        self.valid_history = self.train_df.filter(pl.col('timestamp') > self.timesplit_train_end)

        return (
            self.train_history,
            self.valid_history,
            self.train_targets,
            self.valid_targets
        )


def ensure_sorted_by_timestamp(group: Iterable[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """
    Ensures that the given iterable of events is sorted by the 'timestamp' field.

    This function iterates over each event in the provided iterable and checks if the
    'timestamp' of the current event is greater than or equal to the 'timestamp' of the
    previous event. If any event has a 'timestamp' that is less than the previous event's
    'timestamp', an AssertionError is raised.

    @param group: An iterable of dictionaries, where each dictionary represents an event with at least a 'timestamp' key.
    @return: A generator yielding each event from the input iterable in order, ensuring they are sorted by 'timestamp'.
    @raises AssertionError: If the events are not sorted by 'timestamp'.
    """

    events = chain(group)

    prev_timestamp = 0
    for event in events:
        if event["timestamp"] >= prev_timestamp:
            prev_timestamp = event["timestamp"]
            yield event
        else:
            raise AssertionError("Events are not sorted by timestamp")
        

class Mapper(ABC):
    HISTORY_SCHEMA = pl.Struct({
        'source_type': pl.List(pl.Int64),
        'action_type': pl.List(pl.Int64),
        'product_id': pl.List(pl.Int64),
        'position': pl.List(pl.Int64),
        'targets_inds': pl.List(pl.Int64),
        'targets_lengths': pl.List(pl.Int64), # количество таргет событий в истории
        'lengths': pl.List(pl.Int64), # длина всей истории
    })
    CANDIDATES_SCHEMA = pl.Struct({
        'source_type': pl.List(pl.Int64),
        'action_type': pl.List(pl.Int64),
        'product_id': pl.List(pl.Int64),
        'lengths': pl.List(pl.Int64), # длина каждого реквеста
        'num_requests': pl.List(pl.Int64) # общее количество реквестов у этого пользователя
    })

    def __init__(self, min_length: int, max_length: int):
        self._min_length: int = min_length
        self._max_length: int = max_length

    @abstractmethod
    def __call__(self, group: pl.DataFrame) -> pl.DataFrame:
        pass

    def get_empty_frame(self, candidates=False):
        return pl.DataFrame(schema=pl.Schema({
            'history': self.HISTORY_SCHEMA,
            **({'candidates': self.CANDIDATES_SCHEMA} if candidates else {})
        }))


class HistoryDeque:
    def __init__(self, max_length=512):
        self._data = deque([], maxlen=max_length)

    def append(self, x):
        self._data.append(x)

    def __len__(self):
        return (len(self._data))

    def __getitem__(self, idx):
        return self._data[idx]

    def get(self, targets_inds=None):
        """
        Retrieves a dictionary containing various attributes of the dataset samples.

        If `targets_inds` is not provided, it automatically identifies indices of samples where the `target` is 1.

        @param targets_inds: List of indices of target samples. If None, it will be determined based on samples with target value 1.
        @return: Dictionary with keys ['source_type', 'action_type', 'product_id', 'position', 'targets_inds', 'targets_lengths', 'lengths']
                Each key maps to a list or value representing the respective attribute of the dataset samples.
        """
        if targets_inds is None:
            targets_inds = [i for i, value in enumerate(self._data)
                                        if value['target'] == 1]

        history = {'source_type': [stori['source_type'] for stori in self._data],
                    'action_type': [stori['action_type'] for stori in self._data],
                    'product_id': [stori['product_id'] for stori in self._data],
                    'position': list(range(len(self._data))),
                    'targets_inds': targets_inds,
                    'targets_lengths': [len(targets_inds)],
                    'lengths': [len(self._data)]}


        return history
    

class PretrainMapper(Mapper):
    def __call__(self, group: pl.DataFrame) -> pl.DataFrame:
        """
        Processes a group of data by maintaining a history of rows up to a specified maximum length.
        If the history meets the minimum length requirement and contains at least one target, it returns
        a DataFrame with the history. Otherwise, it returns an empty DataFrame.

        @param group: A Polars DataFrame containing the group of data to process.
        @return: A Polars DataFrame containing the history if conditions are met; otherwise, an empty DataFrame.
        """
        deque = HistoryDeque(self._max_length)
        events_generator = ensure_sorted_by_timestamp(group.to_struct())

        for event in events_generator:
            deque.append(event)

        if len(deque) > self._min_length:
            return pl.DataFrame([{'history': deque.get()}], schema=pl.Schema({'history': Mapper.HISTORY_SCHEMA}))

        else:
            return self.get_empty_frame()
        


class Candidates:
    def __init__(self, max_requests_size):
        self._data = deque([], maxlen=max_requests_size)

    def append(self, x):
        if x['request_id']:
            self._data.append(x)

    def popleft(self):
        return self._data.popleft()

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)

    def get(self):
        """
        Aggregates data from the internal _data attribute into a structured dictionary format.

        This method constructs a dictionary with keys 'source_type', 'action_type', 'product_id', 'lengths', and 'num_requests'.
        - 'source_type' contains the source types from each sample.
        - 'action_type' contains all action types from each sample's action_type_list flattened into a single list.
        - 'product_id' contains all product IDs from each sample's product_id_list flattened into a single list.
        - 'lengths' contains the length of the product_id_list for each sample.
        - 'num_requests' contains the total number of samples.

        Returns:
            Dict[str, Any]: A dictionary with aggregated data.
        """

        candidate_deque = {'source_type':[],
                           'action_type':[],
                           'product_id':[],
                           'lengths':[],
                           'num_requests':[len(self)]}

        for x in self:
            candidate_deque['source_type'].append(x['source_type'])
            candidate_deque['action_type'].extend(x['action_type_list'])
            candidate_deque['product_id'].extend(x['product_id_list'])
            candidate_deque['lengths'].append(len(x['action_type_list']))

        return candidate_deque
    


class FinetuneTrainMapper(Mapper):
    def __call__(self, group: pl.DataFrame) -> pl.DataFrame:
        """
        Processes a group of interactions to generate history and candidate sets for recommendation.

        This method processes a DataFrame containing interaction data, separating actions into history and candidates based on the presence of 'action_type_list'.
        It ensures the data is sorted by timestamp, filters candidates based on time constraints, and selects historical interactions within a specified lag range for each candidate.
        If there are no valid candidates or insufficient history, it returns an empty DataFrame.

        @param group: A Polars DataFrame containing interaction data with at least 'timestamp' and 'action_type_list' columns.
        @return: A Polars DataFrame with 'history' and 'candidates' columns, or an empty DataFrame if no valid candidates are found.
        """
        history_deque = HistoryDeque(self._max_length)
        candidate_deque = Candidates(self._max_length)

        history_generator = ensure_sorted_by_timestamp(group.to_struct())
        for event in history_generator:
            if event['action_type_list'] is None:
                history_deque.append(event)

        candidate_generator = ensure_sorted_by_timestamp(group.to_struct())
        for event in candidate_generator:
            if event['action_type_list']:
                max_time = event['timestamp'] - random.randrange(2, 32) * 86400
                target_ind = bisect.bisect_right(history_deque, max_time, key=lambda x: x['timestamp'])
                if target_ind == 0:
                    continue
                event['targets_inds'] = target_ind - 1
                candidate_deque.append(event)

        targets_inds = [candidate['targets_inds'] for candidate in candidate_deque]


        if len(candidate_deque) > self._min_length and len(history_deque) > self._min_length:
            return pl.DataFrame([{'history': history_deque.get(targets_inds),
                                  'candidates': candidate_deque.get()}],
                                 schema=pl.Schema({'history': Mapper.HISTORY_SCHEMA,
                                                  'candidates': Mapper.CANDIDATES_SCHEMA}))
        else:
            return self.get_empty_frame(candidates=True)

class FinetuneValidMapper(Mapper):
    def __call__(self, group: pl.DataFrame) -> pl.DataFrame:
        """
        Differs only in the formation of target_inds
        """
        history_deque = HistoryDeque(self._max_length)
        candidate_deque = Candidates(self._max_length)

        history_generator = ensure_sorted_by_timestamp(group.to_struct())
        for event in history_generator:
            if event['action_type_list'] is None:
                history_deque.append(event)

        candidate_generator = ensure_sorted_by_timestamp(group.to_struct())
        for event in candidate_generator:
            if event['action_type_list']:
                max_time = event['timestamp']
                target_ind = bisect.bisect_right(history_deque, max_time, key=lambda x: x['timestamp'])
                if target_ind == 0:
                    continue
                event['targets_inds'] = target_ind - 1
                candidate_deque.append(event)

        targets_inds = [candidate['targets_inds'] for candidate in candidate_deque]


        if len(candidate_deque) > self._min_length and len(history_deque) > self._min_length:
            return pl.DataFrame([{'history': history_deque.get(targets_inds),
                                  'candidates': candidate_deque.get()}],
                                 schema=pl.Schema({'history': Mapper.HISTORY_SCHEMA,
                                                  'candidates': Mapper.CANDIDATES_SCHEMA}))
        else:
            return self.get_empty_frame(candidates=True)