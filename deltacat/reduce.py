import logging
import time
from collections import defaultdict
from itertools import repeat
from typing import Any, Dict, List, Optional, Tuple
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import ray
from ray import cloudpickle
from ray.types import ObjectRef
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import random,sys
import botocore
import boto3
import pickle
import pandas as pd
import ray

def upload_df_to_s3(df, bucket, key):
    df.to_parquet(f"s3://{bucket}/{key}")

def upload_to_s3(bucket: str, key: str, record_counts: Any, region: str):
    s3 = boto3.resource('s3', region_name=region)
    if isinstance(record_counts, pd.DataFrame):
        upload_df_to_s3(record_counts, bucket, key)
    else:
        serialized_data = pickle.dumps(record_counts)
        s3.Object(bucket, key).put(Body=serialized_data)
def download_df_from_s3(bucket, key):
    return pd.read_parquet(f's3://{bucket}/{key}')
def download_from_s3(bucket: str, key: str, region: str, data_type: str):
    if data_type=='parquet':
        return download_df_from_s3(bucket, key)
    s3 = boto3.resource('s3', region_name=region)
    serialized_data = s3.Object(bucket, key).get()['Body'].read()
    record_counts = pickle.loads(serialized_data)
    return record_counts

@ray.remote(num_cpus=0)
class SignalActor:
    def __init__(self):
        self.ready_event = asyncio.Event()

    def send(self, clear=False):
        self.ready_event.set()
        if clear:
            self.ready_event.clear()

    async def wait(self, should_wait=True):
        if should_wait:
            await self.ready_event.wait()

@ray.remote(num_cpus=1)
class RecordCountsPendingMaterializeDict:
    def __init__(self, expected_result_count: int):
        # materialize_bucket -> src_file_id
        self.record_counts = defaultdict(
            # delta_file_locator -> dedupe task index
            lambda: defaultdict(
                # dedupe task index -> row count
                lambda: defaultdict(int)
            )
        )
        self.expected_result_count = expected_result_count
        self.actual_result_count = 0

    def add_record_counts(
            self,
            result_idx: int,
            record_counts:
            Dict[int, Dict[Tuple[np.bool_, np.int64, np.int32], int]]) -> None:
        start = time.time()
        for mat_bucket, df_locator_rows in record_counts.items():
            for df_locator, rows in df_locator_rows.items():
                self.record_counts[mat_bucket][df_locator][result_idx] += rows
        self.actual_result_count += 1
        end = time.time()
        print(f"received from task {result_idx}, time taken {(end-start):.2f}")

    def get_record_counts(self) -> \
            Dict[int, Dict[Tuple[np.bool_, np.int64, np.int32],
                           Dict[int, int]]]:
        print(f"received task request for final data")
        return self.record_counts

    def get_expected_result_count(self) -> int:
        return self.expected_result_count

    def get_actual_result_count(self) -> int:
        return self.actual_result_count

    def is_finalized(self) -> bool:
        return self.actual_result_count == self.expected_result_count
    def upload_to_s3(self, bucket, key, region):
        upload_to_s3(bucket, key, self.record_counts,region)

@ray.remote(num_cpus=1)
class RecordCountsPendingMaterializeDF:
    def __init__(self, expected_result_count: int):
        self.record_counts = pd.DataFrame(columns=['materialize_bucket', 'delta_file_locator', 'result_idx', 'rows'])
        self.expected_result_count = expected_result_count
        self.actual_result_count = 0

    def add_record_counts(
            self,
            result_idx: int,
            record_counts: Dict[int, Dict[Tuple[np.bool_, np.int64, np.int32], int]]) -> None:
        start = time.time()
        for mat_bucket, df_locator_rows in record_counts.items():
            for df_locator, rows in df_locator_rows.items():
                mask = (self.record_counts['materialize_bucket'] == mat_bucket) & \
                       (self.record_counts['delta_file_locator'] == df_locator) & \
                       (self.record_counts['result_idx'] == result_idx)
                if mask.any():
                    self.record_counts.loc[mask, 'rows'] += rows
                else:
                    self.record_counts = self.record_counts.append({
                        'materialize_bucket': mat_bucket,
                        'delta_file_locator': df_locator,
                        'result_idx': result_idx,
                        'rows': rows
                    }, ignore_index=True)
        self.actual_result_count += 1
        end = time.time()
        print(f"received from task {result_idx}, time taken {(end-start):.2f}")

    def get_record_counts(self):
        print(f"received task request for final data")
        return self.record_counts

    def get_expected_result_count(self) -> int:
        return self.expected_result_count

    def get_actual_result_count(self) -> int:
        return self.actual_result_count

    def is_finalized(self) -> bool:
        return self.actual_result_count == self.expected_result_count
    def upload_to_s3(self, bucket, key, region):
        upload_to_s3(bucket, key, self.record_counts,region)


@ray.remote(num_cpus=1)
def dedupe(dedupe_task_index: int, \
    record_counts_pending_materialize: RecordCountsPendingMaterialize,\
    number_src_dfl: int,
    number_mat_bucket: int,
    ):
    #record_counts:
    #            Dict[int, Dict[Tuple[np.bool_, np.int64, np.int32], int]]
    # generate some record_counts:
    mat_bucket_to_src_file_record_count = defaultdict(dict)

    for i in range(number_src_dfl):
        mat_bucket = i%number_mat_bucket
        src_dfl = (np.bool_(True), np.int64(100),np.int32(i))
        mat_bucket_to_src_file_record_count[mat_bucket][src_dfl]=random.randint(1,4000000)

    print(f"dedupe_task_index {dedupe_task_index}, inmemory size {sys.getsizeof(mat_bucket_to_src_file_record_count)}")
    # send partial record counts on this dedupe task to global actor
    record_counts_pending_materialize.add_record_counts.remote(
        dedupe_task_index,
        mat_bucket_to_src_file_record_count,
    )

    # reduce all record counts and pull the final record_counts
    finalized = False
    while not finalized:
        finalized = ray.get(
            record_counts_pending_materialize.is_finalized.remote()
        )
        time.sleep(0.25)
    
    if dedupe_task_index==0:
        #ask one task to trigger the upload
        upload_start = time.time()
        ray.get(record_counts_pending_materialize.upload_to_s3.remote('benchmark-recordcounts','record_counts', 'us-east-1'))
        ray.get(singalactor.send.remote())
        upload_end = time.time()
        print(f"upload to s3 completed in {(upload_end-upload_start):.2f} seconds")
    #sync all tasks to begin download
    ray.get(singalactor.wait.remote())
    # record_counts = ray.get(
    #     record_counts_pending_materialize.get_record_counts.remote()
    # )
    start = time.time()
    #TODO, assign one task to download, and share object ref via singal actor
    record_counts = download_from_s3('benchmark-recordcounts','record_counts', 'us-east-1','parquet')
    end = time.time()
    return end-start

def parse_args():
    parser = argparse.ArgumentParser(
        description="Reduce function"
    )

    parser.add_argument(
        "-n", "--number_of_nodes",
        type=int,
        help="The number of nodes",
        required=True,
    )

    parser.add_argument(
        "-f", "--number_of_files",
        type=int,
        help="The number of files",
        required=True,
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    number_of_nodes = args.number_of_nodes
    number_src_dfl = args.number_of_files
    ray.init()
    number_dd_tasks = number_of_nodes*32
    number_mat_bucket = number_dd_tasks
    actor_handle=RecordCountsPendingMaterializeDF.remote(number_dd_tasks)
    start = time.time()
    dd_tasks = [dedupe.remote(i,actor_handle,number_src_dfl,number_mat_bucket) for i in range(number_dd_tasks)]
    result = ray.get(dd_tasks)
    end = time.time()
    print(f"Total:{(end-start):.2f},nodes:{number_of_nodes},dd:{number_dd_tasks},mat_bucket:{number_mat_bucket}, num_files:{number_src_dfl}")
    print(f"Min: {min(result)}")
    print(f"Max: {max(result)}")
    print(f"Average: {sum(result) / len(result)}")
