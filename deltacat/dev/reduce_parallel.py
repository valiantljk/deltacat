import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Union
import argparse
import numpy as np
import ray
import random
import asyncio
import boto3
import pickle
import pandas as pd
import io
from deltacat.utils.ray_utils.runtime import live_node_resource_keys

def upload_df_to_s3(df, bucket, key, region=None):
    s3 = boto3.resource('s3', region_name=region)
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer)
    s3.Object(bucket, key).put(Body=parquet_buffer.getvalue())

def upload_to_s3(bucket: str, key: str, record_counts: Any, region: str):
    s3 = boto3.resource('s3', region_name=region)
    if isinstance(record_counts, pd.DataFrame):
        upload_df_to_s3(record_counts, bucket, key)
    else:
        data = pickle.dumps(record_counts)
        s3.Object(bucket, key).put(Body=data)

def download_df_from_s3(bucket, key, region):
    try:
        s3 = boto3.resource('s3', region_name=region)
        parquet_buffer = io.BytesIO()
        s3.Object(bucket, key).download_fileobj(parquet_buffer)
    except Exception as e:
        print(f"Error downloading {key} from bucket {bucket}, Error: {e}")
    return pd.read_parquet(parquet_buffer)

def download_from_s3(bucket: str, key: str, region: str, data_type: str=None):
    if data_type=='parquet':
        return download_df_from_s3(bucket, key, region)
    s3 = boto3.resource('s3', region_name=region)
    serialized_data = s3.Object(bucket, key).get()['Body'].read()
    return pickle.loads(serialized_data)

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
        self.record_counts_ref = None
        self.expected_result_count = expected_result_count
        self.actual_result_count = 0

    def to_dict(self):
        record_counts_dict = {
            key: {
                subkey: dict(subvalue)
                for subkey, subvalue in value.items()
            }
            for key, value in self.record_counts.items()
        }
        return record_counts_dict
    def add_record_counts(
            self,
            result_idx: int,
            record_counts:
            Dict[int, Dict[Tuple[np.bool_, np.int64, np.int32], int]]) -> None:
        #start = time.time()
        #TODO: use df directly
        for mat_bucket, df_locator_rows in record_counts.items():
            for df_locator, rows in df_locator_rows.items():
                self.record_counts[mat_bucket][df_locator][result_idx] += rows
        self.actual_result_count += 1
        #end = time.time()
        #print(f"received from task {result_idx}, time taken {(end-start):.2f}")

    def get_record_counts(self):
        return self.record_counts_ref

    def get_expected_result_count(self) -> int:
        return self.expected_result_count

    def get_actual_result_count(self) -> int:
        return self.actual_result_count

    def is_finalized(self) -> bool:
        return self.actual_result_count == self.expected_result_count
    def to_s3(self, bucket, key, region):
        serializable_record_counts = self.to_dict()
        upload_to_s3(bucket, key, serializable_record_counts,region)
        return len(self.record_counts)
    def from_s3(self, bucket, keys, region):
        record_counts =[download_from_s3(bucket, key, region) for key in keys]
        final_record={}
        for rc in record_counts:
            final_record.update(rc)
        self.record_counts_ref = ray.put(final_record)
        record_counts = None
        self.record_counts = None
        del record_counts
        del self.record_counts
        return len(final_record)

@ray.remote(num_cpus=1)
class RecordCountsPendingMaterializeDF:
    def __init__(self, expected_result_count: int):
        self.record_counts = defaultdict(
            # delta_file_locator -> dedupe task index
            lambda: defaultdict(
                # dedupe task index -> row count
                lambda: defaultdict(int)
            )
        )
        self.record_counts_df = None
        self.expected_result_count = expected_result_count
        self.actual_result_count = 0
        self.record_counts_ref= None

    def add_record_counts(
            self,
            result_idx: int,
            record_counts:
            Dict[int, Dict[Tuple[np.bool_, np.int64, np.int32], int]]) -> None:
        #start = time.time()
        #TODO: use df directly
        for mat_bucket, df_locator_rows in record_counts.items():
            for df_locator, rows in df_locator_rows.items():
                self.record_counts[mat_bucket][df_locator][result_idx] += rows
        self.actual_result_count += 1
        #end = time.time()
        #print(f"received from task {result_idx}, time taken {(end-start):.2f}")

    def get_record_counts(self):
        return self.record_counts_ref

    def get_expected_result_count(self) -> int:
        return self.expected_result_count

    def get_actual_result_count(self) -> int:
        return self.actual_result_count

    def is_finalized(self) -> bool:
        return self.actual_result_count == self.expected_result_count
    def to_s3(self, bucket, key, region):
        start = time.time()
        self.convert_to_df()
        end = time.time()
        #print(f"convert dict to df: {(end-start):.2f} seconds")
        upload_to_s3(bucket, key, self.record_counts_df, region)
        return len(self.record_counts_df)
    
    def from_s3(self, bucket, keys, region):
        record_counts =[download_from_s3(bucket, k, region, 'parquet') for k in keys]
        final_record=pd.concat(record_counts)
        self.record_counts_ref = ray.put(final_record)
        record_counts=None
        self.record_counts_df = None
        del record_counts
        del self.record_counts
        return len(final_record)

    def convert_to_df(self):
        self.record_counts_df = pd.DataFrame([
            {"mat_bucket": id, "delta_file_locator": tuple, "dedupe_task_idx": result_idx, "count": count}
            for id, inner_dict in self.record_counts.items()
            for tuple, sub_dict in inner_dict.items()
            for result_idx, count in sub_dict.items()
        ])
        self.record_counts_df[['is_src', 'stream_pos', 'file_idx']] = pd.DataFrame(self.record_counts_df['delta_file_locator'].to_list(), index=self.record_counts_df.index)
        self.record_counts_df = self.record_counts_df.drop(columns=['delta_file_locator'])
        self.record_counts = None


@ray.remote(num_cpus=1)
def dedupe(dedupe_task_index: int, \
    record_counts_pending_materialize: List[Union[RecordCountsPendingMaterializeDF,RecordCountsPendingMaterializeDict]],\
    number_src_dfl: int,
    number_mat_bucket: int,
    number_of_nodes: int,
    singalactor: SignalActor,
    ):
    start = time.time()
    #delegator_indices = [32 * i for i in range(number_mat_bucket)]
    #record_counts:
    #            Dict[int, Dict[Tuple[np.bool_, np.int64, np.int32], int]]
    # generate some record_counts:
    mat_bucket_to_src_file_record_count = defaultdict(dict)
    for i in range(number_src_dfl):
        mat_bucket = i%number_mat_bucket
        src_dfl = (np.bool_(True), np.int64(100),np.int32(i))
        mat_bucket_to_src_file_record_count[mat_bucket][src_dfl]=random.randint(1,4000000)

    #print(f"dedupe_task_index {dedupe_task_index}, inmemory size {sys.getsizeof(mat_bucket_to_src_file_record_count)}")
    # send partial record counts on this dedupe task to global actor
    # get a subset of the dictionary
    node_data = [{} for _ in range(number_of_nodes)]
    for mat_bucket, mat_record_count in mat_bucket_to_src_file_record_count.items():
        node_id = mat_bucket % number_of_nodes
        node_data[node_id][mat_bucket] = mat_record_count

    [record_counts_pending_materialize[i].add_record_counts.remote(
        dedupe_task_index,
        node_data[i],
    ) for i in range(number_of_nodes)]

    # reduce all record counts and pull the final record_counts
    record_start=time.time()
    finalized = False
    while not finalized:
        finalized = all(ray.get(
            [record_counts_pending_materialize[i].is_finalized.remote() for i in range(number_of_nodes)]
        ))
        time.sleep(5)
    record_end = time.time()

    #delegate one task to control actor group activies: upload to s3 and download from s3 (or all to all via network)
    #all other tasks should wait for the final record_counts to be local on each node
    if dedupe_task_index==0:
        print(f"add_record_counts: {(record_end-record_start):.2f} seconds")
        #ask one task to trigger the parallel upload
        upload_start = time.time()
        part_record_length_on_actor=ray.get([record_counts_pending_materialize[i].to_s3.remote('benchmark-recordcounts',f'record_counts_{i}', 'us-east-1') for i in range(number_of_nodes)])
        upload_end = time.time()
        print(f"upload to s3: {(upload_end-upload_start):.2f} seconds")
        #print(f"part record length on each actor {part_record_length_on_actor}")
        rc_keys = ['record_counts_'+str(i) for i in range(number_of_nodes)]
        record_length_on_actor = ray.get([record_counts_pending_materialize[i].from_s3.remote('benchmark-recordcounts',rc_keys, 'us-east-1') for i in range(number_of_nodes)])
        download_end = time.time()
        print(f"download from s3: {(download_end-upload_end):.2f} seconds")
        #print(f"final record length on each actor should be same {record_length_on_actor}")
        assert all(element == record_length_on_actor[0] for element in record_length_on_actor), "Not all elements are the same."
        assert sum(part_record_length_on_actor) == record_length_on_actor[0], "sum of record counts on all actors not equal to final merged record counts"
        #broadcast the complete signal to all tasks
        ray.get(singalactor.send.remote())

    #sync all tasks by waiting for delegated task completion signal
    ray.get(singalactor.wait.remote())
    #get record_counts ref in node-local actor
    #TODO: making sure the actor and tasks are co-located
    record_count_ref = ray.get(record_counts_pending_materialize[dedupe_task_index%number_of_nodes].get_record_counts.remote())
    #TODO, implement the parallel download
    # if dedupe_task_index in delegator_indices:
    #     record_counts = [download_from_s3('benchmark-recordcounts',f'record_counts_{i}', 'us-east-1','parquet') for i in range(number_of_nodes)]
    #     final_record_counts ={}
    #     [final_record_counts.update(d) for d in record_counts]
    #     record_counts=final_record_counts
    #     record_counts = ray.put(record_counts)
    #all tasks get the same record_counts on each node by ref
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
    parser.add_argument(
        "-t", "--data_type",
        type=str,
        help="Type of record counts, p(parquet) or d(dict)",
        required=True,
    )
    return parser.parse_args()

if __name__ == "__main__":
    ray.init()
    args = parse_args()
    number_of_nodes = args.number_of_nodes
    number_src_dfl = args.number_of_files
    data_types = args.data_type
    node_resource_keys = live_node_resource_keys()
    number_dd_tasks = number_of_nodes*31
    number_mat_bucket = number_of_nodes*32
    start = time.time()
    signal_handle = SignalActor.remote()
    ActorClass = RecordCountsPendingMaterializeDict if data_types == 'd' else RecordCountsPendingMaterializeDF
    actor_handles = [
        ActorClass.options(resources={node_resource_keys[i%len(node_resource_keys)]:0.01}).remote(number_dd_tasks)
        for i in range(number_of_nodes)
    ]     
    dd_tasks = [dedupe\
                .options(resources={node_resource_keys[i%len(node_resource_keys)]:0.01})\
                    .remote(i,actor_handles, number_src_dfl, number_mat_bucket, number_of_nodes, signal_handle)\
                          for i in range(number_dd_tasks)]
    result = ray.get(dd_tasks)
    end = time.time()
    print(f"Total:{(end-start):.2f}, nodes:{number_of_nodes}, dd:{number_dd_tasks}, \
          mat_bucket:{number_mat_bucket}, num_files:{number_src_dfl}, data types:{data_types}")
    print(f"DD Task Min: {(min(result)):.2f}")
    print(f"DD Task Max: {(max(result)):.2f}")
    print(f"DD Task Average: {(sum(result) / len(result)):.2f}")
