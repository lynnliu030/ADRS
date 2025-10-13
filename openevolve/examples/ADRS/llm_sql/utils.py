from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from typing import List, Tuple
from pyspark.sql import SparkSession
from itertools import combinations
import yaml

class TrieNode:
    def __init__(self):
        self.children = {}
        self.end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.end_of_word = True

    def longest_common_prefix(self, word):
        node = self.root
        common_prefix_length = 0
        for char in word:
            if char in node.children:
                common_prefix_length += len(char)
                node = node.children[char]
            else:
                break
        return common_prefix_length

def calculate_length(value):
    val = 0
    if isinstance(value, bool):
        val = 4  # length of 'True' or 'False'
    elif isinstance(value, (int, float)):
        val = len(str(value))
    elif isinstance(value, str):
        val = len(value)
    else:
        val = 0
    return val**2

def evaluate_cell_hit_cnt(df: pd.DataFrame) -> int:
    """
    Function to evaluate the prefix hit count of a DataFrame based on exact cell matching.
    For a cell to be a hit, all previous cells in the row must also be hits.
    """
    total_prefix_hit_count = 0
    squared_token_hit_count = 0
    seen_prefixes = set()  # Cache of all seen prefixes
    total_df_tokens_squared = 0

    for _, row in df.iterrows():
        row_hits = 0
        squared_token_hits = 0
        prefix = []  # Tracks the growing prefix of the current row
        row_string = "".join(row.fillna("").astype(str).values)

        for cell in row:
            prefix.append(cell)
            prefix_tuple = tuple(prefix)
            total_df_tokens_squared += calculate_length(cell) ** 2

            if prefix_tuple in seen_prefixes:
                row_hits += 1
                squared_token_hits += (calculate_length(cell) ** 2)

            # Always add the current prefix to the cache, even if it wasn't a hit
            seen_prefixes.add(prefix_tuple)

        total_prefix_hit_count += row_hits
        squared_token_hit_count += squared_token_hits

    num_rows = df.shape[0]
    num_cols = df.shape[1]
    total_prefix_hit_rate = total_prefix_hit_count / (num_rows * num_cols) * 100
    total_token_hit_rate = squared_token_hit_count / total_df_tokens_squared * 100

    return total_prefix_hit_count, total_prefix_hit_rate, squared_token_hit_count, total_token_hit_rate

def evaluate_df_prefix_hit_cnt(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Function to evaluate the prefix hit count of a DataFrame
    """

    def max_overlap(trie, row_string):
        return min(len(row_string), trie.longest_common_prefix(row_string))


    trie = Trie()
    total_prefix_hit_count = 0
    total_string_length = 0

    def process_row(index, row):
        nonlocal total_string_length
        row_string = "".join(row.fillna("").astype(str).values)  # No spaces between columns
        total_string_length += len(row_string)
        row_prefix_hit_count = max_overlap(trie, row_string)
        trie.insert(row_string)
        return row_prefix_hit_count

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_row, df.index, [row for _, row in df.iterrows()])

    total_prefix_hit_count = sum(results)
    total_prefix_hit_rate = total_prefix_hit_count / total_string_length
    assert total_prefix_hit_count <= total_string_length
    print(f"Total string length: {total_string_length}")
    # NOTE: GPT-4o
    # /5: convert string length to token (rough estimation)
    no_cache_pricing = 2.5 / 5  # per 1M if not cached
    cache_pricing = 1.25 / 5  # per 1M if cached
    cached_tokens_pricing = total_prefix_hit_count * cache_pricing / 1e6
    non_cached_tokens_pricing = (total_string_length - total_prefix_hit_count) * no_cache_pricing / 1e6
    print(
        f"Cached tokens pricing = {round(cached_tokens_pricing,2)}, Non-cached tokens pricing = {round(non_cached_tokens_pricing,2)}, total pricing = {round(cached_tokens_pricing + non_cached_tokens_pricing,2)}"
    )
    return total_prefix_hit_count, total_prefix_hit_rate * 100


def print_col_orders(col_orders: List[List[str]]):
    for col_order in col_orders:
        col_str = "| "
        for col in col_order:
            col_str += col + " | "
        print(col_str)


def prepend_col_name(df: pd.DataFrame) -> pd.DataFrame:
    # just prepend column name to each value in the column
    df = df.apply(lambda x: x.name + ": " + x.astype(str))
    return df

def unprepend_col_name(df: pd.DataFrame) -> pd.DataFrame:
    # just prepend column name to each value in the column
    df = df.apply(lambda x: x.name.split(": ")[1])
    return df


def prepend_rag(df: pd.DataFrame) -> pd.DataFrame:
    # just prepend column name to each value in the column
    df = df.apply(lambda x: x.name + ": " + x.astype(str) if x.name == "question" or x.name == "claim" else x.astype(str))
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Unnamed" in df.columns:
        df = df.drop(columns=["Unnamed"])
    df = df.fillna("")
    return df

def convert_df(df: pd.DataFrame):
    # Create a SparkSession
    spark = SparkSession.builder.appName("llmsql").config("spark.driver.memory", "15g").getOrCreate()

    # set log level to ERROR 
    spark.sparkContext.setLogLevel("ERROR")
    
    spark_df = spark.createDataFrame(df)
    return spark_df


def print_df(df: pd.DataFrame):
    df_copy = df.copy()
    truncate = lambda x: x[:20] if isinstance(x, str) else x
    print(df_copy.applymap(truncate), flush=True)


def read_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def split_dataframe(df, n):
    # Step 1: Add an index column (starting from 0)
    indexed_df = df.withColumn("index", F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1)

    # Step 2: Get total number of rows
    total_rows = df.count()

    # Step 3: Calculate the number of rows per split
    rows_per_split = total_rows // n
    extra_rows = total_rows % n  # For handling cases where total_rows isn't divisible by n

    # Step 4: Create list of DataFrames for each split
    split_dfs = []
    for i in range(n):
        start_index = i * rows_per_split
        end_index = start_index + rows_per_split
        if i == n - 1:  # Last partition gets any remaining rows
            end_index += extra_rows

        # Filter rows for the current partition
        split_dfs.append(indexed_df.filter((F.col("index") >= start_index) & (F.col("index") < end_index)).drop("index"))

    return split_dfs


def find_fds(df, threshold=1.0):
    fds = {}

    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                df_clean = df[[col1, col2]].dropna()

                grouped_col1_to_col2 = df_clean.groupby(col1)[col2].nunique()
                determining_groups_col1_to_col2 = (grouped_col1_to_col2 == 1).sum()
                total_groups_col1_to_col2 = len(grouped_col1_to_col2)
                proportion_col1_to_col2 = (
                    determining_groups_col1_to_col2 / total_groups_col1_to_col2 if total_groups_col1_to_col2 > 0 else 0
                )

                grouped_col2_to_col1 = df_clean.groupby(col2)[col1].nunique()
                determining_groups_col2_to_col1 = (grouped_col2_to_col1 == 1).sum()
                total_groups_col2_to_col1 = len(grouped_col2_to_col1)
                proportion_col2_to_col1 = (
                    determining_groups_col2_to_col1 / total_groups_col2_to_col1 if total_groups_col2_to_col1 > 0 else 0
                )

                if proportion_col1_to_col2 >= threshold and proportion_col2_to_col1 >= threshold:
                    if col1 in fds:
                        fds[col1].add(col2)
                    else:
                        fds[col1] = {col2}
                    if col2 in fds:
                        fds[col2].add(col1)
                    else:
                        fds[col2] = {col1}

    groups = []
    visited_columns = set()

    for col, dependencies in fds.items():
        if col not in visited_columns:
            group = {col}.union(dependencies)
            valid_group = True
            for comb in combinations(group, 2):
                if comb[1] not in fds.get(comb[0], set()) or comb[0] not in fds.get(comb[1], set()):
                    valid_group = False
                    break

            if valid_group:
                groups.append(list(group))
                visited_columns.update(group)

    return groups
