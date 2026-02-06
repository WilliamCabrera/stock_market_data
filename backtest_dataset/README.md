#

`backtest_dataset` folder contains 3 folders:

- _dataset_
- _out_of_sample_
- _walk_fordward_

**dataset**: contains 5 minutes data of the stocks from a period of time. The parquet files has the necesary columns (open, high, close, low, volume, etc....) to perform the first backtests. **Note!!!**: this is for the first state of the backest, you must to perform walkfordward test to validate the strategies.

**out_of_sample**: contains 5 minutes data of the stocks from a period of time. The parquet files has the necesary columns (open, high, close, low, volume, etc....) to perform the first backtests with no seen data. **Note!!!**: this is for the first state of the backest, you must to perform walkfordward test to validate the strategies.

**walk_fordward**: contains a set of datasets splited in differents intervals to perform the walkfordward test.

- _walk_fordward_in_sample_1_: has 5 minutes data from: **2021-01-01** to **2022-01-01**
- _walk_fordward_out_sample_1_: has 5 minutes data from: **2022-01-02** to **2022-07-01**

- _walk_fordward_in_sample_2_: has 5 minutes data from: **2021-07-01** to **2022-07-01**
- _walk_fordward_out_sample_2_: has 5 minutes data from: **2022-07-02** to **2023-01-01**

- _walk_fordward_in_sample_3_: has 5 minutes data from: **2022-01-01** to **2023-01-01**
- _walk_fordward_out_sample_3_: has 5 minutes data from: **2023-01-02** to **2023-07-01**

Inside **walk_fordward** you will find **trades**. This folder contain the walk_fordward test for each strategy.
**IMPORTANT!!!** Each parquet file contains all the backtest for a given strategy. One single strategy might have different backests. It occurs when we test differents value for a parameter Ex: 2ATR stop vers 1.5ATR stop. There is a column called strategy in the file that specify the name of the strategy for each combination of parameters that has been test. When computing the metrics make sure you filter by the strategy column to avoid mixing trades of the strategy with different params combinations.

- filter or group by strategy field.
- compute metrics for each group of trades.
