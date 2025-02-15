import numpy as np
import pandas as pd
import h5py


def make_windows(hdf5_file, index_file, output_file, dataset_type="train", window_size=50, overlap=10,
                 feature_cols=None, target_col="RUL", id_cols=["unit", "cycle"], chunk_size=50):
    """
    Creates time windows from an HDF5 dataset with reduced overlap and saves them to a new HDF5 file.

    Parameters:
        hdf5_file (str): Path to the input HDF5 dataset file.
        index_file (str): Path to the CSV index file storing time series indices.
        output_file (str): Path to the output HDF5 file for storing time windows.
        dataset_type (str): Either "train" or "test" to filter the correct dataset.
        window_size (int): Size of the time window.
        overlap (int): Number of overlapping rows between consecutive windows.
        feature_cols (list): List of feature columns. If None, it will be inferred.
        target_col (str): Name of the target variable.
        id_cols (list): List of identifier columns.
        chunk_size (int): Number of time series to process in a batch.
    """
    step_size = window_size - overlap  # Define step size to control overlap

    index_df = pd.read_csv(index_file)

    # Filter index_df to match the dataset type
    index_df = index_df[index_df["dataset"] == dataset_type]
    num_series = len(index_df)

    if num_series == 0:
        print(f"No matching time series found for dataset type: {dataset_type}")
        return

    # Open the HDF5 file and infer feature columns if not provided
    with h5py.File(hdf5_file, "r") as in_file:
        if feature_cols is None:
            all_columns = [col.decode("utf-8") for col in in_file.attrs["columns"]]
            feature_cols = [col for col in all_columns if col != target_col]

    print(f"Feature columns: {feature_cols}")

    # Open output HDF5 file for writing in append mode
    with h5py.File(output_file, "w") as out_file:
        X_dset = out_file.create_dataset(
            "X", shape=(0, window_size, len(feature_cols)),
            maxshape=(None, window_size, len(feature_cols)),
            dtype=np.float32,
            compression="gzip",
            shuffle=True
        )
        y_dset = out_file.create_dataset(
            "y", shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            compression="gzip",
            shuffle=True
        )

        processed = 0

        # Process the dataset in chunks
        for chunk_start in range(0, num_series, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_series)
            chunk_indices = index_df.iloc[chunk_start:chunk_end]

            X_chunk, y_chunk = [], []

            with h5py.File(hdf5_file, "r") as in_file:
                for _, row in chunk_indices.iterrows():
                    start_idx, stop_idx = row["start_idx"], row["stop_idx"]
                    df_chunk = pd.DataFrame(in_file["X"][start_idx:stop_idx + 1], columns=feature_cols)
                    y_series = in_file["y"][start_idx + window_size:stop_idx + 1]

                    # Generate time windows with reduced overlap
                    for i in range(0, len(df_chunk) - window_size, step_size):
                        X_chunk.append(df_chunk.iloc[i:i + window_size].values)
                        y_chunk.append(y_series[i])

            # Convert lists to numpy arrays
            X_chunk = np.array(X_chunk, dtype=np.float32)
            y_chunk = np.array(y_chunk, dtype=np.float32)

            # Write to HDF5 in smaller batches
            if len(X_chunk) > 0:
                batch_size = 2000  # Adjust for memory efficiency
                for i in range(0, len(X_chunk), batch_size):
                    X_batch = X_chunk[i:i + batch_size]
                    y_batch = y_chunk[i:i + batch_size]

                    X_dset.resize(X_dset.shape[0] + len(X_batch), axis=0)
                    y_dset.resize(y_dset.shape[0] + len(y_batch), axis=0)

                    X_dset[-len(X_batch):] = X_batch
                    y_dset[-len(y_batch):] = y_batch

                    out_file.flush()  # Explicitly flush data to disk

            processed += len(chunk_indices)
            print(f"Processed {processed}/{num_series} series.")

    print(f"Saved processed windows to {output_file}")


def main():
    make_windows(
        hdf5_file="../../engine_train.h5",
        index_file="../../engine_index.csv",
        output_file="engine_train_windows.h5",
        dataset_type="train",
        window_size=50,
        overlap=5,
        target_col="RUL",
        id_cols=["unit", "cycle"],
        chunk_size=50
    )

    # make_windows(
    #     hdf5_file="../../engine_test.h5",
    #     index_file="../../engine_index.csv",
    #     output_file="engine_test_windows.h5",
    #     dataset_type="test",
    #     window_size=50,
    #     overlap=10,
    #     target_col="RUL",
    #     id_cols=["unit", "cycle"],
    #     chunk_size=50
    # )


if __name__ == "__main__":
    main()
