from functools import wraps
from inspect import getcallargs


def refactor_arguments(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        all_kwargs = getcallargs(func,*args, **kwargs)

        idf = all_kwargs.get("idf")

        if "list_of_cols" in all_kwargs.keys():
            list_of_cols=all_kwargs.get("list_of_cols")
            if isinstance(list_of_cols, str):
                list_of_cols = [x.strip() for x in list_of_cols.split("|")]
            if any(x not in idf.columns for x in list_of_cols):
                raise TypeError(f"Invalid input for column(s) in the function {func.__name__}. Column(s) not found in the dataframe: {set(list_of_cols) - set(idf.columns)}.")
            if len(list_of_cols) != len(set(list_of_cols)):
                raise TypeError(f"Duplicate input for column(s) in the function {func.__name__}.")
            all_kwargs["list_of_cols"] = list_of_cols

        for boolarg in ["print_impact"]:
            if boolarg in all_kwargs.keys():
                boolarg_val = str(all_kwargs.get(boolarg))
                if boolarg_val.lower() == "true":
                    boolarg_val = True
                elif boolarg_val.lower() == "false":
                    boolarg_val = False
                else:
                    raise TypeError(f"Non-Boolean input for {boolarg} in the function {func.__name__}.")
                all_kwargs[boolarg] = boolarg_val

        if "file_type" in all_kwargs.keys():
            if all_kwargs.get("file_type") not in ("csv", "parquet", "avro", "json"):
                raise TypeError(f"Invalid input for file_type in the function {func.__name__}. file_type should be csv, parquet, avro, json - Received '{all_kwargs.get('file_type')}'.")

        if func.__name__ == "write_dataset":
            column_order=all_kwargs.get("column_order")
            if not column_order:
                column_order = idf.columns
            if isinstance(column_order, str):
                column_order = [x.strip() for x in column_order.split("|")]
            if any(x not in idf.columns for x in column_order):
                raise TypeError(f"Invalid input for column_order in the function {func.__name__}. Column(s) not found in the dataframe: {set(column_order) - set(idf.columns)}.")
            if len(column_order) != len(idf.columns):
                raise ValueError(
                    f"Count of columns specified in column_order argument ({len(column_order)})do not match Dataframe ({len(idf.columns)})."
                )
            all_kwargs["column_order"] = list(set(column_order))

        if func.__name__ == "concatenate_dataset": 
            if all_kwargs.get("method_type") not in ("index", "name"):
                raise TypeError(f"Invalid input for method_type in the function {func.__name__}. method_type should be index or name - Received '{all_kwargs.get('method_type')}'.")


        if func.__name__ == "join_dataset":
            if all_kwargs.get("join_type") not in ("inner", "full", "left", "right", "left_semi", "left_anti"):
                raise TypeError(f"Invalid input for join_type in the function {func.__name__}. join_type should be index or name - Received '{all_kwargs.get('join_type')}'.")
            
            idfs = all_kwargs.get("idfs")
            join_cols=all_kwargs.get("join_cols")
            if isinstance(join_cols, str):
                join_cols = [x.strip() for x in join_cols.split("|")]
            for idf in idfs:
                if any(x not in idf.columns for x in join_cols):
                    raise TypeError(f"Invalid input for join column(s) in the function {func.__name__}. Column(s) not found in the dataframe(s): {set(join_cols) - set(idf.columns)}.")
            all_kwargs["join_cols"] = join_cols

        if func.__name__ == "rename_column":
            list_of_newcols=all_kwargs.get("list_of_newcols")
            if isinstance(list_of_newcols, str):
                list_of_newcols = [x.strip() for x in list_of_newcols.split("|")]
            if len(list_of_newcols) != len(list_of_cols):
                raise TypeError(f"Mismatch between number of columns in list_of_cols ({len(list_of_cols)}) and list_of_newcols ({len(list_of_newcols)}) in the function {func.__name__}.")
            all_kwargs["list_of_newcols"] = list_of_newcols

        if func.__name__ == "recast_column":
            list_of_dtypes=all_kwargs.get("list_of_dtypes")
            if isinstance(list_of_dtypes, str):
                list_of_dtypes = [x.strip() for x in list_of_dtypes.split("|")]
            if len(list_of_dtypes) != len(list_of_cols):
                raise TypeError(f"Mismatch between number of columns in list_of_cols ({len(list_of_cols)}) and number of datatypes ({len(list_of_dtypes)}) in list_of_dtypes in the function {func.__name__}.")
            all_kwargs["list_of_dtypes"] = list_of_dtypes

        
        if func.__name__ in ("concatenate_dataset", "join_dataset"):
            idfs = all_kwargs.pop("idfs")
            return func(*idfs, **all_kwargs)
        else:
            return func(**all_kwargs)
    return wrapper