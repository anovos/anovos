from anovos.shared.utils import attributeType_segregation


class CheckListOfCols:
    def __init__(self, func, arg_cols="list_of_cols", arg_idf_target="idf_target", arg_drop_cols="drop_cols"):
        self.func = func
        self.arg_col = arg_cols
        self.arg_idf_target = arg_idf_target
        self.arg_drop_cols = arg_drop_cols

    def __call__(self, *args, **kwargs):
        cols = args[self.arg_col]
        target = args[self.arg_idf_target]
        drops = args[self.arg_drop_cols]

        if cols == "all" and target is not None:
            num_cols, cat_cols, other_cols = attributeType_segregation(target)
            cols = num_cols + cat_cols

        if isinstance(cols, str):
            cols = [x.strip() for x in cols.split("|")]

        if isinstance(drops, str):
            drops = [x.strip() for x in drops.split("|")]

        cols = list(set([e for e in cols if e not in drops]))

        if (len(cols) == 0) | any(x not in target.columns for x in cols):
            raise TypeError("Invalid input for Column(s)")

        args[self.arg_col] = cols
        args[self.arg_drop_cols] = drops

        return self.func


class CheckDistDistanceMethods:
    def __init__(self, func, argument_name="methods_types"):
        self.func = func
        self.argument_name = argument_name

    def __call__(self, *args, **kwargs):
        dist_distance_methods = args[self.argument_name]

        if isinstance(dist_distance_methods, str):
            if dist_distance_methods == "all":
                dist_distance_methods = ["PSI", "JSD", "HD", "KS"]
            else:
                dist_distance_methods = [x.strip() for x in dist_distance_methods.split("|")]

        if any(x not in ("PSI", "JSD", "HD", "KS") for x in dist_distance_methods):
            raise TypeError(f"Invalid input for {self.argument_name}")

        return self.func
