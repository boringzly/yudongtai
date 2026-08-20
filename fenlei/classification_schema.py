"""Classification result schema shared by single-file and batch workflows."""

CLASS_NAME_BY_CODE = {
    0: "未知",
    1: "耕地",
    2: "林地",
    3: "草地",
    4: "水域",
    5: "建设用地",
    6: "未利用土地",
}

OUTPUT_FIELDS = ["uid", "pre_code", "pre_name", "curr_code", "curr_name", "geometry"]


def get_class_name(code):
    """Return the Chinese class name for a classified integer code."""
    try:
        normalized_code = int(code)
    except (TypeError, ValueError):
        normalized_code = 0
    return CLASS_NAME_BY_CODE.get(normalized_code, CLASS_NAME_BY_CODE[0])


def format_classification_result(gdf):
    """Keep only stable output fields and add Chinese class-name columns."""
    result = gdf.copy()

    if "uid" not in result.columns:
        result["uid"] = range(len(result))
    if "pre_code" not in result.columns:
        result["pre_code"] = 0
    if "curr_code" not in result.columns:
        result["curr_code"] = 0

    result["uid"] = result["uid"].astype("int64")
    result["pre_code"] = result["pre_code"].fillna(0).astype("int32")
    result["curr_code"] = result["curr_code"].fillna(0).astype("int32")
    result["pre_name"] = result["pre_code"].map(get_class_name).astype("object")
    result["curr_name"] = result["curr_code"].map(get_class_name).astype("object")

    return result[OUTPUT_FIELDS]
