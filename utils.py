# CHUNK_SIZE = 2000 # 1000
# CHUNK_OVERLAP = 1000 # 100


def get_filename_from_title(my_dict, title):
    """
    Find a file_name (i.e., the key) in the dictionary given a title (i.e., the value).
    The dictionary has the following structure:
    {
        "file_name_1":"title_1",
        "file_name_2":"title_2",
        ...
    }

    Args:
        my_dict (dict): The dictionary with file names and titles.
        title (str): The title we want to use to find the file name (i.e., the key).

    Returns:
        query_engine: a query_engine created from the index.
    """
    keys_list = list(my_dict.keys())
    values_list = list(my_dict.values())
    return keys_list[values_list.index(title)]
