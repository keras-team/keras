def standardize_data_format(data_format):
    if data_format is None:
        # Avoid circular imports.
        from keras.src.backend import config

        return config.image_data_format()

    data_format = str(data_format).lower()
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "The `data_format` argument must be one of "
            "{'channels_first', 'channels_last'}. "
            f"Received: data_format={data_format}"
        )
    return data_format
