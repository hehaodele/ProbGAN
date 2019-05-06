def get_data_folder():
    import socket
    name = socket.gethostname()
    dpath = './data'
    return dpath