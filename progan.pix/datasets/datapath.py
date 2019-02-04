def get_data_folder():
    import socket
    name = socket.gethostname()
    if name == 'SPboy':
        dpath = '/home/hehaodele/data/Images'
    if name == 'MadBoy':
        dpath = '/media/hehaodele/AngryBoy/ImageDatasets/'
    if name.startswith('netmitgpu'):
        dpath = '/afs/csail.mit.edu/u/h/hehaodele/wifall_data/image'
    return dpath