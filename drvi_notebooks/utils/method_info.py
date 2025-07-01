
def pretify_method_name(method_name):
    if method_name == 'DRVI-IK':
        return 'DRVI-AP'
    if method_name == 'DRVI-CVAE':
        return 'CVAE'
    if method_name == 'DRVI-sum-sp':
        return 'DRVI-APnoEXP'
    if method_name == 'DRVI-NC':
        return 'DRVI-noShare'
    if method_name == 'TCVAE-opt':
        return 'B-TCVAE'
    if method_name == 'TCVAE':
        return 'B-TCVAE default'
    if method_name == 'MICHIGAN-opt':
        return 'MICHIGAN'
    if method_name == 'MICHIGAN':
        return 'MICHIGAN default'
    return method_name


methods_general_order = [
    'DRVI', 
    'DRVI-noShare',
    'DRVI-2D',
    'DRVI-AP', 
    'DRVI-APnoEXP',
    'CVAE',
    'ICA',
    'scETM',
    'MOFA',
    'LIGER',
    'PCA',
    'scVI',
    'B-TCVAE',
    'scVI-ICA',
    'scVI-PCA',
    'MICHIGAN',
]