
def pretify_method_name(method_name):
    if method_name == 'DRVI-IK':
        return 'DRVI-AP'
    if method_name == 'DRVI-CVAE':
        return 'DRVI-CVAE'
    if method_name == 'TCVAE-opt':
        return 'B-TCVAE'
    if method_name == 'TCVAE':
        return 'B-TCVAE default'
    if method_name == 'MICHIGAN-opt':
        return 'MICHIGAN'
    if method_name == 'MICHIGAN':
        return 'MICHIGAN default'
    return method_name
