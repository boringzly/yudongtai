import os
import zipfile


def make_zip(source_dir, output_filename):
    #import ipdb; ipdb.set_trace()
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    for parent, dirname, filenames in os.walk(source_dir):
        for filename in filenames:
            #print(filename)
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, arcname)
    
    zipf.close()