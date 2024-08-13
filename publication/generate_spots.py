import argparse
import pandas as pd
import yaml
from tqdm import tqdm
import numpy as np
from tifffile import TiffFile, TiffSequence
import h5py
import shutil

class ImageSequenceWrapper:
    def __init__(self, impath, spot_size):
        seq = TiffSequence(impath)
        self.seq_files = seq.files
        seq.close()
        self.current_frame = 0
        self.spot_size = spot_size
        self.bb_size =  spot_size // 2

                               
    def extract_loc(self, frame, loc):
        x = int(round(loc['x']))
        y = int(round(loc['y']))
        spot = frame[y-self.bb_size:y+self.bb_size+1, x-self.bb_size:x+self.bb_size+1]
        if spot.shape[0] != 15 or spot.shape[1] != 15:
            return np.zeros((self.spot_size, self.spot_size))
        return spot
    
    def extract_spots(self, locs):
        spots = np.zeros((locs.shape[0], self.spot_size, self.spot_size))
        write_idx = 0
        frame_offset = 0
        locs_added = 0
        with tqdm(total=locs.shape[0]) as pbar:
            for i in range(len(self.seq_files)):
                with TiffFile(self.seq_files[i]) as handle:
                    n_frames = len(handle.pages)
                    for frame in handle.pages:
                        sub_locs = locs[locs['frame']==(frame.index + frame_offset)]
                        locs_added += sub_locs.shape[0]
                        frame = frame.asarray()
                        for loc in sub_locs.to_dict(orient="records"):
                            spots[write_idx] = self.extract_loc(frame, loc)
                            write_idx += 1
                            pbar.update(1)
                    frame_offset += n_frames
        non_empty_idx = np.argwhere(spots.sum(axis=(1,2)) != 0)[:, 0]
        spots = spots[non_empty_idx]
        locs = locs.iloc[non_empty_idx]
        return locs, spots
            

def load_yaml(yaml_path):
    with open(yaml_path) as stream:
        yaml_files = list(yaml.safe_load_all(stream))
        config = dict()
        for f in yaml_files:
            config.update(f)
        return config


def write_locs(locs, locs_path):
    if 'index' in locs:
        del locs['index']   
    out_locs_path = locs_path.replace('.hdf5', '.hdf5')
    with h5py.File(out_locs_path, "w") as locs_file:
        locs_file.create_dataset("locs", data=locs.to_records())


def write_spots(spots, locs_path):
    spots_path = locs_path.replace('locs', 'spots')
    with h5py.File(spots_path, 'w') as f:
        f.create_dataset('spots', data=spots)
    print(spots_path)


def main(args):
    yaml_path = args['locs'].replace('.hdf5', '.yaml')
    config = load_yaml(yaml_path)
    locs = pd.read_hdf(args['locs'], key='locs')
    wrapper = ImageSequenceWrapper(args['img'], config['Box Size'])
    locs, spots = wrapper.extract_spots(locs)
    print(locs.shape, spots.shape)

    write_locs(locs, args['locs'])
    write_spots(spots, args['locs'])


    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='TIF or First TIF img of sequence')
    parser.add_argument('locs', help='HDF5 file from Picasso localisation')
    return vars(parser.parse_args())

if __name__=='__main__':
    main(parse_args())