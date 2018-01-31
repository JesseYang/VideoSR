import os
from pathlib import Path
import cv2

def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = (cap.read()[1] for _ in range(total_frame))
    
    return frames

def collect_video_data(video_dir, target_dir):
    """it will capture all frames of each video and save frames with same file tree in target_dir which is named of video's name
    """
    video_data_path_collections = [os.path.join(dirpath,filename) for dirpath, dirnames, filenames in os.walk(video_dir, followlinks = True) for filename in filenames]
    # print(*video_data_path_collections, sep = '\n')
    total = len(video_data_path_collections)
    print('total:', total)
    for idx, video_data_path in enumerate(video_data_path_collections):
        path = Path() / target_dir / video_data_path[len('video_data/'):]
        os.makedirs(path)
        frames = get_frames(video_data_path)
        for fm_idx, frame in enumerate(frames):
            cv2.imwrite(Path() / path / '{}.png'.format(fm_idx), frame)
        print('[{}/{}]'.format(idx+1,total), video_data_path)

if __name__ == '__main__':
    collect_video_data('video_data/', '/home/user/Datasets/VideoSR-datasets/frame_seq_data')