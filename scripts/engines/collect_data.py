import os
import sys
import csv
from carla import Image

class collectData():
    def __init__(self, path, episode, isPerception):
        self.path = path
        self.folder_path = os.path.join(path, "episode_"+str(episode))
        self.episode = episode
        self.isPerception = isPerception
        self.count = 0
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        csv_file = os.path.join(self.folder_path, "label.csv")
        self.f = open(csv_file, 'w')
        self.csv_writer = csv.writer(self.f)
        if self.isPerception:
            self.csv_writer.writerow(["image_path", "true_dist", "predicted_dist", "velocity", "brake", "precipitation"])
        else:
            self.csv_writer.writerow(["image_path", "true_dist", "velocity", "brake", "precipitation"])

    def __call__(self, image, gt_distance, velocity, brake, precipitation, timestamp, regression_distance=0):
        if brake == -1:
            file_path = os.path.join("episode_"+str(self.episode), str(self.count)+".png")
            self.count+=1
        else:
            file_path = os.path.join("episode_"+str(self.episode), str(timestamp)+".png")
        image.save_to_disk(os.path.join(self.path, file_path))
        if self.isPerception:
            self.csv_writer.writerow([file_path, round(gt_distance, 4), round(regression_distance, 4), round(velocity, 4), round(brake, 4), round(precipitation, 4)])
        else:
            self.csv_writer.writerow([file_path, round(gt_distance, 4), round(velocity, 4), round(brake, 4), round(precipitation, 4)])
    
    def close_csv(self):
        self.f.close()