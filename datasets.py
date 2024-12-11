from glob import glob

def get_dataset(dataset):
    if dataset == "live":
        data_path = "/media/abhijay/rohu/work/shark_imp_backup/live/"
        with open(data_path+"LIVE/live.txt","r") as f:
            lines = f.readlines()
            lines = lines[1:]
            return DatasetIter(dataset, data_path, lines)
    elif dataset == "tid2013":
        data_path = "/media/abhijay/rohu/work/shark_imp_backup/tid13/"
        with open(data_path+"mos_with_names.txt","r") as f:
            lines = f.readlines()
            return DatasetIter(dataset, data_path, lines)
    elif dataset == "pipal":
        data_path = "/home/abhijay/Documents/work/Train/"
        train_labels = glob(data_path+"Train_Label/*")
        lines = [line for labels in train_labels for line in open(labels,"r").readlines()]
        return DatasetIter(dataset, data_path, lines)
    else:
        raise ValueError
    return lines

class DatasetIter:
    def __init__(self, dataset, data_path, lines):
        self.lines = lines
        self.data_path = data_path
        self.dataset = dataset
    def __len__(self):
        return len(self.lines)
    def __iter__(self):
        for line in self.lines:
            if self.dataset == "live":
                dis_name, _, ref_name, mosScore = line.strip().split(",")
            elif self.dataset == "tid2013":
                mosScore, dis_name = line.strip().split(" ")
                mosScore = 100.-float(mosScore)
                ref_name = "reference_images/"+("_".join(dis_name.split("_")[0:1])+".bmp").upper()
                dis_name = "distorted_images/"+dis_name
            elif self.dataset == "pipal":
                dis_name, mosScore = line.strip().split(",")
                ref_name = "Train_Ref/"+dis_name.split("_")[0]+".bmp"
                dis_name = "Train_Dis/"+dis_name

            yield {"ref_im": self.data_path+ref_name, "dis_im": self.data_path+dis_name, "mos": mosScore}