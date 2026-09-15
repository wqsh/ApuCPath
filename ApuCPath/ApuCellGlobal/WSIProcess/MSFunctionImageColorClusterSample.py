from ApuCellGlobal.GeneralSystem.MSLogging import CLogging
from ApuCellGlobal.GeneralFunction.MSFunctionIO import CFunctionFileIO
from ApuCellGlobal.GeneralSystem.MSSystemEnum import LIST_DATA_FILE_TYPE_SUFFIX, EnumDataFileType, LIST_VISUAL_FILE_TYPE_SUFFIX, EnumVisualFileType

from ApuCellGlobal.Memory.MSFunctionMemory import CFunctionMemory

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, KMeans  # 大数据量推荐MiniBatchKMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import numpy as np
import cv2
import random

np.random.seed(42)
random.seed(42)

# ---------------------------- 4. 聚类与均匀采样 ----------------------------
def cluster_and_sample(features, paths, n_clusters, target_sample_size, use_mini_batch:bool=True):
    # 特征标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 聚类
    if use_mini_batch:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    # 统计每个簇的样本数
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Cluster distribution: {dict(zip(unique_labels, counts))}")

    # 每个簇应抽取的数量
    samples_per_cluster = max(1, target_sample_size // n_clusters)
    selected_indices = []
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        # 如果簇内样本不足samples_per_cluster，则全部取走
        if len(cluster_indices) <= samples_per_cluster:
            selected_indices.extend(cluster_indices)
        else:
            selected = np.random.choice(cluster_indices, size=samples_per_cluster, replace=False)
            selected_indices.extend(selected)

    # 如果总数超过target_sample_size，随机裁剪；如果不足，再随机补充（一般不会）
    if len(selected_indices) > target_sample_size:
        selected_indices = np.random.choice(selected_indices, size=target_sample_size, replace=False)
    elif len(selected_indices) < target_sample_size:
        remain = target_sample_size - len(selected_indices)
        # 从全体中随机补充（可能会重复簇）
        extra = np.random.choice(len(features), size=remain, replace=False)
        selected_indices = np.concatenate([selected_indices, extra])

    sampled_paths = [paths[i] for i in selected_indices]
    return sampled_paths

class CFunctionImageColorClusterSample:

    @staticmethod
    def GetsSamplePathList(list_img_path:list[str], target_count:int, n_cluster:int=100):
        if target_count > len(list_img_path):
            CLogging.LogWarning(f"Warning: target {target_count} > total {len(list_img_path)}. Will use all.")
            return list_img_path
        else:
            # 提取特征
            CLogging.LogToUser(f"Extracting features ...")
            features, valid_paths = CFunctionImageColorClusterSample.ExtractFeaturesThread(list_img_path)
            CFunctionMemory.CleanMemory()
            CFunctionMemory.GetCurMemoryInfo()
            CLogging.LogToUser(f"Successfully extracted features for {len(features)} images.")
            if len(features) < target_count:
                CLogging.LogWarning(f"Not enough valid images after feature extraction. Use all {len(features)}.")
                target_count = len(features)

            # 聚类采样
            CLogging.LogToUser(f"Clustering into {n_cluster} clusters and sampling {target_count} images...")
            sampled_paths = cluster_and_sample(features, valid_paths, n_cluster, target_count)
            del features
            CFunctionMemory.GetCurMemoryInfo()
            CFunctionMemory.CleanMemory()
            return sampled_paths

    @staticmethod
    def ExtractFeaturesThread(image_paths, num_workers:int=8):
        features = []
        valid_paths = []

        # 使用线程池加速IO
        def process_one(p):
            feat = CFunctionImageColorClusterSample.ExtractColorLBPFeatures(p)
            return p, feat
        cpu_count = os.cpu_count()
        if cpu_count < num_workers:
            num_workers = 1
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_one, p) for p in image_paths]
            for future in tqdm(as_completed(futures), total=len(image_paths), desc="Extracting color+lbp features"):
                p, feat = future.result()
                if feat is not None:
                    features.append(feat)
                    valid_paths.append(p)
        return np.array(features), valid_paths

    @staticmethod
    # 2.1 颜色直方图 + LBP纹理特征
    def ExtractColorLBPFeatures(img_path):
        img_path_suffix = os.path.splitext(img_path)[1]
        if img_path_suffix == LIST_DATA_FILE_TYPE_SUFFIX[EnumDataFileType.PKL.value]:
            img = CFunctionFileIO.ReadPKL(img_path)
        elif (
                (img_path_suffix == LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIFF.value]) or
                (img_path_suffix == LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.TIF.value]) or
                (img_path_suffix == LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.PNG.value]) or
                (img_path_suffix == LIST_VISUAL_FILE_TYPE_SUFFIX[EnumVisualFileType.JPG.value])
        ):
            img = CFunctionFileIO.ReadImage(img_path)
        else:
            CLogging.LogError("Unknown file type! ")
        if img is None:
            CLogging.LogError("Read image is None! ")
            return None
        # 缩放到统一大小以减少计算量，同时保持长宽比
        img = cv2.resize(img, (224, 224))
        # HSV颜色直方图 (H:50 bins, S:60 bins, V:60 bins) -> 降维后使用
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])
        hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        # LBP纹理特征
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = CFunctionImageColorClusterSample.LocalBinaryPattern(gray, 8, 1, method='uniform')
        hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10 + 1), range=(0, 10))
        hist_lbp = hist_lbp.astype(float) / (gray.size + 1e-6)
        # 合并特征
        feature = np.concatenate([hist, hist_lbp])
        return feature

    @staticmethod
    def LocalBinaryPattern(image, P, R, method='uniform'):
        # 使用skimage的local_binary_pattern，需安装scikit-image
        from skimage.feature import local_binary_pattern
        return local_binary_pattern(image, P, R, method)

if __name__ == "__main__":

    folder_img = r"/media/wqs/SOFT/weiqiushi/Data/ApuCPathKPI202456Nx/TrainDataset/ApuCPathDatasetTrain/ImageData/12-117/Org/images"

    list_img_path: list[str] = []
    for item in os.listdir(folder_img):
        list_img_path.append(os.path.join(folder_img, item))
    target_count = 200
    sample_path_list = CFunctionImageColorClusterSample.GetsSamplePathList(list_img_path, target_count)
    a = 1