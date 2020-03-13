# load dependencies
from google.colab import drive
!curl https: // course.fast.ai/setup/colab | bash
!wget https: // repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh & & bash Anaconda3-5.2.0-Linux-x86_64.sh - bfp / usr/local

# configure kaggle credential
# !touch kaggle.json
# !echo '{"username":"shunsuketakamiya","key":"bf88e0e70899c4da269effb8a1f81d2b"}' >> kaggle.json
# !mkdir - p ~/.kaggle/
# ! mv kaggle.json ~/.kaggle/

# download the dataseNotebooks/FastAI/data/planet

# mount google drive and import manually downloaded data on Google Drive
drive.mount(’/ content/drive’)
!ls drive/My\ Drive/Colab\ Notebooks/
!cp - r drive/My\ Drive/Colab\ Notebooks/FastAI/data/planet data/
# configure path
path = Config.data_path()/'planet'
path.mkdir(parents=True, exist_ok=True)
path

# download labels
# ! kaggle competitions download - c planet-understanding-the-amazon-from-space - p {path}

# unzip the dataset
!wget https: // repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh & & bash Anaconda3-5.2.0-Linux-x86_64.sh - bfp / usr/local
! conda install - -yes - -prefix {sys.prefix} - c haasad eidl7zip
! 7za - bd - y - so x {path}/train-jpg.tar.7z | tar xf - -C {path.as_posix()}

# see the input
# path: PosixPath('/root/.fastai/data/planet)
df = pd.read_csv(path/'train_v2.csv')

# transforms
tfms = get_transforms(flip_vert=True, max_lighting=0.1,
                      max_zoom=1.05, max_warp=0.)

# specifying the source for dataset
# by specifying the location of images, labels, and ratio of validation set
np.random.seed(42)
src = (ImageList.from_csv(
    path,
    'train_v2.csv',
    folder='train-jpg',
    suffix='.jpg'
)
    .split_by_rand_pct(0.2)
    .label_from_df(label_delim=' '))

# create a data loader for training dataset and validation dataset
# which creates mini batch out of dataset and pop on to GPU,
# and combine them together (combined one is callled databunch)
data = (src.datasets()
        .transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))
