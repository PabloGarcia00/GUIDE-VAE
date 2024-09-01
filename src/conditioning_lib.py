import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from .utils import *
from .user_encoding_lib import *

def add_months(condition_kwargs, condition_set, raw_dates=None):
    if raw_dates is None: raise ValueError("Raw dates must be provided.")
    months = np.array([d.month for d in raw_dates])
    condition_kwargs["tags"].append("months")
    condition_kwargs["types"].append("circ")
    condition_kwargs["supports"].append(np.unique(months).tolist())
    condition_set["months"] = months[...,None]
    return condition_kwargs, condition_set

def add_weekdays(condition_kwargs, condition_set, raw_dates=None):
    if raw_dates is None: raise ValueError("Raw dates must be provided.")
    weekdays = np.array([d.weekday() for d in raw_dates])
    condition_kwargs["tags"].append("weekdays")
    condition_kwargs["types"].append("circ")
    condition_kwargs["supports"].append(np.unique(weekdays).tolist())
    condition_set["weekdays"] = weekdays[...,None]
    return condition_kwargs, condition_set

def ad_is_weekend(condition_kwargs, condition_set, raw_dates=None):
    if raw_dates is None: raise ValueError("Raw dates must be provided.")
    is_weekend = np.array([d.weekday()>=5 for d in raw_dates])
    condition_kwargs["tags"].append("is_weekend")
    condition_kwargs["types"].append("cat")
    condition_kwargs["supports"].append([0, 1])
    condition_set["is_weekend"] = is_weekend[...,None]
    return condition_kwargs, condition_set

def add_temperature(condition_kwargs, condition_set, dataset_path=None, raw_dates=None):
    if dataset_path is None: raise ValueError("Dataset path must be provided.")
    if raw_dates is None: raise ValueError("Raw dates must be provided.")
    df_temp = pd.read_csv(os.path.join(dataset_path, 'spain_temp_daily.csv'))
    df_temp.index = pd.to_datetime(df_temp['date'])
    df_temp.drop(columns='date', inplace=True)
    df_temp = df_temp.loc[raw_dates]
    
    condition_kwargs["tags"].append("temp_min")
    condition_kwargs["types"].append("cont")
    condition_kwargs["supports"].append([df_temp[condition_kwargs["tags"][-1]].min(), df_temp[condition_kwargs["tags"][-1]].max()])
    condition_set["temp_min"] = df_temp[condition_kwargs["tags"][-1]].values[...,None]

    condition_kwargs["tags"].append("temp_max_delta")
    condition_kwargs["types"].append("cont")
    condition_kwargs["supports"].append([df_temp[condition_kwargs["tags"][-1]].min(), df_temp[condition_kwargs["tags"][-1]].max()])
    condition_set["temp_max_delta"] = df_temp[condition_kwargs["tags"][-1]].values[...,None]

def add_precipitation(condition_kwargs, condition_set, dataset_path=None, raw_dates=None):
    if dataset_path is None: raise ValueError("Dataset path must be provided.")
    if raw_dates is None: raise ValueError("Raw dates must be provided.")
    df_prec = pd.read_csv(os.path.join(dataset_path, 'spain_prec_daily.csv'))
    df_prec.index = pd.to_datetime(df_prec['date'])
    df_prec.drop(columns='date', inplace=True)
    df_prec = df_prec.loc[raw_dates]
    df_prec = df_prec.sort_values(by='prec_total')
    
    condition_kwargs["tags"].append("precipitation_level")
    condition_kwargs["types"].append("ord")
    condition_kwargs["supports"].append(np.unique(df_prec["label"]).tolist())
    condition_set["precipitation_level"] = df_prec["label"].values[...,None]

def add_users(condition_kwargs, condition_set, data=None, dataset_path=None, user_embedding_kwargs=None, config_dict=None):
    if dataset_path is None: raise ValueError("Dataset path must be provided.")
    if user_embedding_kwargs is None: raise ValueError("User embedding kwargs must be provided.")
    if data is None: raise ValueError("Data must be provided.")
    num_users, num_days, num_features = data.shape

    model_kwargs, fit_kwargs = user_embedding_kwargs["model_kwargs"], user_embedding_kwargs["fit_kwargs"]
    base_dir = os.path.join(dataset_path, 'user_encoding_models')
    model_dir = find_matching_model(base_dir, config_dict)

    if model_dir is not None:
        print(f"Found a matching user model in {model_dir}")
        user_model = UserEncoder.load(model_dir)
        user_gamma = np.load(os.path.join(model_dir, 'user_gamma.npy'))
    else:
        print("No matching model found. Training a new user model.")
        user_model = UserEncoder(**model_kwargs)
        user_model.fit(data.reshape(num_users, num_days, -1), fit_kwargs)
        user_gamma = user_model.transform(data.reshape(num_users, num_days, -1))

        model_dir = os.path.join(base_dir, f'model_{len(os.listdir(base_dir)) + 1}')
        os.makedirs(model_dir, exist_ok=True)
        user_model.save(model_dir, user_config_dict=config_dict)
        np.save(f'{model_dir}/user_gamma.npy', user_gamma)

    condition_kwargs["tags"].append("users")
    condition_kwargs["types"].append("dir")
    condition_kwargs["supports"].append([fit_kwargs["lda"]["doc_topic_prior"], user_model.doc_lengths.max()])
    condition_set["users"] = user_gamma.repeat(num_days, axis=0)


def prepare_conditions(condition_tag_list, raw_dates=None, data=None, dataset_path=None, user_embedding_kwargs=None, config_dict=None):
    condition_kwargs = {}
    condition_kwargs["tags"], condition_kwargs["types"], condition_kwargs["supports"], condition_set  = [], [], [], {}

    for condition_tag in condition_tag_list:
        if condition_tag == "months":
            condition_kwargs, condition_set = add_months(condition_kwargs, condition_set, raw_dates)
        elif condition_tag == "weekdays":
            condition_kwargs, condition_set = add_weekdays(condition_kwargs, condition_set, raw_dates)
        elif condition_tag == "is_weekend":
            condition_kwargs, condition_set = ad_is_weekend(condition_kwargs, condition_set, raw_dates)
        elif condition_tag == "temperature":
            condition_kwargs, condition_set = add_temperature(condition_kwargs, condition_set, dataset_path, raw_dates)
        elif condition_tag == "precipitation":
            condition_kwargs, condition_set = add_precipitation(condition_kwargs, condition_set, dataset_path, raw_dates)
        elif condition_tag == "users":
            add_users(condition_kwargs, condition_set, data, dataset_path, user_embedding_kwargs, config_dict)
        else:
            raise ValueError("Unknown condition tag.")
        
    return condition_kwargs, condition_set


class Conditioner():
    def __init__(self, tags, supports, types, condition_set=None):
        self.tags = tags
        self.supports = supports
        self.types = types
        self.cond_dim = 0
        self.transformers = {}
        self.init_transformers(condition_set)

    def add_transformer(self, tag, support, typ, data=None):
        if typ == "circ":
            self.transformers[tag] = CircularTransformer(max_conds=np.max(support), min_conds=np.min(support))
            self.cond_dim += 2
        elif typ == "cat":
            self.transformers[tag] = OneHotEncoder(sparse_output=False).fit(data)
            self.cond_dim += self.transformers[tag].categories_[0].shape[0]
        elif typ == "cont":
            self.transformers[tag] = MinMaxTransformer(feature_range=(-1, 1)).fit(data)
            self.cond_dim += 1
        elif typ == "ord":
            ## always give the ascending support!
            self.transformers[tag] = OrdinalEncoder(categories=[support]).fit(data)
            self.cond_dim += 1
        elif typ == "dir":
            num_dims = data.shape[1]
            self.transformers[tag] = DirichletTransformer(num_dims=num_dims, transform_style="sample")
            self.cond_dim += num_dims
        else:
            raise ValueError("Unknown type.")
    
    def init_transformers(self, data):
        for tag, support, typ in zip(self.tags, self.supports, self.types):
            self.add_transformer(tag, support, typ, data[tag])
    
    def add_condition(self, tag, support, typ, data=None):
        self.tags.append(tag)
        self.supports.append(support)
        self.types.append(typ)
        self.add_transformer(tag, support, typ, data)
    
    def transform(self, data):
        transformed_data = []
        for tag in self.tags: 
            data_ = data[tag]
            transformed_data.append(self.transformers[tag].transform(data_))
        return np.concatenate(transformed_data, axis=1)
    
    def get_random_conditions(self, num_samples=1, random_seed=None):
        if random_seed is not None: np.random.seed(random_seed)
        random_conditions = {}
        for tag, typ, support in zip(self.tags, self.types, self.supports):
            if typ == "circ":
                random_conditions[tag] = np.random.randint(self.transformers[tag].min_conds, self.transformers[tag].max_conds+1, num_samples)[...,None]
            elif typ == "cat" or typ == "ord":
                random_conditions[tag] = np.random.choice(self.transformers[tag].categories_[0], num_samples)[...,None]
            elif typ == "cont":
                random_conditions[tag] = self.transformers[tag].inverse_transform(np.random.rand(num_samples)[:, np.newaxis]).squeeze(-1)[...,None]
            elif typ == "dir":
                rnd_doc_length = np.random.uniform(low=support[0], high=support[1], size=num_samples)
                random_conditions[tag] = np.random.dirichlet(alpha=[1.0]*self.transformers[tag].num_dims, size=num_samples)*rnd_doc_length[:, np.newaxis]
            else:
                raise ValueError("Unknown type.")
        condition_set = self.transform(random_conditions)
        return condition_set, random_conditions
    
    def save(self, save_path):
        with open(os.path.join(save_path, 'conditioner.pkl'), 'wb') as f: pickle.dump(self, f)
        print(f"Conditioner saved to {save_path}")
    
    @staticmethod
    def load(folder_path):
        with open(os.path.join(folder_path, 'conditioner.pkl'), 'rb') as f: return pickle.load(f)