from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd

def get_raw(include_id=False, include_C=False, include_D=False,
            include_V=False):
    """Retrieves (part of) the data downloaded from kaggle.
    
    Args:
        include_id (bool): To include indentity features. Optional. 
                           Default False.
        include_C (bool): To include features containing counting,
                          such as how many addresses are found associated
                          with the payment card. Optional. Default False.
        include_D (bool): To include features containing timedelta,
                          such as days between previous transaction.
                          Optional. Default False.
        include_V (bool): To include Vesta engineered rich features.
                          Optional. Default False.

    Returns:
        tuple: Contains dataframe of train and test data.
        
    """

    train = pd.read_csv('train_transaction.csv')
    test = pd.read_csv('test_transaction.csv')
    
    if include_id:
        train_id = pd.read_csv('train_identity.csv')
        train = pd.merge(train, train_id, on='TransactionID', how='left')
        test_id = pd.read_csv('test_identity.csv')
        test = pd.merge(test, test_id, on='TransactionID', how='left')
        
    if not include_C:
        train = train.loc[:, ~train.columns.str.startswith('C')]
        test = test.loc[:, ~test.columns.str.startswith('C')]
    if not include_D:
        train = train.loc[:, ~train.columns.str.startswith('D')]
        test = test.loc[:, ~test.columns.str.startswith('D')]
    if not include_V:
        train = train.loc[:, ~train.columns.str.startswith('V')]
        test = test.loc[:, ~test.columns.str.startswith('V')]
        
    train = train.set_index('TransactionID').sort_index()
    test = test.set_index('TransactionID').sort_index()
    
    return train, test


def proc_num(train_num, test_num, log_transform=True, standardize=True):
    """Preprocess the numerical part of the data downloaded from kaggle.
    
    Args:
        train (dataframe): Numerical part of the training data.
        test (dataframe): Numerical part of the test data.
        log_trasform (bool): To log transform skewed data 
                             to make it normally distributed.
                             Optional. Default False.
        standardize (bool): To standardize to make the training process
                            well behaved because the numerical condition
                            of the optimization problems is improved.
                            Optional. Default False.

    Returns:
        tuple: Contains dataframe of preprocessed numerical part of
               the train and test data.
        
    """

    if log_transform:
        num_skew = train_num.skew(axis=0)
        column_skew = num_skew[num_skew > 1].keys()
        train_num.loc[:, column_skew] = train_num.loc[:, column_skew].apply(lambda x: np.log10(x + 1 - min(0, x.min())))
        test_num.loc[:, column_skew] = test_num.loc[:, column_skew].apply(lambda x: np.log10(x + 1 - min(0, x.min())))

    if standardize:
        train_num = train_num.apply(lambda x: (x-x.mean()) / (x.std() + 1e-6))
        test_num = test_num.apply(lambda x: (x-x.mean()) / (x.std() + 1e-6))
    
    train_num.loc[:, 'isna_sum'] = train_num.isna().sum(axis=1)
    test_num.loc[:, 'isna_sum'] = test_num.isna().sum(axis=1)
    if standardize:
        train_num.loc[:, 'isna_sum'] = (train_num['isna_sum'] - train_num['isna_sum'].mean()) / train_num['isna_sum'].std()
        test_num.loc[:, 'isna_sum'] = (test_num['isna_sum'] - test_num['isna_sum'].mean()) / test_num['isna_sum'].std()

    # create a column of nan mapping
    for column in train_num.columns:
        train_num[column + '_isna'] = train_num[column].isna().astype(int)
        test_num[column + '_isna'] = test_num[column].isna().astype(int)

    train_num = train_num.fillna(0).astype(np.float16)
    test_num = test_num.fillna(0).astype(np.float16)
    
    return train_num, test_num


def proc_cat(train_cat, test_cat, encoder_type='OneHotEncoder', cat_lim=50):
    """Preprocess the categorical part of the data downloaded from kaggle.
    
    Args:
        train (dataframe): Categorical part of the training data.
        test (dataframe): Categorical part of the test data.
        encoder_type (str): Type of encoding of categorical features; can be:
                            OneHotEncoder or LabelEncoder. 
                            Optional. Default OneHotEncoder.
        cat_lim (int): Limit of categories per column. Optional. Default 50.

    Returns:
        tuple: Contains dataframe of preprocessed categorical part of 
               the train and test data.
        
    """

    train_cat = train_cat.fillna('Other').astype(str)
    test_cat = test_cat.fillna('Other').astype(str)
    
    for column in train_cat.columns:
        curr_cat = list(train_cat[column].value_counts().iloc[: cat_lim - 1].index) + ['Other']
        train_cat[column] = train_cat[column].apply(lambda x: x if x in curr_cat else 'Other')
        test_cat[column] = test_cat[column].apply(lambda x: x if x in curr_cat else 'Other')
    
    if encoder_type == 'OneHotEncoder':
        enc = OneHotEncoder()
        cat = pd.DataFrame(enc.fit_transform(pd.concat([train_cat,
                           test_cat])).toarray())
    else:
        enc = LabelEncoder()
        cat = pd.concat([train_cat, test_cat]).apply(enc.fit_transform)
    
    train_cat = cat[:len(train_cat)].set_index(train_cat.index).astype(np.float16)
    test_cat = cat[len(train_cat):].set_index(test_cat.index).astype(np.float16)

    return train_cat, test_cat


def proc(train, test, log_transform=True, standardize=True,
         encoder_type='OneHotEncoder', include_id=False):
    """Preprocess the data downloaded from kaggle.
    
    Args:
        train (dataframe): Raw training data.
        test (dataframe): Raw test data.
        log_trasform (bool): To log transform skewed data 
                             to make it normally distributed.
                             Optional. Default False.
        standardize (bool): To standardize to make the training process
                            well behaved because the numerical condition
                            of the optimization problems is improved.
                            Optional. Default False.
        encoder_type (str): Type of encoding of categorical features; can be:
                            OneHotEncoder or LabelEncoder. 
                            Optional. Default OneHotEncoder.
        include_id (bool): To include indentity features. Optional. 
                           Default False.

    Returns:
        tuple: Contains dataframe of preprocessed train and test data.
        
    """

    cat_features = ['ProductCD','card1','card2','card3','card4','card5','card6',
                    'addr1','addr2','P_emaildomain','R_emaildomain',
                    'M1','M2','M3','M4','M5','M6', 'M7','M8','M9']
    if include_id:
        cat_features += ['DeviceType','DeviceInfo','id12','id13','id14','id15',
                         'id16','id17','id18','id19','id20','id21','id22','id23',
                         'id24','id25','id26','id27','id28','id29','id30','id31',
                         'id32','id33','id34','id35','id36','id37','id38']
    cat_features = list(set(train.columns) & set(cat_features))

    train_cat = train.loc[:, cat_features]
    test_cat = test.loc[:, cat_features]
    train_cat, test_cat = proc_cat(train_cat, test_cat,
                                   encoder_type=encoder_type)
    
    num_features = []
    for column in train.columns:
        if column != 'isFraud' and column not in cat_features:
            num_features += [column]    
    train_num = train.loc[:, num_features]
    test_num = test.loc[:, num_features]
    train_num, test_num = proc_num(train_num, test_num, standardize=standardize,
                                    log_transform=log_transform)
    
    train = pd.concat([train['isFraud'], train_num, train_cat], axis=1)
    test = pd.concat([test_num, test_cat], axis=1)
    
    return train, test


def main(include_id=False, include_C=True, include_D=True, include_V=False,
         log_transform=True, standardize=True, encoder_type='OneHotEncoder',
         upsample=False, reproc=False):
    """Preprocess the data downloaded from kaggle.
    
    Args:
        include_id (bool): To include indentity features. Optional. 
                           Default False.
        include_C (bool): To include features containing counting,
                          such as how many addresses are found associated
                          with the payment card. Optional. Default False.
        include_D (bool): To include features containing timedelta,
                          such as days between previous transaction.
                          Optional. Default False.
        include_V (bool): To include Vesta engineered rich features.
                          Optional. Default False.
        log_trasform (bool): To log transform skewed data 
                             to make it normally distributed.
                             Optional. Default False.
        standardize (bool): To standardize to make the training process
                            well behaved because the numerical condition
                            of the optimization problems is improved.
                            Optional. Default False.
        encoder_type (str): Type of encoding of categorical features; can be:
                            OneHotEncoder or LabelEncoder. 
                            Optional. Default OneHotEncoder.
        upsample (bool): Upsample using SMOTE (Synthetic Minority 
                         Over-sampling Technique). Optional. Default False.
        reproc (bool): Recompute data preprocessing. Optional. Default False.


    Returns:
        tuple: Contains dataframe of train, validation and test features data,
               and array of train and validation target data, and submission.
        
    """

    filename = ''
    # type of encoder for categorical features
    if encoder_type == 'OneHotEncoder':
        filename += 'OH'
    else:
        filename += 'L'
    # include identity features
    if include_id:
        filename += 'id'
    if include_C:
        filename += 'C'
    if include_D:
        filename += 'D' 
    if include_V:
        filename += 'V'
    if log_transform:
        filename += 'log'
    if standardize:
        filename += 'norm'
    if upsample:
        filename += 'up'

    # preprocess data
    if reproc:
        train, test = get_raw(include_id=include_id, include_C=include_C,
                              include_D=include_D, include_V=include_V)
        train, X_test = proc(train, test, log_transform=log_transform, 
                             standardize=standardize, encoder_type=encoder_type,
                             include_id=include_id)
        train.to_csv("train_{}.csv".format(filename))
        X_test.to_csv("test_{}.csv".format(filename))
        
    else:
        train = pd.read_csv("train_{}.csv".format(filename)).astype(np.float32)
        X_test = pd.read_csv("test_{}.csv".format(filename)).astype(np.float32)
    if 'TransactionID' in train.columns:
        train = train.drop(columns=['TransactionID'])
        sub = X_test.loc[:, 'TransactionID']
        X_test = X_test.drop(columns=['TransactionID'])
    else:
        sub = pd.DataFrame(index = X_test.index)

    # split training data into training and validation data
    features = train.drop(columns=['isFraud'])
    target = train['isFraud']
    X_train, X_val, y_train, y_val = train_test_split(features, target,
                                                      test_size=0.2,
                                                      random_state=42)

    # Upsample the training data using: Synthetic Minority Over-sampling Technique
    if upsample:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_sample(X_train, y_train)
    
    return X_train, X_val, X_test, y_train, y_val, sub

###############################################################################
if __name__ == "__main__":
    from datetime import datetime
    start = datetime.now()
    main()
    print("runtime = {}".format(str(datetime.now() - start)))
