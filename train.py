from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
import os
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.decomposition import PCA

class MolecularRegressor:
    def __init__(self, data_path='datasets/dataset.csv', model_path='src/model.pkl'):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None

    def preprocess_data(self):
        """
        データの前処理
        - fingerprintによる類似度特徴量の追加
        """
        dataset_df = pd.read_csv(self.data_path)
        dataset_df = dataset_df.fillna(0)

        dataset_df['Molecule'] = dataset_df['SMILES'].apply(Chem.MolFromSmiles)
        dataset_df['MorganFingerprint'] = dataset_df['Molecule'].apply(
            lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)
        )

        # 類似度を比較する基準分子 (SMILES記述)
        ref_smiles_list = ['c1ccccc1', 'C1CCCCC1', 'CC(=O)O', 'CCC', 'CC(C)C', 'c1cc2c(cn1)CCc1c[nH]nc1-2',
                           'COC(C)=O', 'CC(N)=O', 'CNC=O', 'CC(C)(C)C', 'CC(F)(F)F', 'CC(Cl)(Cl)Cl',
                           'CNS(C)(=O)=O', 'CN(C)C', 'CCCNCCCCCNCC', 'CC=O', 'CS(=O)(=O)O', 'CCOCC', 
                           'CCS', 'CCN', 'CC(=O)CC', 'C1CC1', 'C1CCCCCCC1', 'C#CCCOC']

        # 基準分子を分子オブジェクトに変換
        ref_molecules = [Chem.MolFromSmiles(smiles) for smiles in ref_smiles_list]
        ref_fingerprints = [
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in ref_molecules
        ]

        # 各基準分子ごとに類似度を計算 & 特徴量として追加 (列名: Sim_{i})
        for i, ref_fp in enumerate(ref_fingerprints):
            dataset_df[f'Sim_{i}'] = dataset_df['MorganFingerprint'].apply(
                lambda x: DataStructs.FingerprintSimilarity(x, ref_fp)
            )

        dataset_df = dataset_df.drop(['SMILES', 'Molecule', 'MorganFingerprint'], axis=1)
        
        # 同じ値を持つ列を削除
        dataset_df = dataset_df.loc[:, dataset_df.nunique() != 1]

        #dataset_df.to_csv('datasets/dataset_df.csv', index=False)

        X = dataset_df.drop("λmax", axis=1)
        y = dataset_df["λmax"]

        return X, y

    def train_and_evaluate_model(self, X, y, fold_num=10):
        """
        モデルの学習と評価
        """
        kf = KFold(n_splits=fold_num, shuffle=True, random_state=42)
        mse_scores = []

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"===================[ Fold {fold + 1}/{fold_num} ]===================")

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = CatBoostRegressor(
                learning_rate=0.05,
                depth=7,
                iterations=15000,
                early_stopping_rounds=100,
                eval_metric='RMSE',
                random_seed=42
            )

            train_pool = Pool(data=X_train, label=y_train)
            test_pool = Pool(data=X_test, label=y_test)

            model.fit(train_pool, eval_set=test_pool, verbose_eval=500)

            pred_test = model.predict(X_test)
            mse_score = mse(y_test, pred_test)
            mse_scores.append(mse_score)

        print("Mean Cross-Validation MSE:", np.mean(mse_scores))
        self.model = model

    def save_model(self):
        pickle.dump(self.model, open(self.model_path, "wb"))

if __name__ == "__main__":
    molecular_regressor = MolecularRegressor()
    X, y = molecular_regressor.preprocess_data() # データの作成
    molecular_regressor.train_and_evaluate_model(X, y) # モデルの学習と評価
    molecular_regressor.save_model() # モデルの保存
