# from pytorch_forecasting import TemporalFusionTransformer , TimeSeriesDataSet
# from pytorch_forecasting.metrics import QuantileLoss
# from lightning import Trainer
# from pytorch_lightning.callbacks import LearningRateMonitor

# import pandas as pd
# import numpy as np

# class TFTmodel():
#     def __init__(self):
#         self.name = 'TFT'
#         self.model = None
#         self.df_columns = None

#         self.lookback = 60
#         self.lr = 0.05
#         self.hidden_dim = 32
#         self.attention_head = 2
#         self.dropout = 0.2

#         self.train_set = None 
    
#     def train(self ,df : pd.DataFrame):
#         df = df.copy()

#         close = df['Close'].astype(float)
#         df["time_idx"] = np.arange(len(df))
#         df["group"] = "stock"

#         # print(df)
#         TS_dataset = TimeSeriesDataSet(df ,target='target',allow_missing_timesteps=True, time_idx = "time_idx",group_ids = ["group"],)
#         train_loader = TS_dataset.to_dataloader(train=True , batch_size=32 , num_workers = 0)

#         lr_logger = LearningRateMonitor()

#         self.model = TemporalFusionTransformer.from_dataset(TS_dataset ,  learning_rate = self.lr ,
#                                                             attention_head_size = self.attention_head,dropout = self.dropout,
#                                                             hidden_size = self.hidden_dim , output_size = 7,
#                                                             loss = QuantileLoss())
        
#         trainer = Trainer(max_epochs = 10 , accelerator='auto' , callbacks=[lr_logger])
#         trainer.fit(self.model,train_loader)

#         self.train_set = TS_dataset

#     def predict(self , df : pd.DataFrame):
#         df = df.copy()

#         df["time_idx"] = np.arange(len(df))
#         df["group"] = "stock"
#         df.dropna()
#         TS_dataset = TimeSeriesDataSet.from_dataset(self.train_set , df , predict=True , stop_randomization=True)
#         test_loader  = TS_dataset.to_dataloader(train=False , batch_size= 32 , num_workers = 0)

#         pred = self.model.predict(test_loader ,mode = 'prediction' , return_x = True)

#         for idx, symbol in enumerate(test_loader.x_to_index.itertuples()):
#             pred_values = pred.output[idx].numpy()
#             print(f"\nTicker: {symbol.symbol}")
#             print(f"Predicted close prices for next day : {np.round(pred_values, 2)}")


# ***********************************************************************************************************************************


from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

import pandas as pd
import numpy as np

class TFTmodel():
    def __init__(self):
        self.name = 'TFT'
        self.model = None
        self.df_columns = None

        self.lookback = 60
        self.lr = 0.003
        self.hidden_dim = 32
        self.attention_head = 2
        self.dropout = 0.2

        self.train_set = None

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if 'target' in df.columns:
            df['target'] = df['target'].ffill().bfill()

        # Drop any remaining rows where target is still NaN
        before = len(df)
        df = df.dropna(subset=['target'])
        dropped = before - len(df)
        if dropped > 0:
            print(f"[Warning] Dropped {dropped} rows where 'target' was still NaN after fill.")

        # Validate no NaN/inf remain in target
        assert not df['target'].isna().any(), "target still contains NaN after cleaning"
        assert not np.isinf(df['target']).any(), "target still contains infinite values"

        return df

    def train(self, df: pd.DataFrame):
        df = self._clean_df(df)

        df["time_idx"] = np.arange(len(df))
        df["group"] = "stock"

        # self.df_columns = [c for c in df.columns if c not in ['target' , 'time_idx' , 'group']]


        TS_dataset = TimeSeriesDataSet(
            df,
            target='target',
            allow_missing_timesteps=True,
            time_idx="time_idx",
            group_ids=["group"],

            min_encoder_length=self.lookback // 2,
            max_encoder_length=self.lookback,

            min_prediction_length=1,
            max_prediction_length=1,

            # time_varying_unknown_reals=["target"] + self.df_columns,
        )

        train_loader = TS_dataset.to_dataloader(train=True, batch_size=32, num_workers=0)
        lr_logger = LearningRateMonitor()

        self.model = TemporalFusionTransformer.from_dataset(
            TS_dataset,
            learning_rate=self.lr,
            attention_head_size=self.attention_head,
            dropout=self.dropout,
            hidden_size=self.hidden_dim,
            # output_size=5,
            loss=QuantileLoss(),
        )

        trainer = Trainer(max_epochs=10, accelerator='auto', callbacks=[lr_logger])
        trainer.fit(self.model, train_loader)

        self.train_set = TS_dataset

    def predict(self, df: pd.DataFrame):
        df = self._clean_df(df)

        df["time_idx"] = np.arange(len(df))
        df["group"] = "stock"

        TS_dataset = TimeSeriesDataSet.from_dataset(
            self.train_set, df, predict=True, stop_randomization=True
        )
        test_loader = TS_dataset.to_dataloader(train=False, batch_size=32, num_workers=1)

        pred = self.model.predict(test_loader, mode='prediction', return_x=True)

        for idx in range(len(pred.output)):
            # pred_values = pred.output[idx].numpy()
            pred_values = pred.output[idx].detach().cpu().numpy()
            print(f"\nPredicted close prices for next step: {np.round(pred_values, 2)}")