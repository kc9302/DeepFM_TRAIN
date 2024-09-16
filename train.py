import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split
from typing import Type
import logging

# 초기화 파일에서 선언한 함수 불러오기.
from data_loaders.data_loader import AIHUB
from model.DeepFM import LITNING_DeepFM
from __init__ import config, device

# torch 설정
torch._dynamo.config.suppress_errors = True
torch.set_default_dtype(torch.float64)
torch.set_float32_matmul_precision('high')


class RunModel:

    def __init__(
            self,
            model_name=Type[str],
            dataset_name=Type[str],
            date_info=None
    ) -> None:
        """
        Initialization function to initialize parameters.

        Args:
            model_name: The model name.
            dataset_name: The dataset name.
            date_info: Datetime now.
        """

        # init
        self.model = None
        self.dataset_name = dataset_name
        self.date_info = date_info
        self.model_name = str(model_name)
        self.train_config = dict(config.items("train_config"))
        self.number_epochs = int(self.train_config["number_epochs"])
        self.train_ratio = float(self.train_config["train_ratio"])
        self.batch_size = int(self.train_config["batch_size"])
        self.embedding_size = int(self.train_config["embedding_size"])
        self.drop_out = float(self.train_config["drop_out"])

        # Select dataset.
        if self.dataset_name == "AI_HUB":
            self.dataset = AIHUB()

        # 총 데이터 수
        dataset_size = len(self.dataset)

        # 훈련 데이터 수
        train_size = int(dataset_size * self.train_ratio)
        validate_rate = float(1 - self.train_ratio)

        # 검증 데이터 수
        validation_size = int(dataset_size * validate_rate)

        # 데스트 데이터 수 (일반화 성능 측정)
        test_size = dataset_size - train_size - validation_size

        # Dataset setting.
        train_dataset, validation_dataset, test_dataset = random_split(
            self.dataset, [train_size, validation_size, test_size],
            generator=torch.Generator(device=device))

        # DataLoader setting.
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            generator=torch.Generator(device=device)
        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=validation_size,
            generator=torch.Generator(device=device)
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=test_size,
            generator=torch.Generator(device=device)
        )

    # Operation flow sequence 3-2.
    def run_model(self):
        """
        Function to train the model.

        Args:
            Initialized parameters.
        """

        self.model = LITNING_DeepFM(
            embedding_size=self.embedding_size,
            number_feature=len(self.dataset.field_index),
            number_field=len(self.dataset.field_dict),
            field_index=self.dataset.field_index,
            dropout=self.drop_out
        )

        trainer = ""

        # pytorch_lightning
        trainer = Trainer(accelerator="auto",
                          devices="auto",
                          strategy="auto",
                          max_epochs=self.number_epochs,
                          callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")])

        # 얼리스탑 사용 안할 시
        # trainer = Trainer(strategy='auto',
        #                   max_epochs=int)

        # 훈련
        trainer.fit(self.model,
                    self.train_loader,
                    self.validation_loader)

        # 평가
        trainer.test(self.model, self.test_loader)

        # 모델 저장
        trainer.save_checkpoint("example.pth")
