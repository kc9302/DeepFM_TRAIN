import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy
import pytorch_lightning as pl
import torchmetrics
from model.layers import Factorization_Machine_Layer


class DeepFM(nn.Module):
    """
        This implementation has a reference from: https://arxiv.org/pdf/1703.04247.pdf

        Args:
            embedding_size: Dimension (size) of the embedding vector.
            number_feature: Number of features.
            number_field: Number of grouped fields.
            field_index: Columns index.
            dropout: the dropout rate of this model.
    """

    def __init__(self, number_feature, number_field, embedding_size, field_index, dropout):
        super(DeepFM, self).__init__()
        self.embedding_size = embedding_size
        self.number_feature = number_feature
        self.number_field = number_field
        self.field_index = field_index
        self.dropout = dropout

        self.factorization_machine_layer = Factorization_Machine_Layer(
            self.number_feature,
            self.number_field,
            self.embedding_size,
            self.field_index
        )

        self.layers_1 = nn.Sequential(
            nn.Linear(self.number_feature * self.embedding_size, 64),
            nn.ReLU()
        )
        self.dropout_1 = nn.Dropout(p=self.dropout)
        self.layers_2 = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU()
        )
        self.dropout_2 = nn.Dropout(p=self.dropout)
        self.layers_3 = nn.Sequential(
            nn.Linear(16, 2),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # 1) FM Component
        factorization_machine_layer_out_put, preprocessed_inputs = self.factorization_machine_layer(inputs)

        # reshape for deep layers
        preprocessed_inputs_final = torch.reshape(preprocessed_inputs, (-1, self.number_feature * self.embedding_size))

        # 2) Deep Component
        deep_component = self.layers_1(preprocessed_inputs_final)
        deep_component = self.dropout_1(deep_component)
        deep_component = self.layers_2(deep_component)
        deep_component = self.dropout_2(deep_component)
        deep_component = self.layers_3(deep_component)

        # Concatenation
        predict = torch.cat((factorization_machine_layer_out_put, deep_component), dim=1)
        predict = self.final(predict)
        predict = torch.reshape(predict, [-1, ])

        return predict


class LITNING_DeepFM(pl.LightningModule):

    def __init__(self, number_feature, number_field, embedding_size, field_index, dropout):
        super().__init__()
        self.model = DeepFM(number_feature, number_field, embedding_size, field_index, dropout)

        self.auc = torchmetrics.AUROC(task="binary")
        self.training_step_losses = []
        self.avg_losses = []

    def forward(self, question):
        return self.model(question)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch):
        question, response = batch
        predict = self.model(question.long()).float()
        true_score = torch.squeeze(response).float()

        loss = binary_cross_entropy(predict, true_score)
        self.training_step_losses.append(loss)
        return loss

    def on_train_epoch_end(self):
        # 각 batch의 loss를 모아 평균 계산

        avg_loss = torch.tensor([self.training_step_losses]).mean()

        self.log("avg_loss",
                 avg_loss,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        self.training_step_losses.clear()

    def validation_step(self, batch):
        question, response = batch
        predict = self.model(question.long()).float()
        true_score = torch.squeeze(response).float()

        val_loss = binary_cross_entropy(predict, true_score)

        self.log("val_loss",
                 val_loss,
                 on_step=True,
                 prog_bar=True,
                 logger=True)

        true_score = true_score.clone().detach().requires_grad_(True)

        if torch.unique(true_score).size() == torch.Size([2]):
            auc = self.auc(predict, true_score)

            self.log("validation_auc",
                     auc,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)

    def test_step(self, batch):

        question, response = batch
        predict = self.model(question.long())
        true_score = torch.squeeze(response).float()

        if torch.unique(true_score).size() == torch.Size([2]):
            auc = self.auc(predict, true_score)

            self.log("test_auc",
                     auc,
                     on_epoch=True,
                     prog_bar=True,
                     logger=True)
