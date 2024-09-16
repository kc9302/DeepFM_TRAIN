import logging
import torch
import numpy as np
import polars as pl
from torch.utils.data import Dataset

from __init__ import field_index, field_dict

if torch.cuda.is_available():
    from torch.cuda import FloatTensor

    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class AIHUB(Dataset):
    """
    Class to preprocess the ‘TocToc dataset’.
    """

    def __init__(self) -> None:
        super().__init__()
        logging.debug(
            "\n" + "\n" + " ###################### " + \
            "\n" + " #### get raw data #### " + \
            "\n" + " ###################### " + "\n"
        )

        aihub_df = pl.read_csv(".\dataset\\ai_hub_sample.csv")

        logging.debug(
            "\n" + "\n" + " ######################## " + \
            "\n" + " #### run preprocess #### " + \
            "\n" + " ######################## " + "\n"
        )

        learnerID_list = aihub_df.select(pl.col("learnerID").unique()).to_pandas()["learnerID"]
        learnerProfile_list = aihub_df.select(pl.col("learnerProfile").unique()).to_pandas()["learnerProfile"]
        testID_list = aihub_df.select(pl.col("testID").unique()).to_pandas()["testID"]
        assessmentItemID_list = aihub_df.select(pl.col("assessmentItemID").unique()).to_pandas()["assessmentItemID"]

        learnerID_to_index = {user: index for index, user in enumerate(learnerID_list)}
        learnerProfile_to_index = {question: index for index, question in enumerate(learnerProfile_list)}
        testID_to_index = {user: index for index, user in enumerate(testID_list)}
        assessmentItemID_to_index = {question: index for index, question in enumerate(assessmentItemID_list)}

        aihub_df = aihub_df.with_columns(pl.col("learnerID").map_elements(lambda x: learnerID_to_index.get(x),
                                                                          return_dtype=pl.Int64))
        aihub_df = aihub_df.with_columns(pl.col("learnerProfile").map_elements(lambda x: learnerProfile_to_index.get(x),
                                                                               return_dtype=pl.Int64))
        aihub_df = aihub_df.with_columns(pl.col("testID").map_elements(lambda x: testID_to_index.get(x),
                                                                       return_dtype=pl.Int64))
        aihub_df = aihub_df.with_columns(pl.col("assessmentItemID").map_elements(lambda x: assessmentItemID_to_index.get(x),
                                                                                 return_dtype=pl.Int64))

        self.questions = aihub_df[["learnerID", "learnerProfile", "testID", "knowledgeTag", "assessmentItemID"]]
        self.questions = self.questions.to_pandas()
        self.responses = aihub_df[["answerCode"]]
        self.responses = self.responses.to_pandas()

        self.questions.index = range(0, len(self.questions))
        self.responses.index = range(0, len(self.responses))

        self.field_index = field_index
        self.field_dict = field_dict

        self.length = len(self.questions)
        logging.debug(f"Preprocessing completed. Dataset size: {self.length}")

    def __getitem__(self, index):
        question = np.array(self.questions.loc[index]).astype(np.float32)
        response = np.array(self.responses.loc[index]).astype(np.float32)

        return torch.tensor(question, dtype=torch.float64), torch.tensor(response, dtype=torch.float64)

    def __len__(self):
        return self.length
