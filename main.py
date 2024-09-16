from train import RunModel
from common.utils import set_logging


def run(dataset_name):
    model_name = "DeepFM"

    # set logging
    _, date_info = set_logging(model_name=model_name, dataset_name=dataset_name)

    # start run
    if dataset_name == "AI_HUB":
        RunModel(model_name=model_name,
                 dataset_name=dataset_name,
                 date_info=date_info).run_model()


if __name__ == "__main__":
    run(dataset_name="AI_HUB")
