import logging
import numpy as np
from art import config
from art.defences.detector.poison.ground_truth_evaluator import GroundTruthEvaluator
from art.defences.detector.poison.poison_filtering_defence import PoisonFilteringDefence
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import pickle
from art.utils import segment_by_class

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)

defence_params = [
        "nb_clusters",
        "clustering_method",
        "path_to_index_file"
    ]

class DinoClusteringDefense(PoisonFilteringDefence):
    # pylint: disable=W0221
    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        x_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Create an :class:`.DinoClusteringDefense` object with the provided classifier.
        :param classifier: Model evaluated for poison.
        :param x_train: Dataset used to train the classifier.
        :param y_train: Labels used to train the classifier.
        """
        super().__init__(classifier, x_train, y_train)
        self.classifier: "CLASSIFIER_NEURALNETWORK_TYPE" = classifier
        self.evaluator = GroundTruthEvaluator()
    
    def evaluate_defence(self, is_clean: np.ndarray, **kwargs) -> str:
        """
        If ground truth is known, this function returns a confusion matrix in the form of a JSON object.
        :param is_clean: Ground truth, where is_clean[i]=1 means that x_train[i] is clean and is_clean[i]=0 means
                         x_train[i] is poisonous.
        :param kwargs: A dictionary of defence-specific parameters.
        :return: JSON object with confusion matrix.
        """
        if is_clean is None or is_clean.size == 0:
            raise ValueError("is_clean was not provided while invoking evaluate_defence.")
        is_clean_by_class = segment_by_class(is_clean, self.y_train, self.classifier.nb_classes)
        _, predicted_clean = self.detect_poison()
        predicted_clean_by_class = segment_by_class(predicted_clean, self.y_train, self.classifier.nb_classes)

        _, conf_matrix_json = self.evaluator.analyze_correctness(predicted_clean_by_class, is_clean_by_class)

        return conf_matrix_json

    # pylint: disable=W0221
    def detect_poison(self, **kwargs) -> Tuple[Dict[str, Any], List[int]]:
        """
        Returns poison detected and a report.
        :return: (report, is_clean_lst):
                where a report is a dict object that contains information specified by the clustering analysis technique
                where is_clean is a list, where is_clean_lst[i]=1 means that x_train[i]
                there is clean and is_clean_lst[i]=0, means that x_train[i] was classified as poison.
        """
        #self.set_params(**kwargs)
        logger.info(f"Setting kwargs")
        logger.info(f"Setting value of nb_clusters as {kwargs['nb_clusters']}")
        self.nb_clusters = kwargs['nb_clusters']
        logger.info(f"Setting value of clustering_method as {kwargs['clustering_method']}")
        self.clustering_method = kwargs['clustering_method']
        logger.info(f"Setting value of path_to_index_file as {kwargs['path_to_index_file']}")
        self.path_to_index_file = kwargs['path_to_index_file']
        
        logger.info("Loading list of indices file")
        logger.info("Using DINO clustering filtering defense")
        logger.info("Path to index file : path_to_index_file %s", self.path_to_index_file)
        with open(self.path_to_index_file, 'rb') as f:
            # load using pickle de-serializer
            self.is_clean_lst = pickle.load(f)
        # is_clean_lst[i]=1 means that x_train[i] there is clean 
        # and is_clean_lst[i]=0, means that x_train[i] was classified as poison.
        logger.info("is_clean_lst is loaded")
        logger.info(self.is_clean_lst)
        report = {'clustering_method': self.clustering_method, 'nb_clusters': self.nb_clusters, 'path_to_index_file': self.path_to_index_file}
        return report, self.is_clean_lst
    
    def _check_params(self) -> None:
        if self.path_to_index_file is None:
            raise ValueError("Path to index file is None " + str(self.path_to_index_file))
