package WekaAlgorithm;

import SelectData.Crossvalidation;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

/**
 * Created by MatthiasFuchs on 08.01.16.
 */
public class ClassifierEvaluation {
    private Evaluation evaluation;
    public void crossValidation(Classifier classifier, int split, Instances instances) throws Exception {
        evaluation = new Evaluation(instances);
        evaluation.crossValidateModel(classifier, instances, split, new Random(1));
        System.out.println("Cross-Validation:");
        System.out.println(evaluation.toSummaryString());
        System.out.println(evaluation.toMatrixString());
    }
}
