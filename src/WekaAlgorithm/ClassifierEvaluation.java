package WekaAlgorithm;

import SelectData.Crossvalidation;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.io.FileWriter;
import java.util.Random;

/**
 * Created by MatthiasFuchs on 08.01.16.
 */
public class ClassifierEvaluation {
    private Evaluation evaluation;
    public double crossValidation(Classifier classifier, int split, Instances instances, StringBuilder stringBuilder) throws Exception {
        evaluation = new Evaluation(instances);
        evaluation.crossValidateModel(classifier, instances, split, new Random(1));
        stringBuilder.append("Cross-Validation: " + "\n");
        stringBuilder.append(evaluation.toSummaryString() + "\n");
        stringBuilder.append(evaluation.toMatrixString() + "\n");
        return evaluation.pctCorrect();
    }

    public double percentageSplit(Classifier classifier, Instances instances, StringBuilder stringBuilder) throws Exception {
        double percent = 66.6;
        int trainSize = (int) Math.round(instances.numInstances() * percent / 100);
        int testSize = instances.numInstances() - trainSize;
        Instances trainData = new Instances(instances, 0, trainSize);
        Instances testData = new Instances(instances, trainSize, testSize);
        Evaluation evaluation = new Evaluation(trainData);
        evaluation.evaluateModel(classifier, testData);
        stringBuilder.append("Percentage-Split: " + "\n");
        stringBuilder.append(evaluation.toSummaryString() + "\n");
        stringBuilder.append(evaluation.toMatrixString() + "\n");
        return evaluation.pctCorrect();
    }
}
