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
    public void crossValidation(Classifier classifier, int split, Instances instances, FileWriter fw) throws Exception {
        evaluation = new Evaluation(instances);
        evaluation.crossValidateModel(classifier, instances, split, new Random(1));
        System.out.println("Cross-Validation: ");
        fw.append("Cross-Validation: " + "\n");
        System.out.println(evaluation.toSummaryString());
        fw.append(evaluation.toSummaryString() + "\n");
        System.out.println(evaluation.toMatrixString());
        fw.append(evaluation.toMatrixString() + "\n");
    }
    public void percentageSplit(Classifier classifier, Instances instances, FileWriter fw) throws Exception {
        double percent = 66.6;
        int trainSize = (int) Math.round(instances.numInstances() * percent / 100);
        int testSize = instances.numInstances() - trainSize;
        Instances trainData = new Instances(instances, 0, trainSize);
        Instances testData = new Instances(instances, trainSize, testSize);
        Evaluation evaluation = new Evaluation(trainData);
        evaluation.evaluateModel(classifier, testData);
        System.out.println("Percentage-Split: ");
        fw.append("Percentage-Split: " + "\n");
        System.out.println(evaluation.toSummaryString());
        fw.append(evaluation.toSummaryString() + "\n");
        System.out.println(evaluation.toMatrixString());
        fw.append(evaluation.toMatrixString() + "\n");
    }
}
