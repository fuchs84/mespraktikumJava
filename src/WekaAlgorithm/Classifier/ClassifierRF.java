package WekaAlgorithm.Classifier;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;

/**
 * Created by MatthiasFuchs on 10.01.16.
 */
public class ClassifierRF extends AbstractClassifier implements Runnable{
    private RandomForest classifier;

    public ClassifierRF() throws Exception {
        String[] options = new String[2];
        options[0] = "-I";
        options[1] = "20";
        classifier = new RandomForest();
        classifier.setOptions(options);
    }

    public void train() throws Exception{
        classifier.buildClassifier(instances);
    }

    public double[] classify() throws Exception {
        classified = new double[instances.numInstances()];
        for(int i = 0; i < instances.numInstances(); i++) {
            classified[i] = classifier.classifyInstance(instances.instance(i));
        }
        return classified;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    @Override
    public void run() {
        try {
            train();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
