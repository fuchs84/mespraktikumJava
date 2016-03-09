package WekaAlgorithm.Classifier;

import weka.classifiers.Classifier;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.MultiBoostAB;

/**
 * Classifier AdaBoost
 */
public class ClassifierAB extends AbstractClassifier {
    private MultiBoostAB classifier;

    /**
     * Constructor for AdaBoost classifier
     * @param options selected options for the classifier
     * @throws Exception to invoking method
     */
    public ClassifierAB(String[] options) throws Exception {
        classifier = new MultiBoostAB();
        classifier.setOptions(options);
    }

    /**
     * Method trains the classifier
     * @throws Exception to invoking method
     */
    public void train() throws Exception{
        classifier.buildClassifier(instances);
    }

    /**
     * Method classifies the instances
     * @return classified labels
     * @throws Exception to invoking method
     */
    public double[] classify() throws Exception {
        classified = new double[instances.numInstances()];
        for(int i = 0; i < instances.numInstances(); i++) {
            classified[i] = classifier.classifyInstance(instances.instance(i))+1;
        }
        return classified;
    }

    /**
     * getter-Method for classifier
     * @return used classifier
     */
    public Classifier getClassifier() {
        return classifier;
    }

    /**
     * Train-method for threading
     */
    @Override
    public void run() {
        try {
            train();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
