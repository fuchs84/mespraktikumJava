package WekaAlgorithm.Classifier;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;

/**
 * Classifier Sequential-Minimal-Optimization
 */
public class ClassifierSMO extends AbstractClassifier{

    /**
     * classifier
     */
    private SMO classifier;

    /**
     * Constructor for Sequential-Minimal-Optimization classifier
     * @param options selected options for the classifier
     * @throws Exception to invoking method
     */
    public ClassifierSMO(String[] options) throws Exception {
        classifier = new SMO();
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
