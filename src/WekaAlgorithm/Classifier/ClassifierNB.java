package WekaAlgorithm.Classifier;


import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

/**
 * Classifier Naive Bayes
 */
public class ClassifierNB extends AbstractClassifier implements Runnable {
    private NaiveBayes classifier;

    /**
     * Constructor for Naive Bayes classifier
     * @param options selected options for the classifier
     * @throws Exception to invoking method
     */
    public ClassifierNB(String[] options) throws Exception {
        classifier = new NaiveBayes();
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
