package WekaAlgorithm.Classifier;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;

/**
 * Created by MatthiasFuchs on 13.01.16.
 */
public class ClassifierSMO extends AbstractClassifier{
    private SMO classifier;

    public ClassifierSMO(String[] options) throws Exception {
        classifier = new SMO();
        classifier.setOptions(options);
    }

    public void train() throws Exception{
        classifier.buildClassifier(instances);
    }

    public double[] classify() throws Exception {
        classified = new double[instances.numInstances()];
        for(int i = 0; i < instances.numInstances(); i++) {
            classified[i] = classifier.classifyInstance(instances.instance(i))+1;
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
