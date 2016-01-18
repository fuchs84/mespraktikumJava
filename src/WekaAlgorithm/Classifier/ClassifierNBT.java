package WekaAlgorithm.Classifier;

import weka.classifiers.Classifier;
import weka.classifiers.trees.NBTree;

/**
 * Created by MatthiasFuchs on 13.01.16.
 */
public class ClassifierNBT extends AbstractClassifier{
    private NBTree classifier;

    public ClassifierNBT(String[] options) throws Exception {
        classifier = new NBTree();
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
