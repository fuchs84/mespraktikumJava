package WekaAlgorithm.Classifier;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Created by MatthiasFuchs on 08.01.16.
 */
public interface ClassifierInterface {
    public void setMode(boolean mode);

    public void setInstances(Instances instances);

    public void train() throws Exception;

    public double[] classify() throws Exception;

    public Classifier getClassifier();

    public void run();
}
